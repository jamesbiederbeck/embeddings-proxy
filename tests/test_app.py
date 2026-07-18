import fakeredis.aioredis
import httpx
import pytest

import app as app_module
from registry import HealthStatus, Registry
from vector_store import point_id


class StubBackend:
    type = "stub"

    def __init__(
        self, name="stub", model="stub-model", vector=None, healthy=True, embedding_task=None
    ):
        self.name = name
        self.model = model
        self.embedding_task = embedding_task
        self.vector = vector or [1.0, 2.0, 3.0]
        self.calls = 0
        self.healthy = healthy

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self.vector

    async def check(self) -> None:
        if not self.healthy:
            raise RuntimeError("backend down")


class StubVectorStore:
    type = "stub-store"

    def __init__(self, name="store", model="stub-model", collection="test-collection"):
        self.name = name
        self.model = model
        self.collection = collection
        self.base_metadata = {}
        self.upserted = []
        self.search_results = []
        self.scopes_set = []
        self.deleted = []
        self.search_scopes = []

    async def upsert(self, text, vector, metadata=None, scopes=None):
        self.upserted.append((text, vector, metadata))

    async def search(self, vector, top_k, scope=None):
        self.search_scopes.append(scope)
        return self.search_results

    async def set_scopes(self, point_id, scopes):
        self.scopes_set.append((point_id, scopes))

    async def delete(self, point_id):
        self.deleted.append(point_id)

    async def check(self) -> None:
        pass


def register(reg, *, backend=None, store=None):
    """Add a stub to the registry, marked healthy (as if the poller had seen it up)."""
    if backend is not None:
        reg.embedding.append(backend)
        reg.health[backend.name] = HealthStatus(healthy=True, error=None)
    if store is not None:
        reg.stores.append(store)
        reg.health[store.name] = HealthStatus(healthy=True, error=None)


@pytest.fixture
def stub_backend(monkeypatch):
    backend = StubBackend()
    reg = Registry([], [])
    register(reg, backend=backend)
    monkeypatch.setattr(app_module, "registry", reg)
    monkeypatch.setattr(
        app_module, "cache", fakeredis.aioredis.FakeRedis(decode_responses=True)
    )
    return backend


@pytest.fixture
def stub_vector_store(stub_backend):
    store = StubVectorStore()
    register(app_module.registry, store=store)
    return store


@pytest.fixture
async def api(stub_backend):
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_cache_miss_then_hit(api, stub_backend):
    first = await api.post("/vectors", json={"text": "hello"})
    assert first.status_code == 200
    assert first.json() == {
        "text": "hello",
        "vector": [1.0, 2.0, 3.0],
        "dim": 3,
        "cached": False,
    }

    second = await api.post("/vectors", json={"text": "hello"})
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["vector"] == [1.0, 2.0, 3.0]

    # The backend was only hit once; the second request came from Redis.
    assert stub_backend.calls == 1


async def test_different_texts_are_cached_separately(api, stub_backend):
    await api.post("/vectors", json={"text": "one"})
    await api.post("/vectors", json={"text": "two"})
    assert stub_backend.calls == 2


async def test_search_returns_501_without_vector_store(api):
    resp = await api.post("/search", json={"text": "hello"})
    assert resp.status_code == 501


async def test_search_reshapes_to_documents(api, stub_backend, stub_vector_store):
    stub_vector_store.search_results = [
        {"id": "x", "score": 0.9, "payload": {"text": "hello", "source": "doc1"}}
    ]
    resp = await api.post("/search", json={"text": "hi"})
    assert resp.status_code == 200
    assert resp.json() == {
        "query": "hi",
        "results": [
            {
                "page_content": "hello",
                "metadata": {"source": "doc1"},
                "locations": [],
                "score": 0.9,
            }
        ],
    }


async def test_search_with_no_extra_payload_fields_has_empty_metadata(
    api, stub_backend, stub_vector_store
):
    stub_vector_store.search_results = [{"id": "x", "score": 0.5, "payload": {"text": "hello"}}]
    resp = await api.post("/search", json={"text": "hi"})
    assert resp.json()["results"][0]["metadata"] == {}


async def test_vectors_passes_metadata_to_upsert(api, stub_backend, stub_vector_store):
    resp = await api.post(
        "/vectors", json={"text": "hello", "metadata": {"source": "doc1"}}
    )
    assert resp.status_code == 200
    assert stub_vector_store.upserted == [("hello", [1.0, 2.0, 3.0], {"source": "doc1"})]


async def test_vectors_without_metadata_upserts_none(api, stub_backend, stub_vector_store):
    resp = await api.post("/vectors", json={"text": "hello"})
    assert resp.status_code == 200
    assert stub_vector_store.upserted == [("hello", [1.0, 2.0, 3.0], None)]


async def test_vectors_skips_upsert_on_cache_hit(api, stub_backend, stub_vector_store):
    await api.post("/vectors", json={"text": "hello"})
    await api.post("/vectors", json={"text": "hello"})

    assert len(stub_vector_store.upserted) == 1
    assert stub_backend.calls == 1


async def test_vectors_requires_scope_when_location_given(api, stub_backend, stub_vector_store):
    resp = await api.post("/vectors", json={"text": "hello", "location": "repoA/foo.py"})
    assert resp.status_code == 422


async def test_vectors_with_location_records_provenance_and_pushes_scopes(
    api, stub_backend, stub_vector_store
):
    resp = await api.post(
        "/vectors", json={"text": "hello", "location": "repoA/foo.py", "scope": "repoA"}
    )
    assert resp.status_code == 200

    pid = point_id("hello")
    assert stub_vector_store.scopes_set == [(pid, ["repoA"])]

    locations = await app_module.provenance.get_locations(app_module.cache, [pid])
    assert locations[pid] == [{"uri": "repoA/foo.py", "scope": "repoA"}]


async def test_vectors_location_provenance_runs_on_cache_hit_too(
    api, stub_backend, stub_vector_store
):
    await api.post(
        "/vectors", json={"text": "hello", "location": "repoA/foo.py", "scope": "repoA"}
    )
    resp = await api.post(
        "/vectors", json={"text": "hello", "location": "repoB/foo.py", "scope": "repoB"}
    )
    assert resp.status_code == 200

    # Only one Qdrant vector write (the initial miss), but two scope pushes.
    assert len(stub_vector_store.upserted) == 1
    pid = point_id("hello")
    assert stub_vector_store.scopes_set[-1] == (pid, ["repoA", "repoB"])


async def test_search_passes_scope_through_to_the_store(api, stub_backend, stub_vector_store):
    resp = await api.post("/search", json={"text": "hi", "scope": "repoA"})
    assert resp.status_code == 200
    assert stub_vector_store.search_scopes == ["repoA"]


async def test_search_attaches_locations_from_provenance(api, stub_backend, stub_vector_store):
    pid = point_id("hello")
    await app_module.provenance.add_location(app_module.cache, pid, "repoA/foo.py", "repoA")
    stub_vector_store.search_results = [
        {"id": pid, "score": 0.9, "payload": {"text": "hello"}}
    ]

    resp = await api.post("/search", json={"text": "hi"})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["locations"] == [
        {"uri": "repoA/foo.py", "scope": "repoA"}
    ]


async def test_search_metadata_excludes_reserved_bookkeeping_fields(
    api, stub_backend, stub_vector_store
):
    stub_vector_store.search_results = [
        {
            "id": "x",
            "score": 0.9,
            "payload": {
                "text": "hello",
                "content_hash": "deadbeef",
                "scopes": ["repoA"],
                "source": "doc1",
            },
        }
    ]
    resp = await api.post("/search", json={"text": "hi"})
    assert resp.json()["results"][0]["metadata"] == {"source": "doc1"}


async def test_delete_location_returns_501_without_vector_store(api):
    resp = await api.request("DELETE", "/locations", json={"uri": "repoA/foo.py"})
    assert resp.status_code == 501


async def test_delete_location_updates_scopes_when_point_still_referenced(
    api, stub_backend, stub_vector_store
):
    await api.post(
        "/vectors", json={"text": "hello", "location": "repoA/foo.py", "scope": "repoA"}
    )
    await api.post(
        "/vectors", json={"text": "hello", "location": "repoB/foo.py", "scope": "repoB"}
    )

    resp = await api.request("DELETE", "/locations", json={"uri": "repoA/foo.py"})
    assert resp.status_code == 200
    assert resp.json() == {"uri": "repoA/foo.py", "affected": 1}

    pid = point_id("hello")
    assert stub_vector_store.scopes_set[-1] == (pid, ["repoB"])
    assert stub_vector_store.deleted == []


async def test_delete_location_gcs_orphaned_point(api, stub_backend, stub_vector_store):
    await api.post(
        "/vectors", json={"text": "hello", "location": "repoA/foo.py", "scope": "repoA"}
    )

    resp = await api.request("DELETE", "/locations", json={"uri": "repoA/foo.py"})
    assert resp.status_code == 200

    pid = point_id("hello")
    assert stub_vector_store.deleted == [pid]


async def test_embeddings_single_input(api, stub_backend):
    resp = await api.post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"] == [{"object": "embedding", "embedding": [1.0, 2.0, 3.0], "index": 0}]
    assert body["model"] == "stub-model"
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


async def test_embeddings_batch_input_preserves_order(api, stub_backend):
    resp = await api.post("/v1/embeddings", json={"input": ["one", "two"]})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert stub_backend.calls == 2


async def test_embeddings_unknown_model_is_404(api):
    resp = await api.post("/v1/embeddings", json={"input": "hello", "model": "custom"})
    assert resp.status_code == 404


async def test_embeddings_no_healthy_backend_is_503(api, stub_backend):
    stub_backend.healthy = False
    await app_module.registry.poll_once()
    resp = await api.post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 503


async def test_embeddings_falls_over_to_next_healthy_backend_for_model(api, stub_backend):
    fallback = StubBackend(name="fallback", model="stub-model", vector=[9.0])
    register(app_module.registry, backend=fallback)
    stub_backend.healthy = False
    await app_module.registry.poll_once()

    resp = await api.post("/v1/embeddings", json={"input": "hello", "model": "stub-model"})
    assert resp.status_code == 200
    assert resp.json()["data"][0]["embedding"] == [9.0]


async def test_embeddings_backend_param_selects_by_name(api, stub_backend):
    other = StubBackend(name="other", model="other-model", vector=[7.0])
    register(app_module.registry, backend=other)

    resp = await api.post("/v1/embeddings", json={"input": "hello", "backend": "other"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "other-model"
    assert resp.json()["data"][0]["embedding"] == [7.0]


async def test_embeddings_unknown_backend_name_is_404(api):
    resp = await api.post("/v1/embeddings", json={"input": "hello", "backend": "nope"})
    assert resp.status_code == 404


async def test_embeddings_backend_name_model_mismatch_is_404(api):
    resp = await api.post(
        "/v1/embeddings", json={"input": "hello", "backend": "stub", "model": "other-model"}
    )
    assert resp.status_code == 404


async def test_embeddings_named_but_unhealthy_backend_is_503(api, stub_backend):
    stub_backend.healthy = False
    await app_module.registry.poll_once()
    resp = await api.post("/v1/embeddings", json={"input": "hello", "backend": "stub"})
    assert resp.status_code == 503


async def test_embedding_task_gets_its_own_cache_entries(api, stub_backend):
    # Same type and model as the default stub, but task-tagged: the cached vector for
    # plain "hello" must not be returned for the code-task variant.
    tasked = StubBackend(name="stub-code", model="stub-model", vector=[4.0], embedding_task="code")
    register(app_module.registry, backend=tasked)

    await api.post("/v1/embeddings", json={"input": "hello"})
    resp = await api.post("/v1/embeddings", json={"input": "hello", "backend": "stub-code"})
    assert resp.json()["data"][0]["embedding"] == [4.0]
    assert tasked.calls == 1


async def test_embedding_task_is_stamped_into_upserted_metadata(
    api, stub_backend, stub_vector_store
):
    stub_backend.embedding_task = "code"
    await api.post("/vectors", json={"text": "hello"})
    _text, _vector, metadata = stub_vector_store.upserted[0]
    assert metadata == {"embedding_task": "code"}


async def test_vectors_routes_to_the_store_for_the_resolved_model(
    api, stub_backend, stub_vector_store
):
    other_backend = StubBackend(name="b2", model="m2", vector=[5.0])
    other_store = StubVectorStore(name="s2", model="m2", collection="c2")
    register(app_module.registry, backend=other_backend, store=other_store)

    resp = await api.post("/vectors", json={"text": "hello", "model": "m2"})
    assert resp.status_code == 200
    assert other_store.upserted == [("hello", [5.0], None)]
    assert stub_vector_store.upserted == []


async def test_backends_endpoint_lists_registry_with_health(api, stub_backend, stub_vector_store):
    resp = await api.get("/backends")
    assert resp.status_code == 200
    assert resp.json()["backends"] == [
        {
            "name": "stub",
            "type": "stub",
            "kind": "embedding",
            "model": "stub-model",
            "embedding_task": None,
            "healthy": True,
            "error": None,
        },
        {
            "name": "store",
            "type": "stub-store",
            "kind": "vector_store",
            "model": "stub-model",
            "collection": "test-collection",
            "metadata": {},
            "healthy": True,
            "error": None,
        },
    ]


async def test_backends_endpoint_filters_by_model_and_name(api, stub_backend, stub_vector_store):
    other = StubBackend(name="other", model="other-model")
    register(app_module.registry, backend=other)

    by_model = await api.get("/backends", params={"model": "other-model"})
    assert [e["name"] for e in by_model.json()["backends"]] == ["other"]

    by_name = await api.get("/backends", params={"name": "store"})
    assert [e["kind"] for e in by_name.json()["backends"]] == ["vector_store"]


async def test_backends_endpoint_reports_unhealthy_backends(api, stub_backend):
    stub_backend.healthy = False
    await app_module.registry.poll_once()
    entry = (await api.get("/backends")).json()["backends"][0]
    assert entry["healthy"] is False
    assert "backend down" in entry["error"]


async def test_ready_ok(api):
    resp = await api.get("/.well-known/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"] == {"redis": "ok", "backend:stub": "ok"}


async def test_ready_fails_when_backend_down(api, stub_backend):
    stub_backend.healthy = False
    await app_module.registry.poll_once()
    resp = await api.get("/.well-known/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"]["redis"] == "ok"
    assert resp.json()["checks"]["backend:stub"].startswith("error:")


async def test_live_always_ok(api):
    resp = await api.get("/.well-known/live")
    assert resp.status_code == 200

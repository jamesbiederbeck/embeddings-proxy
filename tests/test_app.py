import fakeredis.aioredis
import httpx
import pytest

import app as app_module


class StubBackend:
    name = "stub"
    model_name = "stub-model"

    def __init__(self, vector=None, healthy=True):
        self.vector = vector or [1.0, 2.0, 3.0]
        self.calls = 0
        self.healthy = healthy

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self.vector

    async def check(self) -> None:
        if not self.healthy:
            raise RuntimeError("backend down")


@pytest.fixture
def stub_backend(monkeypatch):
    backend = StubBackend()
    monkeypatch.setattr(app_module, "backend", backend)
    monkeypatch.setattr(
        app_module, "cache", fakeredis.aioredis.FakeRedis(decode_responses=True)
    )
    return backend


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


async def test_embeddings_model_override(api):
    resp = await api.post("/v1/embeddings", json={"input": "hello", "model": "custom"})
    assert resp.json()["model"] == "custom"


async def test_ready_ok(api):
    resp = await api.get("/.well-known/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"] == {"redis": "ok", "backend": "ok"}


async def test_ready_fails_when_backend_down(api, stub_backend):
    stub_backend.healthy = False
    resp = await api.get("/.well-known/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"]["redis"] == "ok"
    assert resp.json()["checks"]["backend"].startswith("error:")


async def test_live_always_ok(api):
    resp = await api.get("/.well-known/live")
    assert resp.status_code == 200

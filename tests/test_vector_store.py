from unittest.mock import AsyncMock, MagicMock

import pytest

from vector_store import QdrantVectorStore, point_id


class FakeVectorsConfig:
    def __init__(self, size):
        self.size = size


class FakeCollectionInfo:
    def __init__(self, size):
        self.config = MagicMock()
        self.config.params.vectors = FakeVectorsConfig(size)


def make_store(collection_exists=False, existing_dim=None):
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection = "test-collection"
    store.base_metadata = {}
    store._ensured = False
    store.client = MagicMock()
    store.client.collection_exists = AsyncMock(return_value=collection_exists)
    store.client.get_collection = AsyncMock(return_value=FakeCollectionInfo(existing_dim))
    store.client.create_collection = AsyncMock()
    store.client.create_payload_index = AsyncMock()
    store.client.upsert = AsyncMock()
    store.client.query_points = AsyncMock()
    store.client.set_payload = AsyncMock()
    store.client.delete = AsyncMock()
    return store


async def test_ensure_collection_creates_scopes_and_content_hash_indexes():
    store = make_store(collection_exists=False)
    await store._ensure_collection(3)

    indexed_fields = {
        call.kwargs["field_name"] for call in store.client.create_payload_index.await_args_list
    }
    assert indexed_fields == {"scopes", "content_hash"}
    assert store._ensured is True


async def test_ensure_collection_only_indexes_once():
    store = make_store(collection_exists=False)
    await store._ensure_collection(3)
    await store._ensure_collection(3)

    assert store.client.create_payload_index.await_count == 2


async def test_upsert_writes_content_hash_and_scopes():
    store = make_store(collection_exists=True, existing_dim=3)
    await store.upsert("hello", [1.0, 2.0, 3.0], metadata={"source": "doc1"}, scopes=["repoA"])

    points = store.client.upsert.await_args.kwargs["points"]
    payload = points[0].payload
    assert payload["text"] == "hello"
    assert payload["source"] == "doc1"
    assert payload["scopes"] == ["repoA"]
    assert payload["content_hash"] == __import__("hashlib").sha256(b"hello").hexdigest()
    assert points[0].id == point_id("hello")


async def test_upsert_defaults_scopes_to_empty_list():
    store = make_store(collection_exists=True, existing_dim=3)
    await store.upsert("hello", [1.0, 2.0, 3.0])

    points = store.client.upsert.await_args.kwargs["points"]
    assert points[0].payload["scopes"] == []


async def test_set_scopes_overwrites_scopes_field_only():
    store = make_store(collection_exists=True, existing_dim=3)
    store._ensured = True
    await store.set_scopes("some-point-id", ["repoA", "repoB"])

    store.client.set_payload.assert_awaited_once_with(
        "test-collection", payload={"scopes": ["repoA", "repoB"]}, points=["some-point-id"]
    )


async def test_delete_removes_the_point():
    store = make_store(collection_exists=True, existing_dim=3)
    store._ensured = True
    await store.delete("some-point-id")

    store.client.delete.assert_awaited_once_with(
        "test-collection", points_selector=["some-point-id"]
    )


async def test_search_without_scope_omits_filter():
    store = make_store(collection_exists=True, existing_dim=3)
    store.client.query_points.return_value = MagicMock(points=[])
    await store.search([1.0, 2.0, 3.0], top_k=5)

    assert store.client.query_points.await_args.kwargs["query_filter"] is None


async def test_search_with_scope_builds_a_filter():
    store = make_store(collection_exists=True, existing_dim=3)
    store.client.query_points.return_value = MagicMock(points=[])
    await store.search([1.0, 2.0, 3.0], top_k=5, scope="repoA")

    query_filter = store.client.query_points.await_args.kwargs["query_filter"]
    assert query_filter is not None
    assert query_filter.must[0].key == "scopes"
    assert query_filter.must[0].match.value == "repoA"


async def test_ensure_collection_raises_on_dimension_mismatch():
    from vector_store import DimensionMismatchError

    store = make_store(collection_exists=True, existing_dim=7)
    with pytest.raises(DimensionMismatchError):
        await store._ensure_collection(3)

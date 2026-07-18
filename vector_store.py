import hashlib
import os
import uuid
from typing import Any, Protocol


class DimensionMismatchError(Exception):
    """The existing collection's vector size doesn't match what the backend produces."""


class VectorStore(Protocol):
    async def upsert(
        self,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        scopes: list[str] | None = None,
    ) -> None: ...

    async def search(
        self, vector: list[float], top_k: int, scope: str | None = None
    ) -> list[dict[str, Any]]: ...

    async def set_scopes(self, point_id: str, scopes: list[str]) -> None: ...

    async def delete(self, point_id: str) -> None: ...

    async def check(self) -> None: ...


def point_id(text: str) -> str:
    # Deterministic id per text, so re-embedding the same text overwrites its point.
    return str(uuid.UUID(hashlib.sha256(text.encode()).hexdigest()[:32]))


class QdrantVectorStore:
    """Stores vectors in a Qdrant collection, auto-created on first use.

    qdrant-client is imported lazily so the dependency is only needed when
    VECTOR_STORE=qdrant is actually configured.
    """

    def __init__(self, url: str, collection: str):
        from qdrant_client import AsyncQdrantClient

        # check_compatibility=False: don't contact the server at construction time,
        # so the proxy can start while Qdrant is down (indexing is best-effort anyway).
        self.client = AsyncQdrantClient(url=url, timeout=10, check_compatibility=False)
        self.collection = collection
        self._ensured = False

    async def _ensure_collection(self, dim: int) -> None:
        if self._ensured:
            return
        from qdrant_client import models

        if await self.client.collection_exists(self.collection):
            info = await self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            existing = getattr(vectors, "size", None)
            if existing != dim:
                raise DimensionMismatchError(
                    f"collection {self.collection!r} has vector size {existing}, "
                    f"but the backend produces {dim}-dim vectors"
                )
        else:
            await self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
        # Keyword indexes so /search can filter by scope, and points can be looked up by
        # their full content hash without relying on the truncated point-id UUID.
        await self.client.create_payload_index(
            self.collection, field_name="scopes", field_schema=models.PayloadSchemaType.KEYWORD
        )
        await self.client.create_payload_index(
            self.collection,
            field_name="content_hash",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._ensured = True

    async def upsert(
        self,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        await self._ensure_collection(len(vector))
        from qdrant_client import models

        # "text" last so it always wins if caller-supplied metadata happens to collide —
        # every payload this proxy writes must keep a real text key for /search to rely on.
        payload = {
            **(metadata or {}),
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "scopes": scopes or [],
            "text": text,
        }
        await self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=point_id(text), vector=vector, payload=payload)],
        )

    async def search(
        self, vector: list[float], top_k: int, scope: str | None = None
    ) -> list[dict[str, Any]]:
        await self._ensure_collection(len(vector))
        from qdrant_client import models

        query_filter = None
        if scope is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="scopes", match=models.MatchValue(value=scope))]
            )
        result = await self.client.query_points(
            self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in result.points]

    async def set_scopes(self, point_id: str, scopes: list[str]) -> None:
        await self.client.set_payload(
            self.collection, payload={"scopes": scopes}, points=[point_id]
        )

    async def delete(self, point_id: str) -> None:
        await self.client.delete(self.collection, points_selector=[point_id])

    async def check(self) -> None:
        await self.client.get_collections()


def build_vector_store() -> VectorStore | None:
    store_name = os.environ.get("VECTOR_STORE")

    if not store_name:
        return None
    if store_name == "qdrant":
        return QdrantVectorStore(os.environ["QDRANT_URL"], os.environ["QDRANT_COLLECTION"])

    raise ValueError(f"Unknown VECTOR_STORE: {store_name!r}")

import hashlib
import os
import uuid
from typing import Any, Protocol


class DimensionMismatchError(Exception):
    """The existing collection's vector size doesn't match what the backend produces."""


class VectorStore(Protocol):
    async def upsert(self, text: str, vector: list[float]) -> None: ...

    async def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]: ...

    async def check(self) -> None: ...


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

    @staticmethod
    def _point_id(text: str) -> str:
        # Deterministic id per text, so re-embedding the same text overwrites its point.
        return str(uuid.UUID(hashlib.sha256(text.encode()).hexdigest()[:32]))

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
        self._ensured = True

    async def upsert(self, text: str, vector: list[float]) -> None:
        await self._ensure_collection(len(vector))
        from qdrant_client import models

        await self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(id=self._point_id(text), vector=vector, payload={"text": text})
            ],
        )

    async def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        await self._ensure_collection(len(vector))
        result = await self.client.query_points(
            self.collection, query=vector, limit=top_k, with_payload=True
        )
        return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in result.points]

    async def check(self) -> None:
        await self.client.get_collections()


def build_vector_store() -> VectorStore | None:
    store_name = os.environ.get("VECTOR_STORE")

    if not store_name:
        return None
    if store_name == "qdrant":
        return QdrantVectorStore(os.environ["QDRANT_URL"], os.environ["QDRANT_COLLECTION"])

    raise ValueError(f"Unknown VECTOR_STORE: {store_name!r}")

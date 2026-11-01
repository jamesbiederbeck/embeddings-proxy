import hashlib
import uuid
from typing import Any

from backends import Backend
from config import VectorStoreConfig


class DimensionMismatchError(Exception):
    """The existing collection's vector size doesn't match what the backend produces."""


def point_id(text: str) -> str:
    # Deterministic id per text, so re-embedding the same text overwrites its point.
    return str(uuid.UUID(hashlib.sha256(text.encode()).hexdigest()[:32]))


class VectorStoreBackend(Backend):
    """A vector store is a backend too: it has a name, is tied to the embedding
    model whose vectors its collection holds (the routing key for /vectors and
    /search), and gets health-polled like any other backend.
    """

    async def upsert(
        self,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    async def search(
        self,
        vector: list[float],
        top_k: int,
        scope: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def set_scopes(self, point_id: str, scopes: list[str]) -> None:
        raise NotImplementedError

    async def delete(self, point_id: str) -> None:
        raise NotImplementedError


class QdrantVectorStore(VectorStoreBackend):
    """Stores vectors in a Qdrant collection, auto-created on first use.

    qdrant-client is imported lazily so the dependency is only needed when
    a vector store is actually configured.
    """

    type = "qdrant"

    def __init__(
        self,
        url: str,
        collection: str,
        model: str = "",
        name: str | None = None,
        base_metadata: dict[str, Any] | None = None,
    ):
        from qdrant_client import AsyncQdrantClient

        super().__init__(name or self.type, model)
        # check_compatibility=False: don't contact the server at construction time,
        # so the proxy can start while Qdrant is down (indexing is best-effort anyway).
        self.client = AsyncQdrantClient(url=url, timeout=10, check_compatibility=False)
        self.collection = collection
        self.base_metadata = base_metadata or {}
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

        # Precedence: config base_metadata < request metadata < the proxy's own keys.
        # "text" last so it always wins if caller-supplied metadata happens to collide —
        # every payload this proxy writes must keep a real text key for /search to rely on.
        payload = {
            **self.base_metadata,
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
        self,
        vector: list[float],
        top_k: int,
        scope: str | None = None,
        score_threshold: float | None = None,
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
            score_threshold=score_threshold,
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


def build_vector_store(cfg: VectorStoreConfig) -> VectorStoreBackend:
    if cfg.type == "qdrant":
        return QdrantVectorStore(
            cfg.url, cfg.collection, model=cfg.model, name=cfg.name, base_metadata=cfg.metadata
        )
    raise ValueError(f"Unknown vector store type: {cfg.type!r}")


def build_vector_stores(cfgs: list[VectorStoreConfig]) -> list[VectorStoreBackend]:
    return [build_vector_store(cfg) for cfg in cfgs]

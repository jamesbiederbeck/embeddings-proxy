import asyncio
import hashlib
import json
import logging
import os

from contextlib import asynccontextmanager
from typing import Any

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

import provenance
from backends import EmbeddingBackend, build_backends
from config import load_config
from registry import NoHealthyBackendError, NotConfiguredError, Registry
from vector_store import DimensionMismatchError, build_vector_stores, point_id

REDIS_URL = os.environ["REDIS_URL"]

# Readiness probes must stay cheap and bounded — never the shared client's 30s.
PROBE_TIMEOUT = 2.0

logger = logging.getLogger("embeddings-proxy")

config = load_config()
cache = redis.from_url(REDIS_URL, decode_responses=True)
client = httpx.AsyncClient(timeout=30.0)
registry = Registry(
    build_backends(client, config.backends), build_vector_stores(config.vector_stores)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One synchronous sweep before serving so the first request already has health
    # state (each check is time-boxed, so this bounds startup at PROBE_TIMEOUT).
    await registry.poll_once()
    poller = asyncio.create_task(registry.poll_forever(config.health_check_interval))
    yield
    poller.cancel()


app = FastAPI(lifespan=lifespan)


class VectorRequest(BaseModel):
    text: str
    metadata: dict[str, Any] | None = None
    location: str | None = None
    scope: str | None = None
    location_metadata: dict[str, Any] | None = None
    model: str | None = None
    backend: str | None = None

    @model_validator(mode="after")
    def _scope_required_with_location(self):
        if self.location is not None and self.scope is None:
            raise ValueError("scope is required when location is set")
        return self


class SearchRequest(BaseModel):
    text: str
    top_k: int = 5
    scope: str | None = None
    score_threshold: float | None = None
    exact: bool = False
    model: str | None = None
    backend: str | None = None


class LocationRequest(BaseModel):
    uri: str


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    backend: str | None = None


def resolve_backend(name: str | None, model: str | None) -> EmbeddingBackend:
    try:
        return registry.select_embedding(name=name, model=model)
    except NotConfiguredError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NoHealthyBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def cache_key(b: EmbeddingBackend, text: str) -> str:
    # Keyed by type (not instance name) so same-type instances serving the same model
    # share cache — and the bytes stay identical to the pre-0.5 scheme, no invalidation.
    # embedding_task joins the key only when set: task-distinct vectors must not collide.
    parts = [b.type, b.model] + ([b.embedding_task] if b.embedding_task else []) + [text]
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()
    return f"embedding:{digest}"


async def get_vector(b: EmbeddingBackend, text: str) -> tuple[list[float], bool]:
    key = cache_key(b, text)

    cached = await cache.get(key)
    if cached is not None:
        return json.loads(cached), True

    vector = await b.embed(text)
    await cache.set(key, json.dumps(vector))
    return vector, False


@app.post("/vectors")
async def vectors(req: VectorRequest):
    backend = resolve_backend(req.backend, req.model)
    vector, cached = await get_vector(backend, req.text)

    vector_store = registry.store_for_model(backend.model)
    if vector_store is not None:
        try:
            # The vector itself is content-addressed and already in the store once it's
            # been embedded once; only a genuine cache miss needs to (re)write it.
            if not cached:
                metadata = req.metadata
                if backend.embedding_task is not None:
                    metadata = {**(metadata or {}), "embedding_task": backend.embedding_task}
                await vector_store.upsert(req.text, vector, metadata)
            # Location provenance runs on every call, hit or miss — even a cached vector
            # needs its new location recorded.
            if req.location is not None:
                pid = point_id(req.text)
                scopes = await provenance.add_location(
                    cache, pid, req.location, req.scope, req.location_metadata
                )
                await vector_store.set_scopes(pid, scopes)
        except DimensionMismatchError:
            raise
        except Exception:
            # Indexing is best-effort: a store outage must not break the cache path.
            logger.exception("vector store upsert failed; returning embedding anyway")

    return {"text": req.text, "vector": vector, "dim": len(vector), "cached": cached}


@app.post("/search")
async def search(req: SearchRequest):
    backend = resolve_backend(req.backend, req.model)
    vector_store = registry.store_for_model(backend.model)
    if vector_store is None:
        raise HTTPException(
            status_code=501,
            detail=f"No vector store configured for model {backend.model!r}",
        )

    vector, _ = await get_vector(backend, req.text)
    results = await vector_store.search(
        vector, req.top_k, req.scope, req.score_threshold, req.exact
    )
    locations = await provenance.get_locations(cache, [r["id"] for r in results])
    # "content_hash"/"scopes" are this proxy's own bookkeeping fields, not caller metadata.
    reserved = {"text", "content_hash", "scopes"}
    return {
        "query": req.text,
        "results": [
            {
                # payload["text"] assumes every payload was written by this proxy (true for
                # all current write paths); a foreign point without a "text" key 500s here
                # rather than silently returning a malformed Document.
                "page_content": r["payload"]["text"],
                "metadata": {k: v for k, v in r["payload"].items() if k not in reserved},
                "locations": locations.get(r["id"], []),
                "score": r["score"],
            }
            for r in results
        ],
    }


@app.delete("/locations")
async def delete_location(req: LocationRequest):
    if not registry.stores:
        raise HTTPException(status_code=501, detail="No vector store configured")

    affected = await provenance.remove_location(cache, req.uri)
    for pid, scopes, is_orphaned in affected:
        # Provenance doesn't record which collection a point landed in, so the cleanup
        # runs against every store — deleting a nonexistent point is a Qdrant no-op.
        for store in registry.stores:
            try:
                if is_orphaned:
                    await store.delete(pid)
                else:
                    await store.set_scopes(pid, scopes)
            except Exception:
                logger.exception(
                    "vector store cleanup failed for point %s in %s", pid, store.name
                )

    return {"uri": req.uri, "affected": len(affected)}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    backend = resolve_backend(req.backend, req.model)
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = await asyncio.gather(*(get_vector(backend, text) for text in texts))

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": i}
            for i, (vector, _cached) in enumerate(vectors)
        ],
        "model": backend.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/backends")
async def list_backends(model: str | None = None, name: str | None = None):
    entries = []
    for b in registry.embedding:
        entries.append(
            {
                "name": b.name,
                "type": b.type,
                "kind": "embedding",
                "model": b.model,
                "embedding_task": b.embedding_task,
            }
        )
    for s in registry.stores:
        entries.append(
            {
                "name": s.name,
                "type": s.type,
                "kind": "vector_store",
                "model": s.model,
                "collection": s.collection,
                "metadata": s.base_metadata,
            }
        )
    for entry in entries:
        status = registry.health[entry["name"]]
        entry["healthy"] = status.healthy
        entry["error"] = status.error

    if model is not None:
        entries = [e for e in entries if e["model"] == model]
    if name is not None:
        entries = [e for e in entries if e["name"] == name]
    return {"backends": entries}


async def _probe(name: str, coro, checks: dict[str, str]) -> bool:
    try:
        async with asyncio.timeout(PROBE_TIMEOUT):
            await coro
    except Exception as exc:
        checks[name] = f"error: {exc}"
        return False
    checks[name] = "ok"
    return True


@app.get("/.well-known/ready")
async def ready():
    checks: dict[str, str] = {}
    ok = await _probe("redis", cache.ping(), checks)

    # Backends report the poller's last sweep — readiness never probes an upstream.
    for b in (*registry.embedding, *registry.stores):
        status = registry.health[b.name]
        kind = "vector_store" if b in registry.stores else "backend"
        checks[f"{kind}:{b.name}"] = "ok" if status.healthy else f"error: {status.error}"
    ok = ok and any(registry.is_healthy(b.name) for b in registry.embedding)

    if not ok:
        return JSONResponse(status_code=503, content={"status": "unavailable", "checks": checks})
    return {"status": "ok", "checks": checks}


@app.get("/.well-known/live")
async def live():
    return {"status": "ok"}

import asyncio
import hashlib
import json
import logging
import os

from typing import Any

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

import provenance
from backends import build_backend
from vector_store import DimensionMismatchError, build_vector_store, point_id

REDIS_URL = os.environ["REDIS_URL"]

# Readiness probes must stay cheap and bounded — never the shared client's 30s.
PROBE_TIMEOUT = 2.0

logger = logging.getLogger("embeddings-proxy")

app = FastAPI()
cache = redis.from_url(REDIS_URL, decode_responses=True)
client = httpx.AsyncClient(timeout=30.0)
backend = build_backend(client)
vector_store = build_vector_store()


class VectorRequest(BaseModel):
    text: str
    metadata: dict[str, Any] | None = None
    location: str | None = None
    scope: str | None = None
    location_metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _scope_required_with_location(self):
        if self.location is not None and self.scope is None:
            raise ValueError("scope is required when location is set")
        return self


class SearchRequest(BaseModel):
    text: str
    top_k: int = 5
    scope: str | None = None


class LocationRequest(BaseModel):
    uri: str


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


def cache_key(text: str) -> str:
    digest = hashlib.sha256(f"{backend.name}:{backend.model_name}:{text}".encode()).hexdigest()
    return f"embedding:{digest}"


async def get_vector(text: str) -> tuple[list[float], bool]:
    key = cache_key(text)

    cached = await cache.get(key)
    if cached is not None:
        return json.loads(cached), True

    vector = await backend.embed(text)
    await cache.set(key, json.dumps(vector))
    return vector, False


@app.post("/vectors")
async def vectors(req: VectorRequest):
    vector, cached = await get_vector(req.text)

    if vector_store is not None:
        try:
            # The vector itself is content-addressed and already in the store once it's
            # been embedded once; only a genuine cache miss needs to (re)write it.
            if not cached:
                await vector_store.upsert(req.text, vector, req.metadata)
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
    if vector_store is None:
        raise HTTPException(
            status_code=501, detail="No vector store configured (set VECTOR_STORE=qdrant)"
        )

    vector, _ = await get_vector(req.text)
    results = await vector_store.search(vector, req.top_k, req.scope)
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
    if vector_store is None:
        raise HTTPException(
            status_code=501, detail="No vector store configured (set VECTOR_STORE=qdrant)"
        )

    affected = await provenance.remove_location(cache, req.uri)
    for pid, scopes, is_orphaned in affected:
        try:
            if is_orphaned:
                await vector_store.delete(pid)
            else:
                await vector_store.set_scopes(pid, scopes)
        except Exception:
            logger.exception("vector store cleanup failed for point %s", pid)

    return {"uri": req.uri, "affected": len(affected)}


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = await asyncio.gather(*(get_vector(text) for text in texts))

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": i}
            for i, (vector, _cached) in enumerate(vectors)
        ],
        "model": req.model or backend.model_name,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


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
    ok = await _probe("backend", backend.check(), checks) and ok
    if vector_store is not None:
        ok = await _probe("vector_store", vector_store.check(), checks) and ok

    if not ok:
        return JSONResponse(status_code=503, content={"status": "unavailable", "checks": checks})
    return {"status": "ok", "checks": checks}


@app.get("/.well-known/live")
async def live():
    return {"status": "ok"}

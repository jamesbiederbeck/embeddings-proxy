import asyncio
import hashlib
import json
import logging
import os

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backends import build_backend
from vector_store import DimensionMismatchError, build_vector_store

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


class SearchRequest(BaseModel):
    text: str
    top_k: int = 5


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
            await vector_store.upsert(req.text, vector)
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
    results = await vector_store.search(vector, req.top_k)
    return {"text": req.text, "results": results}


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

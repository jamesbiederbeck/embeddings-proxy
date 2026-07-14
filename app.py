import hashlib
import json
import os

import httpx
import redis.asyncio as redis
from fastapi import FastAPI
from pydantic import BaseModel

from backends import build_backend

REDIS_URL = os.environ["REDIS_URL"]

app = FastAPI()
cache = redis.from_url(REDIS_URL, decode_responses=True)
client = httpx.AsyncClient(timeout=30.0)
backend = build_backend(client)


class VectorRequest(BaseModel):
    text: str


def cache_key(text: str) -> str:
    digest = hashlib.sha256(f"{backend.name}:{backend.model_name}:{text}".encode()).hexdigest()
    return f"embedding:{digest}"


@app.post("/vectors")
async def vectors(req: VectorRequest):
    key = cache_key(req.text)

    cached = await cache.get(key)
    if cached is not None:
        vector = json.loads(cached)
        return {"text": req.text, "vector": vector, "dim": len(vector), "cached": True}

    vector = await backend.embed(req.text)

    await cache.set(key, json.dumps(vector))
    return {"text": req.text, "vector": vector, "dim": len(vector), "cached": False}


@app.get("/.well-known/ready")
async def ready():
    return {"status": "ok"}

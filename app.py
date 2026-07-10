import hashlib
import json
import os

import httpx
import redis.asyncio as redis
from fastapi import FastAPI
from pydantic import BaseModel

EMBEDDINGS_URL = os.environ["EMBEDDINGS_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
REDIS_URL = os.environ["REDIS_URL"]

app = FastAPI()
cache = redis.from_url(REDIS_URL, decode_responses=True)
client = httpx.AsyncClient(timeout=30.0)


class VectorRequest(BaseModel):
    text: str


def cache_key(text: str) -> str:
    digest = hashlib.sha256(f"{MODEL_NAME}:{text}".encode()).hexdigest()
    return f"embedding:{digest}"


@app.post("/vectors")
async def vectors(req: VectorRequest):
    key = cache_key(req.text)

    cached = await cache.get(key)
    if cached is not None:
        vector = json.loads(cached)
        return {"text": req.text, "vector": vector, "dim": len(vector), "cached": True}

    resp = await client.post(f"{EMBEDDINGS_URL}/vectors", json={"text": req.text})
    resp.raise_for_status()
    data = resp.json()

    await cache.set(key, json.dumps(data["vector"]))
    return {**data, "cached": False}


@app.get("/.well-known/ready")
async def ready():
    return {"status": "ok"}

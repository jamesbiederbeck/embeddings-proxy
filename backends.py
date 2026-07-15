import os
from typing import Protocol

import httpx


class Backend(Protocol):
    name: str
    model_name: str

    async def embed(self, text: str) -> list[float]: ...


class TransformersInferenceBackend:
    """Backend for semitechnologies/transformers-inference."""

    name = "transformers-inference"

    def __init__(self, client: httpx.AsyncClient):
        self.embeddings_url = os.environ["EMBEDDINGS_URL"]
        self.model_name = os.environ["MODEL_NAME"]
        self.client = client

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post(f"{self.embeddings_url}/vectors", json={"text": text})
        resp.raise_for_status()
        return resp.json()["vector"]


def build_backend(client: httpx.AsyncClient) -> Backend:
    backend_name = os.environ.get("BACKEND", "transformers-inference")

    if backend_name == "transformers-inference":
        return TransformersInferenceBackend(client)

    raise ValueError(f"Unknown BACKEND: {backend_name!r}")

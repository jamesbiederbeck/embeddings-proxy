import os
from typing import Protocol

import httpx

# Short timeout for readiness probes so a slow upstream can't hang the probe.
PROBE_TIMEOUT = 2.0


class Backend(Protocol):
    name: str
    model_name: str

    async def embed(self, text: str) -> list[float]: ...

    async def check(self) -> None: ...


class TransformersInferenceBackend:
    """Backend for semitechnologies/transformers-inference."""

    name = "transformers-inference"

    def __init__(self, client: httpx.AsyncClient, embeddings_url: str, model_name: str):
        self.embeddings_url = embeddings_url.rstrip("/")
        self.model_name = model_name
        self.client = client

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post(f"{self.embeddings_url}/vectors", json={"text": text})
        resp.raise_for_status()
        return resp.json()["vector"]

    async def check(self) -> None:
        resp = await self.client.get(
            f"{self.embeddings_url}/.well-known/ready", timeout=PROBE_TIMEOUT
        )
        resp.raise_for_status()


class OpenAIBackend:
    """Backend for OpenAI-compatible POST /v1/embeddings endpoints."""

    name = "openai"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.client = client

    @property
    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": text, "model": self.model_name},
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def check(self) -> None:
        # Reachability only — any HTTP response (including 401) means the server is up.
        # No raise_for_status, so an auth-guarded endpoint still counts as reachable.
        await self.client.get(
            f"{self.base_url}/v1/models", headers=self._headers, timeout=PROBE_TIMEOUT
        )


class LlamaCppBackend(OpenAIBackend):
    """llama.cpp server, via its OpenAI-compatible /v1/embeddings endpoint.

    The OpenAI-compatible path is used instead of the native POST /embedding
    because its response shape is stable across llama.cpp versions, while the
    native endpoint's shape has changed over time. Requires the server to be
    started with --embeddings.
    """

    name = "llamacpp"


def build_backend(client: httpx.AsyncClient) -> Backend:
    backend_name = os.environ.get("BACKEND", "transformers-inference")

    if backend_name == "transformers-inference":
        return TransformersInferenceBackend(
            client, os.environ["EMBEDDINGS_URL"], os.environ["MODEL_NAME"]
        )
    if backend_name == "openai":
        return OpenAIBackend(
            client,
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
            os.environ["MODEL_NAME"],
            os.environ["OPENAI_API_KEY"],
        )
    if backend_name == "llamacpp":
        return LlamaCppBackend(
            client,
            os.environ["EMBEDDINGS_URL"],
            os.environ["MODEL_NAME"],
            os.environ.get("OPENAI_API_KEY"),
        )

    raise ValueError(f"Unknown BACKEND: {backend_name!r}")

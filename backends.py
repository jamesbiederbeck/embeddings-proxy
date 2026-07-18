import httpx

from config import EmbeddingBackendConfig

# Short timeout for readiness probes so a slow upstream can't hang the probe.
PROBE_TIMEOUT = 2.0


class Backend:
    """Anything the proxy talks to and health-checks: embedding upstreams and
    vector stores alike. `type` is the implementation kind (part of the cache
    key); `name` is the per-instance config name (unique across the registry).
    """

    type: str

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    async def check(self) -> None:
        raise NotImplementedError


class EmbeddingBackend(Backend):
    def __init__(self, name: str, model: str, embedding_task: str | None = None):
        super().__init__(name, model)
        self.embedding_task = embedding_task

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class TransformersInferenceBackend(EmbeddingBackend):
    """Backend for semitechnologies/transformers-inference."""

    type = "transformers-inference"

    def __init__(
        self,
        client: httpx.AsyncClient,
        embeddings_url: str,
        model: str,
        name: str | None = None,
        embedding_task: str | None = None,
    ):
        super().__init__(name or self.type, model, embedding_task)
        self.embeddings_url = embeddings_url.rstrip("/")
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


class OpenAIBackend(EmbeddingBackend):
    """Backend for OpenAI-compatible POST /v1/embeddings endpoints."""

    type = "openai"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        api_key: str | None = None,
        name: str | None = None,
        embedding_task: str | None = None,
    ):
        super().__init__(name or self.type, model, embedding_task)
        self.base_url = base_url.rstrip("/")
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
            json={"input": text, "model": self.model},
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

    type = "llamacpp"


def build_backend(client: httpx.AsyncClient, cfg: EmbeddingBackendConfig) -> EmbeddingBackend:
    if cfg.type == "transformers-inference":
        return TransformersInferenceBackend(
            client, cfg.url, cfg.model, name=cfg.name, embedding_task=cfg.embedding_task
        )
    if cfg.type == "openai":
        return OpenAIBackend(
            client,
            cfg.url,
            cfg.model,
            cfg.api_key,
            name=cfg.name,
            embedding_task=cfg.embedding_task,
        )
    if cfg.type == "llamacpp":
        return LlamaCppBackend(
            client,
            cfg.url,
            cfg.model,
            cfg.api_key,
            name=cfg.name,
            embedding_task=cfg.embedding_task,
        )
    raise ValueError(f"Unknown backend type: {cfg.type!r}")


def build_backends(
    client: httpx.AsyncClient, cfgs: list[EmbeddingBackendConfig]
) -> list[EmbeddingBackend]:
    return [build_backend(client, cfg) for cfg in cfgs]

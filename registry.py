import asyncio
import time
from dataclasses import dataclass

from backends import PROBE_TIMEOUT, Backend, EmbeddingBackend
from vector_store import VectorStoreBackend


class NotConfiguredError(Exception):
    """No configured backend matches the request (mapped to 404)."""


class NoHealthyBackendError(Exception):
    """Matching backends are configured but none is currently healthy (mapped to 503)."""


@dataclass
class HealthStatus:
    healthy: bool = False
    error: str | None = "not checked yet"
    checked_at: float | None = None


class Registry:
    """All configured backends and vector stores, with polled health state.

    Health is only ever written by the poller — request handlers read the last
    polled status and never probe an upstream synchronously. Config order is
    priority order for selection.
    """

    def __init__(self, embedding: list[EmbeddingBackend], stores: list[VectorStoreBackend]):
        self.embedding = embedding
        self.stores = stores
        self.health = {b.name: HealthStatus() for b in (*embedding, *stores)}

    def is_healthy(self, name: str) -> bool:
        return self.health[name].healthy

    def select_embedding(
        self, name: str | None = None, model: str | None = None
    ) -> EmbeddingBackend:
        if name is not None:
            candidate = next((b for b in self.embedding if b.name == name), None)
            if candidate is None:
                raise NotConfiguredError(f"no backend named {name!r} is configured")
            if model is not None and candidate.model != model:
                raise NotConfiguredError(
                    f"backend {name!r} serves model {candidate.model!r}, not {model!r}"
                )
            if not self.is_healthy(candidate.name):
                raise NoHealthyBackendError(
                    f"backend {name!r} is unhealthy: {self.health[candidate.name].error}"
                )
            return candidate

        candidates = (
            self.embedding
            if model is None
            else [b for b in self.embedding if b.model == model]
        )
        if not candidates:
            raise NotConfiguredError(f"no backend is configured with model {model!r}")
        for backend in candidates:
            if self.is_healthy(backend.name):
                return backend
        raise NoHealthyBackendError(
            "no healthy backend available"
            + (f" for model {model!r}" if model is not None else "")
        )

    def store_for_model(self, model: str) -> VectorStoreBackend | None:
        return next((s for s in self.stores if s.model == model), None)

    async def _check_one(self, backend: Backend) -> None:
        try:
            async with asyncio.timeout(PROBE_TIMEOUT):
                await backend.check()
        except Exception as exc:
            self.health[backend.name] = HealthStatus(
                False, str(exc) or type(exc).__name__, time.time()
            )
        else:
            self.health[backend.name] = HealthStatus(True, None, time.time())

    async def poll_once(self) -> None:
        await asyncio.gather(*(self._check_one(b) for b in (*self.embedding, *self.stores)))

    async def poll_forever(self, interval: float) -> None:
        # The startup sweep has already run by the time this task starts, so sleep first.
        while True:
            await asyncio.sleep(interval)
            await self.poll_once()

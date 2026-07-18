"""Startup configuration.

Two sources, checked in order:

1. Legacy env vars — if MODEL_NAME is set, a single-backend config is synthesized
   from the pre-0.5 vars (BACKEND, MODEL_NAME, EMBEDDINGS_URL, OPENAI_*,
   VECTOR_STORE, QDRANT_*), so existing deployments keep working unchanged.
2. A YAML config file at CONFIG_FILE (default ./config.yaml), which is where
   multi-backend setups are declared.

HEALTH_CHECK_INTERVAL from the environment overrides the file in either mode.
REDIS_URL stays env-only (infrastructure, not routing).
"""

import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_FILE = "config.yaml"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_HEALTH_CHECK_INTERVAL = 30.0


class EmbeddingBackendConfig(BaseModel):
    type: Literal["transformers-inference", "openai", "llamacpp"]
    model: str
    name: str = ""  # defaults to `type`; must be set explicitly to disambiguate duplicates
    url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    # Task type the upstream is serving for instruct-aware models (e.g. "code",
    # "search_document"). Declarative: the upstream is assumed to be configured for
    # it — the proxy joins it into the cache key and payloads but never rewrites text.
    embedding_task: str | None = None

    @model_validator(mode="after")
    def _resolve(self):
        if not self.name:
            self.name = self.type
        if self.api_key is None and self.api_key_env:
            if self.api_key_env not in os.environ:
                raise ValueError(
                    f"backend {self.name!r}: api_key_env {self.api_key_env!r} is not set"
                )
            self.api_key = os.environ[self.api_key_env]
        if self.type == "openai":
            if self.url is None:
                self.url = OPENAI_DEFAULT_BASE_URL
            if self.api_key is None:
                raise ValueError(f"backend {self.name!r}: openai backends require an api_key")
        elif self.url is None:
            raise ValueError(f"backend {self.name!r}: url is required for type {self.type!r}")
        return self


class VectorStoreConfig(BaseModel):
    type: Literal["qdrant"]
    url: str
    collection: str
    # The embedding model whose vectors this collection holds — the routing key that
    # sends /vectors and /search traffic for that model to this collection.
    model: str
    name: str = ""
    # Free-form corpus bookkeeping, stamped into every upserted payload beneath
    # per-request metadata and echoed on GET /backends.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve(self):
        if not self.name:
            self.name = self.type
        return self


class AppConfig(BaseModel):
    health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL
    backends: list[EmbeddingBackendConfig] = Field(min_length=1)
    vector_stores: list[VectorStoreConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self):
        names = [b.name for b in self.backends] + [s.name for s in self.vector_stores]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate backend names: {dupes}")
        models = [s.model for s in self.vector_stores]
        dupes = sorted({m for m in models if models.count(m) > 1})
        if dupes:
            raise ValueError(f"multiple vector stores claim the same model: {dupes}")
        return self


def _from_legacy_env() -> AppConfig:
    # os.environ[...] lookups keep the pre-0.5 fail-fast KeyError on missing vars.
    backend_type = os.environ.get("BACKEND", "transformers-inference")
    model = os.environ["MODEL_NAME"]

    if backend_type == "openai":
        backend = EmbeddingBackendConfig(
            type=backend_type,
            model=model,
            url=os.environ.get("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE_URL),
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif backend_type in ("transformers-inference", "llamacpp"):
        backend = EmbeddingBackendConfig(
            type=backend_type,
            model=model,
            url=os.environ["EMBEDDINGS_URL"],
            api_key=os.environ.get("OPENAI_API_KEY") if backend_type == "llamacpp" else None,
        )
    else:
        raise ValueError(f"Unknown BACKEND: {backend_type!r}")

    stores: list[VectorStoreConfig] = []
    store_type = os.environ.get("VECTOR_STORE")
    if store_type:
        if store_type != "qdrant":
            raise ValueError(f"Unknown VECTOR_STORE: {store_type!r}")
        stores.append(
            VectorStoreConfig(
                type="qdrant",
                url=os.environ["QDRANT_URL"],
                collection=os.environ["QDRANT_COLLECTION"],
                model=model,
            )
        )

    return AppConfig(backends=[backend], vector_stores=stores)


def load_config() -> AppConfig:
    if "MODEL_NAME" in os.environ:
        config = _from_legacy_env()
    else:
        path = os.environ.get("CONFIG_FILE", DEFAULT_CONFIG_FILE)
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No config found: set MODEL_NAME (legacy env mode) or provide a config "
                f"file at {path!r} (override the path with CONFIG_FILE)"
            ) from None
        config = AppConfig.model_validate(raw)

    if "HEALTH_CHECK_INTERVAL" in os.environ:
        config.health_check_interval = float(os.environ["HEALTH_CHECK_INTERVAL"])
    return config

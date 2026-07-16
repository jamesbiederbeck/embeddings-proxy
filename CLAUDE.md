# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small FastAPI proxy that sits in front of an embeddings backend and caches vectors in Redis, keyed by a hash of `(backend, model, text)`. Optionally upserts every embedded vector into Qdrant and exposes semantic search.

```
client → embeddings-proxy → Redis (cache check)
                          ↘ embedding backend (cache miss)
                          ↘ Qdrant (optional, upsert + search)
```

Three modules:

- `app.py` — FastAPI app, Redis cache path, endpoints, readiness/liveness probes.
- `backends.py` — the `Backend` protocol (`embed()`, `check()`) and implementations: `TransformersInferenceBackend`, `OpenAIBackend`, `LlamaCppBackend` (a thin subclass of the OpenAI one using llama.cpp's OpenAI-compatible endpoint). `build_backend()` selects via the `BACKEND` env var, defaulting to `transformers-inference`.
- `vector_store.py` — the `VectorStore` protocol (`upsert()`, `search()`, `check()`) and `QdrantVectorStore`. `build_vector_store()` returns `None` unless `VECTOR_STORE` is set; `qdrant-client` is imported lazily so it's only needed when the feature is enabled.

## Endpoints

- `POST /vectors` — `{"text": "..."}` → `{"text", "vector", "dim", "cached"}`. Redis first; on miss calls `backend.embed()` and writes back. If a vector store is configured, upserts the vector (best-effort — store failures are logged, not surfaced, except dimension mismatches which fail loudly).
- `POST /search` — `{"text": "...", "top_k": N}` → nearest neighbours with score + payload. `501` when no vector store is configured.
- `POST /v1/embeddings` — OpenAI-compatible adapter: `{"input": "..." | ["...", ...], "model": "..."?}` → `{"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": i}, ...], "model", "usage"}`. Goes through the same Redis cache/backend path as `/vectors` (batched with `asyncio.gather`, order preserved) but has no vector-store side effect — it's a pure embeddings passthrough for OpenAI-client-compatible callers (LangChain, LiteLLM, etc.). `model` in the response echoes the request's `model` if given, else `backend.model_name`; it's not validated against the configured backend. `usage` token counts are always `0` — no tokenizer is wired up.
- `GET /.well-known/ready` — readiness: pings Redis + backend (+ vector store if configured), each time-boxed to `PROBE_TIMEOUT` (2s); `503` with per-check detail on failure.
- `GET /.well-known/live` — liveness: always `200`. Use for restart probes so backend blips don't restart-loop the container.

## Cache key

`sha256("<backend.name>:<MODEL_NAME>:<text>")` (see `cache_key()`), so different backends/models can never collide in Redis. This replaced the older `sha256("<MODEL_NAME>:<text>")` scheme — the change invalidated pre-existing cache entries once (they just get re-embedded). Cache is unbounded — no TTL/eviction here; Redis persistence (`--appendonly yes`) is configured on the Redis side.

## Configuration

Config is read at startup; missing required vars raise `KeyError` (fail-fast). Only the vars the selected backend/store needs are required.

| Variable | Description |
|---|---|
| `REDIS_URL` | Redis connection string. Always required. |
| `BACKEND` | `transformers-inference` (default), `openai`, or `llamacpp`. |
| `MODEL_NAME` | Model identifier — part of the cache key; sent as `model` by the OpenAI-style backends. Always required. |
| `EMBEDDINGS_URL` | Upstream base URL for `transformers-inference` and `llamacpp` backends. |
| `OPENAI_BASE_URL` | Base URL for the `openai` backend (default `https://api.openai.com`). |
| `OPENAI_API_KEY` | Bearer token. Required for `openai`, optional for `llamacpp`. |
| `VECTOR_STORE` | Optional; `qdrant` enables the vector store. Unset = pure caching proxy, unchanged behavior. |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Required when `VECTOR_STORE=qdrant`. Collection auto-created on first use with the backend's vector dim; a dim mismatch with an existing collection fails loudly. |

## Development

Dependencies are managed with `uv` (see `pyproject.toml` / `uv.lock`). Dev tools live in the `dev` dependency group; `qdrant-client` is the `qdrant` extra.

```
uv sync                                              # install deps + dev group
uv run ruff check .                                  # lint (must pass)
uv run pytest                                        # unit tests — no real Redis/backend needed
uv run uvicorn app:app --host 0.0.0.0 --port 8081 --reload   # run locally
```

Tests mock HTTP via `httpx.MockTransport` and Redis via `fakeredis`; `tests/conftest.py` sets the env vars `app.py` requires at import time. When adding a backend, add a unit test asserting the exact request shape sent and the vector extracted from a canned response — wrong JSON field names still return 200 and cache nonsense otherwise.

## Deployment

Built as a Docker image (Python 3.12-slim, deps installed via `uv sync --locked --no-dev`). Exposes port `8081`. Not intended to be run standalone outside a container. Build with `--build-arg UV_SYNC_EXTRA_ARGS="--extra qdrant"` to include vector-store support in the image — the CI build workflow passes this, so the published image (`v0.2.1+`) supports `VECTOR_STORE=qdrant` out of the box.

CI: pushes to any branch run ruff + pytest (`.gitea/workflows/test.yml`); pushing a tag matching `v*` triggers the build workflow (`.gitea/workflows/build.yml`) that pushes the image to `git.hiddencove.xyz/victor/embeddings-proxy`.

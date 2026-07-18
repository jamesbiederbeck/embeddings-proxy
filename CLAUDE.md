# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small FastAPI proxy that sits in front of an embeddings backend and caches vectors in Redis, keyed by a hash of `(backend, model, text)`. Optionally upserts every embedded vector into Qdrant and exposes semantic search.

```
client → embeddings-proxy → Redis (cache check)
                          ↘ embedding backend (cache miss)
                          ↘ Qdrant (optional, upsert + search)
```

Four modules:

- `app.py` — FastAPI app, Redis cache path, endpoints, readiness/liveness probes.
- `backends.py` — the `Backend` protocol (`embed()`, `check()`) and implementations: `TransformersInferenceBackend`, `OpenAIBackend`, `LlamaCppBackend` (a thin subclass of the OpenAI one using llama.cpp's OpenAI-compatible endpoint). `build_backend()` selects via the `BACKEND` env var, defaulting to `transformers-inference`.
- `vector_store.py` — the `VectorStore` protocol (`upsert()`, `search()`, `set_scopes()`, `delete()`, `check()`) and `QdrantVectorStore`. `build_vector_store()` returns `None` unless `VECTOR_STORE` is set; `qdrant-client` is imported lazily so it's only needed when the feature is enabled. `point_id(text)` (module-level, shared with `app.py`) is the deterministic Qdrant point id for a piece of text — content-addressed, so re-embedding identical text always resolves to the same point. `upsert()` takes optional `metadata` (merged into the payload, `text` always wins on key collision) and `scopes` (a list of coarse tags — e.g. codebase/corpus names — written to an indexed `scopes` payload field so `/search` can filter by it); it also stamps the payload's `content_hash` (full `sha256(text)`, distinct from the truncated point-id UUID). `search()` takes an optional `scope` to filter via a Qdrant payload-indexed match. `set_scopes()` does a targeted overwrite of just the `scopes` payload key (used after a provenance change, not on every upsert); `delete()` removes a point outright (used when its last location is unlinked).
- `provenance.py` — Redis-backed location bookkeeping, kept separate from the Qdrant payload specifically because Qdrant has no atomic array-append: since a point is content-addressed (one per unique text), the same content can legitimately live at many file paths/URNs, and each needs to be added/removed without racing concurrent writers. `prov:<point_id>` is a Redis hash of `location URI → JSON(scope + metadata)`; `loc:<uri>` is a reverse-index set of point ids, letting a location be unlinked without knowing its content hash. `add_location()`/`remove_location()` return the point's up-to-date scope set so the caller can push it to Qdrant via `set_scopes()`; `get_locations()` is a pipelined batch fetch for `/search`.

## Endpoints

- `POST /vectors` — `{"text": "...", "metadata": {...}?, "location": "..."?, "scope": "..."?, "location_metadata": {...}?}` → `{"text", "vector", "dim", "cached"}`. Redis first; on miss calls `backend.embed()` and writes back. If a vector store is configured: the Qdrant vector write only happens on a genuine cache miss (a cache hit means the point already exists); `location`/`scope` (scope required if location is given, `422` otherwise) are recorded in Redis provenance on every call, hit or miss, and the point's `scopes` payload field is recomputed and pushed to Qdrant. All of this is best-effort — store failures are logged, not surfaced, except dimension mismatches which fail loudly.
- `POST /search` — `{"text": "...", "top_k": N, "scope": "..."?}` → `{"query", "results": [{"page_content", "metadata", "locations", "score"}, ...]}` — a LangChain `Document`-like shape (`page_content` is the stored text, `metadata` is any other payload fields excluding this proxy's own `content_hash`/`scopes` bookkeeping, `locations` is the list of `{"uri", "scope", ...}` provenance entries for that point). `scope`, if given, filters results to points whose `scopes` field contains it. `501` when no vector store is configured.
- `DELETE /locations` — `{"uri": "..."}` → `{"uri", "affected"}`. Unlinks one location from every point that references it (via Redis provenance); a point left with zero locations is deleted from Qdrant outright, otherwise its `scopes` payload field is recomputed. This is how a file's index entry is removed on delete, or cleared before re-indexing an edited file. `501` when no vector store is configured.
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

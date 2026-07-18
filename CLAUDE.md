# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small FastAPI proxy that sits in front of one or more embeddings backends and caches vectors in Redis, keyed by a hash of `(backend type, model, [task,] text)`. Optionally upserts every embedded vector into Qdrant and exposes semantic search. Multiple backends and Qdrant collections can be configured at once; requests are routed to the first *healthy* backend serving the requested model (health is polled in the background, never checked on the request path), and the vector store is picked by the resolved model.

```
client → embeddings-proxy → Redis (cache check)
                          ↘ embedding backend (cache miss; first healthy match)
                          ↘ Qdrant (optional, upsert + search; collection per model)
```

Six modules:

- `app.py` — FastAPI app, Redis cache path, endpoints, request routing (404 = not configured, 503 = configured but unhealthy), lifespan (startup health sweep + background poller task), readiness/liveness probes.
- `config.py` — startup configuration: legacy single-backend env vars (used whenever `MODEL_NAME` is set) or a YAML file (`CONFIG_FILE`, default `./config.yaml`) declaring arrays of embedding backends and vector stores. Pydantic-validated, fail-fast (see `config.example.yaml`).
- `registry.py` — `Registry` holds all built backends/stores plus a health map written only by the poller (`poll_once()`/`poll_forever()`; each check time-boxed to `PROBE_TIMEOUT`). `select_embedding(name=, model=)` implements the routing contract (`NotConfiguredError` → 404, `NoHealthyBackendError` → 503; config order = priority); `store_for_model()` picks the collection.
- `backends.py` — `Backend` base class (`name` = per-instance config name, `type` = implementation kind, `model`, `check()`), `EmbeddingBackend` (adds `embed()` and optional `embedding_task` — a declarative task-type tag for instruct-aware models that joins the cache key and upserted payloads; the upstream is assumed to be configured for it, the proxy never rewrites text) and implementations: `TransformersInferenceBackend`, `OpenAIBackend`, `LlamaCppBackend` (a thin subclass of the OpenAI one using llama.cpp's OpenAI-compatible endpoint). `build_backends()` constructs them from config entries.
- `vector_store.py` — `VectorStoreBackend` (a `Backend` subclass: `upsert()`, `search()`, `set_scopes()`, `delete()`; its `model` is the routing key tying a collection to the embedding model whose vectors it holds) and `QdrantVectorStore`; `qdrant-client` is imported lazily so it's only needed when a store is configured. `point_id(text)` (module-level, shared with `app.py`) is the deterministic Qdrant point id for a piece of text — content-addressed, so re-embedding identical text always resolves to the same point. `upsert()` takes optional `metadata` (merged into the payload above the store's config `metadata` (`base_metadata`); `text` always wins on key collision) and `scopes` (a list of coarse tags — e.g. codebase/corpus names — written to an indexed `scopes` payload field so `/search` can filter by it); it also stamps the payload's `content_hash` (full `sha256(text)`, distinct from the truncated point-id UUID). `search()` takes an optional `scope` to filter via a Qdrant payload-indexed match. `set_scopes()` does a targeted overwrite of just the `scopes` payload key (used after a provenance change, not on every upsert); `delete()` removes a point outright (used when its last location is unlinked).
- `provenance.py` — Redis-backed location bookkeeping, kept separate from the Qdrant payload specifically because Qdrant has no atomic array-append: since a point is content-addressed (one per unique text), the same content can legitimately live at many file paths/URNs, and each needs to be added/removed without racing concurrent writers. `prov:<point_id>` is a Redis hash of `location URI → JSON(scope + metadata)`; `loc:<uri>` is a reverse-index set of point ids, letting a location be unlinked without knowing its content hash. `add_location()`/`remove_location()` return the point's up-to-date scope set so the caller can push it to Qdrant via `set_scopes()`; `get_locations()` is a pipelined batch fetch for `/search`.

## Endpoints

Routing (applies to `/vectors`, `/search`, `/v1/embeddings`): optional `backend` (instance name) and/or `model` request fields select the embedding backend. Named backend not configured, model served by no configured backend, or name/model mismatch → `404`; configured but no healthy candidate → `503`; no params → first healthy backend in config order. Health comes from the background poller, never a synchronous check.

- `POST /vectors` — `{"text": "...", "metadata": {...}?, "location": "..."?, "scope": "..."?, "location_metadata": {...}?, "model": "..."?, "backend": "..."?}` → `{"text", "vector", "dim", "cached"}`. Redis first; on miss calls the resolved backend's `embed()` and writes back. If a vector store is configured for the resolved model: the Qdrant vector write only happens on a genuine cache miss (a cache hit means the point already exists); `location`/`scope` (scope required if location is given, `422` otherwise) are recorded in Redis provenance on every call, hit or miss, and the point's `scopes` payload field is recomputed and pushed to Qdrant. All of this is best-effort — store failures are logged, not surfaced, except dimension mismatches which fail loudly.
- `POST /search` — `{"text": "...", "top_k": N, "scope": "..."?, "model": "..."?, "backend": "..."?}` → `{"query", "results": [{"page_content", "metadata", "locations", "score"}, ...]}` — a LangChain `Document`-like shape (`page_content` is the stored text, `metadata` is any other payload fields excluding this proxy's own `content_hash`/`scopes` bookkeeping, `locations` is the list of `{"uri", "scope", ...}` provenance entries for that point). `scope`, if given, filters results to points whose `scopes` field contains it. Searches the collection tied to the resolved model; `501` when no vector store is configured for it.
- `DELETE /locations` — `{"uri": "..."}` → `{"uri", "affected"}`. Unlinks one location from every point that references it (via Redis provenance); a point left with zero locations is deleted from Qdrant outright, otherwise its `scopes` payload field is recomputed. Provenance doesn't record which collection a point landed in, so the cleanup runs against every configured store (deleting a nonexistent point is a Qdrant no-op). This is how a file's index entry is removed on delete, or cleared before re-indexing an edited file. `501` when no vector store is configured.
- `POST /v1/embeddings` — OpenAI-compatible adapter: `{"input": "..." | ["...", ...], "model": "..."?, "backend": "..."?}` → `{"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": i}, ...], "model", "usage"}`. Goes through the same Redis cache/backend path as `/vectors` (batched with `asyncio.gather`, order preserved) but has no vector-store side effect — it's a pure embeddings passthrough for OpenAI-client-compatible callers (LangChain, LiteLLM, etc.). `model` routes per the contract above (pre-0.5 it was echo-only; an unknown model now 404s) and the response echoes the resolved backend's model. `usage` token counts are always `0` — no tokenizer is wired up.
- `GET /backends` — the registry with live health: `{"backends": [{"name", "type", "kind": "embedding"|"vector_store", "model", "embedding_task"? , "collection"?, "metadata"?, "healthy", "error"}]}`. `?model=` / `?name=` filter — this answers both "which backends serve model X" and "what does backend Y serve".
- `GET /.well-known/ready` — readiness: pings Redis live (time-boxed to `PROBE_TIMEOUT`, 2s) and reports the poller's last sweep per backend/store; `503` unless Redis is up and ≥1 embedding backend is healthy.
- `GET /.well-known/live` — liveness: always `200`. Use for restart probes so backend blips don't restart-loop the container.

## Cache key

`sha256("<backend type>:<model>:<text>")`, or `sha256("<type>:<model>:<embedding_task>:<text>")` when a task is configured (see `cache_key()`), so different backend types/models/tasks can never collide in Redis. Keyed by `type`, not instance name, so two instances of the same type serving the same model share cache (cheap failover) — and the bytes match the pre-0.5 `<backend.name>:<MODEL_NAME>` scheme exactly, so upgrading invalidates nothing. Cache is unbounded — no TTL/eviction here; Redis persistence (`--appendonly yes`) is configured on the Redis side.

## Configuration

Two modes, checked in order at startup (fail-fast on anything missing or invalid):

1. **Legacy env vars** — used whenever `MODEL_NAME` is set; behaves exactly like pre-0.5 (one backend, optionally one Qdrant collection tied to that model).
2. **YAML config file** — `CONFIG_FILE` (default `./config.yaml`), declaring `backends:` (array; `name`, `type`, `model`, `url`, `api_key`/`api_key_env`, `embedding_task`?) and `vector_stores:` (array; `name`, `type: qdrant`, `url`, `collection`, `model`, `metadata`?). Array order is routing priority. See `config.example.yaml` for the full annotated schema. Names must be unique and at most one store per model.

`HEALTH_CHECK_INTERVAL` (env, seconds; or `health_check_interval` in the file, default 30) sets the poll cadence.

| Variable | Description |
|---|---|
| `REDIS_URL` | Redis connection string. Always required. |
| `CONFIG_FILE` | Path to the YAML config (default `./config.yaml`). Ignored when `MODEL_NAME` is set. |
| `HEALTH_CHECK_INTERVAL` | Seconds between health sweeps; overrides the config file. |
| `BACKEND` | Legacy mode: `transformers-inference` (default), `openai`, or `llamacpp`. |
| `MODEL_NAME` | Legacy mode trigger + model identifier — part of the cache key; sent as `model` by the OpenAI-style backends. |
| `EMBEDDINGS_URL` | Legacy mode: upstream base URL for `transformers-inference` and `llamacpp` backends. |
| `OPENAI_BASE_URL` | Legacy mode: base URL for the `openai` backend (default `https://api.openai.com`). |
| `OPENAI_API_KEY` | Legacy mode: bearer token. Required for `openai`, optional for `llamacpp`. |
| `VECTOR_STORE` | Legacy mode, optional; `qdrant` enables the vector store. Unset = pure caching proxy. |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Legacy mode, required when `VECTOR_STORE=qdrant`. Collection auto-created on first use with the backend's vector dim; a dim mismatch with an existing collection fails loudly. |

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

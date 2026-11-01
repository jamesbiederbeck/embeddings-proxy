# embeddings-proxy

A small FastAPI proxy that sits in front of an embeddings server and caches vectors in Redis, keyed by a hash of `(backend, model, text)`. Supports multiple upstream backends ([`semitechnologies/transformers-inference`](https://hub.docker.com/r/semitechnologies/transformers-inference), OpenAI-compatible APIs, llama.cpp) and, optionally, semantic search via Qdrant.

## Why

Embedding inference is the expensive part of an embeddings pipeline. If the same text gets embedded repeatedly (e.g. repeated queries, reprocessing, retries), this proxy skips the model call and returns the cached vector instead.

## How it works

```
client → embeddings-proxy → Redis (cache check)
                          ↘ embedding backend (cache miss)
                          ↘ Qdrant (optional, upsert + search)
```

- `POST /vectors` — accepts `{"text": "..."}`, returns `{"text", "vector", "dim", "cached"}`. Checks Redis first; on miss, calls the configured backend and writes the result back to Redis.
- `POST /search` — accepts `{"text": "...", "top_k": 5}`, embeds the query (using the cache) and returns nearest neighbours from the vector store. Returns `501` if no vector store is configured.
- `GET /.well-known/ready` — readiness: pings Redis, the backend, and the vector store (if configured); returns `503` with per-dependency detail when any required dependency is unreachable. Checks are time-boxed to 2s each.
- `GET /.well-known/live` — liveness: always returns `{"status": "ok"}`. Point restart-triggering probes here so a transient backend blip doesn't restart-loop the container; point traffic-gating readiness probes at `/.well-known/ready`.
- Cache key is `sha256("<backend>:<model>:<text>")`, so vectors from different backends or models can never collide. (Older deployments used `sha256("<model>:<text>")` — upgrading invalidates existing cache entries once; they'll simply be re-embedded.)
- Cache is unbounded — Redis is run with `--appendonly yes` for persistence, no TTL/eviction is configured.

## Backends

Selected with the `BACKEND` env var. Defaults to `transformers-inference`, so existing deployments need no config change.

| `BACKEND` | Upstream API | Required config |
|---|---|---|
| `transformers-inference` (default) | `POST {EMBEDDINGS_URL}/vectors` | `EMBEDDINGS_URL`, `MODEL_NAME` |
| `openai` | `POST {OPENAI_BASE_URL}/v1/embeddings` | `OPENAI_API_KEY`, `MODEL_NAME`; `OPENAI_BASE_URL` optional (default `https://api.openai.com`) |
| `llamacpp` | `POST {EMBEDDINGS_URL}/v1/embeddings` | `EMBEDDINGS_URL`, `MODEL_NAME`; `OPENAI_API_KEY` optional |

Notes:

- `openai` works with anything speaking the OpenAI embeddings shape: OpenAI itself, LiteLLM, vLLM, text-embeddings-inference, etc.
- `llamacpp` uses the server's OpenAI-compatible `/v1/embeddings` endpoint (stable across llama.cpp versions, unlike the native `/embedding` shape). Run the server with `--embeddings`. It's effectively the `openai` backend pointed at `EMBEDDINGS_URL` with no API key required — but it has its own cache namespace.
- `MODEL_NAME` is always required: it is part of the cache key, and for OpenAI-style backends it's also sent as the request `model`.

## Configuration

| Variable | Description |
|---|---|
| `REDIS_URL` | Redis connection string (e.g. `redis://redis:6379/0`). Always required. |
| `BACKEND` | Backend selector: `transformers-inference` (default), `openai`, or `llamacpp`. |
| `MODEL_NAME` | Model identifier — part of the cache key; must match the model actually served upstream. Always required. |
| `EMBEDDINGS_URL` | Base URL of the upstream server, for the `transformers-inference` and `llamacpp` backends. |
| `OPENAI_BASE_URL` | Base URL for the `openai` backend (default `https://api.openai.com`). |
| `OPENAI_API_KEY` | API key, sent as `Authorization: Bearer ...`. Required for `openai`, optional for `llamacpp`. |
| `VECTOR_STORE` | Optional. Set to `qdrant` to enable the vector store (see below). Unset = disabled. |
| `QDRANT_URL` | Qdrant base URL (e.g. `http://qdrant:6333`). Required when `VECTOR_STORE=qdrant`. |
| `QDRANT_COLLECTION` | Qdrant collection name. Required when `VECTOR_STORE=qdrant`. |

Missing required variables raise `KeyError` at startup (fail-fast); only the variables the selected backend/store needs are required.

## Vector store / search (optional)

Off by default: with no `VECTOR_STORE` set, the proxy behaves exactly as a caching proxy — no Qdrant client is constructed and `qdrant-client` doesn't need to be installed.

With `VECTOR_STORE=qdrant` + `QDRANT_URL` + `QDRANT_COLLECTION`:

- Every text embedded via `/vectors` is upserted into the collection with a deterministic id (derived from the text hash) and the text stored in the payload. Indexing is **best-effort**: a Qdrant outage is logged but does not fail the `/vectors` response.
- `POST /search {"text": "...", "top_k": 5}` returns `{"text", "results": [{"id", "score", "payload"}]}`.
- `score_threshold` (optional float) drops results scoring below it before `top_k` is applied — e.g. `{"text": "...", "top_k": 5, "score_threshold": 0.75}`. It's a raw cosine-similarity cutoff (the collection's distance metric is `COSINE`), not a normalized/calibrated confidence value, so a sensible threshold depends on your embedding model. Omitting it preserves today's behavior (no filtering beyond `top_k`); setting it can mean `results` comes back empty when nothing clears the bar.
- `exact` (optional bool, default `false`) forces a brute-force cosine-similarity scan of every point in the collection instead of the default approximate HNSW search. Useful for occasional threshold-tuning/introspection queries (e.g. building a similarity histogram from real scores) where you want the true nearest neighbors, not HNSW's approximation — not for routine search traffic, since cost scales with collection size rather than sublinearly like HNSW. `top_k`/`limit` still caps how many results come back; `exact=True` does not by itself return the whole collection — pass a `top_k` at least as large as the scope's point count if you want (near-)all scores.
- `histogram` (optional bool, default `false`) adds a `histogram` field to the response: an adaptive-bin histogram plus distributional stats (`mean`, `std`, `median`, `skewness`, `excess_kurtosis`, a D'Agostino-Pearson normality test, and `above_mean_plus_2std`/`above_mean_plus_3std` counts) computed over cosine similarity scores — useful for showing how distinctive a match is, or for threshold-tuning, without shipping a per-document score array to the caller. This **always forces an exact (brute-force) scan, regardless of the request's own `exact` value**, and is computed over up to `histogram_limit` points (default `10000`) in the resolved `scope` — cost scales with however many points that is, so this isn't free on a large collection; it's meant for occasional introspection, not routine per-keystroke search traffic. `results`/`top_k` are unaffected; the histogram is computed from a separate scan. Returns `histogram: null` if the scope has no points. See `histogram.py` for the exact stats (ported from `photo-embedder`'s `histogram_from_sims()`).
- The collection is auto-created on first use, sized to the backend's vector dimension (cosine distance). If an existing collection's dimension doesn't match the backend's, requests fail loudly rather than storing garbage.

`qdrant-client` is an optional extra (`uv sync --extra qdrant` locally, or build the image with `--build-arg UV_SYNC_EXTRA_ARGS="--extra qdrant"`); it also pulls in `numpy`/`scipy`, used only by the `histogram` field above. The CI-published image (`v0.2.1+`) is built with this extra baked in, so `VECTOR_STORE=qdrant` works out of the box; a plain `docker build` without the build arg stays lean.

## Development

```
uv sync                 # install deps incl. dev group (pytest, ruff, ...)
uv run ruff check .     # lint
uv run pytest           # unit tests (no Redis/backends needed — HTTP and Redis are mocked)
uv run uvicorn app:app --host 0.0.0.0 --port 8081 --reload
```

## Running

Built and run as a container; not intended to be run standalone. Expects a Redis instance and the configured backend reachable over the network. Exposes port `8081`.

`docker-compose.yml` runs the full stack — proxy, Redis (`--appendonly yes`), and a `transformers-inference` backend with `all-MiniLM-L6-v2`:

```
docker compose up                    # caching proxy only
docker compose --profile search up   # also start Qdrant (uncomment the VECTOR_STORE env vars)
```

The proxy's compose healthcheck hits `/.well-known/ready`, so the container only reports healthy when Redis and the backend are reachable.

## CI

- Pushes to any branch run lint + tests (`.github/workflows/test.yml`, mirrored from `.gitea/workflows/test.yml`).
- The author's own deployment additionally uses a Gitea Actions workflow (`.gitea/workflows/build.yml`) that builds and pushes a Docker image to a private registry on tagged (`v*`) pushes — that workflow is specific to that deployment, not required for local development or contributions.

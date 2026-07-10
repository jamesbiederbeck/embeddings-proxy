# embeddings-proxy

A small FastAPI proxy that sits in front of a [`semitechnologies/transformers-inference`](https://hub.docker.com/r/semitechnologies/transformers-inference) embeddings server and caches vectors in Redis, keyed by a hash of `(model, text)`.

## Why

Sentence-transformer inference is the expensive part of an embeddings pipeline. If the same text gets embedded repeatedly (e.g. repeated queries, reprocessing, retries), this proxy skips the model call and returns the cached vector instead.

## How it works

```
client → embeddings-proxy → Redis (cache check)
                          ↘ transformers-inference (cache miss)
```

- `POST /vectors` — accepts `{"text": "..."}`, returns `{"text", "vector", "dim", "cached"}`.
- Cache key is `sha256("<MODEL_NAME>:<text>")`. Changing the model changes the key space, so switching models can't return stale vectors from a different model.
- Cache is unbounded — Redis is run with `--appendonly yes` for persistence, no TTL/eviction is configured.
- `GET /.well-known/ready` — health check, always returns `{"status": "ok"}`.

## Configuration

Set via environment variables:

| Variable | Description |
|---|---|
| `EMBEDDINGS_URL` | Base URL of the upstream `transformers-inference` server (e.g. `http://embeddings:8080`) |
| `MODEL_NAME` | Identifier for the model being served upstream — must match whatever model is actually running, since it's part of the cache key |
| `REDIS_URL` | Redis connection string (e.g. `redis://redis:6379/0`) |

## Running

Built and run as a container; not intended to be run standalone. Expects a Redis instance and a `transformers-inference` instance reachable over the network. See the Dockerfile for the image build (Python 3.12, dependencies managed with `uv`).

Exposes port `8081`.

## CI

Tagged pushes (`v*`) trigger a Gitea Actions build that pushes an image to `git.hiddencove.xyz/victor/embeddings-proxy`.

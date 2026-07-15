# Roadmap

Direction for growing `embeddings-proxy` from a single-backend caching proxy into a small,
general-purpose embeddings gateway. Homelab / personal-infra project: one maintainer, small
scope, low ceremony. Prefer minimal, composable additions over frameworks.

## Themes

1. **Multiple embedding backends** — make the upstream pluggable (transformers-inference, OpenAI-compatible, llama.cpp) instead of hardcoded.
2. **Optional vector store / search** — an opt-in "store the vector and let me search it" feature (Qdrant), off by default, that never becomes mandatory.
3. **Enough quality + ops hardening to support the above** — a light test/lint harness and a readiness check that tells the truth.

## Sequencing

The backend abstraction is the enabling refactor and the first real architectural fork — it
must land before any second backend. The vector store is a separate axis that composes with any
backend and should be built independently. Tests are most valuable once there are multiple
backend adapters to protect.

```
TICKET-001 (backend abstraction)
    ├─ TICKET-002 (OpenAI-compatible backend)
    ├─ TICKET-003 (llama.cpp backend)
    └─ TICKET-004 (tests + lint)        # protects the adapters
TICKET-005 (optional Qdrant vector store + /search)   # independent axis
TICKET-006 (meaningful readiness check)               # ops hardening, any time
```

## Backward compatibility guardrails

- The existing deployment sets only `EMBEDDINGS_URL` / `MODEL_NAME` / `REDIS_URL`. `BACKEND`
  must default to the transformers-inference backend so that deployment keeps working untouched.
- The Docker build copies `app.py` explicitly; splitting into modules requires updating the
  `COPY` line (or packaging) or the image build breaks.
- New dependencies must regenerate `uv.lock` (the image builds with `uv sync --locked`).
- Optional features (Qdrant, OpenAI SDK if used) should be extras so the base image stays lean.
- CI builds only on `v*` tags via Gitea Actions — additive test workflows should not disturb it.

## Recently Ticketed

- **TICKET-001** — Introduce a pluggable backend abstraction (refactor, no behavior change). *In Progress / Ready.*
- **TICKET-002** — Add an OpenAI-compatible embeddings backend.
- **TICKET-003** — Add a llama.cpp server embeddings backend.
- **TICKET-004** — Add a lightweight test + lint foundation for backend adapters.
- **TICKET-005** — Optional pluggable vector store with Qdrant + a `/search` endpoint.
- **TICKET-006** — Make the readiness check actually verify dependencies.

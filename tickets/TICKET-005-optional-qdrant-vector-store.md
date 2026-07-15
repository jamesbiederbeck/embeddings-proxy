# TICKET-005: Optional pluggable vector store with Qdrant + a /search endpoint

**Status**: Ready
**Priority**: Medium
**Type**: Feature
**Labels**: vector-store, qdrant, search, optional
**Estimated Effort**: L
**Related Roadmap Item**: Optional vector store / search
**Related Issues**: N/A

---

## User Story

As a **homelab operator building a small retrieval feature**, I want the proxy to optionally persist embedded vectors into a vector store and let me query nearest neighbours by text so that I get lightweight semantic search without standing up a separate indexing service.

## Context & Background

Today the proxy only caches vectors in Redis for reuse; it has no search capability. The user wants a *minimal, optional* "store the vector and let me search it" feature, e.g. backed by Qdrant, that stays fully optional and does not become mandatory infrastructure. The natural MVP: the proxy already computes every vector on the `/vectors` path, so it can upsert those into a vector store when one is configured, and expose a `/search` endpoint. When no store is configured, the proxy behaves exactly as it does today.

## Acceptance Criteria

- [ ] A vector-store interface exists (upsert + search), mirroring the backend-abstraction style from TICKET-001, with a `QdrantVectorStore` implementation.
- [ ] The vector store is **off by default**: with no `VECTOR_STORE` config set, the proxy runs exactly as today (caching proxy only), the store client is never constructed, and `qdrant-client` is not required to start.
- [ ] When enabled (e.g. `VECTOR_STORE=qdrant` + `QDRANT_URL` + a collection name), each successful `/vectors` embedding is upserted into the store with the text (or a hash) as id and the text stored in the payload.
- [ ] A `POST /search {"text": "...", "top_k": N}` endpoint embeds the query via the configured backend (using the cache) and returns the nearest neighbours with their score and stored payload.
- [ ] When the store is disabled, `/search` returns a clear 404/501 rather than a 500.
- [ ] Collection dimensionality/creation is handled (auto-create on first use with the backend's vector dimension, or documented manual setup).
- [ ] `qdrant-client` is an optional/extra dependency so the base image stays lean when the feature is unused; `uv.lock` updated.
- [ ] `README.md` / `CLAUDE.md` document the feature as optional, including the enable path and the "disabled = unchanged behavior" guarantee.

## Implementation Notes

- Keep the MVP to "auto-index everything embedded + search". Explicit store semantics (caller-supplied ids, payload metadata, deletes, multiple collections) are a deliberate follow-up, not this ticket.
- Upsert failures should not break the `/vectors` cache path — a store write failing shouldn't fail the embedding response (log and continue, or make indexing best-effort). Decide and document.
- Reuse the backend's vector dimension to size the Qdrant collection; a mismatch between backend model and existing collection dim must fail loudly.
- This is a separate architectural axis from backends — it composes with any backend but should not entangle the backend interface.

## Out of Scope

- Caller-supplied document ids, metadata/payload schemas, deletion, and multi-collection routing.
- Reranking, hybrid (keyword+vector) search, filtering.
- Vector stores other than Qdrant (the interface should allow them; implementing them is future work).

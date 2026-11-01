# TICKET-007: Support a score threshold on /search

**Status**: Ready
**Priority**: Medium
**Type**: Feature
**Labels**: search, vector-store
**Estimated Effort**: S
**Related Roadmap Item**: N/A
**Related Issues**: N/A (prerequisite for `project-management`'s planned "find a repo" skill — see that repo's search feature)

---

## User Story

As a **caller of `/search`** (e.g. `project-management`'s semantic/hybrid project search), I want to pass a minimum similarity score so that weak, effectively-unrelated matches are dropped server-side instead of always getting `top_k` results back regardless of how poor the best match is.

## Context & Background

`POST /search` (`app.py`) always returns up to `top_k` nearest points, ranked, with no way to say "nothing scored high enough to be a real match." For a caller trying to answer "is there anything relevant at all," the only signal today is eyeballing the raw `score` values in the response and guessing a cutoff per call site.

Qdrant's `query_points` already accepts a `score_threshold` parameter (confirmed present in the installed `qdrant-client`'s signature) that filters server-side — it is simply not threaded through this proxy's `SearchRequest` → `QdrantVectorStore.search()` → `query_points` call chain.

## Acceptance Criteria

- [ ] `SearchRequest` (`app.py`) gains an optional `score_threshold: float | None = None` field.
- [ ] `VectorStoreBackend.search()` / `QdrantVectorStore.search()` (`vector_store.py`) accept and forward `score_threshold` to `query_points`.
- [ ] When `score_threshold` is omitted, behavior is unchanged (no filtering beyond `top_k`), so this is backward compatible for existing callers.
- [ ] When set, points scoring below it are excluded from the response — including the case where this means `results` comes back empty.
- [ ] Documented in `README.md`'s `/search` section, including that the threshold is a raw cosine-similarity cutoff (collection distance metric is `COSINE`), not a normalized/calibrated confidence value.

## Implementation Notes

- This is a straight-through parameter addition, not new search logic — no new endpoint needed.
- No histogram/distribution endpoint exists in Qdrant for similarity scores (its facet API only covers payload field value counts), so this ticket does not attempt to add one. A threshold is the only server-side match-quality lever available; tuning the right cutoff value is a per-caller concern, not this proxy's.

## Out of Scope

- A score-distribution/histogram endpoint (not supported by Qdrant for vector similarity; would require fetching a large result set and bucketing client-side, which is a caller-side concern if ever needed).
- Changing `project-management`'s `semantic_search()` to pass/use this — that's a follow-up change in that repo once this ships.

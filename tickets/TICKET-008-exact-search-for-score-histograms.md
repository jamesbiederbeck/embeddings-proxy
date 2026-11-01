# TICKET-008: Support exact (brute-force) search for score-distribution introspection

**Status**: Ready
**Priority**: Low
**Type**: Feature
**Labels**: search, vector-store
**Estimated Effort**: S
**Related Roadmap Item**: N/A
**Related Issues**: Related to TICKET-007 (score threshold) — that ticket filters results server-side; this one is about getting the raw similarity distribution across (near-)all points for a query, to decide where a threshold should even go.

---

## User Story

As a **caller tuning match-quality thresholds** (e.g. `project-management`'s project search), I want to request the true cosine similarity of a query against every point in a scope — not an HNSW-approximated top-k — so I can build a real histogram of scores and pick a sensible threshold from actual data instead of guessing.

## Context & Background

`QdrantVectorStore.search()` (`vector_store.py`) calls `query_points` with no `search_params`, so it always traverses the HNSW index: approximate nearest neighbors, and raising `limit` toward the collection size does not make it exact — HNSW can still miss points a brute-force scan would find, especially on an under-tuned or small graph.

Qdrant's `SearchParams` model exposes `exact: bool` (confirmed present on the installed `qdrant-client`'s `SearchParams`), which forces brute-force cosine similarity against every point in the collection, bypassing HNSW entirely. For a collection this proxy's current biggest consumer (`project-management`) actually has — a few hundred points total (README sections + descriptions across ~60 repos) — brute force is cheap, so there's no real cost objection to exposing this.

## Acceptance Criteria

- [ ] `SearchRequest` (`app.py`) gains an optional `exact: bool = False` field.
- [ ] `VectorStoreBackend.search()` / `QdrantVectorStore.search()` (`vector_store.py`) accept and forward it as `search_params=SearchParams(exact=True)` when set; unset behaves exactly as today (HNSW, unchanged).
- [ ] `top_k`/`limit` continues to cap the number of results returned — i.e. getting "all" scores is the caller's responsibility (pass a `top_k` ≥ the scope's point count), not a new "give me everything" mode. Document this clearly so callers don't assume `exact=True` alone returns the full collection.
- [ ] Documented in `README.md`'s `/search` section: what `exact` does, that it's a per-call brute-force scan (cost scales with collection size, not sublinear like HNSW), and that it's intended for occasional threshold-tuning/introspection queries, not routine search traffic.

## Implementation Notes

- This is additive and orthogonal to TICKET-007 (`score_threshold`) — a caller building a histogram wants `exact=True` with a large `limit` and no threshold; a caller doing routine search wants a threshold and the default HNSW path. Both params should be independently usable on the same request.
- No new endpoint needed — `score` is already returned per result on `/search`; `exact` + a large `limit` is sufficient for a caller to bucket scores into a histogram itself.
- Worth a one-line callout in `README.md` that repeated `exact=True` calls with a large `limit` are O(collection size) each — fine for today's scale (hundreds of points), but a caller should not poll this on every keystroke of a search-as-you-type UI.

## Out of Scope

- A dedicated histogram/bucketing endpoint in this proxy — bucketing the returned scores is a caller-side concern (e.g. `project-management` deciding its own bucket width for a one-off tuning pass).
- Any change to the default (non-exact) `/search` path's behavior or performance characteristics.
- Making `exact=True` imply "return every point regardless of `limit`" — callers must still pass an explicit `limit`.

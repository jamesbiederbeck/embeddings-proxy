# TICKET-009: Server-side similarity histogram / relevant-population estimate for /search

**Status**: Ready — scoped per `photo-embedder`'s existing client-side implementation, which serves as the reference design and API shape.
**Priority**: Medium
**Type**: Feature
**Labels**: search, vector-store
**Estimated Effort**: M
**Related Roadmap Item**: N/A
**Related Issues**: Builds on TICKET-007 (`score_threshold`) and TICKET-008 (`exact`) — this ticket is about getting a *summary* of the score distribution server-side, rather than a caller pulling back a near-full-collection result set (TICKET-008's `exact` + large `top_k`) and bucketing scores itself.

---

## User Story

As a **caller showing match quality to a user or tuning thresholds** (e.g. `photo-embedder`'s live search UI, `project-management`'s calibration pass), I want the proxy to compute a similarity-distribution summary (histogram + distributional stats) server-side, so the work isn't duplicated in every client and doesn't compete with other load on a resource-constrained host.

## Context & Background

With TICKET-008's `exact: true` plus a large `top_k`, a caller can already get the true cosine similarity of a query against every point in a scope and compute its own histogram/stats. `photo-embedder` already does exactly this client-side, on every live search, and does it well — but two things changed the calculus for building it into the proxy instead:

1. **Hardware reality on this deployment.** `nvidia-smi` on the box running this proxy's embedding backend shows a single consumer GPU (GTX 1060 6GB) with ~300MB free, shared between a chat LLM and the `embeddinggemma-2` embedding model (`llama-server`, port 5007), both left running persistently. `photo-embedder`'s "local numpy beats Qdrant over the network" benchmark was run on different (less contended) hardware — it doesn't follow that redundant per-client brute-force/stats work is free here. Centralizing the computation in the proxy means it happens once per query, not once per client per query, which matters when multiple consumers (project-management, photo-embedder, future ones) share this same constrained host.
2. **A real reference implementation already exists and is proven in production**: `photo-embedder`'s `histogram_from_sims()`. Rather than design this from scratch (and risk the wrong assumptions — see TICKET-008/009 history: fixed bucket counts and a bimodal split were both flagged as wrong), this ticket adopts that implementation's shape directly.

## Reference implementation (from `photo-embedder`)

Port `histogram_from_sims()`'s behavior and output shape into `embeddings-proxy`, computed over the scores from an exact (brute-force) search:

- **Adaptive binning**: bucket width via Freedman-Diaconis on the IQR of the observed scores, clamped to 15–60 buckets. No fixed/hardcoded bucket count or range — cosine similarities cluster tightly, and a fixed range dumps everything into 1-2 bins.
- **Distributional stats**: `mean`, `std`, `median`, `skewness`, `excess_kurtosis`.
- **Normality test**: D'Agostino-Pearson, surfaced as a `looks_normal` boolean. No bimodal assumption is baked in anywhere — population estimates below are a derived convenience stat, not a claim about distribution shape.
- **Population estimates**: `above_mean_plus_2std` and `above_mean_plus_3std` — counts of points scoring above those thresholds, not a valley-finding/bimodal-split heuristic.
- **Scope correctness**: unlike `photo-embedder`'s current client-side version (which computes over the whole collection even when results are scope/album-filtered — a known gap they flagged), this server-side version must apply the request's `scope` filter to the histogram/stats computation, not just to the returned `results`.

## Acceptance Criteria

- [ ] `SearchRequest` (`app.py`) gains an optional `histogram: bool = False` field (or equivalent — e.g. a dedicated response field populated only when requested; avoid a new endpoint unless the implementation proves awkward as a `/search` field).
- [ ] When set, the response includes a histogram/stats block computed from an exact (brute-force) scan of the resolved scope — reusing TICKET-008's `exact=True` path internally regardless of the request's own `exact` value, since distributional stats are meaningless computed over an HNSW-approximated subset.
- [ ] Bucketing is adaptive (Freedman-Diaconis, clamped 15–60 buckets), not fixed — matching the reference implementation.
- [ ] Response includes `mean`, `std`, `median`, `skewness`, `excess_kurtosis`, `looks_normal`, `above_mean_plus_2std`, `above_mean_plus_3std`, and the bucket histogram itself (edges + counts).
- [ ] The histogram/stats computation respects the request's `scope` filter — this must work correctly from day one, not reproduce `photo-embedder`'s existing scope-filtering gap.
- [ ] When `histogram` is omitted/false, behavior and response shape are unchanged (backward compatible).
- [ ] Documented in `README.md`'s `/search` section, including that this forces an exact/brute-force scan under the hood (cost scales with scope size) regardless of the request's `exact` flag.

## Implementation Notes

- Reuses `QdrantVectorStore.search()`'s `exact=True` path (TICKET-008) with a `top_k`/`limit` large enough to cover the scope (or a dedicated internal path that doesn't bother returning full point payloads when only stats are needed — worth checking whether Qdrant can return scores without payload/vectors for less overhead).
- The binning/stats math itself (Freedman-Diaconis, skew/kurtosis, D'Agostino-Pearson) is pure numpy/scipy over the returned score array — port `photo-embedder`'s `histogram_from_sims()` logic rather than reimplementing from first principles.
- Consider whether this should be computed from the same exact-scan call already made for `results`, vs. a separate call — avoid scanning twice per request if `exact` and `histogram` are both set.

## Out of Scope

- Sending the full per-document `(index, score)` array as part of the histogram response — only summary stats/buckets, consistent with the original goal of not shipping a row per document.
- Any client-side bucketing — this ticket replaces the need for that, server-side.
- Fixing `photo-embedder`'s own client-side scope-filtering bug in their app — not this proxy's code to change, though they may choose to switch to this endpoint and drop their local implementation once it's correctly scoped.

## Consumer Feedback (for context — see Reference implementation above for how it shaped the design)

**`project-management`** (2026-11-01): Doesn't need a live histogram endpoint for their own use case — a fixed `score_threshold` (TICKET-007), calibrated once offline via `exact=True` + a large `top_k` (TICKET-008), covers their "find a repo" skill. If they ever use this endpoint, they'd want raw bucket counts rather than a population estimate (no confirmed distribution shape for their small corpus yet), per-scope (`scope="project-registry"`), and only for occasional manual calibration — cost isn't a concern at their scale (low hundreds of points).

**`photo-embedder`** (2026-11-01): Has `histogram_from_sims()` client-side today, run on every live search (not occasional), serving as the reference implementation above. Their caution was about *server-side exact-scan cost at their scale* (~32k-82k points, every query) given their own numpy-vs-network benchmark — this ticket takes that cost concern seriously (hence "exact" scoped tightly to when `histogram` is requested, and noting cost scales with scope size in the README) rather than dismissing it, while betting that centralizing the computation is still worth it given this deployment's actual shared/constrained hardware (see Context & Background).

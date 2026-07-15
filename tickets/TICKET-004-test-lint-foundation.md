# TICKET-004: Add a lightweight test + lint foundation for backend adapters

**Status**: Ready
**Priority**: Medium
**Type**: Chore
**Labels**: testing, tooling, ci
**Estimated Effort**: S
**Related Roadmap Item**: Multiple embedding backends (enabling)
**Related Issues**: N/A

---

## User Story

As the **maintainer**, I want fast unit tests for each backend adapter and a linter so that the request/response mapping for a backend can't silently break when I add or change another one.

## Context & Background

There are currently no tests, linter, or type checker in the repo. Once there are 2-3 backends with different request/response shapes (TICKET-001..003), those mappings are exactly the kind of thing that breaks quietly — a wrong JSON field name still returns 200 and caches nonsense. A small test harness is worth it here; heavy CI ceremony is not.

## Acceptance Criteria

- [ ] `pytest` is added as a dev dependency and `ruff` as a linter; both run via `uv run`.
- [ ] Each backend adapter has a unit test that mocks the HTTP call (e.g. via `httpx` `MockTransport` or `respx`) and asserts the request shape sent and the vector extracted from a canned response.
- [ ] A cache-hit/cache-miss test covers the Redis path (fakeredis or a mocked cache is fine — no real Redis required in the test run).
- [ ] `ruff check` passes on the codebase.
- [ ] Dev dependencies live in a dev group so the production image (built with `uv sync --no-dev`) does not include them; `uv.lock` is regenerated and committed.
- [ ] Optional: a Gitea Actions job runs `ruff` + `pytest` on push (not just on `v*` tags). If added, keep it in a separate workflow so the tag-triggered build workflow is untouched.

## Implementation Notes

- Keep it proportionate — a handful of focused tests, not exhaustive coverage.
- A type checker (ty/mypy) is explicitly optional; add only if it earns its keep.
- The existing build workflow only triggers on `v*` tags; a push-triggered test workflow is additive and low-risk.

## Out of Scope

- End-to-end tests against real backend servers.
- Coverage gating / badges.

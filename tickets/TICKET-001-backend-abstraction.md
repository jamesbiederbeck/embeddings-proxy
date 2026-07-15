# TICKET-001: Introduce a pluggable backend abstraction (refactor, no behavior change)

**Status**: Ready
**Priority**: High
**Type**: Improvement
**Labels**: architecture, refactor, backend
**Estimated Effort**: M
**Related Roadmap Item**: Multiple embedding backends
**Related Issues**: N/A

---

## User Story

As the **maintainer of embeddings-proxy**, I want the upstream embeddings call to sit behind a small backend interface so that I can add other backends (OpenAI, llama.cpp) later without rewriting the request path or breaking my existing deployment.

## Context & Background

`app.py` currently hardcodes a single upstream shape inline in the `/vectors` handler: `POST {EMBEDDINGS_URL}/vectors {"text": ...} -> {"vector": [...]}` (the `semitechnologies/transformers-inference` API). There is zero "backend" concept in the code — this is the first real architectural fork before any second backend can be added. This ticket extracts the existing behavior into a `TransformersInferenceBackend` behind a common interface, with no change in runtime behavior, so subsequent backend tickets are pure additions.

## Acceptance Criteria

- [ ] A backend interface exists (e.g. a `Protocol` or small ABC) with an async method roughly `async def embed(text: str) -> list[float]`.
- [ ] The current transformers-inference logic is moved into a `TransformersInferenceBackend` implementing that interface; the `/vectors` handler calls `backend.embed(...)` instead of inlining the HTTP call.
- [ ] A `BACKEND` env var selects the implementation, **defaulting to the transformers-inference backend** so an existing deployment that sets only `EMBEDDINGS_URL` / `MODEL_NAME` / `REDIS_URL` keeps working with no config change.
- [ ] The cache key is derived from backend identity **and** model, not `MODEL_NAME` alone, so vectors from different backends/models cannot collide in Redis (see Implementation Notes). Existing keys may be invalidated by this change — acceptable, document it.
- [ ] `/vectors` response shape is unchanged: `{"text", "vector", "dim", "cached"}`.
- [ ] If the code is split into multiple modules, the `Dockerfile` `COPY app.py .` line is updated to copy the new files (or the app is turned into a package that is copied wholesale) — the image must still build.
- [ ] `CLAUDE.md` and `README.md` are updated to describe the backend concept and the `BACKEND` variable.

## Implementation Notes

- Keep it minimal: a single `backends.py` module with the interface + one implementation is plenty; a full package is optional.
- Cache key today is `sha256(f"{MODEL_NAME}:{text}")`. Suggest `sha256(f"{BACKEND}:{MODEL_NAME}:{text}")` or have each backend contribute a stable `cache_namespace`. Confirm the chosen scheme is documented since it changes the key space.
- Config is currently read at import time with `os.environ[...]` (raises `KeyError` if missing). Preserve fail-fast behavior, but only require the vars the *selected* backend actually needs.
- Watch the module-level singletons (`cache`, `client`) — backend construction should happen once at startup, not per request.

## Out of Scope

- Adding any second backend (OpenAI, llama.cpp) — those are TICKET-002 / TICKET-003.
- Vector store / search functionality.
- Introducing a test suite (TICKET-004), though writing this refactor test-first is welcome.

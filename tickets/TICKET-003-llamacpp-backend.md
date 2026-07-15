# TICKET-003: Add a llama.cpp server embeddings backend

**Status**: Ready
**Priority**: Medium
**Type**: Feature
**Labels**: backend, llama.cpp
**Estimated Effort**: S
**Related Roadmap Item**: Multiple embedding backends
**Related Issues**: N/A

---

## User Story

As a **homelab operator running local GGUF models**, I want embeddings-proxy to talk to a `llama.cpp` server's embeddings endpoint so that I can serve and cache embeddings from local quantized models without a separate proxy.

## Context & Background

`llama.cpp`'s server exposes embeddings two ways: a native `POST /embedding {"content": "..."}` endpoint and an OpenAI-compatible `POST /v1/embeddings`. This depends on TICKET-001 and may substantially reuse TICKET-002.

## Acceptance Criteria

- [ ] Selecting `BACKEND=llamacpp` produces embeddings from a running llama.cpp server.
- [ ] The endpoint/shape used is documented (either reuse the OpenAI-compatible path from TICKET-002 pointed at the llama.cpp base URL, or implement the native `/embedding` shape — pick one and note why).
- [ ] Config reuses `EMBEDDINGS_URL` (or a clearly-named base-URL var) for the llama.cpp server address; API key is optional/absent.
- [ ] Response vector is extracted correctly for the chosen endpoint (native `/embedding` returns `{"embedding": [...]}`, which differs from both transformers-inference and OpenAI shapes).
- [ ] Cache key distinguishes this backend/model from others per TICKET-001.
- [ ] Config documented in `README.md` / `CLAUDE.md`.

## Implementation Notes

- If llama.cpp is run with OpenAI-compat mode, this ticket may collapse to "document that `BACKEND=openai` with `OPENAI_BASE_URL=http://llamacpp:8080` works" plus a thin alias — that's a legitimate, low-effort resolution. Prefer it if it holds up in testing.
- The native `/embedding` endpoint keys on `content`, not `text`/`input` — mind the field name.

## Out of Scope

- Managing/launching the llama.cpp server itself.
- Reranking or completion endpoints.

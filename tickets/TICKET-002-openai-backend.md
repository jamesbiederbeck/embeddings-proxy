# TICKET-002: Add an OpenAI-compatible embeddings backend

**Status**: Ready
**Priority**: High
**Type**: Feature
**Labels**: backend, openai
**Estimated Effort**: M
**Related Roadmap Item**: Multiple embedding backends
**Related Issues**: N/A

---

## User Story

As a **homelab operator wiring up a new project**, I want to point embeddings-proxy at an OpenAI-compatible embeddings API so that I can cache and reuse vectors from hosted or self-hosted OpenAI-style endpoints through the same proxy.

## Context & Background

OpenAI's `/v1/embeddings` shape is the de-facto standard and is implemented by many servers (OpenAI itself, LiteLLM, vLLM, text-embeddings-inference, llama.cpp in OpenAI mode). Supporting it makes the proxy broadly useful. This depends on the backend interface from TICKET-001.

## Acceptance Criteria

- [ ] Selecting `BACKEND=openai` uses a backend that calls `POST {OPENAI_BASE_URL}/v1/embeddings` with body `{"input": text, "model": <model>}` and reads the vector from `data[0].embedding`.
- [ ] Config: `OPENAI_BASE_URL` (default `https://api.openai.com`), `OPENAI_API_KEY` (sent as `Authorization: Bearer ...`; required only when the backend is selected), and the model name (reuse `MODEL_NAME`, sent as the request `model`).
- [ ] Non-2xx responses raise/propagate a clear error rather than caching garbage; the vector is only written to Redis on success.
- [ ] The cache key correctly distinguishes this backend from others (per TICKET-001), so switching a deployment from transformers to OpenAI does not return stale cross-backend vectors.
- [ ] `README.md` / `CLAUDE.md` config tables list the OpenAI variables.
- [ ] `openai`-specific config is documented as optional; no new heavy dependency is required (use the existing `httpx` client rather than the `openai` SDK unless there's a clear reason).

## Implementation Notes

- Prefer raw `httpx` against the REST shape — it keeps the image lean and avoids SDK version churn. The request/response mapping is small.
- Batching (`input` as a list) is out of scope; the proxy embeds one text per request today.
- Remember to keep `MODEL_NAME` meaningful as part of the cache key — different OpenAI models (e.g. `text-embedding-3-small` vs `-large`) must not collide.

## Out of Scope

- Multi-text batch requests.
- Streaming, dimensions parameter, or other OpenAI embedding options (can be a follow-up if needed).

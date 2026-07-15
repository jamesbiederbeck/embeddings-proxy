# TICKET-006: Make the readiness check actually verify dependencies

**Status**: Ready
**Priority**: Low
**Type**: Improvement
**Labels**: ops, health-check
**Estimated Effort**: S
**Related Roadmap Item**: N/A (operational hardening)
**Related Issues**: N/A

---

## User Story

As a **homelab operator relying on container orchestration**, I want `/.well-known/ready` to reflect whether the proxy can actually reach Redis and its embedding backend so that a broken dependency surfaces as unready instead of a falsely-healthy container.

## Context & Background

`GET /.well-known/ready` currently always returns `{"status": "ok"}` regardless of whether Redis or the upstream backend are reachable. With multiple backends and an optional vector store in play (TICKET-001..005), a readiness probe that never fails means orchestrators can't tell a wedged instance from a healthy one.

## Acceptance Criteria

- [ ] `/.well-known/ready` pings Redis (e.g. `PING`) and performs a cheap reachability check of the selected embedding backend, returning non-200 when a required dependency is unreachable.
- [ ] The check is cheap and bounded (short timeout) so the probe itself can't hang the container.
- [ ] If a vector store is configured (TICKET-005), its reachability is included; if not configured, it is not checked.
- [ ] A separate liveness-style endpoint or a lightweight mode remains available if needed so restart loops aren't triggered by a transient backend blip — decide and document the intended probe semantics.
- [ ] Behavior documented in `README.md` / `CLAUDE.md`.

## Implementation Notes

- Keep dependency checks best-effort and time-boxed; a slow backend shouldn't turn readiness into a 30s hang (the shared `httpx` client currently has a 30s timeout — use a shorter one for the probe).
- Consider whether the embedding-backend check should be a real (tiny) call or just a TCP/HTTP reachability ping — the latter avoids spending model compute on health checks.

## Out of Scope

- Full metrics/observability endpoints.
- Alerting integration.

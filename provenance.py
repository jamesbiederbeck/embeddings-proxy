import json
from typing import Any

# Redis-backed location provenance for content-addressed vectors: a single point (one
# per unique text) can be referenced from many locations (file paths / URNs), each
# carrying its own scope (codebase/corpus) and metadata. All mutations use atomic
# Redis ops (HSET/HDEL/SADD) so concurrent writers touching different locations of the
# same shared content never race each other the way a Qdrant payload read-modify-write
# would.


def _prov_key(point_id: str) -> str:
    return f"prov:{point_id}"


def _loc_key(uri: str) -> str:
    return f"loc:{uri}"


async def _scopes_for_point(redis_client, point_id: str) -> list[str]:
    raw = await redis_client.hgetall(_prov_key(point_id))
    scopes = {json.loads(v)["scope"] for v in raw.values() if json.loads(v).get("scope")}
    return sorted(scopes)


async def add_location(
    redis_client,
    point_id: str,
    uri: str,
    scope: str,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Record that `point_id` is embedded at `uri` (in `scope`). Returns the point's
    full current scope set, for the caller to push to the vector store's payload."""
    value = json.dumps({"scope": scope, **(metadata or {})})
    await redis_client.hset(_prov_key(point_id), uri, value)
    await redis_client.sadd(_loc_key(uri), point_id)
    return await _scopes_for_point(redis_client, point_id)


async def remove_location(redis_client, uri: str) -> list[tuple[str, list[str], bool]]:
    """Unlink `uri` from every point that references it. Returns, per point,
    (point_id, remaining_scopes, is_orphaned) — is_orphaned means no location
    references that point anymore, so the caller should delete it from the vector
    store; otherwise the caller should push remaining_scopes to the vector store."""
    point_ids = await redis_client.smembers(_loc_key(uri))
    results = []
    for point_id in point_ids:
        await redis_client.hdel(_prov_key(point_id), uri)
        remaining = await redis_client.hlen(_prov_key(point_id))
        scopes = await _scopes_for_point(redis_client, point_id) if remaining else []
        results.append((point_id, scopes, remaining == 0))
    await redis_client.delete(_loc_key(uri))
    return results


async def get_locations(redis_client, point_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Fetch all known locations for each point id, pipelined into one round-trip."""
    if not point_ids:
        return {}
    pipe = redis_client.pipeline()
    for point_id in point_ids:
        pipe.hgetall(_prov_key(point_id))
    raw_results = await pipe.execute()
    return {
        point_id: [{"uri": uri, **json.loads(v)} for uri, v in raw.items()]
        for point_id, raw in zip(point_ids, raw_results)
    }

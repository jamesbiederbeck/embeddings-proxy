import fakeredis.aioredis
import pytest

import provenance


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


async def test_add_location_returns_the_scope(redis_client):
    scopes = await provenance.add_location(redis_client, "pid1", "repoA/foo.py", "repoA")
    assert scopes == ["repoA"]


async def test_add_location_accumulates_multiple_scopes_for_shared_content(redis_client):
    await provenance.add_location(redis_client, "pid1", "repoA/foo.py", "repoA")
    scopes = await provenance.add_location(redis_client, "pid1", "repoB/foo.py", "repoB")
    assert scopes == ["repoA", "repoB"]


async def test_add_location_stores_metadata(redis_client):
    await provenance.add_location(
        redis_client, "pid1", "repoA/foo.py", "repoA", {"line": 42}
    )
    locations = await provenance.get_locations(redis_client, ["pid1"])
    assert locations["pid1"] == [{"uri": "repoA/foo.py", "scope": "repoA", "line": 42}]


async def test_get_locations_is_pipelined_and_returns_empty_for_unknown_points(redis_client):
    await provenance.add_location(redis_client, "pid1", "repoA/foo.py", "repoA")
    locations = await provenance.get_locations(redis_client, ["pid1", "unknown"])
    assert locations["pid1"] == [{"uri": "repoA/foo.py", "scope": "repoA"}]
    assert locations["unknown"] == []


async def test_get_locations_with_no_point_ids_returns_empty_dict(redis_client):
    assert await provenance.get_locations(redis_client, []) == {}


async def test_remove_location_unlinks_and_reports_remaining_scopes(redis_client):
    await provenance.add_location(redis_client, "pid1", "repoA/foo.py", "repoA")
    await provenance.add_location(redis_client, "pid1", "repoB/foo.py", "repoB")

    results = await provenance.remove_location(redis_client, "repoA/foo.py")

    assert results == [("pid1", ["repoB"], False)]
    locations = await provenance.get_locations(redis_client, ["pid1"])
    assert locations["pid1"] == [{"uri": "repoB/foo.py", "scope": "repoB"}]


async def test_remove_location_reports_orphaned_point_when_last_location_removed(redis_client):
    await provenance.add_location(redis_client, "pid1", "repoA/foo.py", "repoA")

    results = await provenance.remove_location(redis_client, "repoA/foo.py")

    assert results == [("pid1", [], True)]
    locations = await provenance.get_locations(redis_client, ["pid1"])
    assert locations["pid1"] == []


async def test_remove_location_for_unknown_uri_returns_empty(redis_client):
    assert await provenance.remove_location(redis_client, "never/seen.py") == []


async def test_remove_location_deletes_the_reverse_index_key(redis_client):
    await provenance.add_location(redis_client, "pid1", "repoA/foo.py", "repoA")
    await provenance.remove_location(redis_client, "repoA/foo.py")

    assert await redis_client.exists("loc:repoA/foo.py") == 0

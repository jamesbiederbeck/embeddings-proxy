import pytest

from registry import HealthStatus, NoHealthyBackendError, NotConfiguredError, Registry


class Stub:
    type = "stub"

    def __init__(self, name, model, healthy=True):
        self.name = name
        self.model = model
        self.embedding_task = None
        self.healthy = healthy

    async def check(self) -> None:
        if not self.healthy:
            raise RuntimeError("down")


def make_registry(*backends, stores=()):
    reg = Registry(list(backends), list(stores))
    for b in (*backends, *stores):
        reg.health[b.name] = HealthStatus(healthy=True, error=None)
    return reg


def test_unknown_name_is_not_configured():
    reg = make_registry(Stub("a", "m"))
    with pytest.raises(NotConfiguredError):
        reg.select_embedding(name="b")


def test_name_and_mismatched_model_is_not_configured():
    reg = make_registry(Stub("a", "m"))
    with pytest.raises(NotConfiguredError):
        reg.select_embedding(name="a", model="other")


def test_named_but_unhealthy_backend_raises_no_healthy():
    reg = make_registry(Stub("a", "m"))
    reg.health["a"] = HealthStatus(healthy=False, error="down")
    with pytest.raises(NoHealthyBackendError):
        reg.select_embedding(name="a")


def test_model_not_configured():
    reg = make_registry(Stub("a", "m"))
    with pytest.raises(NotConfiguredError):
        reg.select_embedding(model="other")


def test_first_healthy_backend_for_model_wins():
    first, second = Stub("first", "m"), Stub("second", "m")
    reg = make_registry(first, second)
    assert reg.select_embedding(model="m") is first

    reg.health["first"] = HealthStatus(healthy=False, error="down")
    assert reg.select_embedding(model="m") is second


def test_no_params_selects_first_healthy_overall():
    first, second = Stub("first", "m1"), Stub("second", "m2")
    reg = make_registry(first, second)
    reg.health["first"] = HealthStatus(healthy=False, error="down")
    assert reg.select_embedding() is second


def test_all_unhealthy_raises_no_healthy():
    reg = make_registry(Stub("a", "m"))
    reg.health["a"] = HealthStatus(healthy=False, error="down")
    with pytest.raises(NoHealthyBackendError):
        reg.select_embedding()


def test_unpolled_backends_start_unhealthy():
    reg = Registry([Stub("a", "m")], [])
    with pytest.raises(NoHealthyBackendError):
        reg.select_embedding()


def test_store_for_model():
    store = Stub("s", "m")
    reg = make_registry(Stub("a", "m"), stores=[store])
    assert reg.store_for_model("m") is store
    assert reg.store_for_model("other") is None


async def test_poll_once_updates_health_both_ways():
    backend = Stub("a", "m")
    reg = Registry([backend], [])
    assert not reg.is_healthy("a")

    await reg.poll_once()
    assert reg.is_healthy("a")
    assert reg.health["a"].error is None

    backend.healthy = False
    await reg.poll_once()
    assert not reg.is_healthy("a")
    assert "down" in reg.health["a"].error

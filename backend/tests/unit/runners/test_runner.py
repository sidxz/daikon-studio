from daikonstudio.domain.runners.runner import Runner


def _runner(**overrides):
    defaults = dict(name="gpu-01", lanes=("gpu",), token_hash="a" * 64)
    defaults.update(overrides)
    return Runner(**defaults)


def test_new_runner_is_not_revoked() -> None:
    assert _runner().is_revoked is False


def test_revoke_sets_timestamp_once() -> None:
    runner = _runner()
    runner.revoke()
    first = runner.revoked_at
    assert runner.is_revoked and first is not None
    runner.revoke()
    assert runner.revoked_at == first  # idempotent, keeps the original instant


def test_lanes_are_a_tuple() -> None:
    assert _runner(lanes=("default", "gpu")).lanes == ("default", "gpu")

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


def test_build_identity_is_none_until_an_agent_reports_it() -> None:
    runner = _runner()
    assert (runner.build_version, runner.build_git_sha, runner.build_date) == (None, None, None)


def test_record_build_stores_what_the_agent_reported() -> None:
    runner = _runner()
    runner.record_build(version="0.8.0", git_sha="abc1234", build_date="2026-10-20T09:00:00Z")
    assert runner.build_version == "0.8.0"
    assert runner.build_git_sha == "abc1234"
    assert runner.build_date == "2026-10-20T09:00:00Z"


def test_record_build_overwrites_the_previous_process_report() -> None:
    """An agent restarted on a new image reports the new build; the old one is
    not history worth keeping, it is a stale answer."""
    runner = _runner()
    runner.record_build(version="0.7.0", git_sha="old", build_date="2026-10-01T00:00:00Z")
    runner.record_build(version="0.8.0", git_sha="new", build_date="2026-10-20T00:00:00Z")
    assert (runner.build_version, runner.build_git_sha) == ("0.8.0", "new")


def test_an_unreported_build_does_not_make_a_runner_invalid() -> None:
    """A runner that has not restarted since the studio began asking still
    claims runs: nothing about the aggregate depends on knowing its build."""
    runner = _runner()
    assert runner.is_revoked is False
    assert runner.build_version is None

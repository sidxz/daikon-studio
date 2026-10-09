"""The claim response's deadline is the claimed run's lane's, not one number for
every lane: a chemprop fit on a real dataset does not finish in the default-lane
budget, and raising the global number for it would let a hung ECFP4 fit hold a
runner for as long."""

import uuid

from daikonstudio.application.execution.claim_run import ClaimRun
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.runners.runner import Runner


def _run(lane: str | None) -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="k",
        lane=lane,
    )


class _Queue:
    def __init__(self, run_id: uuid.UUID) -> None:
        self._run_id = run_id

    async def sweep(self, *, max_attempts: int) -> None:
        return None

    async def claim_next(self, **_: object) -> uuid.UUID:
        return self._run_id


class _Runs:
    def __init__(self, run: Run) -> None:
        self._run = run

    async def get_by_id(self, run_id: uuid.UUID) -> Run:
        return self._run


def _claim(run: Run, *, max_deadline_seconds: int = 432_000) -> ClaimRun:
    return ClaimRun(
        _Queue(run.id),  # type: ignore[arg-type]
        _Runs(run),  # type: ignore[arg-type]
        lease_seconds=600,
        max_active_per_workspace=10,
        max_attempts=3,
        deadline_seconds=1800,
        deadline_by_lane={"gpu": 7200},
        max_deadline_seconds=max_deadline_seconds,
    )


async def test_the_claimed_runs_lane_picks_its_deadline():
    run = _run("gpu")
    runner = Runner(name="gpu-box", lanes=("gpu",), token_hash="h")
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 7200, 600)


async def test_a_lane_without_an_override_gets_the_default():
    run = _run("default")
    runner = Runner(name="cpu-box", lanes=("default",), token_hash="h")
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 1800, 600)


async def test_a_fan_out_run_gets_the_lane_deadline_once_per_target():
    run = _run("gpu")
    run.params = {"deadline_scale": 4}
    runner = Runner(name="gpu-box", lanes=("gpu",), token_hash="h")
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 4 * 7200, 600)


async def test_the_scaled_deadline_is_capped():
    """The cap is the whole point of the ceiling: a run holding many fits must not be
    handed a deadline so long that the backstop never fires. 252 lane budgets is the
    measured worst case -- a 12-target dataset with a 10-member ensemble -- and on the
    GPU lane that is 1,260 days before the cap."""
    run = _run("gpu")
    run.params = {"deadline_scale": 252}
    runner = Runner(name="gpu-box", lanes=("gpu",), token_hash="h")
    assert 252 * 7200 > 432_000  # the thing being prevented
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 432_000, 600)


async def test_the_cap_applies_to_an_unscaled_run_on_a_long_lane():
    """A lane configured above the ceiling is capped too, with no `deadline_scale` in
    play: the ceiling bounds the deadline served, not the multiplier."""
    run = _run("gpu")
    runner = Runner(name="gpu-box", lanes=("gpu",), token_hash="h")
    claim = _claim(run, max_deadline_seconds=3600)
    assert (await claim(runner=runner)).unwrap() == (run, 3600, 600)

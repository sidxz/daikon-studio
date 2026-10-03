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


def _claim(run: Run) -> ClaimRun:
    return ClaimRun(
        _Queue(run.id),  # type: ignore[arg-type]
        _Runs(run),  # type: ignore[arg-type]
        lease_seconds=600,
        max_active_per_workspace=10,
        max_attempts=3,
        deadline_seconds=1800,
        deadline_by_lane={"gpu": 7200},
    )


async def test_the_claimed_runs_lane_picks_its_deadline():
    run = _run("gpu")
    runner = Runner(name="gpu-box", lanes=("gpu",), token_hash="h")
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 7200, 600)


async def test_a_lane_without_an_override_gets_the_default():
    run = _run("default")
    runner = Runner(name="cpu-box", lanes=("default",), token_hash="h")
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 1800, 600)

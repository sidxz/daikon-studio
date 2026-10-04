"""ClaimRun -- a self-hosted runner asking the queue for its next job.

Sweeps expired leases first (a runner that missed its lease is eligible again
before anyone else claims past it -- `RunQueue.claim_next`'s own docstring
notes its SQL implementation already does this atomically, but the use case
calls `sweep` explicitly too: the `RunQueue` Protocol itself makes no such
promise, and this is the one call site the contract has to hold for).

`runner.is_revoked` is checked here purely as defense in depth: the
`get_runner` auth dependency already 401s a revoked runner's token before
this use case is ever reached, so a revoked runner reaching `__call__` would
mean that check was bypassed or is buggy -- worth failing loudly on, not
worth trusting away.
"""

from __future__ import annotations

from returns.result import Failure, Result, Success

from daikonstudio.application.engines.manifest import DEFAULT_LANE
from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import Run
from daikonstudio.domain.runners.runner import Runner
from daikonstudio.domain.shared.errors import AuthorizationError, DomainError


class ClaimRun:
    def __init__(
        self,
        queue: RunQueue,
        runs: RunRepository,
        *,
        lease_seconds: int,
        max_active_per_workspace: int,
        max_attempts: int,
        deadline_seconds: int,
        deadline_by_lane: dict[str, int] | None = None,
    ) -> None:
        self._queue = queue
        self._runs = runs
        self._lease_seconds = lease_seconds
        self._max_active_per_workspace = max_active_per_workspace
        self._max_attempts = max_attempts
        self._deadline_seconds = deadline_seconds
        self._deadline_by_lane = deadline_by_lane or {}

    async def __call__(
        self, *, runner: Runner
    ) -> Result[tuple[Run, int, int] | None, DomainError]:
        if runner.is_revoked:
            return Failure(AuthorizationError(f"Runner '{runner.id}' is revoked"))

        await self._queue.sweep(max_attempts=self._max_attempts)
        run_id = await self._queue.claim_next(
            runner_id=runner.id,
            lanes=runner.lanes,
            lease_seconds=self._lease_seconds,
            max_active_per_workspace=self._max_active_per_workspace,
            max_attempts=self._max_attempts,
        )
        if run_id is None:
            return Success(None)

        run = await self._runs.get_by_id(run_id)
        # The row `claim_next` just claimed cannot have vanished in between --
        # only DeleteProtocol and DeleteDataset delete Runs, and never a pending or
        # running one.
        assert run is not None, f"claimed run '{run_id}' was not found by get_by_id"
        # deadline_seconds (the job's own soft deadline) and lease_seconds (how
        # long the CLAIM survives unrenewed) are different numbers the runner
        # must not conflate -- see `ClaimResponse`'s own docstring for why
        # (Important 1+2, final review). The deadline is the claimed run's lane's,
        # falling back to the server-wide default.
        deadline = self._deadline_by_lane.get(run.lane or DEFAULT_LANE, self._deadline_seconds)
        # A fan-out training run fits once per target; `TrainProtocol` stamped how
        # many lane budgets that is worth. Absent on prediction runs and on runs
        # enqueued before targets could be several, where it is 1.
        deadline *= int(run.params.get("deadline_scale", 1))
        return Success((run, deadline, self._lease_seconds))

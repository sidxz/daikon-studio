"""Port for the runs table as a job queue.

Self-hosted runners claim work from `runs` directly rather than through a
separate jobs table -- see `SqlAlchemyRunQueue` for the SKIP LOCKED SQL this
Protocol is a thin, storage-agnostic face for.
"""

import uuid
from collections.abc import Sequence
from typing import Protocol


class RunQueue(Protocol):
    async def sweep(self, *, max_attempts: int) -> None:
        """Requeue expired leases; fail runs that exhausted their attempts."""
        ...

    async def claim_next(
        self,
        *,
        runner_id: uuid.UUID,
        lanes: Sequence[str],
        lease_seconds: int,
        max_active_per_workspace: int,
        max_attempts: int,
    ) -> uuid.UUID | None: ...

    async def verify_claim(
        self,
        run_id: uuid.UUID,
        *,
        runner_id: uuid.UUID,
        lease_seconds: int,
        require_active: bool,
    ) -> bool:
        """True iff `runner_id` holds the claim on `run_id`; extends the lease
        as a side effect (every authenticated run-scoped call is a heartbeat).
        With require_active=True, also demands status pending/running -- used
        by write endpoints so nothing mutates a terminal run's satellites."""
        ...

    async def set_lane(self, run_id: uuid.UUID, lane: str) -> None: ...

    async def active_run_by_runner(self) -> dict[uuid.UUID, uuid.UUID]:
        """runner_id -> the run it currently holds (claimed pending or running)."""
        ...

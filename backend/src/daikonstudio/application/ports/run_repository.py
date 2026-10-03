"""Persistence port for the Run aggregate.

`add` creates the row the enqueued job will find; `update` is how a worker's
progress reaches the polling client, since there is no channel between the two
other than the row itself. `Run.params` is deliberately absent from what
`update` persists (see the repository) -- a handler must not be able to rewrite
its own instructions mid-flight.

`get` and `find_by_cache_key` are added for Task 17: reading a Run back by id
(the `GET /runs/{id}` poll endpoint) and looking up a prior Run by its cache
key (content-addressed prediction caching) are both application-layer needs
that Task 14 had no reason to expose yet -- the port grows the methods its use
cases actually call and not one before, exactly as `ProtocolRepository`'s own
docstring explains.
"""

import builtins
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from daikonstudio.domain.execution.run import Run, RunKind


@dataclass(frozen=True, kw_only=True)
class SweepSummary:
    """One row of the sweeps list, assembled by a GROUP BY rather than read
    from a `sweeps` table -- a sweep's only state is its members.

    `name` and `dataset_id` are read out of any member's `params` (every member
    of a sweep carries the same two, written once at submission). Both are
    nullable only to survive a malformed row; a sweep created by `SubmitSweep`
    always has them.

    `by_status` holds only the statuses actually present, so a caller reads
    `.get(status, 0)` rather than trusting a fixed key set that a new
    `RunStatus` member would silently invalidate.
    """

    sweep_id: UUID
    name: str | None
    dataset_id: UUID | None
    created_at: datetime
    total: int
    by_status: dict[str, int]


class RunRepository(Protocol):
    async def add(self, run: Run) -> None: ...

    async def update(self, run: Run) -> None: ...

    async def delete_many(self, workspace_id: UUID, run_ids: Sequence[UUID]) -> None:
        """Only for the runs of something being deleted: a draft Protocol's training
        run, or a Dataset's failed and cancelled training runs."""
        ...

    async def list_training_for_dataset(
        self, workspace_id: UUID, dataset_id: UUID
    ) -> builtins.list[Run]:
        """Every training run on a Dataset. Unpaginated: they are submitted by hand."""
        ...

    async def get(self, workspace_id: UUID, run_id: UUID) -> Run | None: ...

    async def get_by_id(self, run_id: UUID) -> Run | None:
        """Unscoped, for the worker only -- it is handed a bare `run_id` with no tenant
        context of its own, and the Run was already scoped to its workspace by the use
        case that created the row. On the port rather than only on the SQLAlchemy class
        because `RunTraining`'s cancellation checkpoint calls it, and `application` may
        not import `infrastructure`."""
        ...

    async def find_by_cache_key(self, workspace_id: UUID, cache_key: str) -> Run | None: ...

    async def list(
        self,
        workspace_id: UUID,
        *,
        kind: RunKind | None = None,
        protocol_id: UUID | None = None,
        cursor: tuple[datetime, UUID] | None = None,
        limit: int = 50,
    ) -> builtins.list[Run]: ...

    # `builtins.list[...]`, not the bare generic: this Protocol already has a
    # method named `list` above, and Python 3.14's lazy annotation evaluation
    # (PEP 649) resolves an unqualified `list` used after that point to the
    # *method*, not the builtin, raising `TypeError: 'function' object is not
    # subscriptable` the first time anything introspects these signatures
    # (mypy catches it statically as "Function ... list is not valid as a
    # type"; `typing.get_type_hints` would hit it at runtime). Reordering
    # doesn't help -- both methods share one class-wide annotation scope.
    async def list_by_sweep(self, workspace_id: UUID, sweep_id: UUID) -> builtins.list[Run]:
        """Every member of one sweep, oldest first -- submission order, which
        is the order the configs were given in and therefore the order a user
        recognises. Unpaginated on purpose: a sweep is bounded by what a human
        typed into a form, and paging a comparison defeats the comparison."""
        ...

    async def sweep_summaries(
        self, workspace_id: UUID, *, limit: int = 50
    ) -> builtins.list[SweepSummary]: ...

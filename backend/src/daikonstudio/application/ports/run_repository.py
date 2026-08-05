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

from datetime import datetime
from typing import Protocol
from uuid import UUID

from daikonstudio.domain.execution.run import Run, RunKind


class RunRepository(Protocol):
    async def add(self, run: Run) -> None: ...

    async def update(self, run: Run) -> None: ...

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
    ) -> list[Run]: ...

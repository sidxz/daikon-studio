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

from typing import Protocol
from uuid import UUID

from daikonstudio.domain.execution.run import Run


class RunRepository(Protocol):
    async def add(self, run: Run) -> None: ...

    async def update(self, run: Run) -> None: ...

    async def get(self, workspace_id: UUID, run_id: UUID) -> Run | None: ...

    async def find_by_cache_key(self, workspace_id: UUID, cache_key: str) -> Run | None: ...

"""Persistence port for the Run aggregate.

`add` creates the row the enqueued job will find; `update` is how a worker's
progress reaches the polling client, since there is no channel between the two
other than the row itself. `Run.params` is deliberately absent from what
`update` persists (see the repository) -- a handler must not be able to rewrite
its own instructions mid-flight.
"""

from typing import Protocol

from daikonstudio.domain.execution.run import Run


class RunRepository(Protocol):
    async def add(self, run: Run) -> None: ...

    async def update(self, run: Run) -> None: ...

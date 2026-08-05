"""Persistence port for the Runner aggregate.

`touch_last_seen` and `revoke` are targeted updates rather than round trips
through `add`/`update` -- see `SqlAlchemyRunnerRepository`'s docstring for why
a heartbeat does not bump `version`.
"""

import uuid
from typing import Protocol

from daikonstudio.domain.runners.runner import Runner


class RunnerRepository(Protocol):
    async def add(self, runner: Runner) -> None: ...

    async def get(self, runner_id: uuid.UUID) -> Runner | None: ...

    async def get_by_token_hash(self, token_hash: str) -> Runner | None: ...

    async def list(self) -> list[Runner]: ...

    async def touch_last_seen(self, runner_id: uuid.UUID) -> None: ...

    async def revoke(self, runner_id: uuid.UUID) -> None: ...

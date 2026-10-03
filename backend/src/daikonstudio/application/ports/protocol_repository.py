"""Persistence port for the InSilicoProtocol aggregate.

Task 14 needed only `add` -- training is the one thing in Phase 1 that
*creates* a Protocol. Task 16 adds `get`/`list` (reading and listing) and
`update` (publishing flips `status` in place on the same aggregate, exactly
like `RunRepository.update` persists a Run's progress): the port grows the
methods its use cases actually call and not one before.

`delete` removes a draft; `DeleteProtocol` is the only caller and refuses a
published one.
"""

import uuid
from datetime import datetime
from typing import Protocol

from daikonstudio.domain.catalog.protocol import InSilicoProtocol


class ProtocolRepository(Protocol):
    async def add(self, protocol: InSilicoProtocol) -> None: ...

    async def update(self, protocol: InSilicoProtocol) -> None: ...

    async def delete(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None: ...

    async def get(
        self, workspace_id: uuid.UUID, protocol_id: uuid.UUID
    ) -> InSilicoProtocol | None: ...

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
        dataset_id: uuid.UUID | None = None,
    ) -> list[InSilicoProtocol]: ...

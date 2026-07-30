"""Persistence port for the Collection aggregate.

Read-only after creation, like `DatasetRepository`: a Collection is a frozen
triage decision, so there is no `update`.
"""

import uuid
from datetime import datetime
from typing import Protocol

from daikonstudio.domain.data.collection import Collection


class CollectionRepository(Protocol):
    async def add(self, collection: Collection) -> None: ...

    async def get(
        self, workspace_id: uuid.UUID, collection_id: uuid.UUID
    ) -> Collection | None: ...

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[Collection]: ...

"""Persistence port for the Collection aggregate.

Read-only after creation, like `DatasetRepository`: a Collection is a frozen
triage decision, so there is no `update`.
"""

import uuid
from typing import Protocol

from daikonstudio.domain.data.collection import Collection


class CollectionRepository(Protocol):
    async def add(self, collection: Collection) -> None: ...

    async def get(
        self, workspace_id: uuid.UUID, collection_id: uuid.UUID
    ) -> Collection | None: ...

"""Persistence port for the Dataset aggregate.

Every method takes `workspace_id` first and is expected to push it into the SQL
WHERE clause rather than filter afterwards, so a cross-tenant read is not
something the caller has to remember to prevent. There is no `update`: a
Dataset is immutable once created. `delete` removes one that nothing depends
on; `DeleteDataset` is the only caller and checks that.
"""

import uuid
from datetime import datetime
from typing import Protocol

from daikonstudio.domain.data.dataset import Dataset


class DatasetRepository(Protocol):
    async def add(self, dataset: Dataset) -> None: ...

    async def get(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> Dataset | None: ...

    async def delete(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> None: ...

    async def find_by_content_hash(
        self, workspace_id: uuid.UUID, content_hash: str
    ) -> Dataset | None: ...

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[Dataset]: ...

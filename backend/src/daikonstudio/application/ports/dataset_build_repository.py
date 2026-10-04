"""Persistence port for DatasetBuild. `save` writes the whole row, last write wins:
only the one task running a build ever writes it (and a stale read marks it failed)."""

import uuid
from typing import Protocol

from daikonstudio.domain.data.dataset_build import DatasetBuild


class DatasetBuildRepository(Protocol):
    async def add(self, build: DatasetBuild) -> None: ...
    async def get(self, build_id: uuid.UUID) -> DatasetBuild | None: ...
    async def save(self, build: DatasetBuild) -> None: ...

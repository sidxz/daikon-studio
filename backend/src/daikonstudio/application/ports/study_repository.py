"""Persistence port for the Study aggregate."""

from __future__ import annotations

import builtins
from typing import Protocol
from uuid import UUID

from daikonstudio.domain.studies.study import Study


class StudyRepository(Protocol):
    async def add(self, study: Study) -> None: ...

    async def update(self, study: Study) -> None:
        """Optimistic on `version`; raises ConcurrencyConflictError("Study", id) on a miss."""
        ...

    async def get(self, workspace_id: UUID, study_id: UUID) -> Study | None: ...

    # `builtins.list[...]`: this Protocol has a method named `list` (PEP 649).
    async def list(
        self, workspace_id: UUID, *, created_by: UUID | None = None, limit: int = 50
    ) -> builtins.list[Study]:
        """Newest first."""
        ...

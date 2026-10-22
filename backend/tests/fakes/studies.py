"""An in-memory `StudyRepository`."""

from __future__ import annotations

import builtins
import copy
import uuid

from daikonstudio.domain.studies.study import Study


class InMemoryStudies:
    """A dict of copies, so a study changes only when it is updated."""

    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, Study] = {}

    async def add(self, study: Study) -> None:
        self.rows[study.id] = copy.deepcopy(study)

    async def update(self, study: Study) -> None:
        study.version += 1
        self.rows[study.id] = copy.deepcopy(study)

    async def get(self, workspace_id: uuid.UUID, study_id: uuid.UUID) -> Study | None:
        row = self.rows.get(study_id)
        return copy.deepcopy(row) if row and row.workspace_id == workspace_id else None

    async def list(
        self, workspace_id: uuid.UUID, *, created_by: uuid.UUID | None = None, limit: int = 50
    ) -> builtins.list[Study]:
        return [copy.deepcopy(s) for s in self.rows.values() if s.workspace_id == workspace_id]

"""A dataset being built in the background: what stage it is at and how it ended.

Creating a dataset canonicalizes every structure with RDKit (and a scaffold split
does it again), which takes minutes on a few hundred thousand rows. The build runs
on a worker thread inside the API and reports here, so the wizard can show real
progress and a reload can pick the build back up.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from daikonstudio.domain.shared.entity import AggregateRoot

# The build loop writes progress every second, so a running build silent for this
# long has lost the process that ran it (an API restart or crash). Generous, so a
# slow database write never fails a healthy build.
STALE_AFTER = timedelta(seconds=60)

INTERRUPTED = {
    "error": "BuildInterrupted",
    "message": (
        "The server restarted before this dataset finished building. Create it again; "
        "nothing was saved."
    ),
}


class BuildStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DatasetBuild(AggregateRoot):
    def __init__(
        self,
        *,
        workspace_id: uuid.UUID,
        created_by: uuid.UUID,
        name: str,
        status: BuildStatus = BuildStatus.RUNNING,
        stage: str = "Reading the file",
        done: int = 0,
        total: int = 0,
        dataset_id: uuid.UUID | None = None,
        error: dict[str, Any] | None = None,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.workspace_id = workspace_id
        self.created_by = created_by
        self.name = name
        self.status = status
        self.stage = stage
        self.done = done
        self.total = total
        self.dataset_id = dataset_id
        # The same JSON body the synchronous `POST /datasets` answers with, so the
        # wizard renders a failed build exactly as it renders a rejected request.
        self.error = error

    def report(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total
        self.updated_at = datetime.now(UTC)

    def succeed(self, dataset_id: uuid.UUID) -> None:
        self.status, self.dataset_id = BuildStatus.SUCCEEDED, dataset_id
        self.updated_at = datetime.now(UTC)

    def fail(self, error: dict[str, Any]) -> None:
        self.status, self.error = BuildStatus.FAILED, error
        self.updated_at = datetime.now(UTC)

    def is_stale(self, now: datetime) -> bool:
        return self.status is BuildStatus.RUNNING and now - self.updated_at > STALE_AFTER

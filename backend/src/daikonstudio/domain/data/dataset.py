"""The Dataset aggregate -- a frozen, validated, content-addressed training set.

A Dataset is immutable by construction. There is no mutating method here and no
update endpoint, deliberately: every Protocol, Run and Scorecard that a later
task hangs off a Dataset cites it by id, and a citation that can change underneath
its citers is not a citation. Correcting the data means uploading a new Dataset,
which gets a new id and a new hash, leaving the old evidence intact.

`content_hash` is the sha256 of the frozen Parquet snapshot (see
`application/data/snapshot.py`), so two Datasets holding byte-identical data hash
identically. The unique index on `(workspace_id, content_hash)` is what turns that
property into an enforced fact rather than a coincidence.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from daikonstudio.domain.data.split import SplitSpec
from daikonstudio.domain.data.target import TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.domain.shared.entity import AggregateRoot
from daikonstudio.domain.shared.errors import ConflictError


class Dataset(AggregateRoot):
    def __init__(
        self,
        *,
        workspace_id: uuid.UUID,
        name: str,
        target: TargetSpec,
        split: SplitSpec,
        content_hash: str,
        snapshot_uri: str,
        row_count: int,
        validation_report: ValidationReport,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.workspace_id = workspace_id
        self.name = name
        self.target = target
        self.split = split
        self.content_hash = content_hash
        self.snapshot_uri = snapshot_uri
        self.row_count = row_count
        self.validation_report = validation_report


class DuplicateDatasetError(ConflictError):
    """The same frozen data is already stored in this workspace.

    Carries the existing Dataset's id in the HTTP body so the common cause -- a
    scientist re-submitting after a browser refresh -- is one redirect away from
    the Dataset they already have, instead of a dead end.
    """

    def __init__(self, existing_dataset_id: uuid.UUID) -> None:
        self.existing_dataset_id = existing_dataset_id
        super().__init__(
            "This data is already stored in this workspace",
            detail=(
                "Datasets are content-addressed: identical data is one Dataset. "
                f"Use dataset {existing_dataset_id}, or change the data to create a new one."
            ),
        )

    def body_extras(self) -> dict[str, Any]:
        return {"existing_dataset_id": str(self.existing_dataset_id)}

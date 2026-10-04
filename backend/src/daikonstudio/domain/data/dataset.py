"""The Dataset aggregate -- a frozen, validated, content-addressed training set.

A Dataset is immutable by construction. There is no mutating method here and no
update endpoint, deliberately: every Protocol, Run and Scorecard that a later
task hangs off a Dataset cites it by id, and a citation that can change underneath
its citers is not a citation. Correcting the data means uploading a new Dataset,
which gets a new id and a new hash, leaving the old evidence intact.

`content_hash` is the sha256 of the frozen Parquet snapshot (see
`application/data/snapshot.py`), and that snapshot is taken *after* the split
column is assigned. So the hash identifies the data **and** the split together:
the same CSV uploaded twice under different seeds, or under scaffold instead of
random, produces two different hashes and two separate Datasets. That is
deliberate rather than a leak -- a Dataset is the data plus the split decision,
and a Run's numbers are only reproducible if both are pinned. What the unique
index on `(workspace_id, content_hash)` enforces is therefore narrower than
"same data": it is "same data, split the same way, stored once".
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from daikonstudio.domain.data.split import SplitSpec
from daikonstudio.domain.data.target import TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.domain.shared.entity import AggregateRoot
from daikonstudio.domain.shared.errors import ConflictError, ValidationError


class Dataset(AggregateRoot):
    def __init__(
        self,
        *,
        workspace_id: uuid.UUID,
        name: str,
        structure_column: str,
        targets: tuple[TargetSpec, ...],
        split: SplitSpec,
        content_hash: str,
        snapshot_uri: str,
        row_count: int,
        validation_report: ValidationReport,
        created_by: uuid.UUID | None = None,
        id_column: str | None = None,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.workspace_id = workspace_id
        self.name = name
        # Which column of the snapshot holds the structures. Recorded rather than
        # re-derived: the Parquet keeps the uploader's own column name, and every
        # later consumer -- training, prediction, scaffold analysis -- must
        # featurize exactly the column the validation pass canonicalized.
        self.structure_column = structure_column
        # What the scientist is predicting, in the order they chose the columns, and
        # never empty (`check_targets`, at creation). There is deliberately no `target`
        # shortcut to the first one: a caller that took it would train on one target
        # and silently drop the rest.
        self.targets = targets
        self.split = split
        self.content_hash = content_hash
        self.snapshot_uri = snapshot_uri
        self.row_count = row_count
        self.validation_report = validation_report
        # Who created it, for the delete permission. None for datasets made before
        # migration 011 recorded it: those are admin-only.
        self.created_by = created_by
        # Which snapshot column holds the compounds' own IDs, if any. Display
        # metadata: not part of `content_hash`, and changeable after freezing.
        self.id_column = id_column

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(target.column for target in self.targets)


def check_id_column(
    columns: Sequence[str],
    *,
    id_column: str,
    structure_column: str,
    target_columns: Sequence[str],
) -> None:
    """An identifier is any stored column but the ones that already mean something."""
    if id_column in (structure_column, "split") or id_column in target_columns:
        raise ValidationError(
            "Choose an identifier column other than the structure, target or split column."
        )
    if id_column not in columns:
        raise ValidationError(f"Column '{id_column}' is not in the uploaded file.")


class DuplicateDatasetError(ConflictError):
    """This data, split this way, is already stored in this workspace.

    Carries the existing Dataset's id in the HTTP body so the common cause -- a
    scientist re-submitting after a browser refresh -- is one redirect away from
    the Dataset they already have, instead of a dead end.
    """

    def __init__(self, existing_dataset_id: uuid.UUID) -> None:
        self.existing_dataset_id = existing_dataset_id
        super().__init__(
            "A dataset with identical data and split already exists in this workspace.",
            detail="Open the existing dataset, or change the data or the split settings.",
        )

    def body_extras(self) -> dict[str, Any]:
        return {"existing_dataset_id": str(self.existing_dataset_id)}

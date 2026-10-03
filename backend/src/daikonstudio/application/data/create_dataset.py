"""Turning an uploaded CSV into a frozen Dataset -- the two halves of creation.

`StoreUpload` parks the raw bytes; `CreateDataset` reads them back, validates,
splits, freezes and persists. They are in one module because they are the only
two places that know an upload's blob key, and a drift between writer and reader
would be a silent "upload not found" -- `upload_key` is the single definition.

`upload_ref` is deliberately a bare UUID rather than a path. The key is composed
from `auth.workspace_id` and that UUID, so a ref belonging to another tenant --
or a `../..` -- resolves to a key inside the caller's own prefix and simply is
not there. The workspace is never read from the request body or the URL.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.application.data.prepare_frame import prepare_frame, read_csv_upload
from daikonstudio.application.data.snapshot import write_snapshot
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.dataset import Dataset, DuplicateDatasetError
from daikonstudio.domain.data.split import SplitSpec
from daikonstudio.domain.data.target import RESERVED_TARGET_COLUMNS, TargetKind, TargetSpec
from daikonstudio.domain.data.validation import InvalidDatasetError
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError


def upload_key(workspace_id: uuid.UUID, upload_ref: uuid.UUID) -> str:
    return f"{workspace_id}/uploads/{upload_ref}.csv"


class StoreUpload:
    def __init__(self, store: BlobStore) -> None:
        self._store = store

    async def __call__(
        self, data: bytes, auth: AuthContext | None = None
    ) -> Result[uuid.UUID, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        if not data:
            return Failure(ValidationError("The uploaded file is empty"))
        upload_ref = uuid.uuid4()
        self._store.put_bytes(upload_key(auth.workspace_id, upload_ref), data)
        return Success(upload_ref)


@dataclass(frozen=True, kw_only=True)
class CreateDatasetCommand:
    name: str
    upload_ref: str
    structure_column: str
    target: TargetSpec
    split: SplitSpec


class CreateDataset:
    def __init__(
        self,
        repository: DatasetRepository,
        store: BlobStore,
        normalizer: StructureNormalizer,
    ) -> None:
        self._repository = repository
        self._store = store
        self._normalizer = normalizer

    async def __call__(
        self, command: CreateDatasetCommand, auth: AuthContext | None = None
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        workspace_id = auth.workspace_id

        # C1 (whole-branch review, Critical): a target column named the same as
        # one of these is not a naming quirk -- it is a silent data-corruption
        # bug. `derive_readouts` names the predicted Readout after
        # `target.column`, and every one of these names is a column the
        # pipeline itself writes downstream (see `RESERVED_TARGET_COLUMNS`'s own
        # comment for exactly where): whichever write happens last wins, so
        # either the served prediction becomes the model's uncertainty/an
        # unrelated provenance value, or -- for a target named "structure" --
        # the compound identity column is overwritten by the predicted value
        # instead. Checked here, before any file is even read, because this is
        # the one and only place a rejection can still prevent the damage; the
        # export-time collision guard (`export_collection.py`) is downstream of
        # a Protocol that has already been trained and published on the bad
        # column.
        if command.target.column in RESERVED_TARGET_COLUMNS:
            return Failure(
                ValidationError(
                    f"'{command.target.column}' cannot be used as a target column",
                    detail=(
                        "This name is reserved for a column the pipeline itself writes "
                        "downstream (prediction results, exports, or the train/test split "
                        "column) -- using it as a target would let that column silently "
                        "overwrite the target's own values in every prediction and export. "
                        f"Reserved names: {', '.join(sorted(RESERVED_TARGET_COLUMNS))}."
                    ),
                )
            )

        try:
            upload_ref = uuid.UUID(command.upload_ref)
        except ValueError:
            return Failure(ValidationError("upload_ref is not a valid upload reference"))

        key = upload_key(workspace_id, upload_ref)
        if not self._store.exists(key):
            return Failure(NotFoundError("Upload", str(upload_ref)))

        try:
            frame = read_csv_upload(self._store.get_bytes(key))
        except ValidationError as error:
            return Failure(error)

        missing = [
            column
            for column in (command.structure_column, command.target.column)
            if column not in frame.columns
        ]
        if missing:
            return Failure(
                ValidationError(
                    f"Column(s) not present in the uploaded file: {', '.join(missing)}",
                    detail=f"Available columns: {', '.join(frame.columns)}",
                )
            )

        try:
            prepared, report = prepare_frame(
                frame, command.structure_column, command.target, self._normalizer
            )
        except pl.exceptions.PolarsError as error:
            # The target gate inside prepare_frame catches what we know about; this
            # is the net for a column shape nobody has met yet -- a 422 naming the
            # file's problem, never a 500.
            return Failure(
                ValidationError(f"The file could not be interpreted: {error}", detail=None)
            )
        if report.valid_rows == 0:
            # `prepared` is deliberately untouched on this path. prepare_frame's
            # zero-valid-rows return hands back a frame whose structure column has
            # degraded to polars' Null dtype (the result of filtering a String
            # column with an all-False mask); the report is the whole payload here.
            return Failure(
                InvalidDatasetError("No valid structures in the uploaded file", report=report)
            )

        try:
            split_frame = assign_split(
                prepared, command.structure_column, command.split, self._normalizer
            )
        except ValidationError as error:
            # A scaffold split that cannot honour the requested fractions is a real
            # outcome for a congeneric series, not a bug. The message names the
            # dominant family and what to do instead, so it travels to the client
            # exactly as written.
            return Failure(error)

        degenerate = _degenerate_partition(split_frame, command.target)
        if degenerate is not None:
            return Failure(degenerate)

        dataset_id = uuid.uuid4()
        snapshot_uri, content_hash = write_snapshot(
            self._store, str(workspace_id), str(dataset_id), split_frame
        )

        existing = await self._repository.find_by_content_hash(workspace_id, content_hash)
        if existing is not None:
            # ponytail: the snapshot just written is now orphaned -- byte-identical
            # to the one the existing Dataset already points at, under an id no row
            # references. Harmless and inside the workspace's own prefix. Reclaim it
            # with a sweep over datasets/* keys with no matching row if it ever adds up.
            return Failure(DuplicateDatasetError(existing.id))

        dataset = Dataset(
            id=dataset_id,
            workspace_id=workspace_id,
            name=command.name,
            structure_column=command.structure_column,
            target=command.target,
            split=command.split,
            content_hash=content_hash,
            snapshot_uri=snapshot_uri,
            row_count=split_frame.height,
            validation_report=report,
        )
        await self._repository.add(dataset)
        return Success(dataset)


def _degenerate_partition(frame: pl.DataFrame, target: TargetSpec) -> ValidationError | None:
    """I1 (whole-branch review, Important): a train partition whose target
    column carries only one distinct value (either kind), or a test
    partition whose *numeric* target is constant.

    `assign_split` already raises when a *requested* partition (a nonzero
    fraction) comes back entirely empty -- but a partition that is merely
    single-class (BINARY) or constant-valued (NUMERIC, the regression
    analogue of single-class) still has rows, so that guard never fires.
    What happens next depends on *which* partition and *which* target kind,
    and the two do not fail the same way:

    - A single-class or constant **train** partition, either kind, can only
      ever learn one answer, so every future prediction comes back as that
      one class/value with a fabricated `uncertainty` of exactly `0.0`
      (maximally confident on every compound) -- a triage grid has no way to
      tell that from a real, well-trained model. Always rejected here.
    - A constant-valued **test** partition on a NUMERIC target makes
      `r2_score` return `0.0` rather than an undefined value, and nothing
      downstream reports that as undefined the way classification's metrics
      do -- so a regression Scorecard would show "R2 = 0.0" as if it were
      measured, silently. Rejected here.
    - A single-class **test** partition on a BINARY target is different: a
      seed sweep against real balanced/imbalanced datasets (20 rows 50/50,
      40 rows at 10%/20% actives) showed this firing on 9-15 of 25 seeds,
      *never* on `train` -- rejecting a model that is perfectly trainable and
      useful. And `_scoring.py`'s single-class branch already reports every
      classification metric as `NaN`, which `train_protocol.py`'s
      `_undefined_reasons` turns into `metrics_undefined` with an actionable
      message ("add positives (or negatives), or split it differently") --
      an earlier version of this guard replaced that honest null-with-reason
      with a hard refusal, which is worse than the bug it was fixing. **Not**
      rejected here; left to the Scorecard, which already handles it.

    Only "train" and "test" are checked at all: "validation" is assigned a
    label by `assign_split` but nothing downstream (`train_protocol.py`,
    both ECFP4 engines) ever reads it for fitting or scoring, so a degenerate
    validation partition has no honesty consequence to guard against.

    A partition of fewer than two rows is skipped, not flagged: a single row
    trivially has exactly one distinct value regardless of whether the
    dataset has a real balance problem, so treating that as "degenerate"
    would reject the many tiny (e.g. four-row) fixtures already exercising
    other, unrelated behaviour throughout this codebase's test suite --
    a real single-class/constant partition this guard needs to catch has
    several rows, not one.

    `assign_split` deliberately carries no knowledge of the target column (a
    decision already reviewed and accepted), so this runs here instead, once
    `command.target` and the split it produced are both in hand.
    """
    is_binary = target.kind is TargetKind.BINARY
    # BINARY's test partition is deliberately excluded: see the docstring
    # above for why rejecting it would replace an honest, already-working
    # null-with-reason with an over-eager hard refusal.
    partitions = ("train",) if is_binary else ("train", "test")
    for partition in partitions:
        rows = frame.filter(pl.col("split") == partition)
        if rows.height < 2:
            # Either an explicitly requested zero-fraction partition, a case
            # `assign_split` has already rejected, or too small for "only one
            # distinct value" to mean anything -- none of them this guard's
            # concern.
            continue
        if rows[target.column].n_unique() >= 2:
            continue
        kind = "class" if is_binary else "value"
        return ValidationError(
            f"The '{partition}' partition has only one distinct target {kind} after "
            "splitting, which would train or score a maximally confident but "
            "meaningless model",
            detail=(
                f"Every row in the '{partition}' partition has the same "
                f"'{target.column}' value. Use a different split seed, "
                "SplitStrategy.RANDOM instead of a scaffold split, or add more "
                "diverse compounds/measurements to the dataset."
            ),
        )
    return None

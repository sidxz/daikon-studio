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

import asyncio
import logging
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
from daikonstudio.domain.data.dataset import Dataset, DuplicateDatasetError, check_id_column
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec, check_targets
from daikonstudio.domain.data.validation import InvalidDatasetError, ValidationReport
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)


def upload_key(workspace_id: uuid.UUID, upload_ref: uuid.UUID) -> str:
    return f"{workspace_id}/uploads/{upload_ref}.csv"


@dataclass
class BuildProgress:
    """Where a build is: written by its worker thread, read by the event loop that
    reports it. Plain attribute writes, so a reader is at most one update behind."""

    stage: str = "Reading the file"
    done: int = 0
    total: int = 0

    def begin(self, stage: str, total: int = 0) -> None:
        self.stage, self.done, self.total = stage, 0, total

    def advance(self, done: int) -> None:
        self.done = done


@dataclass(frozen=True, kw_only=True)
class _Built:
    dataset_id: uuid.UUID
    snapshot_uri: str
    content_hash: str
    row_count: int
    report: ValidationReport


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
        # Off the loop: tens of MB to network storage would stall every other request.
        await asyncio.to_thread(
            self._store.put_bytes, upload_key(auth.workspace_id, upload_ref), data
        )
        return Success(upload_ref)


@dataclass(frozen=True, kw_only=True)
class CreateDatasetCommand:
    name: str
    upload_ref: str
    structure_column: str
    targets: tuple[TargetSpec, ...]
    split: SplitSpec
    id_column: str | None = None


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
        self,
        command: CreateDatasetCommand,
        auth: AuthContext | None = None,
        progress: BuildProgress | None = None,
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        workspace_id = auth.workspace_id

        # C1 (whole-branch review, Critical): a target column named the same as
        # one of the names `check_targets` reserves is not a naming quirk -- it is a
        # silent data-corruption bug. `derive_readouts` names the predicted Readout
        # after `target.column`, and every reserved name is a column the pipeline
        # itself writes downstream (see `RESERVED_TARGET_COLUMNS`'s own comment for
        # exactly where): whichever write happens last wins, so either the served
        # prediction becomes the model's uncertainty/an unrelated provenance value,
        # or -- for a target named "structure" -- the compound identity column is
        # overwritten by the predicted value instead. Checked here, before any file
        # is even read, because this is the one and only place a rejection can still
        # prevent the damage; the export-time collision guard (`export_collection.py`)
        # is downstream of a Protocol that has already been trained and published on
        # the bad column.
        try:
            check_targets(command.targets)
        except ValidationError as error:
            return Failure(error)
        target_columns = [target.column for target in command.targets]
        if command.structure_column in target_columns:
            return Failure(
                ValidationError("The structure column cannot also be a column to predict.")
            )

        try:
            upload_ref = uuid.UUID(command.upload_ref)
        except ValueError:
            return Failure(
                ValidationError("The upload reference is invalid. Upload the file again.")
            )

        # Everything from reading the upload to writing the snapshot is synchronous
        # blob I/O and RDKit over every row -- minutes on a few hundred thousand rows.
        # On the event loop it froze every other request: a 26 MB upload in prod
        # stopped the API answering its health check until Swarm killed it.
        built = await asyncio.to_thread(
            self._build,
            command,
            workspace_id,
            upload_ref,
            target_columns,
            progress or BuildProgress(),
        )
        if isinstance(built, DomainError):
            return Failure(built)
        dataset_id, content_hash = built.dataset_id, built.content_hash

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
            targets=command.targets,
            split=command.split,
            content_hash=content_hash,
            snapshot_uri=built.snapshot_uri,
            row_count=built.row_count,
            validation_report=built.report,
            created_by=auth.user_id,
            id_column=command.id_column,
        )
        await self._repository.add(dataset)
        # The upload has served its purpose: the frozen snapshot is the dataset.
        # Kept on every failure above, so a failed create can be retried against the
        # same bytes; deleted here so that deleting the dataset frees all its storage
        # (nothing records which upload made which dataset).
        key = upload_key(workspace_id, upload_ref)
        try:
            self._store.delete(key)
        except Exception:
            logger.exception("Deleting upload %s failed; it is orphaned", key)
        return Success(dataset)

    def _build(
        self,
        command: CreateDatasetCommand,
        workspace_id: uuid.UUID,
        upload_ref: uuid.UUID,
        target_columns: list[str],
        progress: BuildProgress,
    ) -> _Built | DomainError:
        """The synchronous half of creation, run on a worker thread. Returns the
        error rather than raising it, so a rejection is a value, not a traceback."""
        key = upload_key(workspace_id, upload_ref)
        if not self._store.exists(key):
            return NotFoundError("Upload", str(upload_ref))

        try:
            frame = read_csv_upload(self._store.get_bytes(key))
        except ValidationError as error:
            return error

        missing = [
            column
            for column in (command.structure_column, *target_columns)
            if column not in frame.columns
        ]
        if missing:
            return ValidationError(
                f"Columns not found in the uploaded file: {', '.join(missing)}.",
                detail=f"Available columns: {', '.join(frame.columns)}",
            )

        if command.id_column is not None:
            try:
                check_id_column(
                    frame.columns,
                    id_column=command.id_column,
                    structure_column=command.structure_column,
                    target_columns=target_columns,
                )
            except ValidationError as error:
                return error

        progress.begin("Checking structures", frame.height)
        try:
            prepared, report = prepare_frame(
                frame,
                command.structure_column,
                command.targets,
                self._normalizer,
                on_row=progress.advance,
            )
        except pl.exceptions.PolarsError as error:
            # The target gate inside prepare_frame catches what we know about; this
            # is the net for a column shape nobody has met yet -- a 422 naming the
            # file's problem, never a 500.
            return ValidationError(f"The file could not be interpreted: {error}", detail=None)
        if report.valid_rows == 0:
            # `prepared` is deliberately untouched on this path. prepare_frame's
            # zero-valid-rows return hands back a frame whose structure column has
            # degraded to polars' Null dtype (the result of filtering a String
            # column with an all-False mask); the report is the whole payload here.
            return InvalidDatasetError(
                "The uploaded file has no usable rows. Every row failed structure or "
                "target validation; see the validation report for reasons.",
                report=report,
            )

        if command.split.strategy is SplitStrategy.SCAFFOLD:
            progress.begin("Grouping by scaffold", prepared.height)
        try:
            split_frame = assign_split(
                prepared,
                command.structure_column,
                command.split,
                self._normalizer,
                on_row=progress.advance,
            )
        except ValidationError as error:
            # A scaffold split that cannot honour the requested fractions is a real
            # outcome for a congeneric series, not a bug. The message names the
            # dominant family and what to do instead, so it travels to the client
            # exactly as written.
            return error

        for target in command.targets:
            degenerate = _degenerate_partition(split_frame, target)
            if degenerate is not None:
                return degenerate

        progress.begin("Saving")
        dataset_id = uuid.uuid4()
        snapshot_uri, content_hash = write_snapshot(
            self._store, str(workspace_id), str(dataset_id), split_frame
        )
        return _Built(
            dataset_id=dataset_id,
            snapshot_uri=snapshot_uri,
            content_hash=content_hash,
            row_count=split_frame.height,
            report=report,
        )


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
    `command.targets` and the split they produced are both in hand.
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
            f"After splitting, every compound in the {partition} set has the same "
            f"'{target.column}' {kind}. A model trained or evaluated on it would not be "
            "meaningful.",
            detail=(
                f"All rows in the {partition} set share the same '{target.column}' value. "
                "Use a different split seed or a random split, or add compounds with "
                "more varied measurements."
            ),
        )
    return None

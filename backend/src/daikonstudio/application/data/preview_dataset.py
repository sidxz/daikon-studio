"""Prepare a dataset for review, then freeze exactly the bytes that were reviewed.

Progress uses the existing durable build records. Prepared bytes and their manifest
live in workspace-scoped blob storage, so review survives a page reload and no
Dataset is created until the explicit freeze request. Uncommitted reviews expire
after a day; their blobs can be reclaimed by the storage lifecycle policy.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.data.build_dataset import GetDatasetBuild
from daikonstudio.application.data.create_dataset import (
    BuildProgress,
    CreateDataset,
    CreateDatasetCommand,
    PreparedDataset,
    upload_key,
)
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.engines.context import MIN_CUTOFF_CLASS_COUNT
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_build_repository import DatasetBuildRepository
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.domain.data.dataset import Dataset, DuplicateDatasetError
from daikonstudio.domain.data.dataset_build import BuildStatus, DatasetBuild
from daikonstudio.domain.data.split import split_from_dict, split_to_dict
from daikonstudio.domain.data.target import (
    TargetKind,
    TargetSpec,
    target_from_dict,
    target_to_dict,
)
from daikonstudio.domain.data.validation import report_from_dict, report_to_dict
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)
_running: set[asyncio.Task[None]] = set()
REVIEW_LIFETIME = timedelta(days=1)
READINESS_VERSION = 1


def preview_key(workspace_id: uuid.UUID, build_id: uuid.UUID, name: str) -> str:
    return f"{workspace_id}/dataset-previews/{build_id}/{name}"


def readiness_key(workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    return f"{workspace_id}/datasets/{dataset_id}/readiness.json"


@dataclass(frozen=True)
class TargetClassBalance:
    column: str
    split: str
    positive: int
    negative: int


@dataclass(frozen=True)
class DatasetReadiness:
    row_count: int
    partition_counts: dict[str, int]
    class_balance: list[TargetClassBalance]
    warnings: list[str]


def dataset_readiness(frame: pl.DataFrame, targets: tuple[TargetSpec, ...]) -> DatasetReadiness:
    counts = {
        part: frame.filter(pl.col("split") == part).height
        for part in ("train", "validation", "test")
    }
    balance: list[TargetClassBalance] = []
    warnings: list[str] = []
    for target in targets:
        if target.kind is TargetKind.BINARY:
            for part in counts:
                values = frame.filter(pl.col("split") == part)[target.column]
                # Both classes counted directly, never one subtracted from the row
                # count: a sparse target is null where it was not measured, and
                # `count - positive` would report every unmeasured row as inactive --
                # turning a thinly measured column into a falsely imbalanced one. Every
                # guard below reads `labelled` for the same reason.
                positive = int((values == 1).sum())
                negative = int((values == 0).sum())
                labelled = positive + negative
                balance.append(TargetClassBalance(target.column, part, positive, negative))
                if labelled and (positive == 0 or negative == 0):
                    warnings.append(
                        f"'{target.column}' has only one class in the {part} set. "
                        "Some evaluation metrics or cutoff tuning may be unavailable."
                    )
                elif labelled and min(positive, negative) / labelled < 0.1:
                    warnings.append(
                        f"'{target.column}' is imbalanced in the {part} set "
                        f"({positive} active, {negative} inactive)."
                    )
                # Deliberately NOT gated on `labelled`: a validation partition with no
                # labelled row for this target still keeps the cutoff at 0.5, which is
                # the thing this warning exists to say. Suppressing it there would hide
                # a true statement in exactly the case that most needs it.
                if part == "validation" and min(positive, negative) < MIN_CUTOFF_CLASS_COUNT:
                    warnings.append(
                        f"'{target.column}' has fewer than {MIN_CUTOFF_CLASS_COUNT} examples "
                        "of a class in validation. Requested cutoff tuning will keep the "
                        "default cutoff of 0.5 for this target."
                    )
        else:
            train = frame.filter(pl.col("split") == "train")[target.column]
            test = frame.filter(pl.col("split") == "test")[target.column]
            std = cast(float | None, train.std())
            if (
                std
                and len(test)
                and abs(cast(float, train.mean()) - cast(float, test.mean())) > std
            ):
                warnings.append(
                    f"'{target.column}' has a train/test mean difference larger than "
                    "one training standard deviation. This may indicate a distribution shift."
                )
    return DatasetReadiness(frame.height, counts, balance, warnings)


@dataclass(frozen=True)
class DatasetPreview:
    build: DatasetBuild
    preparation: dict[str, Any] | None


class StartDatasetPreview:
    def __init__(
        self, builds: DatasetBuildRepository, create: CreateDataset, store: BlobStore
    ) -> None:
        self._builds, self._create, self._store = builds, create, store

    async def __call__(
        self, command: CreateDatasetCommand, auth: AuthContext | None = None
    ) -> Result[DatasetPreview, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None
        try:
            uuid.UUID(command.upload_ref)
        except ValueError:
            return Failure(
                ValidationError("The upload reference is invalid. Upload the file again.")
            )
        build = DatasetBuild(
            workspace_id=auth.workspace_id, created_by=auth.user_id, name=command.name
        )
        await self._builds.add(build)
        task = asyncio.create_task(self._run(build, command))
        _running.add(task)
        task.add_done_callback(_running.discard)
        return Success(DatasetPreview(build, None))

    def _prepare(
        self, build: DatasetBuild, command: CreateDatasetCommand, progress: BuildProgress
    ) -> DomainError | None:
        prepared = self._create.prepare(
            command, build.workspace_id, uuid.UUID(command.upload_ref), progress
        )
        if isinstance(prepared, DomainError):
            return prepared
        assert isinstance(prepared, PreparedDataset)
        readiness = dataset_readiness(prepared.frame, command.targets)
        progress.begin("Preparing the review")
        buffer = io.BytesIO()
        prepared.frame.write_parquet(buffer)
        raw = buffer.getvalue()
        self._store.put_bytes(preview_key(build.workspace_id, build.id, "snapshot.parquet"), raw)
        manifest = {
            "name": command.name,
            "file_name": command.file_name,
            "upload_ref": command.upload_ref,
            "structure_column": command.structure_column,
            "targets": [target_to_dict(t) for t in command.targets],
            "split": split_to_dict(command.split),
            "id_column": command.id_column,
            "validation_report": report_to_dict(prepared.report),
            "readiness": asdict(readiness),
            "content_hash": hashlib.sha256(raw).hexdigest(),
            "expires_at": (datetime.now(UTC) + REVIEW_LIFETIME).isoformat(),
        }
        self._store.put_bytes(
            preview_key(build.workspace_id, build.id, "manifest.json"),
            json.dumps(manifest).encode(),
        )
        return None

    async def _run(self, build: DatasetBuild, command: CreateDatasetCommand) -> None:
        progress = BuildProgress()
        work = asyncio.create_task(asyncio.to_thread(self._prepare, build, command, progress))
        try:
            while not work.done():
                await asyncio.wait({work}, timeout=1)
                if not work.done():
                    build.report(progress.stage, progress.done, progress.total)
                    await self._save(build)
            error = work.result()
            if error is not None:
                build.fail(error.to_body())
            else:
                build.ready_for_review()
        except Exception:
            logger.exception("Preparing dataset review %s failed", build.id)
            build.fail(
                {"error": "InternalError", "message": "Could not prepare the dataset. Try again."}
            )
        await self._save(build)

    async def _save(self, build: DatasetBuild) -> None:
        try:
            await self._builds.save(build)
        except Exception:
            logger.exception("Saving dataset review progress %s failed", build.id)


class GetDatasetPreview:
    def __init__(self, builds: DatasetBuildRepository, store: BlobStore) -> None:
        self._builds, self._store = builds, store

    async def __call__(
        self, build_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[DatasetPreview, DomainError]:
        result = await GetDatasetBuild(self._builds)(build_id, auth=auth)
        if isinstance(result, Failure):
            return Failure(result.failure())
        build = result.unwrap()
        preparation = None
        if build.status is BuildStatus.SUCCEEDED and build.dataset_id is None:
            try:
                raw = await asyncio.to_thread(
                    self._store.get_bytes,
                    preview_key(build.workspace_id, build.id, "manifest.json"),
                )
                preparation = json.loads(raw)
            except FileNotFoundError:
                return Failure(NotFoundError("Dataset review", str(build_id)))
            if datetime.fromisoformat(preparation["expires_at"]) < datetime.now(UTC):
                return Failure(
                    ConflictError("This review has expired. Prepare the dataset again.")
                )
        return Success(DatasetPreview(build, preparation))


class FreezeDatasetPreview:
    def __init__(
        self, builds: DatasetBuildRepository, repository: DatasetRepository, store: BlobStore
    ) -> None:
        self._builds, self._repository, self._store = builds, repository, store

    async def __call__(
        self, build_id: uuid.UUID, name: str, auth: AuthContext | None = None
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None
        if not name.strip():
            return Failure(ValidationError("Enter a dataset name."))
        owned = await GetDatasetBuild(self._builds)(build_id, auth=auth)
        if isinstance(owned, Failure):
            return Failure(owned.failure())
        existing = await self._repository.get(auth.workspace_id, build_id)
        if existing is not None:
            return Success(existing)
        reviewed = await GetDatasetPreview(self._builds, self._store)(build_id, auth=auth)
        if isinstance(reviewed, Failure):
            return Failure(reviewed.failure())
        preview = reviewed.unwrap()
        manifest = preview.preparation
        if preview.build.status is not BuildStatus.SUCCEEDED or manifest is None:
            return Failure(ConflictError("Prepare and review the dataset before creating it."))
        duplicate = await self._repository.find_by_content_hash(
            auth.workspace_id, manifest["content_hash"]
        )
        if duplicate is not None:
            return Failure(DuplicateDatasetError(duplicate.id))
        raw = await asyncio.to_thread(
            self._store.get_bytes, preview_key(auth.workspace_id, build_id, "snapshot.parquet")
        )
        uri = await asyncio.to_thread(
            self._store.put_bytes, snapshot_key(auth.workspace_id, build_id), raw
        )
        dataset = Dataset(
            id=build_id,
            workspace_id=auth.workspace_id,
            created_by=auth.user_id,
            name=name.strip(),
            structure_column=manifest["structure_column"],
            targets=tuple(target_from_dict(t) for t in manifest["targets"]),
            split=split_from_dict(manifest["split"]),
            id_column=manifest["id_column"],
            content_hash=manifest["content_hash"],
            snapshot_uri=uri,
            row_count=manifest["readiness"]["row_count"],
            validation_report=report_from_dict(manifest["validation_report"]),
        )
        try:
            await self._repository.add(dataset)
        except ConflictError:
            existing = await self._repository.get(auth.workspace_id, build_id)
            if existing is not None:
                return Success(existing)
            raise
        preview.build.succeed(dataset.id)
        try:
            await self._builds.save(preview.build)
        except Exception:
            # The dataset is saved; a progress write must not turn creation into a failure.
            logger.exception("Updating the completed dataset review %s failed", build_id)
        # Creation is complete even if optional cleanup is interrupted.
        try:
            await asyncio.to_thread(
                self._store.put_bytes,
                readiness_key(auth.workspace_id, build_id),
                json.dumps({"version": READINESS_VERSION, **manifest["readiness"]}).encode(),
            )
            await asyncio.to_thread(
                self._store.delete,
                upload_key(auth.workspace_id, uuid.UUID(manifest["upload_ref"])),
            )
            await asyncio.to_thread(
                self._store.delete_prefix, preview_key(auth.workspace_id, build_id, "")
            )
        except Exception:
            logger.exception("Cleaning up dataset review %s failed", build_id)
        return Success(dataset)


class GetDatasetReadiness:
    def __init__(self, repository: DatasetRepository, store: BlobStore) -> None:
        self._repository, self._store = repository, store

    async def __call__(
        self, dataset_id: uuid.UUID, auth: AuthContext | None = None
    ) -> Result[DatasetReadiness, DomainError]:
        require_authenticated(auth)
        assert auth is not None
        dataset = await self._repository.get(auth.workspace_id, dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(dataset_id)))

        def read() -> DatasetReadiness:
            key = readiness_key(auth.workspace_id, dataset_id)
            try:
                cached = json.loads(self._store.get_bytes(key))
                if cached.get("version") == READINESS_VERSION:
                    return DatasetReadiness(
                        row_count=cached["row_count"],
                        partition_counts=cached["partition_counts"],
                        class_balance=[
                            TargetClassBalance(**entry) for entry in cached["class_balance"]
                        ],
                        warnings=cached["warnings"],
                    )
            except (FileNotFoundError, ValueError, KeyError, TypeError):
                pass
            raw = self._store.get_bytes(snapshot_key(auth.workspace_id, dataset_id))
            frame = pl.read_parquet(
                io.BytesIO(raw), columns=["split", *(t.column for t in dataset.targets)]
            )
            readiness = dataset_readiness(frame, dataset.targets)
            try:
                self._store.put_bytes(
                    key, json.dumps({"version": READINESS_VERSION, **asdict(readiness)}).encode()
                )
            except Exception:
                logger.exception("Caching dataset readiness %s failed", dataset_id)
            return readiness

        try:
            return Success(await asyncio.to_thread(read))
        except FileNotFoundError:
            return Failure(NotFoundError("Stored dataset file", str(dataset_id)))

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

import io
import uuid
from dataclasses import dataclass

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.application.data.prepare_frame import prepare_frame
from daikonstudio.application.data.snapshot import write_snapshot
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.dataset import Dataset, DuplicateDatasetError
from daikonstudio.domain.data.split import SplitSpec
from daikonstudio.domain.data.target import TargetSpec
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

        try:
            upload_ref = uuid.UUID(command.upload_ref)
        except ValueError:
            return Failure(ValidationError("upload_ref is not a valid upload reference"))

        key = upload_key(workspace_id, upload_ref)
        if not self._store.exists(key):
            return Failure(NotFoundError("Upload", str(upload_ref)))

        try:
            frame = pl.read_csv(io.BytesIO(self._store.get_bytes(key)))
        except pl.exceptions.PolarsError as error:
            return Failure(ValidationError(f"The uploaded file is not readable as CSV: {error}"))

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

        prepared, report = prepare_frame(
            frame, command.structure_column, command.target, self._normalizer
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

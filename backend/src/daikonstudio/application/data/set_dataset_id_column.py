"""Set or clear a Dataset's identifier column, the one setting not frozen with it.

Display metadata only: no snapshot, score or content hash changes, and it can be
changed back, so any editor may set it."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.data.compound_ids import eligible_id_columns, snapshot_columns
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.domain.data.dataset import Dataset, check_id_column
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError


@dataclass(frozen=True, kw_only=True)
class SetDatasetIdColumnCommand:
    dataset_id: uuid.UUID
    id_column: str | None


@dataclass(frozen=True, kw_only=True)
class GetDatasetColumnsQuery:
    dataset_id: uuid.UUID


class SetDatasetIdColumn:
    def __init__(self, repository: DatasetRepository, store: BlobStore) -> None:
        self._repository = repository
        self._store = store

    async def __call__(
        self, command: SetDatasetIdColumnCommand, auth: AuthContext | None = None
    ) -> Result[Dataset, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None
        dataset = await self._repository.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        if command.id_column is not None:
            try:
                columns = snapshot_columns(self._store, dataset)
            except FileNotFoundError:
                return Failure(NotFoundError("Stored dataset file", str(dataset.id)))
            try:
                check_id_column(
                    columns,
                    id_column=command.id_column,
                    structure_column=dataset.structure_column,
                    target_column=dataset.target.column,
                )
            except ValidationError as error:
                return Failure(error)
        await self._repository.set_id_column(auth.workspace_id, dataset.id, command.id_column)
        dataset.id_column = command.id_column
        return Success(dataset)


class GetDatasetColumns:
    """The snapshot columns eligible as an identifier, for the picker."""

    def __init__(self, repository: DatasetRepository, store: BlobStore) -> None:
        self._repository = repository
        self._store = store

    async def __call__(
        self, query: GetDatasetColumnsQuery, auth: AuthContext | None = None
    ) -> Result[list[str], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        dataset = await self._repository.get(auth.workspace_id, query.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(query.dataset_id)))
        try:
            columns = snapshot_columns(self._store, dataset)
        except FileNotFoundError:
            return Failure(NotFoundError("Stored dataset file", str(dataset.id)))
        return Success(eligible_id_columns(columns, dataset))

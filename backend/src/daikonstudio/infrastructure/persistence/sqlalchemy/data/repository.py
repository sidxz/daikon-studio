"""SQLAlchemy Dataset repository.

Every read filters on `workspace_id` inside the SQL, never after it: a
cross-tenant row is not fetched and then discarded, it is never selected.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Select, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import split_from_dict, split_to_dict
from daikonstudio.domain.data.target import target_from_dict, target_to_dict
from daikonstudio.domain.data.validation import report_from_dict, report_to_dict
from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.infrastructure.persistence.sqlalchemy.data.models import DatasetModel


def _to_domain(model: DatasetModel) -> Dataset:
    return Dataset(
        id=model.id,
        workspace_id=model.workspace_id,
        name=model.name,
        target=target_from_dict(model.target),
        split=split_from_dict(model.split),
        content_hash=model.content_hash,
        snapshot_uri=model.snapshot_uri,
        row_count=model.row_count,
        validation_report=report_from_dict(model.validation_report),
        created_at=model.created_at,
        updated_at=model.updated_at,
        version=model.version,
    )


def _to_model(dataset: Dataset) -> DatasetModel:
    return DatasetModel(
        id=dataset.id,
        workspace_id=dataset.workspace_id,
        name=dataset.name,
        target=target_to_dict(dataset.target),
        split=split_to_dict(dataset.split),
        content_hash=dataset.content_hash,
        snapshot_uri=dataset.snapshot_uri,
        row_count=dataset.row_count,
        validation_report=report_to_dict(dataset.validation_report),
        version=dataset.version,
        # Explicit, rather than letting the column's server_default fill them in:
        # otherwise the created_at in the 201 response body (the aggregate's own
        # clock) and the one every later GET returns (the database's) differ by
        # however long the insert took.
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


class SqlAlchemyDatasetRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, dataset: Dataset) -> None:
        async with self._sessions() as session:
            session.add(_to_model(dataset))
            try:
                await session.commit()
            except IntegrityError as error:
                # CreateDataset already checked for an existing content hash; this
                # catches the narrow race where two identical uploads commit at once.
                # A ConflictError is a 409 -- far better than surfacing a 500.
                await session.rollback()
                raise ConflictError(
                    "This data is already stored in this workspace",
                    detail="A concurrent upload of identical data won the race.",
                ) from error

    async def get(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> Dataset | None:
        return await self._one(
            select(DatasetModel).where(
                DatasetModel.id == dataset_id, DatasetModel.workspace_id == workspace_id
            )
        )

    async def find_by_content_hash(
        self, workspace_id: uuid.UUID, content_hash: str
    ) -> Dataset | None:
        return await self._one(
            select(DatasetModel).where(
                DatasetModel.workspace_id == workspace_id,
                DatasetModel.content_hash == content_hash,
            )
        )

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[Dataset]:
        statement = (
            select(DatasetModel)
            .where(DatasetModel.workspace_id == workspace_id)
            .order_by(DatasetModel.created_at.desc(), DatasetModel.id.desc())
        )
        if cursor is not None:
            statement = statement.where(tuple_(DatasetModel.created_at, DatasetModel.id) < cursor)
        async with self._sessions() as session:
            result = await session.execute(statement.limit(limit))
            return [_to_domain(model) for model in result.scalars()]

    async def _one(self, statement: Select[tuple[DatasetModel]]) -> Dataset | None:
        async with self._sessions() as session:
            model = (await session.execute(statement)).scalar_one_or_none()
            return _to_domain(model) if model is not None else None

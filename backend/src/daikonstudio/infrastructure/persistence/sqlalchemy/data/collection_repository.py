"""SQLAlchemy Collection repository.

Read-only after `add`, like `SqlAlchemyDatasetRepository`: a Collection is a
frozen triage decision, so there is no `update`. Every read filters on
`workspace_id` inside the SQL, never after it -- a cross-tenant row is not
fetched and then discarded, it is never selected.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.data.collection import (
    Collection,
    provenance_from_dict,
    provenance_to_dict,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.models import CollectionModel


def _to_domain(model: CollectionModel) -> Collection:
    return Collection(
        id=model.id,
        workspace_id=model.workspace_id,
        name=model.name,
        derived_from_run_id=model.derived_from_run_id,
        member_count=model.member_count,
        snapshot_uri=model.snapshot_uri,
        provenance=provenance_from_dict(model.provenance),
        created_at=model.created_at,
        updated_at=model.updated_at,
        version=model.version,
    )


def _to_model(collection: Collection) -> CollectionModel:
    return CollectionModel(
        id=collection.id,
        workspace_id=collection.workspace_id,
        name=collection.name,
        derived_from_run_id=collection.derived_from_run_id,
        member_count=collection.member_count,
        snapshot_uri=collection.snapshot_uri,
        provenance=provenance_to_dict(collection.provenance),
        version=collection.version,
        # Explicit, not the column's server_default: the 201 response body (the
        # aggregate's own clock) and a later GET's (the database's) must agree.
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )


class SqlAlchemyCollectionRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, collection: Collection) -> None:
        async with self._sessions() as session:
            session.add(_to_model(collection))
            await session.commit()

    async def get(self, workspace_id: uuid.UUID, collection_id: uuid.UUID) -> Collection | None:
        async with self._sessions() as session:
            model = (
                await session.execute(
                    select(CollectionModel).where(
                        CollectionModel.id == collection_id,
                        CollectionModel.workspace_id == workspace_id,
                    )
                )
            ).scalar_one_or_none()
            return _to_domain(model) if model is not None else None

"""SQLAlchemy DatasetBuild repository. `save` is a plain UPDATE, last write wins: only
the task running a build writes it (see application.ports.dataset_build_repository)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.domain.data.dataset_build import BuildStatus, DatasetBuild
from daikonstudio.infrastructure.persistence.sqlalchemy.data.models import DatasetBuildModel


class SqlAlchemyDatasetBuildRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, build: DatasetBuild) -> None:
        async with self._sessions() as session:
            session.add(
                DatasetBuildModel(
                    id=build.id,
                    workspace_id=build.workspace_id,
                    created_by=build.created_by,
                    name=build.name,
                    created_at=build.created_at,
                    updated_at=build.updated_at,
                    version=build.version,
                    **_progress(build),
                )
            )
            await session.commit()

    async def get(self, build_id: uuid.UUID) -> DatasetBuild | None:
        async with self._sessions() as session:
            model = await session.scalar(
                select(DatasetBuildModel).where(DatasetBuildModel.id == build_id)
            )
        if model is None:
            return None
        return DatasetBuild(
            id=model.id,
            workspace_id=model.workspace_id,
            created_by=model.created_by,
            name=model.name,
            status=BuildStatus(model.status),
            stage=model.stage,
            done=model.done,
            total=model.total,
            dataset_id=model.dataset_id,
            error=model.error,
            created_at=model.created_at,
            updated_at=model.updated_at,
            version=model.version,
        )

    async def save(self, build: DatasetBuild) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_update(DatasetBuildModel)
                .where(DatasetBuildModel.id == build.id)
                # The domain's clock, not the column's onupdate: staleness is judged
                # against the last progress report.
                .values(updated_at=build.updated_at, **_progress(build))
            )
            await session.commit()


def _progress(build: DatasetBuild) -> dict[str, object]:
    return {
        "status": build.status.value,
        "stage": build.stage,
        "done": build.done,
        "total": build.total,
        "dataset_id": build.dataset_id,
        "error": build.error,
    }

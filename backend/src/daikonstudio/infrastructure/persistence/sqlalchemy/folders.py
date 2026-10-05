"""SQLAlchemy folder model and repository."""

from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Index, String, Uuid, func, select
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.domain.shared.errors import ConflictError
from daikonstudio.domain.shared.folder import Folder, FolderKind
from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    WorkspaceIdMixin,
)

DUPLICATE = "A folder with this name already exists."


class FolderModel(Base, EntityModelMixin, WorkspaceIdMixin):
    __tablename__ = "folders"

    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)

    __table_args__ = (
        CheckConstraint("kind IN ('dataset','protocol')", name="ck_folders_kind"),
        Index(
            "uq_folders_workspace_kind_name", "workspace_id", "kind", func.lower(name), unique=True
        ),
    )


def _to_domain(model: FolderModel) -> Folder:
    return Folder(
        id=model.id,
        workspace_id=model.workspace_id,
        kind=FolderKind(model.kind),
        name=model.name,
        created_by=model.created_by,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class SqlAlchemyFolderRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, folder: Folder) -> None:
        async with self._sessions() as session:
            session.add(
                FolderModel(
                    id=folder.id,
                    workspace_id=folder.workspace_id,
                    kind=folder.kind.value,
                    name=folder.name,
                    created_by=folder.created_by,
                    created_at=folder.created_at,
                    updated_at=folder.updated_at,
                )
            )
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ConflictError(DUPLICATE) from error

    async def get(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> Folder | None:
        async with self._sessions() as session:
            model = (
                await session.execute(
                    select(FolderModel).where(
                        FolderModel.id == folder_id, FolderModel.workspace_id == workspace_id
                    )
                )
            ).scalar_one_or_none()
            return _to_domain(model) if model is not None else None

    async def list(self, workspace_id: uuid.UUID, kind: FolderKind) -> list[Folder]:
        async with self._sessions() as session:
            result = await session.execute(
                select(FolderModel)
                .where(FolderModel.workspace_id == workspace_id, FolderModel.kind == kind.value)
                .order_by(func.lower(FolderModel.name), FolderModel.id)
            )
            return [_to_domain(model) for model in result.scalars()]

    async def rename(self, workspace_id: uuid.UUID, folder_id: uuid.UUID, name: str) -> None:
        async with self._sessions() as session:
            try:
                await session.execute(
                    sa_update(FolderModel)
                    .where(FolderModel.id == folder_id, FolderModel.workspace_id == workspace_id)
                    .values(name=name, updated_at=func.now())
                )
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ConflictError(DUPLICATE) from error

    async def delete(self, workspace_id: uuid.UUID, folder_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_delete(FolderModel).where(
                    FolderModel.id == folder_id, FolderModel.workspace_id == workspace_id
                )
            )
            await session.commit()

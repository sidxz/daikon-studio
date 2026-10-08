"""SQLAlchemy page models and repository.

A page's owner is a kind and an id with no foreign key (it may be a dataset, a protocol
or a run), so the use cases that delete an owner delete its pages. Revisions cascade
with their page. Bodies live in `page_contents`, keyed by sha256 and shared by every
revision that saved the same body, so they are never deleted with a page.
"""

from __future__ import annotations

import builtins
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    false,
    func,
    select,
)
from sqlalchemy import delete as sa_delete
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.application.pages.manage import STALE
from daikonstudio.domain.shared.errors import ConflictError, NotFoundError
from daikonstudio.domain.shared.page import (
    Page,
    PageBlob,
    PageBody,
    PageOwnerKind,
    PageRevision,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.base import Base, EntityModelMixin


class PageModel(Base, EntityModelMixin):
    __tablename__ = "pages"

    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    owner_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    head_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    last_edited_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    revision_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    __table_args__ = (
        CheckConstraint("owner_kind IN ('dataset','protocol','run')", name="ck_pages_owner_kind"),
        Index("ix_pages_owner", "workspace_id", "owner_kind", "owner_id"),
    )


class PageContentModel(Base):
    __tablename__ = "page_contents"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PageRevisionModel(Base):
    __tablename__ = "page_revisions"

    page_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("pages.id", ondelete="CASCADE"), primary_key=True
    )
    revision_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(
        String(64), ForeignKey("page_contents.sha256"), nullable=False
    )
    author_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PageBlobModel(Base):
    __tablename__ = "page_blobs"

    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    mime: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def _to_domain(model: PageModel) -> Page:
    return Page(
        id=model.id,
        workspace_id=model.workspace_id,
        owner_kind=PageOwnerKind(model.owner_kind),
        owner_id=model.owner_id,
        title=model.title,
        created_by=model.created_by,
        head_sha256=model.head_sha256,
        last_edited_by=model.last_edited_by,
        archived=model.archived,
        revision_count=model.revision_count,
        version=model.version,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class SqlAlchemyPageRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, page: Page) -> None:
        async with self._sessions() as session:
            session.add(
                PageModel(
                    id=page.id,
                    workspace_id=page.workspace_id,
                    owner_kind=page.owner_kind.value,
                    owner_id=page.owner_id,
                    title=page.title,
                    head_sha256=page.head_sha256,
                    created_by=page.created_by,
                    last_edited_by=page.last_edited_by,
                    archived=page.archived,
                    revision_count=page.revision_count,
                    version=page.version,
                    created_at=page.created_at,
                    updated_at=page.updated_at,
                )
            )
            await session.commit()

    async def get(self, workspace_id: uuid.UUID, page_id: uuid.UUID) -> Page | None:
        async with self._sessions() as session:
            model = (
                await session.execute(
                    select(PageModel).where(
                        PageModel.id == page_id, PageModel.workspace_id == workspace_id
                    )
                )
            ).scalar_one_or_none()
            return _to_domain(model) if model is not None else None

    async def list(
        self,
        workspace_id: uuid.UUID,
        owner_kind: PageOwnerKind,
        owner_id: uuid.UUID,
        *,
        archived: bool,
        limit: int,
    ) -> builtins.list[Page]:
        async with self._sessions() as session:
            result = await session.execute(
                select(PageModel)
                .where(
                    PageModel.workspace_id == workspace_id,
                    PageModel.owner_kind == owner_kind.value,
                    PageModel.owner_id == owner_id,
                    PageModel.archived.is_(archived),
                )
                .order_by(PageModel.updated_at.desc(), PageModel.id)
                .limit(limit)
            )
            return [_to_domain(model) for model in result.scalars()]

    async def revise(
        self,
        workspace_id: uuid.UUID,
        page_id: uuid.UUID,
        *,
        expected_version: int,
        body: PageBody,
        author_id: uuid.UUID,
    ) -> Page:
        """One transaction. The conditional UPDATE takes the row lock, so a second save
        from the same version waits, then finds the version moved and matches nothing."""
        async with self._sessions() as session:
            await session.execute(
                pg_insert(PageContentModel)
                .values(sha256=body.sha256, body=body.body, size_bytes=body.size_bytes)
                .on_conflict_do_nothing(index_elements=["sha256"])
            )
            model = (
                await session.execute(
                    sa_update(PageModel)
                    .where(
                        PageModel.id == page_id,
                        PageModel.workspace_id == workspace_id,
                        PageModel.version == expected_version,
                        PageModel.archived.is_(False),
                    )
                    .values(
                        head_sha256=body.sha256,
                        revision_count=PageModel.revision_count + 1,
                        version=PageModel.version + 1,
                        last_edited_by=author_id,
                        updated_at=func.now(),
                    )
                    .returning(PageModel)
                    .execution_options(synchronize_session=False)
                )
            ).scalar_one_or_none()
            if model is None:
                await session.rollback()
                raise ConflictError(STALE)
            page = _to_domain(model)
            session.add(
                PageRevisionModel(
                    page_id=page_id,
                    revision_no=page.revision_count,
                    sha256=body.sha256,
                    author_id=author_id,
                )
            )
            await session.commit()
            return page

    async def _bump(self, workspace_id: uuid.UUID, page_id: uuid.UUID, **values: Any) -> Page:
        async with self._sessions() as session:
            model = (
                await session.execute(
                    sa_update(PageModel)
                    .where(PageModel.id == page_id, PageModel.workspace_id == workspace_id)
                    .values(version=PageModel.version + 1, updated_at=func.now(), **values)
                    .returning(PageModel)
                    .execution_options(synchronize_session=False)
                )
            ).scalar_one_or_none()
            if model is None:
                raise NotFoundError("Page", str(page_id))
            page = _to_domain(model)
            await session.commit()
            return page

    async def retitle(self, workspace_id: uuid.UUID, page_id: uuid.UUID, title: str) -> Page:
        return await self._bump(workspace_id, page_id, title=title)

    async def set_archived(
        self, workspace_id: uuid.UUID, page_id: uuid.UUID, archived: bool
    ) -> Page:
        return await self._bump(workspace_id, page_id, archived=archived)

    async def delete(self, workspace_id: uuid.UUID, page_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            await session.execute(
                sa_delete(PageModel).where(
                    PageModel.id == page_id, PageModel.workspace_id == workspace_id
                )
            )
            await session.commit()

    async def delete_for_owners(
        self, workspace_id: uuid.UUID, owner_kind: PageOwnerKind, owner_ids: Sequence[uuid.UUID]
    ) -> None:
        if not owner_ids:
            return
        async with self._sessions() as session:
            await session.execute(
                sa_delete(PageModel).where(
                    PageModel.workspace_id == workspace_id,
                    PageModel.owner_kind == owner_kind.value,
                    PageModel.owner_id.in_(owner_ids),
                )
            )
            await session.commit()

    async def revisions(self, page_id: uuid.UUID) -> builtins.list[PageRevision]:
        async with self._sessions() as session:
            result = await session.execute(
                select(PageRevisionModel)
                .where(PageRevisionModel.page_id == page_id)
                .order_by(PageRevisionModel.revision_no)
            )
            return [
                PageRevision(
                    revision_no=model.revision_no,
                    sha256=model.sha256,
                    author_id=model.author_id,
                    created_at=model.created_at,
                )
                for model in result.scalars()
            ]

    async def content(self, page_id: uuid.UUID, sha256: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            body: dict[str, Any] | None = await session.scalar(
                select(PageContentModel.body)
                .join(PageRevisionModel, PageRevisionModel.sha256 == PageContentModel.sha256)
                .where(PageRevisionModel.page_id == page_id, PageContentModel.sha256 == sha256)
                .limit(1)
            )
            return body

    async def add_blob(self, blob: PageBlob) -> None:
        async with self._sessions() as session:
            await session.execute(
                pg_insert(PageBlobModel)
                .values(
                    workspace_id=blob.workspace_id,
                    sha256=blob.sha256,
                    mime=blob.mime,
                    size_bytes=blob.size_bytes,
                    created_by=blob.created_by,
                )
                .on_conflict_do_nothing(index_elements=["workspace_id", "sha256"])
            )
            await session.commit()

    async def get_blob(self, workspace_id: uuid.UUID, sha256: str) -> PageBlob | None:
        async with self._sessions() as session:
            model = await session.get(PageBlobModel, (workspace_id, sha256))
            if model is None:
                return None
            return PageBlob(
                workspace_id=model.workspace_id,
                sha256=model.sha256,
                mime=model.mime,
                size_bytes=model.size_bytes,
                created_by=model.created_by,
            )

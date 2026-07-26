"""SQLAlchemy models for the data context."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    VersionMixin,
    WorkspaceIdMixin,
)


class DatasetModel(Base, EntityModelMixin, WorkspaceIdMixin, VersionMixin):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    target: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    split: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_uri: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_report: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        # Content addressing is only a real property if the database enforces it:
        # without this index, two Datasets could hold the same frozen bytes under
        # different ids and "same hash means same data" would become advisory.
        Index("uq_datasets_workspace_content_hash", "workspace_id", "content_hash", unique=True),
        # Backs the newest-first keyset listing.
        Index("ix_datasets_workspace_created_at", "workspace_id", "created_at"),
    )

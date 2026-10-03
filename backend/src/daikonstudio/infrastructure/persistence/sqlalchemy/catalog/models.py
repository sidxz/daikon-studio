"""SQLAlchemy models for the catalog context."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    VersionMixin,
    WorkspaceIdMixin,
)


class InSilicoProtocolModel(Base, EntityModelMixin, WorkspaceIdMixin, VersionMixin):
    __tablename__ = "protocols"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    engine_id: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    # Readouts are the derived output contract (name/type/unit/direction); conditions
    # are the resolved training hyperparameters. Both are free-form and per-Protocol,
    # which is exactly what JSONB is for -- no separate table earns its join here.
    readouts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    conditions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Self-referencing version chain -- no separate ProtocolVersion table, exactly as
    # the sibling assay-protocol project models lab protocol versions.
    parent_protocol_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("protocols.id"), nullable=True
    )
    protocol_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Who created it, for the delete permission. NULL for rows made before 011.
    created_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)

    __table_args__ = (
        CheckConstraint("status IN ('draft','published')", name="ck_protocols_status"),
        # Backs the newest-first keyset listing.
        Index("ix_protocols_workspace_created_at", "workspace_id", "created_at"),
    )

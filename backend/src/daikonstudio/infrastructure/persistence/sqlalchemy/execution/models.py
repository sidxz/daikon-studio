"""SQLAlchemy models for the execution context."""

from __future__ import annotations

import uuid

from sqlalchemy import CheckConstraint, Float, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    VersionMixin,
    WorkspaceIdMixin,
)


class RunModel(Base, EntityModelMixin, WorkspaceIdMixin, VersionMixin):
    __tablename__ = "runs"

    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    # sha256 hex digest from `compute_cache_key` -- same width as Dataset.content_hash.
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    phase: Mapped[str | None] = mapped_column(String(256), nullable=True)
    result_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','ready','failed','cancelled')",
            name="ck_runs_status",
        ),
        CheckConstraint("kind IN ('training','prediction')", name="ck_runs_kind"),
        # Backs Task 17's cache lookup: same workspace, same cache_key means
        # identical work already has (or is getting) a result.
        Index("ix_runs_workspace_cache_key", "workspace_id", "cache_key"),
        # Backs the newest-first keyset listing.
        Index("ix_runs_workspace_created_at", "workspace_id", "created_at"),
    )

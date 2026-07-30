"""SQLAlchemy models for the execution context."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CheckConstraint, Float, Index, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
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
    # The job's own inputs -- which dataset, which engine, which conditions for a
    # training run. Free-form and per-kind, which is what JSONB is for; a column
    # per kind would be a wide table of mostly-nulls.
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    # Nullable by nature, not by laxity: a training Run has no Protocol until it
    # finishes. A bare indexed UUID, not a ForeignKey -- `catalog` and
    # `execution` are separate bounded contexts and cross-context references are
    # plain ids by contract.
    protocol_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
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
        # Backs "which runs belong to this Protocol".
        Index("ix_runs_workspace_protocol_id", "workspace_id", "protocol_id"),
    )

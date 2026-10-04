"""SQLAlchemy models for the execution context."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    text,
)
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
    # NULL for an ordinary solo run. Set at creation and never updated -- see
    # `Run.sweep_id`. A bare indexed UUID with no `sweeps` table behind it:
    # the group's only state is its members, and a `GROUP BY` answers every
    # question the list page asks.
    sweep_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    # The headline metric, denormalised for ranking -- see `Run.record_metrics`.
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    phase: Mapped[str | None] = mapped_column(String(256), nullable=True)
    result_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Queue columns -- see the 2026-08-04 self-hosted-runners spec. `lane` is
    # NULL until the enqueuer sets it; only laned pending runs are claimable.
    lane: Mapped[str | None] = mapped_column(String(32), nullable=True)
    claimed_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

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
        # Backs both sweep queries: the member list, and the grouped summary
        # the sweeps list page reads.
        Index("ix_runs_workspace_sweep_id", "workspace_id", "sweep_id"),
        # Backs the queue: pending runs with a lane set are claimable by runners.
        Index(
            "ix_runs_claimable", "lane", "created_at", postgresql_where=text("status = 'pending'")
        ),
    )


class RunEpochModel(Base):
    """One finished training epoch (application.engines.context.EpochPoint), for the
    run page's live charts. Append-only; `attempt` is the run's `attempts` when the
    point arrived, so a redelivered run is charted from its latest attempt alone."""

    __tablename__ = "run_epochs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    fit: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str | None] = mapped_column(Text, nullable=True)
    member: Mapped[int | None] = mapped_column(Integer, nullable=True)
    members: Mapped[int | None] = mapped_column(Integer, nullable=True)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    device: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

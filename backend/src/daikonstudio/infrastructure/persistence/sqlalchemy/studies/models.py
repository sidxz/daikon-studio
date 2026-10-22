"""SQLAlchemy model for the studies context."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    VersionMixin,
    WorkspaceIdMixin,
)


class StudyModel(Base, EntityModelMixin, WorkspaceIdMixin, VersionMixin):
    __tablename__ = "studies"
    __table_args__ = (Index("ix_studies_workspace_created_at", "workspace_id", "created_at"),)

    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    restated_goal: Mapped[str] = mapped_column(Text, nullable=False)
    checks: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    stop_conditions: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    # ponytail: steps and values are JSONB on the study row; split into tables when a
    # query has to cross studies.
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    values: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    # What the planner asked before it planned, and what the scientist answered.
    answers: Mapped[list[dict[str, str]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # None while a draft; Start sets it.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="run")
    # How many steps had run when the checks or stop conditions last changed after Start.
    amended_after_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The study's one conversation: its turns and the paragraph that stands in for old ones.
    thread: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

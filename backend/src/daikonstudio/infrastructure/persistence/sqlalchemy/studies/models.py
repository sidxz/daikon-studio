"""SQLAlchemy model for the studies context."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Index, Text, Uuid, text
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
    # The plan the study was started from, as the planner returned it.
    planning: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # One conversation per stage, each with the turns and the paragraph that leaves it.
    conversations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

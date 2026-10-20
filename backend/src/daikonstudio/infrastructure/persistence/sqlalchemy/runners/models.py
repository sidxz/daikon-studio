"""Runner rows -- registered machines allowed to claim runs. Instance-level:
no workspace_id, a runner serves lanes, and lanes cross workspaces."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    VersionMixin,
)


class RunnerModel(Base, EntityModelMixin, VersionMixin):
    __tablename__ = "runners"

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    lanes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # The build the agent reports at start (`POST /api/v1/runner/identity`), so an
    # operator can tell a runner on a stale image from one on the current one. All
    # three are nullable on purpose -- a runner that has not checked in since this
    # column existed reports nothing, and must keep claiming runs. `Text`, not a
    # bounded `String`: the values come from the image's own `APP_*` environment,
    # and a bound only buys a DataError (a 500) in exchange for nothing.
    build_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    build_git_sha: Mapped[str | None] = mapped_column(Text, nullable=True)
    build_date: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", name="uq_runners_name"),
        UniqueConstraint("token_hash", name="uq_runners_token_hash"),
    )

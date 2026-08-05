"""The Runner aggregate -- a registered machine allowed to claim runs.

Instance-level, not workspace-scoped: a runner serves lanes, and lanes cross
workspaces (see `RunnerModel`'s docstring).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from daikonstudio.domain.shared.entity import AggregateRoot


class Runner(AggregateRoot):
    def __init__(
        self,
        *,
        name: str,
        lanes: tuple[str, ...],
        token_hash: str,
        last_seen_at: datetime | None = None,
        revoked_at: datetime | None = None,
        id: uuid.UUID | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(id=id, created_at=created_at, updated_at=updated_at, version=version)
        self.name = name
        self.lanes = lanes
        self.token_hash = token_hash
        self.last_seen_at = last_seen_at
        self.revoked_at = revoked_at

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def revoke(self) -> None:
        """Idempotent: a second call keeps the first `revoked_at` instant
        rather than sliding it forward."""
        if self.revoked_at is None:
            self.revoked_at = datetime.now(UTC)
            self._touch()

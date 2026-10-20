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
        build_version: str | None = None,
        build_git_sha: str | None = None,
        build_date: str | None = None,
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
        # Which build the agent process is running, as it reported at start --
        # NOT `version`, which is this aggregate's optimistic-concurrency
        # counter. All three are None until an agent reports them: a runner
        # that has not restarted since this field existed, or one running an
        # older agent, says nothing and must keep working, so the absence is a
        # fact to show rather than a value to invent.
        self.build_version = build_version
        self.build_git_sha = build_git_sha
        self.build_date = build_date

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def record_build(self, *, version: str, git_sha: str, build_date: str) -> None:
        """Record the build the agent reports. Build identity can only change by
        restarting the agent process, so the agent sends it once at start and
        this overwrites whatever the previous process left."""
        self.build_version = version
        self.build_git_sha = git_sha
        self.build_date = build_date
        self._touch()

    def revoke(self) -> None:
        """Idempotent: a second call keeps the first `revoked_at` instant
        rather than sliding it forward."""
        if self.revoked_at is None:
            self.revoked_at = datetime.now(UTC)
            self._touch()

"""Runner management -- register, list, and revoke self-hosted runners.

Instance-level, not workspace-scoped -- same reason `domain/runners/runner.py`
gives: a runner serves lanes, and lanes cross workspaces. Creating and revoking
need the admin role in *some* workspace (not a specific one): a runner token
reaches every tenant's runs, so it is an operator's credential, not an editor's.
Listing needs only an authenticated caller.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_admin, require_authenticated
from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.domain.runners.runner import Runner
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError

# `secrets.token_urlsafe(32)` -- 256 bits, the same budget the stdlib's own docs
# recommend for a security token. The `drt_` prefix (**d**aikon **r**unner
# **t**oken) is not for security, only so a token accidentally pasted into a
# log or a bug report is recognisable at a glance, the way GitHub's `ghp_`
# prefix is.
_TOKEN_PREFIX = "drt_"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class CreateRunnerCommand:
    name: str
    lanes: tuple[str, ...]


@dataclass(frozen=True)
class CreatedRunner:
    runner: Runner
    token: str  # plaintext, surfaced exactly once -- only the hash is ever stored


class CreateRunner:
    def __init__(self, runners: RunnerRepository) -> None:
        self._runners = runners

    async def __call__(
        self, command: CreateRunnerCommand, *, auth: AuthContext | None
    ) -> Result[CreatedRunner, DomainError]:
        require_authenticated(auth)
        # Admin, not editor: a runner token claims runs from *every* workspace and
        # reads and writes every blob in the workspaces it serves ("one lane = one
        # trust domain", runners design). Minting one is not an editor's call.
        require_admin(auth)
        if not command.name.strip():
            return Failure(ValidationError("name must not be empty"))
        if not command.lanes:
            return Failure(ValidationError("lanes must not be empty"))
        if any(not lane.strip() for lane in command.lanes):
            return Failure(ValidationError("lanes must not contain blank entries"))

        token = f"{_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
        runner = Runner(name=command.name, lanes=command.lanes, token_hash=_hash_token(token))
        # Raises ConflictError on a duplicate name; that propagates as-is rather
        # than being wrapped in a Failure -- register_error_handlers renders any
        # DomainError raised during a request the same way it renders a Failure.
        await self._runners.add(runner)
        return Success(CreatedRunner(runner=runner, token=token))


@dataclass(frozen=True)
class RunnerStatusView:
    runner: Runner
    # last_seen_at within online_threshold_seconds of now, or, for a runner holding a
    # run, within busy_threshold_seconds (see ListRunners).
    online: bool
    current_run_id: uuid.UUID | None


class ListRunners:
    def __init__(
        self,
        runners: RunnerRepository,
        queue: RunQueue,
        *,
        online_threshold_seconds: int,
        busy_threshold_seconds: int,
    ) -> None:
        self._runners = runners
        self._queue = queue
        self._online_threshold = timedelta(seconds=online_threshold_seconds)
        # An idle runner polls every few seconds; a busy one stops polling and only
        # heartbeats every lease/3 (runner/agent.py), so judging it by the idle
        # threshold showed it Offline for most of every job. Given the whole lease,
        # a runner that dies mid-job still reads Offline once its lease lapses --
        # which is also when the sweep requeues its run.
        self._busy_threshold = timedelta(seconds=busy_threshold_seconds)

    async def __call__(
        self, *, auth: AuthContext | None
    ) -> Result[list[RunnerStatusView], DomainError]:
        require_authenticated(auth)
        runners = await self._runners.list()
        active_by_runner = await self._queue.active_run_by_runner()
        now = datetime.now(UTC)
        return Success(
            [
                RunnerStatusView(
                    runner=runner,
                    online=(
                        runner.last_seen_at is not None
                        and now - runner.last_seen_at
                        <= (
                            self._busy_threshold
                            if runner.id in active_by_runner
                            else self._online_threshold
                        )
                    ),
                    current_run_id=active_by_runner.get(runner.id),
                )
                for runner in runners
            ]
        )


@dataclass(frozen=True)
class RevokeRunnerCommand:
    runner_id: uuid.UUID


class RevokeRunner:
    def __init__(self, runners: RunnerRepository) -> None:
        self._runners = runners

    async def __call__(
        self, command: RevokeRunnerCommand, *, auth: AuthContext | None
    ) -> Result[None, DomainError]:
        require_authenticated(auth)
        require_admin(auth)  # same reasoning as CreateRunner
        runner = await self._runners.get(command.runner_id)
        if runner is None:
            return Failure(NotFoundError("Runner", str(command.runner_id)))
        await self._runners.revoke(command.runner_id)
        return Success(None)

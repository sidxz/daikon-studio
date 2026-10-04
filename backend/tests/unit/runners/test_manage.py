"""Unit coverage for the runner-management use cases.

Fakes everything around them (no real Postgres) -- `RunnerRepository` and
`RunQueue` are Protocols, so a plain class recording calls satisfies them.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from daikonstudio.application.runners.manage import (
    CreatedRunner,
    CreateRunner,
    CreateRunnerCommand,
    ListRunners,
    RevokeRunner,
    RevokeRunnerCommand,
)
from daikonstudio.domain.runners.runner import Runner
from daikonstudio.domain.shared.errors import AuthorizationError, NotFoundError, ValidationError
from tests.fakes.auth import FakeAuth


class FakeRunnerRepository:
    def __init__(self) -> None:
        self.added: list[Runner] = []
        self._by_id: dict[uuid.UUID, Runner] = {}
        self.revoked_ids: list[uuid.UUID] = []

    async def add(self, runner: Runner) -> None:
        self.added.append(runner)
        self._by_id[runner.id] = runner

    async def get(self, runner_id: uuid.UUID) -> Runner | None:
        return self._by_id.get(runner_id)

    async def get_by_token_hash(self, token_hash: str) -> Runner | None:
        return next((r for r in self._by_id.values() if r.token_hash == token_hash), None)

    async def list(self) -> list[Runner]:
        return list(self._by_id.values())

    async def touch_last_seen(self, runner_id: uuid.UUID) -> None:
        raise NotImplementedError

    async def revoke(self, runner_id: uuid.UUID) -> None:
        self.revoked_ids.append(runner_id)


class FakeRunQueue:
    def __init__(self, active_run_by_runner: dict[uuid.UUID, uuid.UUID] | None = None) -> None:
        self._active = active_run_by_runner or {}

    async def sweep(self, *, max_attempts: int) -> None:
        raise NotImplementedError

    async def claim_next(self, **kwargs: object) -> uuid.UUID | None:
        raise NotImplementedError

    async def verify_claim(self, *args: object, **kwargs: object) -> bool:
        raise NotImplementedError

    async def set_lane(self, run_id: uuid.UUID, lane: str) -> None:
        raise NotImplementedError

    async def active_run_by_runner(self) -> dict[uuid.UUID, uuid.UUID]:
        return self._active


# --- CreateRunner ------------------------------------------------------------


async def test_create_returns_token_matching_the_stored_hash() -> None:
    repo = FakeRunnerRepository()
    use_case = CreateRunner(repo)

    result = await use_case(
        CreateRunnerCommand(name="gpu-01", lanes=("default",)),
        auth=FakeAuth(workspace_role="admin"),
    )

    created = result.unwrap()
    assert isinstance(created, CreatedRunner)
    assert created.token.startswith("drt_")
    assert hashlib.sha256(created.token.encode()).hexdigest() == created.runner.token_hash
    assert repo.added == [created.runner]


async def test_create_rejects_empty_lanes() -> None:
    use_case = CreateRunner(FakeRunnerRepository())

    result = await use_case(
        CreateRunnerCommand(name="gpu-01", lanes=()), auth=FakeAuth(workspace_role="admin")
    )

    assert isinstance(result.failure(), ValidationError)


async def test_create_rejects_blank_lane_entries() -> None:
    use_case = CreateRunner(FakeRunnerRepository())

    result = await use_case(
        CreateRunnerCommand(name="gpu-01", lanes=("default", "  ")),
        auth=FakeAuth(workspace_role="admin"),
    )

    assert isinstance(result.failure(), ValidationError)


async def test_create_rejects_empty_name() -> None:
    use_case = CreateRunner(FakeRunnerRepository())

    result = await use_case(
        CreateRunnerCommand(name="  ", lanes=("default",)), auth=FakeAuth(workspace_role="admin")
    )

    assert isinstance(result.failure(), ValidationError)


async def test_create_refuses_viewer_role() -> None:
    use_case = CreateRunner(FakeRunnerRepository())

    with pytest.raises(AuthorizationError):
        await use_case(
            CreateRunnerCommand(name="gpu-01", lanes=("default",)),
            auth=FakeAuth(workspace_role="viewer"),
        )


async def test_create_refuses_unauthenticated_caller() -> None:
    """`require_editor(None)` alone would treat `auth=None` as a trusted
    system/worker call -- `require_authenticated` must run first so an
    anonymous caller cannot mint a runner token."""
    use_case = CreateRunner(FakeRunnerRepository())

    with pytest.raises(AuthorizationError):
        await use_case(CreateRunnerCommand(name="gpu-01", lanes=("default",)), auth=None)


# --- ListRunners ---------------------------------------------------------------


def _runner(**overrides: object) -> Runner:
    defaults: dict[str, object] = dict(name="gpu-01", lanes=("default",), token_hash="a" * 64)
    defaults.update(overrides)
    return Runner(**defaults)  # type: ignore[arg-type]


async def test_list_reports_online_from_last_seen_and_joins_current_run() -> None:
    now = datetime.now(UTC)
    online_runner = _runner(name="online", last_seen_at=now - timedelta(seconds=5))
    offline_runner = _runner(name="offline", last_seen_at=now - timedelta(seconds=60))
    never_seen_runner = _runner(name="never-seen", last_seen_at=None)

    repo = FakeRunnerRepository()
    for runner in (online_runner, offline_runner, never_seen_runner):
        await repo.add(runner)

    run_id = uuid.uuid4()
    queue = FakeRunQueue({online_runner.id: run_id})

    use_case = ListRunners(repo, queue, online_threshold_seconds=15, busy_threshold_seconds=600)
    views = (await use_case(auth=FakeAuth(workspace_role="admin"))).unwrap()
    by_name = {view.runner.name: view for view in views}

    assert by_name["online"].online is True
    assert by_name["online"].current_run_id == run_id
    assert by_name["offline"].online is False
    assert by_name["offline"].current_run_id is None
    assert by_name["never-seen"].online is False
    assert by_name["never-seen"].current_run_id is None


async def test_a_busy_runner_stays_online_between_heartbeats_until_its_lease_lapses() -> None:
    """A runner holding a run stops polling and heartbeats every lease/3 (200 s), so
    the idle 15 s threshold showed it Offline for most of every job (prod, 2026-10-04).
    Silent for longer than the lease, it is gone: the run is about to be requeued."""
    now = datetime.now(UTC)
    working = _runner(name="working", last_seen_at=now - timedelta(seconds=180))
    dead = _runner(name="dead", last_seen_at=now - timedelta(seconds=700))
    repo = FakeRunnerRepository()
    for runner in (working, dead):
        await repo.add(runner)
    queue = FakeRunQueue({working.id: uuid.uuid4(), dead.id: uuid.uuid4()})

    use_case = ListRunners(repo, queue, online_threshold_seconds=15, busy_threshold_seconds=600)
    views = (await use_case(auth=FakeAuth(workspace_role="admin"))).unwrap()
    by_name = {view.runner.name: view for view in views}

    assert by_name["working"].online is True
    assert by_name["dead"].online is False


# --- RevokeRunner ----------------------------------------------------------------


async def test_revoke_unknown_runner_is_not_found() -> None:
    use_case = RevokeRunner(FakeRunnerRepository())

    result = await use_case(
        RevokeRunnerCommand(runner_id=uuid.uuid4()), auth=FakeAuth(workspace_role="admin")
    )

    assert isinstance(result.failure(), NotFoundError)


async def test_revoke_known_runner_calls_repository_revoke() -> None:
    repo = FakeRunnerRepository()
    runner = _runner()
    await repo.add(runner)
    use_case = RevokeRunner(repo)

    result = await use_case(
        RevokeRunnerCommand(runner_id=runner.id), auth=FakeAuth(workspace_role="admin")
    )

    assert result.unwrap() is None
    assert repo.revoked_ids == [runner.id]


async def test_revoke_refuses_viewer_role() -> None:
    repo = FakeRunnerRepository()
    runner = _runner()
    await repo.add(runner)
    use_case = RevokeRunner(repo)

    with pytest.raises(AuthorizationError):
        await use_case(
            RevokeRunnerCommand(runner_id=runner.id), auth=FakeAuth(workspace_role="viewer")
        )


async def test_revoke_refuses_unauthenticated_caller() -> None:
    repo = FakeRunnerRepository()
    runner = _runner()
    await repo.add(runner)
    use_case = RevokeRunner(repo)

    with pytest.raises(AuthorizationError):
        await use_case(RevokeRunnerCommand(runner_id=runner.id), auth=None)


async def test_create_refuses_editor_role() -> None:
    """A runner token reaches every workspace's runs, so minting one is an
    admin's call, not an editor's (runners design: one lane = one trust domain)."""
    use_case = CreateRunner(FakeRunnerRepository())
    with pytest.raises(AuthorizationError):
        await use_case(
            CreateRunnerCommand(name="gpu-01", lanes=("default",)),
            auth=FakeAuth(workspace_role="editor"),
        )


async def test_revoke_refuses_editor_role() -> None:
    repository = FakeRunnerRepository()
    runner = Runner(name="gpu-01", lanes=("default",), token_hash="h")
    await repository.add(runner)
    with pytest.raises(AuthorizationError):
        await RevokeRunner(repository)(
            RevokeRunnerCommand(runner_id=runner.id), auth=FakeAuth(workspace_role="editor")
        )

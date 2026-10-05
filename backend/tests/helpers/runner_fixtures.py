"""Shared runner-protocol test plumbing -- registering a runner and seeding a
claimable run straight through the app's own container, the way both
`tests/api/test_runner_protocol.py` (Task 7) and `tests/api/test_runner_ports.py`
(Task 8) need to set up a run a runner can claim without going through
`POST /api/v1/runs` or `POST /api/v1/protocols`.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.runners.manage import CreateRunner, CreateRunnerCommand
from daikonstudio.domain.execution.run import Run, RunKind
from tests.fakes.auth import FakeAuth


async def register_runner(app, lanes: list[str]) -> tuple[uuid.UUID, dict[str, str]]:
    """Register a runner via `CreateRunner` resolved from the container, and
    hand back its id plus a header dict carrying its bearer token."""
    create_runner = app.state.container[CreateRunner]
    result = await create_runner(
        CreateRunnerCommand(name=f"runner-{uuid.uuid4()}", lanes=tuple(lanes)),
        auth=FakeAuth(workspace_role="admin"),
    )
    created = result.unwrap()
    return created.runner.id, {"Authorization": f"Bearer {created.token}"}


async def seed_run(
    app,
    workspace_id: uuid.UUID,
    *,
    kind: RunKind = RunKind.PREDICTION,
    lane: str | None = "default",
    params: dict[str, Any] | None = None,
    requested_by: uuid.UUID | None = None,
) -> Run:
    run = Run(
        kind=kind,
        workspace_id=workspace_id,
        requested_by=requested_by or uuid.uuid4(),
        cache_key=f"runner-protocol-test-{uuid.uuid4()}",
        params=params or {},
    )
    await app.state.container[RunRepository].add(run)
    if lane is not None:
        await app.state.container[RunQueue].set_lane(run.id, lane)
    return run


async def claim(anonymous_client, headers: dict[str, str]) -> dict[str, Any]:
    response = await anonymous_client.post("/api/v1/runner/claim", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


@contextlib.asynccontextmanager
async def cleanup_registered_runners(engine: AsyncEngine) -> AsyncIterator[None]:
    """Deletes any `runners` row registered during the `yield` -- for a fixture
    bound straight to the session-scoped `NullPool` engine instead of the
    savepoint-pinned per-test connection (`test_runner_ports.py`'s `blob_app`,
    `test_runner_full_loop.py`'s `app`; both need it so `SyncAsgiTransport`'s
    own event loop can drive a request).

    `runners` is instance-level, not workspace-scoped (`RunnerRepository`'s own
    docstring), so unlike every other write those fixtures make there is no
    workspace_id a targeted cleanup could scope by, and the NullPool trade-off
    already means no savepoint rollback undoes it either. A runner left behind
    outlives its test and corrupts
    `test_runner_repository.py::test_list_returns_all`, which asserts `list()`
    returns *exactly* the rows it just added -- so snapshot `runners` ids
    before the test body runs, diff after, and delete only what was added.
    """
    async with engine.connect() as probe:
        before = {row[0] for row in (await probe.execute(text("SELECT id FROM runners"))).all()}
    yield
    async with engine.connect() as probe:
        after = {row[0] for row in (await probe.execute(text("SELECT id FROM runners"))).all()}
    created = after - before
    if created:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM runners WHERE id = ANY(:ids)"), {"ids": list(created)}
            )

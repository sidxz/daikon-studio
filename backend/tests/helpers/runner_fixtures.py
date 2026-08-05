"""Shared runner-protocol test plumbing -- registering a runner and seeding a
claimable run straight through the app's own container, the way both
`tests/api/test_runner_protocol.py` (Task 7) and `tests/api/test_runner_ports.py`
(Task 8) need to set up a run a runner can claim without going through
`POST /api/v1/runs` or `POST /api/v1/protocols`.
"""

from __future__ import annotations

import uuid
from typing import Any

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
        CreateRunnerCommand(name=f"runner-{uuid.uuid4()}", lanes=tuple(lanes)), auth=FakeAuth()
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
) -> Run:
    run = Run(
        kind=kind,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
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

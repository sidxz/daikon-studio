"""API coverage for `/api/v1/runners`: register, list, revoke."""

from __future__ import annotations

import uuid

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.queue import SqlAlchemyRunQueue
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)


async def test_create_returns_a_token_exactly_once(client):
    response = await client.post("/api/v1/runners", json={"name": "gpu-01", "lanes": ["default"]})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "gpu-01"
    assert body["lanes"] == ["default"]
    assert body["token"].startswith("drt_")
    assert "id" in body and "created_at" in body


async def test_list_does_not_include_the_token(client):
    created = await client.post("/api/v1/runners", json={"name": "gpu-02", "lanes": ["default"]})
    assert created.status_code == 201, created.text

    response = await client.get("/api/v1/runners")
    assert response.status_code == 200, response.text
    items = response.json()
    item = next(item for item in items if item["name"] == "gpu-02")
    assert "token" not in item
    assert item["revoked"] is False
    assert item["online"] is False
    assert item["current_run_id"] is None


async def test_revoke_then_list_shows_revoked(client):
    created = await client.post("/api/v1/runners", json={"name": "gpu-03", "lanes": ["default"]})
    runner_id = created.json()["id"]

    revoke = await client.post(f"/api/v1/runners/{runner_id}/revoke")
    assert revoke.status_code == 204, revoke.text
    assert revoke.content == b""

    response = await client.get("/api/v1/runners")
    item = next(item for item in response.json() if item["id"] == runner_id)
    assert item["revoked"] is True


async def test_viewer_cannot_create(viewer_client):
    response = await viewer_client.post(
        "/api/v1/runners", json={"name": "gpu-04", "lanes": ["default"]}
    )
    assert response.status_code == 403, response.text


async def test_duplicate_name_conflicts(client):
    first = await client.post("/api/v1/runners", json={"name": "gpu-05", "lanes": ["default"]})
    assert first.status_code == 201, first.text

    second = await client.post("/api/v1/runners", json={"name": "gpu-05", "lanes": ["gpu"]})
    assert second.status_code == 409, second.text


async def test_list_reports_the_run_a_runner_currently_holds(
    client, session_factory, workspace_id
):
    """`RunQueue.active_run_by_runner` has no other test anywhere -- this is
    its first real exercise, through a genuine claim rather than a mock."""
    created = await client.post("/api/v1/runners", json={"name": "gpu-06", "lanes": ["default"]})
    runner_id = uuid.UUID(created.json()["id"])

    run = Run(
        kind=RunKind.PREDICTION,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="runners-api-test",
    )
    await SqlAlchemyRunRepository(session_factory).add(run)
    queue = SqlAlchemyRunQueue(session_factory)
    await queue.set_lane(run.id, "default")
    claimed_id = await queue.claim_next(
        runner_id=runner_id,
        lanes=["default"],
        lease_seconds=600,
        max_active_per_workspace=10,
        max_attempts=3,
    )
    assert claimed_id == run.id

    response = await client.get("/api/v1/runners")
    item = next(item for item in response.json() if item["id"] == str(runner_id))
    assert item["current_run_id"] == str(run.id)

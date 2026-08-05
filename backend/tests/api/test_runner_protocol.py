"""API coverage for the machine-facing runner protocol, `/api/v1/runner/*`.

This module is the security-boundary proof for the whole self-hosted-runner
feature, so it deliberately does NOT use the `client` fixture: that fixture
carries a Sentinel token pair, and every route under test here must work with
*only* a runner bearer token (and must reject everything else). `anonymous_client`
carries no default headers at all, so each request below states its own.

Runs and runners are seeded straight through the app's own container
(`app.state.container[...]`) rather than through their HTTP endpoints -- the
brief for this task is explicit that these tests never enqueue and never go
through `POST /api/v1/runs` or `POST /api/v1/protocols`.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.fakes.auth import FakeAuth

from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.application.runners.manage import CreateRunner, CreateRunnerCommand
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings

SOLUBILITY_CSV = b"smiles,y\nCCO,1.0\nc1ccccc1,5.0\nCCN,2.0\nc1ccncc1,6.0\n"

_UPDATE_BODY: dict[str, Any] = {
    "status": "running",
    "progress": 0.0,
    "phase": None,
    "result_uri": None,
    "error_message": None,
    "protocol_id": None,
    "expected_version": 1,
}


async def _register_runner(app, lanes: list[str]) -> tuple[uuid.UUID, dict[str, str]]:
    """Register a runner via `CreateRunner` resolved from the container, and
    hand back its id plus a header dict carrying its bearer token."""
    create_runner = app.state.container[CreateRunner]
    result = await create_runner(
        CreateRunnerCommand(name=f"runner-{uuid.uuid4()}", lanes=tuple(lanes)), auth=FakeAuth()
    )
    created = result.unwrap()
    return created.runner.id, {"Authorization": f"Bearer {created.token}"}


async def _seed_run(
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


async def _claim(anonymous_client, headers: dict[str, str]) -> dict[str, Any]:
    response = await anonymous_client.post("/api/v1/runner/claim", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# Auth: no Sentinel headers required, and only a live non-revoked token works.
# --------------------------------------------------------------------------


async def test_claim_without_a_token_is_401(anonymous_client):
    response = await anonymous_client.post("/api/v1/runner/claim")
    assert response.status_code == 401, response.text
    assert response.headers["www-authenticate"] == "Bearer"


async def test_claim_with_a_garbage_token_is_401(anonymous_client):
    response = await anonymous_client.post(
        "/api/v1/runner/claim", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401, response.text


async def test_claim_with_a_revoked_token_is_401(anonymous_client, app):
    runner_id, headers = await _register_runner(app, ["default"])
    await app.state.container[RunnerRepository].revoke(runner_id)

    response = await anonymous_client.post("/api/v1/runner/claim", headers=headers)
    assert response.status_code == 401, response.text


async def test_claim_needs_no_sentinel_headers(anonymous_client, app):
    """The exclude_paths proof: the request carries only the runner token."""
    _, headers = await _register_runner(app, ["default"])
    response = await anonymous_client.post("/api/v1/runner/claim", headers=headers)
    assert response.status_code == 204, response.text


# --------------------------------------------------------------------------
# POST /claim
# --------------------------------------------------------------------------


async def test_claim_against_an_empty_queue_is_204(anonymous_client, app):
    _, headers = await _register_runner(app, ["default"])
    response = await anonymous_client.post("/api/v1/runner/claim", headers=headers)
    assert response.status_code == 204, response.text
    assert response.content == b""


async def test_claim_a_pending_default_lane_run(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id, lane="default")
    _, headers = await _register_runner(app, ["default"])

    body = await _claim(anonymous_client, headers)
    assert body["run"]["id"] == str(run.id)
    assert body["deadline_seconds"] == Settings().worker_job_timeout


async def test_claim_ignores_a_run_in_a_lane_the_runner_does_not_serve(
    anonymous_client, app, workspace_id
):
    """Lanes are registered on the runner, not something a claim can pick."""
    await _seed_run(app, workspace_id, lane="gpu")
    _, headers = await _register_runner(app, ["default"])

    response = await anonymous_client.post("/api/v1/runner/claim", headers=headers)
    assert response.status_code == 204, response.text


# --------------------------------------------------------------------------
# GET /runs/{id}
# --------------------------------------------------------------------------


async def test_get_run_by_the_claimant_returns_the_full_envelope(
    anonymous_client, app, workspace_id
):
    run = await _seed_run(app, workspace_id, params={"foo": "bar"})
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    response = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == str(run.id)
    assert body["params"] == {"foo": "bar"}


async def test_get_run_by_a_non_claimant_runner_is_403(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id)
    _, claimant_headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, claimant_headers)

    _, other_headers = await _register_runner(app, ["default"])
    response = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=other_headers)
    assert response.status_code == 403, response.text


async def test_get_an_unclaimed_unknown_run_is_404_not_403(anonymous_client, app):
    """`claimed_run` must check `get_by_id` before `verify_claim` -- an id
    nobody ever claimed is 404, never a 403 that would confirm it exists."""
    _, headers = await _register_runner(app, ["default"])
    response = await anonymous_client.get(f"/api/v1/runner/runs/{uuid.uuid4()}", headers=headers)
    assert response.status_code == 404, response.text


# --------------------------------------------------------------------------
# POST /runs/{id} -- status updates, lease fencing, the never-cancel rule.
# --------------------------------------------------------------------------


async def test_update_running_then_ready_increments_the_version(
    anonymous_client, app, workspace_id
):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)
    version = claimed["run"]["version"]

    running = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={**_UPDATE_BODY, "status": "running", "progress": 0.5, "expected_version": version},
    )
    assert running.status_code == 200, running.text
    assert running.json()["version"] == version + 1

    ready = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            **_UPDATE_BODY,
            "status": "ready",
            "progress": 1.0,
            "result_uri": "file:///nowhere/result.json",
            "expected_version": version + 1,
        },
    )
    assert ready.status_code == 200, ready.text
    assert ready.json()["version"] == version + 2


async def test_update_to_cancelled_is_422(anonymous_client, app, workspace_id):
    """Runners never cancel -- that is a client-initiated stop signal only,
    via `POST /api/v1/runs/{id}/cancel`."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    version = claimed["run"]["version"]
    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={**_UPDATE_BODY, "status": "cancelled", "expected_version": version},
    )
    assert response.status_code == 422, response.text


async def test_update_with_a_garbage_status_is_422(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={**_UPDATE_BODY, "status": "sideways", "expected_version": claimed["run"]["version"]},
    )
    assert response.status_code == 422, response.text


async def test_update_with_a_stale_expected_version_is_409(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={**_UPDATE_BODY, "expected_version": claimed["run"]["version"] + 5},
    )
    assert response.status_code == 409, response.text


async def test_update_on_a_run_another_runner_claimed_is_403(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id)
    _, claimant_headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, claimant_headers)

    _, other_headers = await _register_runner(app, ["default"])
    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=other_headers,
        json={**_UPDATE_BODY, "expected_version": claimed["run"]["version"]},
    )
    assert response.status_code == 403, response.text


# --------------------------------------------------------------------------
# Blobs -- the workspace-prefix guard, upload size cap, and require_active.
# --------------------------------------------------------------------------


async def test_blob_get_outside_the_workspace_prefix_is_403(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    key = f"{uuid.uuid4()}/escaped.txt"
    response = await anonymous_client.get(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers
    )
    assert response.status_code == 403, response.text


async def test_blob_put_then_get_round_trips_inside_the_prefix(
    anonymous_client, app, workspace_id
):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    key = f"{workspace_id}/runs/{run.id}/log.txt"
    put = await anonymous_client.put(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"hello runner"
    )
    assert put.status_code == 200, put.text
    assert put.json()["uri"]

    get = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers)
    assert get.status_code == 200, get.text
    assert get.content == b"hello runner"


async def test_blob_get_of_a_missing_key_is_404(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    key = f"{workspace_id}/runs/{run.id}/never-written.txt"
    response = await anonymous_client.get(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers
    )
    assert response.status_code == 404, response.text


async def test_blob_put_after_the_run_is_ready_is_403(anonymous_client, app, workspace_id):
    """The write dependency's `require_active=True` -- a terminal run's
    satellites cannot be mutated any more."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    ready = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            **_UPDATE_BODY,
            "status": "ready",
            "progress": 1.0,
            "result_uri": "file:///nowhere/result.json",
            "expected_version": claimed["run"]["version"],
        },
    )
    assert ready.status_code == 200, ready.text

    key = f"{workspace_id}/runs/{run.id}/late.txt"
    put = await anonymous_client.put(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"too late"
    )
    assert put.status_code == 403, put.text


@pytest.fixture
def small_cap_app(tmp_path, session_factory):
    """A second app instance, identical to the `app` fixture's, except
    `runner_upload_max_bytes` is small enough to exercise the 413 path
    without uploading a real gigabyte in a test."""
    application = create_app()
    container = Container(
        create_container(
            Settings(
                blob_base_url=f"file://{tmp_path}",
                inline_jobs=True,
                runner_upload_max_bytes=64,
            )
        )
    )
    container.define(async_sessionmaker, Singleton(lambda: session_factory))
    application.state.container = container
    return application


async def test_blob_put_over_the_cap_is_413(small_cap_app, workspace_id):
    run = await _seed_run(small_cap_app, workspace_id)
    _, headers = await _register_runner(small_cap_app, ["default"])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=small_cap_app), base_url="http://testserver"
    ) as anon:
        claim = await anon.post("/api/v1/runner/claim", headers=headers)
        assert claim.status_code == 200, claim.text

        key = f"{workspace_id}/runs/{run.id}/big.bin"
        response = await anon.put(
            f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"x" * 100
        )
        assert response.status_code == 413, response.text


# --------------------------------------------------------------------------
# GET /runs/{id}/dataset
# --------------------------------------------------------------------------


async def test_get_dataset_matches_the_dataset_created_via_the_normal_api(
    anonymous_client, app, workspace_id, client, csv_upload
):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    created = await client.post(
        "/api/v1/datasets",
        json={
            "name": "solubility",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "target": {"column": "y", "kind": "numeric"},
            "split": {"strategy": "random", "seed": 1},
        },
    )
    assert created.status_code == 201, created.text
    dataset_id = created.json()["id"]

    run = await _seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": dataset_id}
    )
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    response = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}/dataset", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == dataset_id


async def test_get_dataset_without_a_dataset_id_is_404(anonymous_client, app, workspace_id):
    """A prediction-style run has no `dataset_id` in its params."""
    run = await _seed_run(app, workspace_id, kind=RunKind.PREDICTION, params={})
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    response = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}/dataset", headers=headers)
    assert response.status_code == 404, response.text


# --------------------------------------------------------------------------
# GET/POST /runs/{id}/protocol
# --------------------------------------------------------------------------


def _protocol_body(*, workspace_id: uuid.UUID) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "workspace_id": str(workspace_id),
        "name": "runner-reported-protocol",
        "dataset_id": str(uuid.uuid4()),
        "engine_id": "ecfp4-randomforest",
        "artifact_uri": "file:///nowhere/model.joblib",
        "readouts": [],
        "conditions": {},
        "status": "draft",
        "published_at": None,
        "parent_protocol_id": None,
        "protocol_version": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "version": 1,
    }


async def test_get_protocol_is_404_until_the_run_links_one(anonymous_client, app, workspace_id):
    run = await _seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    missing = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}/protocol", headers=headers)
    assert missing.status_code == 404, missing.text

    body = _protocol_body(workspace_id=workspace_id)
    posted = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/protocol", headers=headers, json=body
    )
    assert posted.status_code == 201, posted.text

    linked = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            **_UPDATE_BODY,
            "protocol_id": body["id"],
            "expected_version": claimed["run"]["version"],
        },
    )
    assert linked.status_code == 200, linked.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}/protocol", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["id"] == body["id"]


async def test_post_protocol_with_a_mismatched_workspace_is_422(
    anonymous_client, app, workspace_id
):
    run = await _seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    body = _protocol_body(workspace_id=uuid.uuid4())
    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/protocol", headers=headers, json=body
    )
    assert response.status_code == 422, response.text

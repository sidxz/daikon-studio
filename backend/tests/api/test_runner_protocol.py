"""API coverage for the machine-facing runner protocol, `/api/v1/runner/*`.

This module is the security-boundary proof for the whole self-hosted-runner
feature, so it deliberately does NOT use the `client` fixture: that fixture
carries a Duar token pair, and every route under test here must work with
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
from tests.helpers.runner_fixtures import claim as _claim
from tests.helpers.runner_fixtures import register_runner as _register_runner
from tests.helpers.runner_fixtures import seed_run as _seed_run

from daikonstudio.application.engines.checkpoints import checkpoint_root
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.domain.catalog.protocol import ProtocolStatus
from daikonstudio.domain.execution.run import RunKind
from daikonstudio.infrastructure.di.container import create_container
from daikonstudio.interface.app import create_app
from daikonstudio.settings import Settings

_HEADLINE_METRICS = {
    "targets": [{"column": "y", "primary_metric": "mcc", "value": 0.6, "baseline_value": 0.5}]
}
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


# --------------------------------------------------------------------------
# Auth: no Duar headers required, and only a live non-revoked token works.
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


async def test_claim_needs_no_duar_headers(anonymous_client, app):
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
    assert body["lease_seconds"] == Settings().runner_lease_seconds


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


async def test_update_omitting_progress_and_phase_preserves_them(
    anonymous_client, app, workspace_id
):
    """Security review, Important 1 -- an update that doesn't echo the
    current progress/phase must not null them out."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    first = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "progress": 0.9,
            "phase": "fit",
            "expected_version": claimed["run"]["version"],
        },
    )
    assert first.status_code == 200, first.text

    second = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={"status": "running", "expected_version": first.json()["version"]},
    )
    assert second.status_code == 200, second.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    body = fetched.json()
    assert body["progress"] == 0.9
    assert body["phase"] == "fit"


async def test_update_advances_updated_at(anonymous_client, app, workspace_id):
    """Security review, Minor (a)."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)
    before = claimed["run"]["updated_at"]

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={"status": "running", "expected_version": claimed["run"]["version"]},
    )
    assert response.status_code == 200, response.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert fetched.json()["updated_at"] > before


async def test_update_omitting_protocol_id_leaves_it_intact(anonymous_client, app, workspace_id):
    """Security review, Important 1."""
    run = await _seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    protocol_id = uuid.uuid4()
    linked = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "protocol_id": str(protocol_id),
            "expected_version": claimed["run"]["version"],
        },
    )
    assert linked.status_code == 200, linked.text

    again = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={"status": "running", "expected_version": linked.json()["version"]},
    )
    assert again.status_code == 200, again.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert fetched.json()["protocol_id"] == str(protocol_id)


async def test_update_cannot_repoint_an_already_linked_protocol_id(
    anonymous_client, app, workspace_id
):
    """Security review, Important 1 -- `Run.link_protocol` is write-once;
    the update route must route through it, not assign the field directly."""
    run = await _seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    first_id = uuid.uuid4()
    linked = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "protocol_id": str(first_id),
            "expected_version": claimed["run"]["version"],
        },
    )
    assert linked.status_code == 200, linked.text

    second_id = uuid.uuid4()
    conflict = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "protocol_id": str(second_id),
            "expected_version": linked.json()["version"],
        },
    )
    assert conflict.status_code == 409, conflict.text

    unlink = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "protocol_id": None,
            "expected_version": linked.json()["version"],
        },
    )
    assert unlink.status_code == 409, unlink.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert fetched.json()["protocol_id"] == str(first_id)


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


async def test_blob_put_path_traversal_is_403_and_writes_nothing_outside_the_blob_root(
    anonymous_client, app, workspace_id, tmp_path
):
    """Security review, Critical 1 -- percent-encoded `..` segments, decoded
    by the ASGI layer before routing, must not let a PUT escape the blob
    root even though the raw string still starts with `f"{workspace_id}/"`."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    url = f"/api/v1/runner/runs/{run.id}/blobs/{workspace_id}/%2E%2E/%2E%2E/pwned.txt"
    response = await anonymous_client.put(url, headers=headers, content=b"pwned")
    assert response.status_code == 403, response.text
    assert not (tmp_path.parent / "pwned.txt").exists()
    assert not any(tmp_path.rglob("pwned.txt"))


async def test_blob_get_path_traversal_is_403(anonymous_client, app, workspace_id, tmp_path):
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    # Also proves a traversal read of another workspace's blob root is blocked,
    # not just a write outside the blob root entirely.
    url = f"/api/v1/runner/runs/{run.id}/blobs/{workspace_id}/%2E%2E/%2E%2E/etc/passwd"
    response = await anonymous_client.get(url, headers=headers)
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


async def test_blob_get_accepts_a_full_store_uri_key_for_its_own_workspace(
    anonymous_client, app, workspace_id
):
    """Final review, Critical 1 -- `FsspecBlobStore.put_bytes` returns the
    FULL STORE URI, not the bare key, and `RunPrediction` deliberately reads
    an artifact back BY that URI (`predict_with_protocol.py`). The guard must
    accept that form for the run's own workspace, not just a bare key."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    key = f"{workspace_id}/runs/{run.id}/artifact.bin"
    put = await anonymous_client.put(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"model bytes"
    )
    assert put.status_code == 200, put.text
    full_uri = put.json()["uri"]
    assert full_uri != key  # proves this really is the full store URI, not the bare key

    by_bare_key = await anonymous_client.get(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers
    )
    assert by_bare_key.status_code == 200, by_bare_key.text

    by_full_uri = await anonymous_client.get(
        f"/api/v1/runner/runs/{run.id}/blobs/{full_uri}", headers=headers
    )
    assert by_full_uri.status_code == 200, by_full_uri.text
    assert by_full_uri.content == by_bare_key.content == b"model bytes"


async def test_blob_get_with_a_full_uri_key_for_another_workspace_is_403(
    anonymous_client, app, workspace_id
):
    """Final review, Critical 1 -- normalizing the URI form must not weaken
    the workspace check: a full URI pointing at a DIFFERENT workspace is
    still rejected."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    settings = app.state.container[Settings]
    other_workspace = uuid.uuid4()
    full_uri = (
        f"{settings.blob_base_url.rstrip('/')}/{other_workspace}/protocols/x/artifact/model.joblib"
    )
    response = await anonymous_client.get(
        f"/api/v1/runner/runs/{run.id}/blobs/{full_uri}", headers=headers
    )
    assert response.status_code == 403, response.text


async def test_blob_get_with_a_full_uri_key_containing_traversal_is_403(
    anonymous_client, app, workspace_id
):
    """Final review, Critical 1 -- normalizing the URI form must not weaken
    the traversal check either: a full URI whose remainder contains `..` is
    still rejected."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    settings = app.state.container[Settings]
    full_uri = f"{settings.blob_base_url.rstrip('/')}/{workspace_id}/../../etc/passwd"
    response = await anonymous_client.get(
        f"/api/v1/runner/runs/{run.id}/blobs/{full_uri}", headers=headers
    )
    assert response.status_code == 403, response.text


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


async def test_blob_put_chunked_over_the_cap_is_413(small_cap_app, workspace_id):
    """Security review, Important 3 -- a chunked request carries no
    Content-Length at all, so the fast-fail header check can never catch it;
    only the streaming accumulator can."""
    run = await _seed_run(small_cap_app, workspace_id)
    _, headers = await _register_runner(small_cap_app, ["default"])

    async def _chunks():
        for _ in range(5):
            yield b"x" * 30  # 150 bytes total, over the 64-byte test cap

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=small_cap_app), base_url="http://testserver"
    ) as anon:
        claim = await anon.post("/api/v1/runner/claim", headers=headers)
        assert claim.status_code == 200, claim.text

        key = f"{workspace_id}/runs/{run.id}/chunked.bin"
        response = await anon.put(
            f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=_chunks()
        )
        assert response.status_code == 413, response.text


async def test_blob_put_with_a_negative_content_length_does_not_500(
    anonymous_client, app, workspace_id
):
    """Security review, Important 3 -- a lying/garbage Content-Length must be
    treated as absent, not fed straight to `int()`."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    key = f"{workspace_id}/runs/{run.id}/neg.bin"
    request = anonymous_client.build_request(
        "PUT", f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"hello"
    )
    request.headers["content-length"] = "-1"
    response = await anonymous_client.send(request)
    assert response.status_code != 500, response.text
    assert response.status_code == 200, response.text


async def test_delete_checkpoints_removes_only_this_runs_saved_progress(
    anonymous_client, app, workspace_id
):
    dataset_id = uuid.uuid4()
    run = await _seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": str(dataset_id)}
    )
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)
    store = app.state.container[BlobStore]
    saved = f"{checkpoint_root(workspace_id, dataset_id, run.id)}model/result.json"
    snapshot = f"{workspace_id}/datasets/{dataset_id}/snapshot.parquet"
    store.put_bytes(saved, b"progress")
    store.put_bytes(snapshot, b"keep")

    response = await anonymous_client.delete(
        f"/api/v1/runner/runs/{run.id}/checkpoints", headers=headers
    )

    assert response.status_code == 204, response.text
    assert not store.exists(saved)
    assert store.exists(snapshot)


async def test_delete_checkpoints_without_the_claim_is_403(anonymous_client, app, workspace_id):
    dataset_id = uuid.uuid4()
    run = await _seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": str(dataset_id)}
    )
    _, claimant_headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, claimant_headers)
    store = app.state.container[BlobStore]
    saved = f"{checkpoint_root(workspace_id, dataset_id, run.id)}model/result.json"
    store.put_bytes(saved, b"progress")

    _, other_headers = await _register_runner(app, ["default"])
    response = await anonymous_client.delete(
        f"/api/v1/runner/runs/{run.id}/checkpoints", headers=other_headers
    )

    assert response.status_code == 403, response.text
    assert store.exists(saved)


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
            "targets": [{"column": "y", "kind": "numeric"}],
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


async def test_get_dataset_with_a_non_uuid_dataset_id_is_404_not_500(
    anonymous_client, app, workspace_id
):
    """Security review, Minor (b)."""
    run = await _seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": "not-a-uuid"}
    )
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    response = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}/dataset", headers=headers)
    assert response.status_code == 404, response.text


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


async def test_post_protocol_forces_draft_status_regardless_of_the_body(
    anonymous_client, app, workspace_id
):
    """Security review, Important 2 -- a runner reporting the protocol it
    just trained must never be able to hand back an already-published
    envelope and skip the editor-role `PublishProtocol` gate."""
    run = await _seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    body = _protocol_body(workspace_id=workspace_id)
    body["status"] = "published"
    body["published_at"] = "2026-01-01T00:00:00Z"
    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/protocol", headers=headers, json=body
    )
    assert response.status_code == 201, response.text

    stored = await app.state.container[ProtocolRepository].get(workspace_id, uuid.UUID(body["id"]))
    assert stored is not None
    assert stored.status is ProtocolStatus.DRAFT
    assert stored.published_at is None


async def test_post_protocol_with_a_duplicate_id_is_409_not_500(
    anonymous_client, app, workspace_id
):
    """Security review, Minor (c)."""
    run = await _seed_run(app, workspace_id, kind=RunKind.TRAINING)
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    body = _protocol_body(workspace_id=workspace_id)
    first = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/protocol", headers=headers, json=body
    )
    assert first.status_code == 201, first.text

    second = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}/protocol", headers=headers, json=body
    )
    assert second.status_code == 409, second.text


def test_claim_response_model_reaches_the_openapi_schema(app):
    """Security review, Minor (d)."""
    schema = app.openapi()["paths"]["/api/v1/runner/claim"]["post"]["responses"]["200"]
    assert "ClaimResponse" in schema["content"]["application/json"]["schema"]["$ref"]


async def test_runners_management_api_stays_duar_protected(anonymous_client):
    """Security review, Minor (e) -- nobody can widen `exclude_paths` to
    also swallow the human-facing `/api/v1/runners` unnoticed."""
    assert (await anonymous_client.get("/api/v1/runners")).status_code == 401
    body = {"name": "x", "lanes": ["default"]}
    assert (await anonymous_client.post("/api/v1/runners", json=body)).status_code == 401


# --------------------------------------------------------------------------
# POST /runs/{id} -- metrics
# --------------------------------------------------------------------------


async def test_update_run_applies_metrics(anonymous_client, app, workspace_id):
    """A runner's terminal update carries the headline metric.

    Without this the column is populated only under InlineEnqueuer -- green in
    tests, NULL in production.
    """
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "expected_version": claimed["run"]["version"],
            "metrics": _HEADLINE_METRICS,
        },
    )
    assert response.status_code == 200, response.text


async def test_update_run_without_metrics_does_not_clear_them(anonymous_client, app, workspace_id):
    """A bare progress update must not null out a metric a previous update set
    -- the same failure the security review caught for progress and phase."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    first = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "expected_version": claimed["run"]["version"],
            "metrics": _HEADLINE_METRICS,
        },
    )
    assert first.status_code == 200, first.text

    second = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={"status": "running", "expected_version": first.json()["version"], "progress": 0.5},
    )
    assert second.status_code == 200, second.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert fetched.json()["metrics"] == _HEADLINE_METRICS


async def test_update_run_rejects_a_malformed_metrics_payload(anonymous_client, app, workspace_id):
    """`metrics` is `RunMetricsWire`, not a free `dict[str, Any]` -- an
    unknown key, or a missing required one, is a 422 at the edge rather than
    arbitrary JSON landing verbatim in the `runs.metrics` column."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)

    response = await anonymous_client.post(
        f"/api/v1/runner/runs/{run.id}",
        headers=headers,
        json={
            "status": "running",
            "expected_version": claimed["run"]["version"],
            "metrics": {
                "targets": [{"column": "y", "primary_metric": "mcc", "not_a_real_field": 1}]
            },
        },
    )
    assert response.status_code == 422, response.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert fetched.json()["metrics"] is None


async def test_update_run_rejects_an_unbounded_metrics_payload(
    anonymous_client, app, workspace_id
):
    """The shape is closed, and so is its size: a runner cannot fill `runs.metrics`
    with an arbitrarily long name or an arbitrarily long list."""
    run = await _seed_run(app, workspace_id)
    _, headers = await _register_runner(app, ["default"])
    claimed = await _claim(anonymous_client, headers)
    headline = {"column": "y", "primary_metric": "rmse", "value": 0.5, "baseline_value": 0.7}

    for targets in (
        [{**headline, "column": "c" * 1025}],
        [{**headline, "primary_metric": "m" * 65}],
        [headline] * 4097,
    ):
        response = await anonymous_client.post(
            f"/api/v1/runner/runs/{run.id}",
            headers=headers,
            json={
                "status": "running",
                "expected_version": claimed["run"]["version"],
                "metrics": {"targets": targets},
            },
        )
        assert response.status_code == 422, response.text

    fetched = await anonymous_client.get(f"/api/v1/runner/runs/{run.id}", headers=headers)
    assert fetched.json()["metrics"] is None


async def test_a_runner_may_write_its_own_runs_saved_progress(anonymous_client, app, workspace_id):
    dataset_id = uuid.uuid4()
    run = await _seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": str(dataset_id)}
    )
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    key = f"{checkpoint_root(workspace_id, dataset_id, run.id)}model/lightning/training-state.a"
    response = await anonymous_client.put(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"state"
    )

    assert response.status_code == 200, response.text


async def test_a_runner_may_not_write_another_runs_saved_progress(
    anonymous_client, app, workspace_id
):
    """Saved training state is unpickled when its run resumes: workspace confinement
    alone would let the runner of one run plant state another run then loads."""
    dataset_id = uuid.uuid4()
    run = await _seed_run(
        app, workspace_id, kind=RunKind.TRAINING, params={"dataset_id": str(dataset_id)}
    )
    _, headers = await _register_runner(app, ["default"])
    await _claim(anonymous_client, headers)

    other_run = uuid.uuid4()
    key = f"{checkpoint_root(workspace_id, dataset_id, other_run)}model/lightning/training-state.a"
    response = await anonymous_client.put(
        f"/api/v1/runner/runs/{run.id}/blobs/{key}", headers=headers, content=b"payload"
    )

    assert response.status_code == 403, response.text
    assert not app.state.container[BlobStore].exists(key)

"""Building a dataset in the background: start, poll, and read how it ended -- the
wizard's path, so a large file shows progress instead of freezing the API."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

from tests.api.test_datasets import SOLUBILITY_CSV, create_body

from daikonstudio.application.data import build_dataset
from daikonstudio.application.ports.dataset_build_repository import DatasetBuildRepository
from daikonstudio.domain.data.dataset_build import DatasetBuild


async def _finish(client, build_id: str) -> dict:
    """Await the build's task, then read it once. Not polled: the test database is a
    single connection inside a per-test transaction, and a poll running beside the
    build's own writes would share it concurrently (production pools connections)."""
    await asyncio.gather(*build_dataset._running)
    return (await client.get(f"/api/v1/datasets/builds/{build_id}")).json()


async def test_a_build_ends_in_the_dataset_it_built(client, csv_upload):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    started = await client.post("/api/v1/datasets/builds", json=create_body(upload_ref))
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "running"

    build = await _finish(client, started.json()["id"])

    assert build["status"] == "succeeded", build
    assert build["error"] is None
    dataset = await client.get(f"/api/v1/datasets/{build['dataset_id']}")
    assert dataset.status_code == 200
    assert dataset.json()["row_count"] == 4


async def test_a_rejected_build_carries_the_same_body_as_a_rejected_request(client, csv_upload):
    """The wizard renders a failed build exactly like a 422: the validation report
    is the error's `detail`."""
    upload_ref = await csv_upload(b"smiles,y\nnope,1.0\nalso-nope,2.0\n")
    started = await client.post("/api/v1/datasets/builds", json=create_body(upload_ref))

    build = await _finish(client, started.json()["id"])

    assert build["status"] == "failed"
    assert build["dataset_id"] is None
    assert build["error"]["error"] == "InvalidDatasetError"
    assert build["error"]["detail"]["total_rows"] == 2
    assert build["error"]["detail"]["valid_rows"] == 0


async def test_a_build_is_scoped_to_the_callers_workspace(
    client, other_workspace_client, csv_upload
):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    started = await client.post("/api/v1/datasets/builds", json=create_body(upload_ref))
    build_id = started.json()["id"]
    await _finish(client, build_id)

    response = await other_workspace_client.get(f"/api/v1/datasets/builds/{build_id}")

    assert response.status_code == 404


async def test_a_build_silent_for_a_minute_reads_as_interrupted(app, client, workspace_id):
    """The process running it died (a restart): the row would say `running` forever."""
    stale = DatasetBuild(
        workspace_id=workspace_id,
        created_by=uuid.uuid4(),
        name="lost",
        updated_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    await app.state.container[DatasetBuildRepository].add(stale)

    build = (await client.get(f"/api/v1/datasets/builds/{stale.id}")).json()

    assert build["status"] == "failed"
    assert build["error"]["error"] == "BuildInterrupted"


async def test_a_viewer_cannot_start_a_build(viewer_client, csv_upload):
    upload_ref = await csv_upload(SOLUBILITY_CSV)
    response = await viewer_client.post("/api/v1/datasets/builds", json=create_body(upload_ref))
    assert response.status_code == 403

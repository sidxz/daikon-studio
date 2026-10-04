"""Deleting datasets and draft protocols: who may, and what goes with them."""

import asyncio
import threading
import uuid

from sqlalchemy import text
from tests.api import test_protocols

from daikonstudio.application.data import get_dataset_profile
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)

# A 20-compound dataset with a random split, and a training request, shared
# with the protocol suite.
_create_dataset = test_protocols._create_dataset
_train = test_protocols._train


async def _train_protocol(client, dataset_id: str) -> tuple[str, str]:
    """Train inline (tests use the in-process enqueuer); return (run_id, protocol_id)."""
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    run = (await client.get(f"/api/v1/runs/{run_id}")).json()
    assert run["status"] == "ready", run
    return run_id, run["protocol_id"]


async def test_only_the_creator_and_admins_may_delete_a_dataset(
    client, other_editor_client, admin_client, viewer_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    url = f"/api/v1/datasets/{dataset_id}"

    assert (await client.get(url)).json()["can_delete"] is True
    assert (await admin_client.get(url)).json()["can_delete"] is True
    assert (await other_editor_client.get(url)).json()["can_delete"] is False
    assert (await viewer_client.get(url)).json()["can_delete"] is False


async def test_a_draft_is_deletable_by_its_creator_until_it_is_published(
    client, other_editor_client, admin_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)
    url = f"/api/v1/protocols/{protocol_id}"

    assert (await client.get(url)).json()["can_delete"] is True
    assert (await admin_client.get(url)).json()["can_delete"] is True
    assert (await other_editor_client.get(url)).json()["can_delete"] is False

    assert (await client.post(f"{url}/publish")).status_code == 204
    assert (await client.get(url)).json()["can_delete"] is False
    assert (await admin_client.get(url)).json()["can_delete"] is False


async def test_the_creator_deletes_a_draft_with_its_training_run_and_files(
    client, csv_upload, workspace_id, tmp_path
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _train_protocol(client, dataset_id)
    folder = tmp_path / str(workspace_id) / "protocols" / protocol_id
    assert folder.exists()

    response = await client.delete(f"/api/v1/protocols/{protocol_id}")

    assert response.status_code == 204, response.text
    assert (await client.get(f"/api/v1/protocols/{protocol_id}")).status_code == 404
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404
    assert not folder.exists()
    # A second delete finds nothing: 404, not 500.
    assert (await client.delete(f"/api/v1/protocols/{protocol_id}")).status_code == 404


async def test_a_published_protocol_cannot_be_deleted(client, admin_client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)
    assert (await client.post(f"/api/v1/protocols/{protocol_id}/publish")).status_code == 204

    response = await admin_client.delete(f"/api/v1/protocols/{protocol_id}")

    assert response.status_code == 409
    assert response.json()["message"] == "Published protocols cannot be deleted."


async def test_only_the_creator_or_an_admin_may_delete_a_draft(
    client, other_editor_client, viewer_client, other_workspace_client, admin_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)
    url = f"/api/v1/protocols/{protocol_id}"

    refused = await other_editor_client.delete(url)
    assert refused.status_code == 403
    assert refused.json()["message"] == (
        "Only an admin or the person who created it can delete this."
    )
    assert (await viewer_client.delete(url)).status_code == 403
    assert (await other_workspace_client.delete(url)).status_code == 404
    assert (await admin_client.delete(url)).status_code == 204


async def test_a_draft_whose_training_run_is_still_finishing_is_refused(
    client, csv_upload, session_factory
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id, protocol_id = await _train_protocol(client, dataset_id)
    # The protocol row exists before the run's last phase ("Mapping chemical space") ends.
    async with session_factory() as session:
        await session.execute(
            text("UPDATE runs SET status = 'running' WHERE id = :id"), {"id": run_id}
        )
        await session.commit()

    response = await client.delete(f"/api/v1/protocols/{protocol_id}")

    assert response.status_code == 409
    assert "still finishing" in response.json()["message"]


async def _add_training_run(session_factory, workspace_id, dataset_id: str, status) -> str:
    """A training run on the dataset, in a state the inline test enqueuer never leaves one."""
    run = Run(
        kind=RunKind.TRAINING,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="test",
        params={"dataset_id": dataset_id},
        status=status,
    )
    await SqlAlchemyRunRepository(session_factory).add(run)
    return str(run.id)


async def test_the_creator_deletes_an_unused_dataset_and_can_upload_it_again(
    client, csv_upload, workspace_id, tmp_path
):
    dataset_id = await _create_dataset(client, csv_upload)
    folder = tmp_path / str(workspace_id) / "datasets" / dataset_id
    assert folder.exists()

    response = await client.delete(f"/api/v1/datasets/{dataset_id}")

    assert response.status_code == 204, response.text
    assert (await client.get(f"/api/v1/datasets/{dataset_id}")).status_code == 404
    assert not folder.exists()
    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 404
    # The content-hash unique index no longer holds the old row.
    assert await _create_dataset(client, csv_upload) != dataset_id


async def test_a_dataset_with_a_protocol_is_refused_until_the_draft_is_deleted(client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)

    refused = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert refused.status_code == 409
    assert refused.json()["message"] == (
        "Protocols trained on this dataset must be deleted first. "
        "A dataset used by a published protocol cannot be deleted."
    )
    listing = await client.get("/api/v1/protocols", params={"dataset_id": dataset_id})
    assert [item["id"] for item in listing.json()["items"]] == [protocol_id]

    assert (await client.delete(f"/api/v1/protocols/{protocol_id}")).status_code == 204
    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204


async def test_protocols_can_be_listed_by_the_dataset_they_were_trained_on(client, csv_upload):
    dataset_id = await _create_dataset(client, csv_upload)
    _, protocol_id = await _train_protocol(client, dataset_id)

    mine = await client.get("/api/v1/protocols", params={"dataset_id": dataset_id})
    other = await client.get("/api/v1/protocols", params={"dataset_id": str(uuid.uuid4())})

    assert [item["id"] for item in mine.json()["items"]] == [protocol_id]
    # Without the filter this would list every protocol in the workspace.
    assert other.json()["items"] == []


async def test_a_dataset_with_a_training_run_in_progress_is_refused(
    client, csv_upload, session_factory, workspace_id
):
    dataset_id = await _create_dataset(client, csv_upload)
    await _add_training_run(session_factory, workspace_id, dataset_id, RunStatus.PENDING)

    response = await client.delete(f"/api/v1/datasets/{dataset_id}")

    assert response.status_code == 409
    assert response.json()["message"] == "A training run on this dataset is still in progress."


async def test_failed_training_runs_go_with_their_dataset(
    client, csv_upload, session_factory, workspace_id
):
    dataset_id = await _create_dataset(client, csv_upload)
    run_id = await _add_training_run(session_factory, workspace_id, dataset_id, RunStatus.FAILED)

    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204
    assert (await client.get(f"/api/v1/runs/{run_id}")).status_code == 404


async def test_only_the_creator_or_an_admin_may_delete_a_dataset(
    client, other_editor_client, viewer_client, other_workspace_client, admin_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    url = f"/api/v1/datasets/{dataset_id}"

    assert (await other_editor_client.delete(url)).status_code == 403
    assert (await viewer_client.delete(url)).status_code == 403
    assert (await other_workspace_client.delete(url)).status_code == 404
    assert (await admin_client.delete(url)).status_code == 204


async def test_a_dataset_deleted_while_its_profile_computes_stays_deleted(
    client, csv_upload, monkeypatch, workspace_id, tmp_path
):
    release = threading.Event()
    real = get_dataset_profile.build_profile

    def slow_build(**kwargs):
        release.wait(timeout=10)
        return real(**kwargs)

    monkeypatch.setattr(get_dataset_profile, "build_profile", slow_build)
    dataset_id = await _create_dataset(client, csv_upload)
    assert (await client.get(f"/api/v1/datasets/{dataset_id}/profile")).status_code == 202

    assert (await client.delete(f"/api/v1/datasets/{dataset_id}")).status_code == 204
    release.set()
    for _ in range(200):
        if not get_dataset_profile._RUNNING:
            break
        await asyncio.sleep(0.05)

    assert not (tmp_path / str(workspace_id) / "datasets" / dataset_id).exists()


async def test_the_raw_upload_is_removed_once_its_dataset_exists(
    client, csv_upload, workspace_id, tmp_path
):
    """Otherwise deleting a dataset would leave its raw CSV, usually the largest
    file it ever had, in storage: nothing records which upload made which dataset."""
    upload_ref = await csv_upload(test_protocols._csv())
    upload = tmp_path / str(workspace_id) / "uploads" / f"{upload_ref}.csv"
    body = {
        "name": "solubility",
        "upload_ref": upload_ref,
        "structure_column": "smiles",
        "targets": [{"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"}],
        "split": {"strategy": "random", "seed": 1},
    }

    # A failed create keeps the upload, so it can be retried against the same bytes.
    failed = await client.post("/api/v1/datasets", json={**body, "structure_column": "nope"})
    assert failed.status_code >= 400
    assert upload.exists()

    created = await client.post("/api/v1/datasets", json=body)
    assert created.status_code == 201, created.text
    assert not upload.exists()

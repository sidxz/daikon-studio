"""Deleting datasets and draft protocols: who may, and what goes with them."""

from sqlalchemy import text
from tests.api import test_protocols

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

"""Deleting datasets and draft protocols: who may, and what goes with them."""

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

"""Protocols are registered with Duar when training creates them and forgotten on delete."""

from __future__ import annotations

from tests.api.test_protocols import _create_dataset, _train
from tests.api.test_runs import _PREDICTION_CSV, _predict


async def _trained(client, csv_upload) -> str:
    dataset_id = await _create_dataset(client, csv_upload)
    assert (await _train(client, dataset_id)).status_code == 202
    return str((await client.get("/api/v1/protocols")).json()["items"][0]["id"])


async def test_training_registers_the_protocol_private_to_its_creator(
    client, csv_upload, protocol_access, workspace_id, client_user_id
):
    protocol_id = await _trained(client, csv_upload)
    entry = next(v for k, v in protocol_access.acl.items() if str(k) == protocol_id)
    assert entry[0] == workspace_id
    assert entry[2] == "private"
    assert entry[1] == client_user_id


async def test_deleting_a_draft_deregisters_it(client, csv_upload, protocol_access):
    protocol_id = await _trained(client, csv_upload)
    assert len(protocol_access.acl) == 1
    response = await client.delete(f"/api/v1/protocols/{protocol_id}")
    assert response.status_code == 204, response.text
    assert protocol_access.acl == {}


_API = "/api/v1/protocols"


async def test_a_draft_is_read_by_its_creator_and_admins_only(
    client, other_editor_client, admin_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    assert (await client.get(f"{_API}/{protocol_id}")).status_code == 200
    assert (await admin_client.get(f"{_API}/{protocol_id}")).status_code == 200
    assert (await other_editor_client.get(f"{_API}/{protocol_id}")).status_code == 404


async def test_lists_hide_a_draft_from_colleagues(
    client, other_editor_client, admin_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)

    async def ids(http_client, query: str = "") -> list[str]:
        return [item["id"] for item in (await http_client.get(f"{_API}{query}")).json()["items"]]

    assert protocol_id not in await ids(other_editor_client)
    assert protocol_id in await ids(admin_client)
    assert protocol_id in await ids(client)


async def test_a_draft_scorecard_and_map_are_hidden_from_colleagues(
    client, other_editor_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    for path in ("scorecard", "chemical-space", "chemical-space/compounds?indices=0"):
        assert (await other_editor_client.get(f"{_API}/{protocol_id}/{path}")).status_code == 404
        assert (await client.get(f"{_API}/{protocol_id}/{path}")).status_code != 404


async def test_only_the_creator_can_publish_and_publishing_shares_it(
    client, other_editor_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    assert (await other_editor_client.post(f"{_API}/{protocol_id}/publish")).status_code == 404
    assert (await client.post(f"{_API}/{protocol_id}/publish")).status_code == 204
    assert (await other_editor_client.get(f"{_API}/{protocol_id}")).status_code == 200


async def test_publish_stays_a_draft_when_duar_is_down(client, csv_upload, protocol_access):
    protocol_id = await _trained(client, csv_upload)
    protocol_access.down = True
    assert (await client.post(f"{_API}/{protocol_id}/publish")).status_code == 503
    protocol_access.down = False
    assert (await client.get(f"{_API}/{protocol_id}")).json()["status"] == "draft"
    assert (await client.post(f"{_API}/{protocol_id}/publish")).status_code == 204


async def test_a_colleague_cannot_delete_a_draft_they_cannot_see(
    client, other_editor_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    assert (await other_editor_client.delete(f"{_API}/{protocol_id}")).status_code == 404
    assert (await client.get(f"{_API}/{protocol_id}")).status_code == 200


async def test_mine_lists_only_protocols_the_caller_created(
    client, other_editor_client, admin_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    assert (await client.post(f"{_API}/{protocol_id}/publish")).status_code == 204
    mine = (await client.get(f"{_API}?mine=true")).json()["items"]
    assert [item["id"] for item in mine] == [protocol_id]
    assert (await other_editor_client.get(f"{_API}?mine=true")).json()["items"] == []
    # The admin sees it in the full list, but it is not theirs.
    assert (await admin_client.get(f"{_API}?mine=true")).json()["items"] == []


async def test_a_permissions_outage_is_a_503_not_an_empty_list(client, protocol_access):
    protocol_access.down = True
    assert (await client.get(_API)).status_code == 503


async def test_prediction_runs_still_list_while_permissions_are_down(client, protocol_access):
    protocol_access.down = True
    assert (await client.get("/api/v1/runs?kind=prediction")).status_code == 200


async def test_responses_name_their_creator(client, csv_upload, client_user_id):
    dataset_id = await _create_dataset(client, csv_upload)
    assert (await _train(client, dataset_id)).status_code == 202
    protocol_id = str((await client.get(_API)).json()["items"][0]["id"])
    protocol = (await client.get(f"{_API}/{protocol_id}")).json()
    assert protocol["created_by"] == str(client_user_id)
    dataset = (await client.get(f"/api/v1/datasets/{dataset_id}")).json()
    assert dataset["created_by"] == str(client_user_id)


async def _training_run_id(client) -> str:
    runs = (await client.get("/api/v1/runs?kind=training")).json()["items"]
    return str(runs[0]["id"])


async def test_a_drafts_training_run_is_listed_for_its_creator_and_admins_only(
    client, other_editor_client, admin_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    run_id = await _training_run_id(client)

    async def training_ids(http_client) -> list[str]:
        items = (await http_client.get("/api/v1/runs?kind=training")).json()["items"]
        return [item["id"] for item in items]

    assert await training_ids(other_editor_client) == []
    assert run_id in await training_ids(admin_client)
    assert (await client.post(f"{_API}/{protocol_id}/publish")).status_code == 204
    assert run_id in await training_ids(other_editor_client)


async def test_a_drafts_training_run_is_hidden_from_colleagues_one_at_a_time(
    client, other_editor_client, admin_client, csv_upload
):
    await _trained(client, csv_upload)
    run_id = await _training_run_id(client)
    url = f"/api/v1/runs/{run_id}"
    assert (await other_editor_client.get(url)).status_code == 404
    assert (await other_editor_client.get(f"{url}/epochs")).status_code == 404
    assert (await other_editor_client.post(f"{url}/cancel")).status_code == 404
    assert (await other_editor_client.post(f"{url}/retry")).status_code == 404
    assert (await client.get(url)).status_code == 200
    assert (await admin_client.get(url)).status_code == 200
    assert (await admin_client.get(f"{url}/epochs")).status_code == 200


async def test_predictions_on_a_published_protocol_stay_shared(
    client, other_editor_client, csv_upload
):
    protocol_id = await _trained(client, csv_upload)
    assert (await client.post(f"{_API}/{protocol_id}/publish")).status_code == 204
    upload_ref = await csv_upload(_PREDICTION_CSV)
    run_id = (await _predict(client, protocol_id, upload_ref)).json()["id"]

    assert (await other_editor_client.get(f"/api/v1/runs/{run_id}")).status_code == 200
    listed = (await other_editor_client.get("/api/v1/runs?kind=prediction")).json()["items"]
    assert run_id in [item["id"] for item in listed]

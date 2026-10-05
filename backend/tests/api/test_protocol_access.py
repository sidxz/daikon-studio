"""Protocols are registered with Duar when training creates them and forgotten on delete."""

from __future__ import annotations

from tests.api.test_protocols import _create_dataset, _train


async def _trained(client, csv_upload) -> str:
    dataset_id = await _create_dataset(client, csv_upload)
    assert (await _train(client, dataset_id)).status_code == 202
    return str((await client.get("/api/v1/protocols")).json()["items"][0]["id"])


async def test_training_registers_the_protocol_private_to_its_creator(
    client, csv_upload, protocol_access, workspace_id
):
    protocol_id = await _trained(client, csv_upload)
    entry = next(v for k, v in protocol_access.acl.items() if str(k) == protocol_id)
    assert entry[0] == workspace_id
    assert entry[2] == "private"
    created_by = (await client.get(f"/api/v1/protocols/{protocol_id}")).json().get("created_by")
    if created_by is not None:
        assert str(entry[1]) == created_by


async def test_deleting_a_draft_deregisters_it(client, csv_upload, protocol_access):
    protocol_id = await _trained(client, csv_upload)
    assert len(protocol_access.acl) == 1
    response = await client.delete(f"/api/v1/protocols/{protocol_id}")
    assert response.status_code == 204, response.text
    assert protocol_access.acl == {}

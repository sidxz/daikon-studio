"""Shared folders for datasets and protocols."""

from __future__ import annotations

from tests.api.test_protocols import _create_dataset, _train

_F = "/api/v1/folders"
_DUPLICATE = "A folder with this name already exists."


async def _folder(http, kind: str, name: str) -> dict:
    response = await http.post(_F, json={"kind": kind, "name": name})
    assert response.status_code == 201, response.text
    return response.json()


async def _count(http, kind: str, folder_id: str) -> int:
    items = (await http.get(_F, params={"kind": kind})).json()["items"]
    return next(i for i in items if i["id"] == folder_id)["item_count"]


async def test_names_are_unique_per_kind_ignoring_case(client):
    await _folder(client, "dataset", "Gyrase")
    response = await client.post(_F, json={"kind": "dataset", "name": " gyrase "})
    assert response.status_code == 409
    assert response.json()["message"] == _DUPLICATE
    await _folder(client, "protocol", "Gyrase")


async def test_a_bad_name_is_refused(client):
    response = await client.post(_F, json={"kind": "dataset", "name": "  "})
    assert response.status_code == 422
    assert response.json()["message"] == "A folder name must be 1 to 100 characters."


async def test_a_viewer_reads_but_cannot_create(client, viewer_client):
    await _folder(client, "dataset", "Gyrase")
    assert (await viewer_client.post(_F, json={"kind": "dataset", "name": "x"})).status_code == 403
    listing = (await viewer_client.get(_F, params={"kind": "dataset"})).json()
    assert listing["can_edit"] is False
    assert [i["name"] for i in listing["items"]] == ["Gyrase"]


async def test_filing_a_dataset_counts_and_filters(client, csv_upload):
    folder = await _folder(client, "dataset", "Gyrase")
    await _folder(client, "dataset", "Empty")
    dataset_id = await _create_dataset(client, csv_upload)
    response = await client.put(
        f"/api/v1/datasets/{dataset_id}/folder", json={"folder_id": folder["id"]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["folder_id"] == folder["id"]
    assert await _count(client, "dataset", folder["id"]) == 1
    listed = (await client.get("/api/v1/datasets", params={"folder_id": folder["id"]})).json()
    assert [d["id"] for d in listed["items"]] == [dataset_id]

    response = await client.put(f"/api/v1/datasets/{dataset_id}/folder", json={"folder_id": None})
    assert response.json()["folder_id"] is None
    assert await _count(client, "dataset", folder["id"]) == 0


async def test_filing_into_the_wrong_kind_or_a_foreign_folder_is_422(
    client, other_workspace_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    protocol_folder = await _folder(client, "protocol", "P")
    foreign = await _folder(other_workspace_client, "dataset", "Theirs")
    for folder_id in (protocol_folder["id"], foreign["id"]):
        response = await client.put(
            f"/api/v1/datasets/{dataset_id}/folder", json={"folder_id": folder_id}
        )
        assert response.status_code == 422
        assert response.json()["message"] == "Choose a folder for this kind of item."


async def test_rename(client):
    folder = await _folder(client, "dataset", "A")
    await _folder(client, "dataset", "B")
    response = await client.patch(f"{_F}/{folder['id']}", json={"name": " C "})
    assert response.status_code == 200
    assert response.json()["name"] == "C"
    response = await client.patch(f"{_F}/{folder['id']}", json={"name": "b"})
    assert response.status_code == 409
    assert response.json()["message"] == _DUPLICATE


async def test_deleting_a_folder_unfiles_its_dataset(client, csv_upload):
    folder = await _folder(client, "dataset", "Gyrase")
    dataset_id = await _create_dataset(client, csv_upload)
    await client.put(f"/api/v1/datasets/{dataset_id}/folder", json={"folder_id": folder["id"]})
    assert (await client.delete(f"{_F}/{folder['id']}")).status_code == 204
    dataset = (await client.get(f"/api/v1/datasets/{dataset_id}")).json()
    assert dataset["folder_id"] is None
    assert (await client.delete(f"{_F}/{folder['id']}")).status_code == 404


async def test_a_draft_you_cannot_see_cannot_be_filed_or_counted(
    client, other_editor_client, csv_upload
):
    dataset_id = await _create_dataset(client, csv_upload)
    assert (await _train(client, dataset_id)).status_code == 202
    protocol_id = (await client.get("/api/v1/protocols")).json()["items"][0]["id"]
    folder = await _folder(client, "protocol", "Drafts")

    hidden = await other_editor_client.put(
        f"/api/v1/protocols/{protocol_id}/folder", json={"folder_id": folder["id"]}
    )
    assert hidden.status_code == 404
    filed = await client.put(
        f"/api/v1/protocols/{protocol_id}/folder", json={"folder_id": folder["id"]}
    )
    assert filed.status_code == 200, filed.text
    assert filed.json()["folder_id"] == folder["id"]

    assert await _count(client, "protocol", folder["id"]) == 1
    assert await _count(other_editor_client, "protocol", folder["id"]) == 0
    listed = (await client.get("/api/v1/protocols", params={"folder_id": folder["id"]})).json()
    assert [p["id"] for p in listed["items"]] == [protocol_id]

    assert (await client.post(f"/api/v1/protocols/{protocol_id}/publish")).status_code == 204
    assert await _count(other_editor_client, "protocol", folder["id"]) == 1
    # Filing works on a published, locked protocol.
    again = await client.put(f"/api/v1/protocols/{protocol_id}/folder", json={"folder_id": None})
    assert again.status_code == 200

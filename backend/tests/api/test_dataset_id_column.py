"""A dataset's identifier column: chosen at upload or later, shown wherever its compounds are."""

from tests.api import test_protocols

_STRUCTURES = test_protocols._STRUCTURES


def _csv_with_ids() -> bytes:
    rows = "\n".join(
        f"{smiles},{1.0 + 0.37 * index},cpd-{index},{index}"
        for index, smiles in enumerate(_STRUCTURES)
    )
    return f"smiles,y,name,num\n{rows}\n".encode()


async def _create(client, csv_upload, **overrides):
    upload_ref = await csv_upload(_csv_with_ids())
    body = {
        "name": "solubility",
        "upload_ref": upload_ref,
        "structure_column": "smiles",
        "target": {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
        "split": {"strategy": "random", "seed": 1},
    }
    body.update(overrides)
    return await client.post("/api/v1/datasets", json=body)


async def test_the_identifier_column_can_be_chosen_at_upload(client, csv_upload):
    response = await _create(client, csv_upload, id_column="name")

    assert response.status_code == 201, response.text
    assert response.json()["id_column"] == "name"


async def test_an_identifier_column_must_exist_and_not_be_reserved(client, csv_upload):
    missing = await _create(client, csv_upload, id_column="nope")
    assert missing.status_code == 422
    assert missing.json()["message"] == "Column 'nope' is not in the uploaded file."

    reserved = await _create(client, csv_upload, id_column="smiles")
    assert reserved.status_code == 422
    assert reserved.json()["message"] == (
        "Choose an identifier column other than the structure, target or split column."
    )


async def test_an_editor_sets_changes_and_clears_it_later(client, csv_upload):
    dataset_id = (await _create(client, csv_upload)).json()["id"]
    url = f"/api/v1/datasets/{dataset_id}/id-column"

    assert (await client.get(f"/api/v1/datasets/{dataset_id}/columns")).json() == {
        "columns": ["name", "num"]
    }
    first = await client.put(url, json={"id_column": "name"})
    assert first.status_code == 200, first.text
    assert first.json()["id_column"] == "name"
    assert (await client.get(f"/api/v1/datasets/{dataset_id}")).json()["id_column"] == "name"

    assert (await client.put(url, json={"id_column": "split"})).status_code == 422
    assert (await client.put(url, json={"id_column": None})).json()["id_column"] is None


async def test_a_viewer_cannot_change_it(client, viewer_client, csv_upload):
    dataset_id = (await _create(client, csv_upload)).json()["id"]

    assert (await client.get(f"/api/v1/datasets/{dataset_id}")).json()["can_edit"] is True
    viewer = await viewer_client.get(f"/api/v1/datasets/{dataset_id}")
    assert viewer.json()["can_edit"] is False
    refused = await viewer_client.put(
        f"/api/v1/datasets/{dataset_id}/id-column", json={"id_column": "name"}
    )
    assert refused.status_code == 403

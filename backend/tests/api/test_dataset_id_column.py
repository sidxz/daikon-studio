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
        "targets": [{"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"}],
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


async def _compounds(client, dataset_id: str, **params):
    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/compounds", params={"limit": 200, **params}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_compounds_carry_their_id_and_can_be_searched(client, csv_upload):
    dataset_id = (await _create(client, csv_upload, id_column="name")).json()["id"]

    every = await _compounds(client, dataset_id)
    assert sorted(item["compound_id"] for item in every["items"]) == sorted(
        f"cpd-{index}" for index in range(len(_STRUCTURES))
    )

    found = await _compounds(client, dataset_id, q="  CPD-1 ")
    assert found["total"] == 11  # cpd-1 and cpd-10 to cpd-19
    assert all("cpd-1" in item["compound_id"] for item in found["items"])

    # Whitespace alone is no filter.
    assert (await _compounds(client, dataset_id, q="   "))["total"] == len(_STRUCTURES)


async def test_a_numeric_id_column_reads_as_integers(client, csv_upload):
    dataset_id = (await _create(client, csv_upload, id_column="num")).json()["id"]

    ids = {item["compound_id"] for item in (await _compounds(client, dataset_id))["items"]}
    assert ids == {str(index) for index in range(len(_STRUCTURES))}
    assert (await _compounds(client, dataset_id, q="12"))["total"] == 1


async def test_search_needs_an_identifier_column(client, csv_upload):
    dataset_id = (await _create(client, csv_upload)).json()["id"]

    refused = await client.get(f"/api/v1/datasets/{dataset_id}/compounds", params={"q": "cpd"})
    assert refused.status_code == 422
    assert refused.json()["message"] == "This dataset has no identifier column."
    assert all(
        item["compound_id"] is None for item in (await _compounds(client, dataset_id))["items"]
    )


async def test_a_protocols_errors_and_map_show_ids_and_follow_a_change(client, csv_upload):
    dataset_id = (await _create(client, csv_upload, id_column="name")).json()["id"]
    by_structure = {
        item["structure"]: item["compound_id"]
        for item in (await _compounds(client, dataset_id))["items"]
    }
    train = await test_protocols._train(client, dataset_id)
    assert train.status_code == 202, train.text
    run = (await client.get(f"/api/v1/runs/{train.json()['id']}")).json()
    protocol_id = run["protocol_id"]

    scorecard = (await client.get(f"/api/v1/protocols/{protocol_id}/scorecard")).json()
    assert scorecard["worst_rows"]
    for row in scorecard["worst_rows"]:
        assert row["compound_id"] == by_structure[row["structure"]]

    compounds = (
        await client.get(
            f"/api/v1/protocols/{protocol_id}/chemical-space/compounds",
            params={"indices": [0, 1]},
        )
    ).json()
    assert [item["compound_id"] for item in compounds] == [
        by_structure[item["structure"]] for item in compounds
    ]

    # Switching the column changes the IDs on the trained protocol at once.
    await client.put(f"/api/v1/datasets/{dataset_id}/id-column", json={"id_column": "num"})
    switched = (await client.get(f"/api/v1/protocols/{protocol_id}/scorecard")).json()
    assert all(row["compound_id"].isdigit() for row in switched["worst_rows"])

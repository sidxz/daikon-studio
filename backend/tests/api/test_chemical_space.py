"""The chemical-space map, end to end through the API.

Fixtures come from the protocol and triage suites: a 20-compound random split
(16 train / 2 validation / 2 test) and a finished prediction run on it.
"""

import json
import uuid

import polars as pl
from tests.api import test_protocols, test_triage_round_trip

# Fixtures shared with the protocol and triage suites, bound here so pytest finds them.
dataset_id = test_protocols.dataset_id
trained_protocol_id = test_protocols.trained_protocol_id
ready_run_id = test_triage_round_trip.ready_run_id


def _meta_path(tmp_path, workspace_id: uuid.UUID, protocol_id: str):
    return tmp_path / str(workspace_id) / "protocols" / protocol_id / "chemical-space.json"


async def test_training_writes_a_map_of_every_compound(
    tmp_path, workspace_id, trained_protocol_id
):
    meta = json.loads(_meta_path(tmp_path, workspace_id, trained_protocol_id).read_text())
    assert meta["version"] == 1
    assert meta["method"] == "umap"
    assert meta["counts"] == {"train": 16, "validation": 2, "test": 2}


async def test_a_prediction_keeps_five_neighbours_per_compound(
    client, tmp_path, workspace_id, ready_run_id
):
    run_dir = tmp_path / str(workspace_id) / "runs" / ready_run_id
    neighbours = pl.read_parquet(run_dir / "neighbors.parquet")
    predictions = pl.read_parquet(run_dir / "predictions.parquet")
    assert neighbours.height == predictions.height
    for indices, sims, applicability in zip(
        neighbours["neighbor_index"].to_list(),
        neighbours["neighbor_similarity"].to_list(),
        predictions["applicability"].to_list(),
        strict=True,
    ):
        assert 1 <= len(indices) <= 5
        assert sims == sorted(sims, reverse=True)
        assert abs(sims[0] - applicability) < 1e-6


async def test_the_protocol_map_is_served_in_unit_coordinates(client, trained_protocol_id):
    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/chemical-space")).json()
    assert body["status"] == "ready"
    points = body["points"]
    assert len(points["x"]) == len(points["y"]) == len(points["partition"]) == 20
    assert all(0 <= v <= 1 for v in points["x"] + points["y"])
    assert sorted(set(points["partition"])) == [0, 1, 2]


async def test_a_protocol_without_a_map_says_missing(
    client, tmp_path, workspace_id, trained_protocol_id
):
    _meta_path(tmp_path, workspace_id, trained_protocol_id).unlink()
    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/chemical-space")).json()
    assert body == {
        "status": "missing",
        "method": None,
        "params": None,
        "counts": None,
        "points": None,
    }


async def test_map_compounds_are_looked_up_by_index(client, trained_protocol_id):
    response = await client.get(
        f"/api/v1/protocols/{trained_protocol_id}/chemical-space/compounds",
        params=[("indices", 0), ("indices", 3)],
    )
    assert response.status_code == 200, response.text
    items = response.json()
    assert [item["index"] for item in items] == [0, 3]
    assert all(item["partition"] in {"train", "validation", "test"} for item in items)


async def test_more_than_fifty_lookups_is_refused(client, trained_protocol_id):
    response = await client.get(
        f"/api/v1/protocols/{trained_protocol_id}/chemical-space/compounds",
        params=[("indices", i) for i in range(51)],
    )
    assert response.status_code == 422


async def test_a_run_is_placed_among_its_training_neighbours(client, ready_run_id):
    run = (await client.get(f"/api/v1/runs/{ready_run_id}")).json()
    protocol_map = (
        await client.get(f"/api/v1/protocols/{run['protocol_id']}/chemical-space")
    ).json()
    body = (await client.get(f"/api/v1/runs/{ready_run_id}/chemical-space")).json()
    assert body["status"] == "ready"
    points = body["points"]
    assert len(points["x"]) == body["summary"]["total"]
    partition = protocol_map["points"]["partition"]
    for neighbours in points["neighbors"]:
        assert neighbours and all(partition[i] == 0 for i in neighbours)
    assert body["summary"]["threshold"] == 0.3


async def test_run_compounds_are_looked_up_by_row(client, ready_run_id):
    page = (await client.get(f"/api/v1/runs/{ready_run_id}/results")).json()
    first = page["items"][0]
    items = (
        await client.get(
            f"/api/v1/runs/{ready_run_id}/chemical-space/compounds",
            params=[("rows", first["row_id"])],
        )
    ).json()
    assert items[0]["structure"] == first["structure"]


async def test_a_run_without_neighbours_says_missing(client, tmp_path, workspace_id, ready_run_id):
    (tmp_path / str(workspace_id) / "runs" / ready_run_id / "neighbors.parquet").unlink()
    body = (await client.get(f"/api/v1/runs/{ready_run_id}/chemical-space")).json()
    assert body["status"] == "missing"

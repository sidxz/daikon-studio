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

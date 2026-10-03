"""The chemical-space map, end to end through the API.

Fixtures come from the protocol and triage suites: a 20-compound random split
(16 train / 2 validation / 2 test) and a finished prediction run on it.
"""

import json
import uuid

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

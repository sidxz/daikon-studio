"""End-to-end tests for the Collection routes: save a triage selection from a
`ready` prediction Run's results, read it back, export it as CSV or SDF.

Training and prediction both run through the real `InlineEnqueuer` (see the
`app` fixture in `tests/api/conftest.py`), so a fixture that submits either
already holds a finished Run by the time the request returns -- the same
"no polling needed" shape `test_runs.py` documents.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest_asyncio

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)

# Twenty distinct compounds to train on -- the same shape `test_runs.py` uses,
# for the same reason: a real 16/2/2 random split needs enough rows to hold.
_STRUCTURES = (
    "CCO",
    "CCN",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOC1",
    "C1CCCC1",
    "C1CC1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",
    "C1CCC2CCCCC2C1",
    "c1cnc2ccccc2c1",
    "O=C1CCCCC1",
)

# Four distinct compounds to triage: row_ids 0 and 3 pick the first and last.
_PREDICTION_CSV = b"smiles\nCCO\nCCN\nc1ccccc1\nFc1ccc(F)cc1\n"


def _training_csv() -> bytes:
    rows = "\n".join(f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_STRUCTURES))
    return f"smiles,y\n{rows}\n".encode()


async def _create_dataset(client, csv_upload) -> str:
    upload_ref = await csv_upload(_training_csv())
    response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "solubility",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "target": {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
            "split": {"strategy": "random", "seed": 1},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _train(client, dataset_id: str, **overrides: object) -> Any:
    body: dict[str, object] = {
        "name": "solubility model",
        "dataset_id": dataset_id,
        "engine_id": "ecfp4-xgboost",
        "conditions": {},
    }
    body.update(overrides)
    return await client.post("/api/v1/protocols", json=body)


async def _predict(client, protocol_id: str, upload_ref: str) -> Any:
    return await client.post(
        "/api/v1/runs",
        json={
            "protocol_id": protocol_id,
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "conditions": {},
        },
    )


@pytest_asyncio.fixture
async def published_protocol_id(client, csv_upload) -> str:
    dataset_id = await _create_dataset(client, csv_upload)
    trained = await _train(client, dataset_id)
    assert trained.status_code == 202, trained.text
    listing = await client.get("/api/v1/protocols")
    items = listing.json()["items"]
    assert len(items) == 1, items
    protocol_id = items[0]["id"]

    published = await client.post(f"/api/v1/protocols/{protocol_id}/publish")
    assert published.status_code == 204, published.text
    return str(protocol_id)


@pytest_asyncio.fixture
async def readout(client, published_protocol_id) -> dict[str, Any]:
    """The one readout `ecfp4-xgboost` regression declares -- name, unit,
    direction -- so tests can assert on it without hardcoding what
    `derive_readouts` happens to name a numeric target's readout."""
    protocol = (await client.get(f"/api/v1/protocols/{published_protocol_id}")).json()
    return protocol["readouts"][0]


@pytest_asyncio.fixture
async def ready_run_id(client, csv_upload, published_protocol_id) -> str:
    upload_ref = await csv_upload(_PREDICTION_CSV)
    response = await _predict(client, published_protocol_id, upload_ref)
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]

    polled = await client.get(f"/api/v1/runs/{run_id}")
    assert polled.json()["status"] == "ready", polled.text
    return str(run_id)


@pytest_asyncio.fixture
async def pending_run_id(session_factory, workspace_id) -> str:
    """A `PENDING` prediction Run, inserted directly (bypassing the worker) the
    same way `test_runs.py` builds one -- no code path in this app leaves a
    real request pending, since `InlineEnqueuer` finishes synchronously."""
    run = Run(
        kind=RunKind.PREDICTION,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="pending-collection-k",
    )
    await SqlAlchemyRunRepository(session_factory).add(run)
    return str(run.id)


@pytest_asyncio.fixture
async def collection_id(client, ready_run_id) -> str:
    response = await client.post(
        "/api/v1/collections",
        json={"name": "top 2 for synthesis", "run_id": ready_run_id, "row_ids": [0, 3]},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def test_saving_a_triage_selection_creates_a_collection(client, ready_run_id):
    response = await client.post(
        "/api/v1/collections",
        json={"name": "top 2 for synthesis", "run_id": ready_run_id, "row_ids": [0, 3]},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["member_count"] == 2
    assert body["derived_from_run_id"] == ready_run_id
    assert body["name"] == "top 2 for synthesis"


async def test_csv_export_contains_the_selected_structures(client, collection_id):
    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert "smiles" in response.text.splitlines()[0]
    assert response.text.count("\n") == 3  # header + 2 selected rows (+ trailing newline)


async def test_sdf_export_is_a_valid_molfile_block(client, collection_id):
    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=sdf")
    assert response.status_code == 200, response.text
    assert response.text.count("$$$$") == 2


async def test_predictions_carry_the_ai_predicted_provenance(client, collection_id):
    body = (await client.get(f"/api/v1/collections/{collection_id}")).json()
    assert body["provenance"]["generation_method"] == "ai_predicted"


async def test_selecting_rows_from_an_unfinished_run_is_rejected(client, pending_run_id):
    response = await client.post(
        "/api/v1/collections",
        json={"name": "too early", "run_id": pending_run_id, "row_ids": [0]},
    )
    assert response.status_code == 409, response.text


async def test_csv_column_header_carries_the_readouts_unit(client, collection_id, readout):
    """Decision 3: a spreadsheet reader sees the unit in the column header,
    not just a bare number -- CSV has no per-cell tag the way SDF does."""
    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    header = response.text.splitlines()[0]
    assert f"{readout['name']} ({readout['unit']})" in header
    assert "generation_method" in header
    rows = response.text.splitlines()[1:]
    assert all("ai_predicted" in row for row in rows)


async def test_sdf_tag_carries_the_readouts_value_and_unit(client, collection_id, readout):
    """Decision 3: an SD tag's *value* is labelled with its unit (a chemistry
    tool has no header row to hang a unit on the way a spreadsheet does), and
    a dedicated `generation_method` tag marks every block as a prediction."""
    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=sdf")
    body = response.text
    assert f"<{readout['name']}>" in body
    assert readout["unit"] in body
    assert "<generation_method>" in body
    assert "ai_predicted" in body


async def test_empty_row_ids_is_rejected(client, ready_run_id):
    response = await client.post(
        "/api/v1/collections",
        json={"name": "nothing", "run_id": ready_run_id, "row_ids": []},
    )
    assert response.status_code == 422, response.text


async def test_duplicate_row_ids_is_rejected(client, ready_run_id):
    response = await client.post(
        "/api/v1/collections",
        json={"name": "twice", "run_id": ready_run_id, "row_ids": [0, 0]},
    )
    assert response.status_code == 422, response.text


async def test_out_of_range_row_id_is_rejected(client, ready_run_id):
    """A stale cursor or an id from a different run must not be silently
    dropped -- the run behind `ready_run_id` has 4 result rows (0-3)."""
    response = await client.post(
        "/api/v1/collections",
        json={"name": "too far", "run_id": ready_run_id, "row_ids": [0, 99]},
    )
    assert response.status_code == 422, response.text


async def test_negative_row_id_is_rejected(client, ready_run_id):
    """Negative indices are never silently reinterpreted as "from the end"
    (which is what Python/polars fancy indexing would otherwise do)."""
    response = await client.post(
        "/api/v1/collections",
        json={"name": "negative", "run_id": ready_run_id, "row_ids": [-1]},
    )
    assert response.status_code == 422, response.text


async def test_creating_a_collection_from_an_unknown_run_is_a_404(client):
    response = await client.post(
        "/api/v1/collections",
        json={"name": "ghost", "run_id": str(uuid.uuid4()), "row_ids": [0]},
    )
    assert response.status_code == 404, response.text


async def test_a_missing_collection_id_is_a_404(client):
    response = await client.get(f"/api/v1/collections/{uuid.uuid4()}")
    assert response.status_code == 404, response.text


async def test_workspace_id_in_the_body_is_rejected(client, ready_run_id):
    response = await client.post(
        "/api/v1/collections",
        json={
            "name": "spoofed",
            "run_id": ready_run_id,
            "row_ids": [0],
            "workspace_id": "00000000-0000-0000-0000-000000000001",
        },
    )
    assert response.status_code == 422, response.text


async def test_viewer_cannot_create_a_collection(viewer_client, ready_run_id):
    response = await viewer_client.post(
        "/api/v1/collections",
        json={"name": "not allowed", "run_id": ready_run_id, "row_ids": [0]},
    )
    assert response.status_code == 403, response.text


async def test_viewer_can_read_and_export_a_collection(viewer_client, collection_id):
    assert (await viewer_client.get(f"/api/v1/collections/{collection_id}")).status_code == 200
    export = await viewer_client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    assert export.status_code == 200


async def test_unauthenticated_request_is_rejected(anonymous_client):
    assert (await anonymous_client.post("/api/v1/collections", json={})).status_code == 401
    assert (await anonymous_client.get(f"/api/v1/collections/{uuid.uuid4()}")).status_code == 401


async def test_collections_are_scoped_to_the_callers_workspace(
    client, other_workspace_client, collection_id
):
    assert (
        await other_workspace_client.get(f"/api/v1/collections/{collection_id}")
    ).status_code == 404
    assert (
        await other_workspace_client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    ).status_code == 404


async def test_a_collection_cannot_be_created_from_another_workspaces_run(
    other_workspace_client, ready_run_id
):
    response = await other_workspace_client.post(
        "/api/v1/collections",
        json={"name": "borrowed", "run_id": ready_run_id, "row_ids": [0]},
    )
    assert response.status_code == 404, response.text

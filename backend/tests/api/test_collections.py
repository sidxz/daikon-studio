"""End-to-end tests for the Collection routes: save a triage selection from a
`ready` prediction Run's results, read it back, export it as CSV or SDF.

Training and prediction both run through the real `InlineEnqueuer` (see the
`app` fixture in `tests/api/conftest.py`), so a fixture that submits either
already holds a finished Run by the time the request returns -- the same
"no polling needed" shape `test_runs.py` documents.
"""

from __future__ import annotations

import csv
import io
import uuid
from typing import Any

import pytest_asyncio
from rdkit import Chem

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)

# Mirrors `export_collection.py`'s own `_DIRECTION_LABEL` -- a small, stable
# English-phrase contract, duplicated here deliberately so these tests assert
# against the HTTP-visible contract, not by importing the implementation.
_DIRECTION_LABEL = {"high": "higher is better", "low": "lower is better"}

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


def _training_csv(*, structure_column: str = "smiles", target_column: str = "y") -> bytes:
    rows = "\n".join(f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_STRUCTURES))
    return f"{structure_column},{target_column}\n{rows}\n".encode()


async def _create_dataset(
    client,
    csv_upload,
    *,
    structure_column: str = "smiles",
    target_column: str = "y",
    unit: str | None = "logS",
    direction: str | None = "high",
) -> str:
    """`structure_column` only needs overriding when `target_column` is
    itself `"smiles"` (a CSV header can't repeat a column name); `unit`/
    `direction` default to a normal, disambiguating readout, and are set to
    `None` by the collision tests below to reproduce a readout that renders
    to its *bare* name -- the actual trigger for Important finding 2/round 2."""
    upload_ref = await csv_upload(
        _training_csv(structure_column=structure_column, target_column=target_column)
    )
    response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "solubility",
            "upload_ref": upload_ref,
            "structure_column": structure_column,
            "target": {
                "column": target_column,
                "kind": "numeric",
                "unit": unit,
                "direction": direction,
            },
            "split": {"strategy": "random", "seed": 1},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _result_structures(client, run_id: str) -> list[str]:
    """The run's own results, in file order -- `test_runs.py`'s pagination
    tests already establish this order is stable and offset-addressable."""
    response = await client.get(f"/api/v1/runs/{run_id}/results?limit=50")
    assert response.status_code == 200, response.text
    return [item["structure"] for item in response.json()["items"]]


async def _result_values(client, run_id: str, readout_name: str) -> list[float]:
    response = await client.get(f"/api/v1/runs/{run_id}/results?limit=50")
    assert response.status_code == 200, response.text
    return [item["readouts"][readout_name]["value"] for item in response.json()["items"]]


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
    """Not just "the delimiter appears twice" -- RDKit must actually be able
    to read every block back as a molecule. A malformed atom table or a
    mangled tag delimiter would still pass a bare `$$$$` count."""
    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=sdf")
    assert response.status_code == 200, response.text
    assert response.text.count("$$$$") == 2

    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(response.content)))
    assert len(molecules) == 2
    assert all(mol is not None for mol in molecules)


async def test_predictions_carry_the_ai_predicted_provenance(client, collection_id):
    body = (await client.get(f"/api/v1/collections/{collection_id}")).json()
    assert body["provenance"]["generation_method"] == "ai_predicted"


async def test_selecting_rows_from_an_unfinished_run_is_rejected(client, pending_run_id):
    response = await client.post(
        "/api/v1/collections",
        json={"name": "too early", "run_id": pending_run_id, "row_ids": [0]},
    )
    assert response.status_code == 409, response.text


async def test_csv_column_header_carries_the_readouts_unit_and_direction(
    client, collection_id, readout
):
    """Decision 3 (Important 1 fix): a spreadsheet reader sees the unit
    *and* direction in the column header itself -- CSV has no per-cell tag
    the way SDF does. Parsed with `csv.DictReader`, not a substring search:
    a bare `"logS" in body` would pass even if the unit landed somewhere
    unrelated to the actual value column."""
    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    rows = list(csv.DictReader(response.text.splitlines()))
    assert len(rows) == 2

    direction_label = _DIRECTION_LABEL[readout["direction"]]
    column = f"{readout['name']} ({readout['unit']}, {direction_label})"
    assert column in rows[0]  # KeyError if the header doesn't have exactly this name
    assert all(row["generation_method"] == "ai_predicted" for row in rows)


async def test_sdf_tag_carries_the_readouts_value_unit_and_direction(
    client, ready_run_id, collection_id, readout
):
    """Decision 3 (Important 1 + 4 fix): an SD tag's *value* is labelled with
    its unit and direction together (a chemistry tool has no header row to
    hang either on), and a dedicated `generation_method` tag marks every
    block as a prediction -- not inferable only from the filename.

    Parsed with RDKit's own SD reader and compared with `==` against the
    exact predicted value, not a substring search: `"logS" in body` would
    pass even if the unit were detached from the value entirely.
    """
    values = await _result_values(client, ready_run_id, readout["name"])
    # `collection_id` selected row_ids [0, 3] -- the first of those two.
    first_value = values[0]
    direction_label = _DIRECTION_LABEL[readout["direction"]]

    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=sdf")
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(response.content)))
    assert len(molecules) == 2

    first = molecules[0]
    assert first.GetProp(readout["name"]) == f"{first_value} {readout['unit']} ({direction_label})"
    assert first.GetProp("generation_method") == "ai_predicted"


async def test_export_preserves_the_callers_selection_order_not_sorted(client, ready_run_id):
    """Decision 1's whole point, made concrete: `row_ids` is a ranked "top N"
    the caller chose, not a set. `[3, 0]` must come back as compound 3 then
    compound 0 -- re-sorting ascending would be the exact "believed they
    saved compound A" substitution the decision was written against."""
    structures = await _result_structures(client, ready_run_id)
    assert len(structures) == 4

    async def _exported_smiles(row_ids: list[int]) -> list[str]:
        created = await client.post(
            "/api/v1/collections",
            json={"name": f"order {row_ids}", "run_id": ready_run_id, "row_ids": row_ids},
        )
        assert created.status_code == 201, created.text
        export = await client.get(f"/api/v1/collections/{created.json()['id']}/export?format=csv")
        return [row["smiles"] for row in csv.DictReader(export.text.splitlines())]

    assert await _exported_smiles([0, 3]) == [structures[0], structures[3]]
    assert await _exported_smiles([3, 0]) == [structures[3], structures[0]]
    assert await _exported_smiles([2]) == [structures[2]]


async def _publish_and_predict(client, csv_upload, dataset_id: str) -> str:
    """Train, publish, and run a prediction against `_PREDICTION_CSV` --
    returns the ready run id. Shared by the collision tests below, each of
    which needs its own from-scratch Protocol built on a non-default target
    column/unit/direction."""
    trained = await _train(client, dataset_id)
    assert trained.status_code == 202, trained.text
    listing = await client.get("/api/v1/protocols")
    items = listing.json()["items"]
    assert len(items) == 1, items
    protocol_id = items[0]["id"]
    published = await client.post(f"/api/v1/protocols/{protocol_id}/publish")
    assert published.status_code == 204, published.text

    upload_ref = await csv_upload(_PREDICTION_CSV)
    run = await _predict(client, protocol_id, upload_ref)
    assert run.status_code == 202, run.text
    return str(run.json()["id"])


async def _create_collection(
    client, run_id: str, name: str, row_ids: list[int] | None = None
) -> str:
    created = await client.post(
        "/api/v1/collections",
        json={"name": name, "run_id": run_id, "row_ids": row_ids or [0]},
    )
    assert created.status_code == 201, created.text
    return str(created.json()["id"])


async def test_export_rejects_a_readout_name_that_collides_with_the_provenance_column(
    client, csv_upload
):
    """Important 2 fix: a readout literally named `generation_method`, with
    no unit or direction to disambiguate it, renders to the bare name
    `"generation_method"` in both formats -- colliding with the provenance
    column/tag this module always appends (CSV: renamed-then-overwritten
    column; SDF: the readout's own `SetProp` clobbered by the provenance
    `SetProp` right after it). Guarded before either render runs, not
    discovered as a corrupted file with a 200 status."""
    dataset_id = await _create_dataset(
        client, csv_upload, target_column="generation_method", unit=None, direction=None
    )
    run_id = await _publish_and_predict(client, csv_upload, dataset_id)
    collection_id = await _create_collection(client, run_id, "colliding generation_method")

    for export_format in ("csv", "sdf"):
        response = await client.get(
            f"/api/v1/collections/{collection_id}/export?format={export_format}"
        )
        assert response.status_code == 422, response.text


async def test_export_rejects_a_readout_that_renders_to_the_bare_smiles_column(client, csv_upload):
    """Round 2 finding: CSV always renames `structure` to `smiles`. A
    readout with *no* unit and *no* direction renders to its bare name --
    not exotic, a classification CLASS readout with no direction set
    produces exactly this -- so a target column literally named `smiles`
    collides with the renamed structure column the same way
    `generation_method` collides with the appended provenance column.
    `structure_column="mol"` avoids a duplicate header in the training CSV
    itself; the collision is entirely about the *target* column's name.

    SDF never renames `structure`, so the same Protocol's SDF export is
    unaffected -- proof the guard is computed per format, not a blanket
    rejection of the name `"smiles"` everywhere.
    """
    dataset_id = await _create_dataset(
        client,
        csv_upload,
        structure_column="mol",
        target_column="smiles",
        unit=None,
        direction=None,
    )
    run_id = await _publish_and_predict(client, csv_upload, dataset_id)
    collection_id = await _create_collection(client, run_id, "colliding smiles")

    csv_response = await client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    assert csv_response.status_code == 422, csv_response.text

    sdf_response = await client.get(f"/api/v1/collections/{collection_id}/export?format=sdf")
    assert sdf_response.status_code == 200, sdf_response.text


async def test_a_smiles_named_readout_with_a_unit_still_exports_safely(client, csv_upload):
    """The genuinely safe sibling the guard must not catch: a readout named
    `smiles` that *does* carry a unit and direction renders to
    `"smiles (nM, lower is better)"`, distinct from the renamed structure
    column -- it must keep exporting, not be swept up by an over-broad
    check on the name `"smiles"` alone."""
    dataset_id = await _create_dataset(
        client,
        csv_upload,
        structure_column="mol",
        target_column="smiles",
        unit="nM",
        direction="low",
    )
    run_id = await _publish_and_predict(client, csv_upload, dataset_id)
    collection_id = await _create_collection(client, run_id, "safe smiles")

    response = await client.get(f"/api/v1/collections/{collection_id}/export?format=csv")
    assert response.status_code == 200, response.text
    rows = list(csv.DictReader(response.text.splitlines()))
    assert "smiles (nM, lower is better)" in rows[0]
    assert rows[0]["smiles"]  # the real structure column, untouched by the collision


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

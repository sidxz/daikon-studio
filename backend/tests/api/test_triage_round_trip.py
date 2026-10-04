"""The triage round trip: read a reordered page of results, save what you saw.

This is the one flow where a wrong answer has no visible symptom. A chemist
sorts and filters the triage grid, ticks some rows, and saves them as a
Collection. If the handle travelling from `/runs/{id}/results` to
`POST /collections` ever stops meaning "position in the original results file",
the Collection silently holds *different compounds* than the ones on screen --
no error, no warning, just the wrong molecules going to synthesis.

Two independent things have to hold for that not to happen, and both are
asserted here against a view that is sorted AND filtered:

1. `row_id` is minted before any reordering, so it survives being sorted.
2. The sort is a total order, so the same request twice returns the same page
   -- results are re-read and re-sorted on every request, and an unstable sort
   would let a tied row appear on two pages or none.

`test_result_view.py` covers both properties as pure functions. This covers the
wiring between them: the ids the API hands out are the ids `POST /collections`
accepts, over HTTP, through the real repositories and blob store.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any
from urllib.parse import quote

import pytest_asyncio

# Twenty compounds to train on -- a real 16/2/2 split needs enough rows to hold,
# the same reason `test_collections.py` and `test_runs.py` use twenty.
_STRUCTURES = (
    "CCO",
    "CCN",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "CCOC",
    "CCSC",
    "CC(=O)O",
    "CC(=O)N",
    "CCBr",
    "CCCl",
    "CCF",
    "OCC1CCCCC1",
    "NCC1CCCCC1",
    "c1ccc2ccccc2c1",
    "CC(C)CO",
    "CCCCCCO",
)

# Six to predict on. Enough that a filter can drop some and a sort can reorder
# the rest, so the page under test is never trivially in file order.
_PREDICTION_CSV = b"smiles\nCCO\nCCN\nc1ccccc1\nFc1ccc(F)cc1\nCCCCO\nCC(=O)O\n"


def _training_csv() -> bytes:
    rows = "\n".join(f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_STRUCTURES))
    return f"smiles,y\n{rows}\n".encode()


async def _create_dataset(client, csv_upload) -> str:
    upload_ref = await csv_upload(_training_csv())
    response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "round trip",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "targets": [{"column": "y", "kind": "numeric", "unit": "nM", "direction": "low"}],
            "split": {"strategy": "random", "seed": 7},
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


@pytest_asyncio.fixture
async def ready_run_id(client, csv_upload) -> str:
    """A finished prediction Run. `InlineEnqueuer` runs the work synchronously,
    so this is `ready` by the time the POST returns -- no polling."""
    dataset_id = await _create_dataset(client, csv_upload)
    trained = await client.post(
        "/api/v1/protocols",
        json={
            "name": "round trip protocol",
            "dataset_id": dataset_id,
            "engine_id": "ecfp4-xgboost",
            "conditions": {},
        },
    )
    assert trained.status_code == 202, trained.text

    protocol_id = (await client.get("/api/v1/protocols")).json()["items"][0]["id"]
    published = await client.post(f"/api/v1/protocols/{protocol_id}/publish")
    assert published.status_code == 204, published.text

    upload_ref = await csv_upload(_PREDICTION_CSV)
    submitted = await client.post(
        "/api/v1/runs",
        json={
            "protocol_id": protocol_id,
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "conditions": {},
        },
    )
    assert submitted.status_code == 202, submitted.text
    run_id = submitted.json()["id"]
    assert (await client.get(f"/api/v1/runs/{run_id}")).json()["status"] == "ready"
    return str(run_id)


async def _sorted_filtered_page(client, run_id: str) -> list[dict[str, Any]]:
    """The page a chemist is actually looking at: applicability filtered, and
    sorted descending on a readout, so the rows are in neither file order nor
    ascending order."""
    protocol_id = (await client.get(f"/api/v1/runs/{run_id}")).json()["protocol_id"]
    readout = (await client.get(f"/api/v1/protocols/{protocol_id}")).json()["readouts"][0]["name"]
    filters = quote(json.dumps({"applicability": {"min": 0.0}}))
    response = await client.get(
        f"/api/v1/runs/{run_id}/results?sort_by={readout}&sort_dir=desc&filters={filters}"
    )
    assert response.status_code == 200, response.text
    return response.json()["items"]


async def test_the_compounds_saved_are_the_compounds_that_were_on_screen(client, ready_run_id):
    """The whole point. Take rows from a sorted, filtered page, save exactly
    those row_ids, and get exactly those structures back."""
    page = await _sorted_filtered_page(client, ready_run_id)
    assert len(page) >= 3, page

    # Not the first N: picking from the middle means a saved selection that
    # matches file order by luck cannot pass.
    picked = [page[1], page[2]]
    expected = {row["structure"] for row in picked}

    created = await client.post(
        "/api/v1/collections",
        json={
            "name": "picked under a sort",
            "run_id": ready_run_id,
            "row_ids": [row["row_id"] for row in picked],
        },
    )
    assert created.status_code == 201, created.text

    exported = await client.get(f"/api/v1/collections/{created.json()['id']}/export?format=csv")
    assert exported.status_code == 200, exported.text
    # The export names the structure column `smiles`, not `structure` -- it is
    # written for a chemist opening it in Excel, not for this API's field names.
    saved = {row["smiles"] for row in csv.DictReader(io.StringIO(exported.text))}
    assert saved == expected


async def test_a_sorted_page_is_the_same_page_when_asked_for_twice(client, ready_run_id):
    """Results are re-read and re-sorted on every request, so a page is only a
    stable window if the sort is a total order. An unstable sort would let rows
    that tie on the sort column swap places between requests -- and a client
    paging through would then see one row twice and another never."""
    first = await _sorted_filtered_page(client, ready_run_id)
    second = await _sorted_filtered_page(client, ready_run_id)
    assert [row["row_id"] for row in first] == [row["row_id"] for row in second]


async def test_row_ids_are_file_positions_not_positions_within_the_view(client, ready_run_id):
    """`row_id` must survive reordering: the same compound carries the same id
    whether the page is sorted ascending, descending, or not at all. That is
    what makes it a valid handle for `POST /collections`, which indexes into the
    original results file."""
    unsorted = (await client.get(f"/api/v1/runs/{ready_run_id}/results")).json()["items"]
    descending = await _sorted_filtered_page(client, ready_run_id)

    by_structure = {row["structure"]: row["row_id"] for row in unsorted}
    assert by_structure, unsorted
    for row in descending:
        assert row["row_id"] == by_structure[row["structure"]]

    # And unsorted really is file order, so the assertion above is comparing
    # against something meaningful rather than against another sorted view.
    assert [row["row_id"] for row in unsorted] == list(range(len(unsorted)))

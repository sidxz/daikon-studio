"""End-to-end tests for the Run routes: submit a prediction, poll it, read its
results, cancel it.

Training and prediction both run through the real `InlineEnqueuer` (see the
`app` fixture in `tests/api/conftest.py`), so by the time `POST /api/v1/runs`
returns, the prediction it names has already finished -- the same "no polling
needed" shape `test_protocols.py` documents for training.
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import quote

import pytest_asyncio

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)

# Twenty distinct compounds to train on -- the same shape `test_protocols.py`
# uses and for the same reason: a real 16/2/2 random split.
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

# A colleague's own compounds to run the Protocol against.
_PREDICTION_CSV = b"smiles\nCCO\nc1ccccc1\nFc1ccc(F)cc1\n"


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


def _predict_body(protocol_id: str, upload_ref: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "protocol_id": protocol_id,
        "upload_ref": upload_ref,
        "structure_column": "smiles",
        "conditions": {},
    }
    body.update(overrides)
    return body


async def _predict(client, protocol_id: str, upload_ref: str, **overrides: object) -> Any:
    return await client.post(
        "/api/v1/runs", json=_predict_body(protocol_id, upload_ref, **overrides)
    )


@pytest_asyncio.fixture
async def trained_protocol_id(client, csv_upload) -> str:
    dataset_id = await _create_dataset(client, csv_upload)
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text
    listing = await client.get("/api/v1/protocols")
    items = listing.json()["items"]
    assert len(items) == 1, items
    return str(items[0]["id"])


@pytest_asyncio.fixture
async def published_protocol_id(client, trained_protocol_id) -> str:
    response = await client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    assert response.status_code == 204, response.text
    return trained_protocol_id


@pytest_asyncio.fixture
async def prediction_upload_ref(csv_upload) -> str:
    return await csv_upload(_PREDICTION_CSV)


async def test_prediction_request_returns_202_and_is_immediately_ready(
    client, published_protocol_id, prediction_upload_ref
):
    response = await _predict(client, published_protocol_id, prediction_upload_ref)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["kind"] == "prediction"

    polled = await client.get(f"/api/v1/runs/{body['id']}")
    assert polled.status_code == 200, polled.text
    assert polled.json()["status"] == "ready"


async def test_identical_prediction_requests_reuse_the_cached_run(
    client, published_protocol_id, prediction_upload_ref
):
    first = await _predict(client, published_protocol_id, prediction_upload_ref)
    second = await _predict(client, published_protocol_id, prediction_upload_ref)
    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    assert first.json()["id"] == second.json()["id"]


async def test_results_carry_structure_readouts_uncertainty_and_applicability(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]

    protocol = (await client.get(f"/api/v1/protocols/{published_protocol_id}")).json()
    readout = protocol["readouts"][0]

    response = await client.get(f"/api/v1/runs/{run_id}/results")
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["items"]) == 3  # all three query compounds canonicalize
    row = body["items"][0]
    assert "structure" in row
    assert readout["name"] in row["readouts"]
    # Important 5 (Task 17 review): unit and direction travel with the value,
    # not just a bare number -- otherwise a client can't line a prediction up
    # against a measurement without a second call to fetch the Protocol.
    predicted = row["readouts"][readout["name"]]
    assert isinstance(predicted["value"], float)
    assert predicted["unit"] == readout["unit"]
    assert predicted["direction"] == readout["direction"]
    assert "uncertainty" in row
    # ecfp4-xgboost: no ensemble spread to report, never a fabricated number.
    assert row["uncertainty"] is None
    assert isinstance(row["applicability"], float)
    assert 0.0 <= row["applicability"] <= 1.0


async def test_run_response_carries_the_protocol_id_for_a_prediction(
    client, published_protocol_id, prediction_upload_ref
):
    """Important 5 (Task 17 review): without this, a client holding only a
    run id has no way to construct `GET /protocols/{id}` to fetch the readout
    metadata above."""
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    assert submitted.json()["protocol_id"] == published_protocol_id

    polled = await client.get(f"/api/v1/runs/{submitted.json()['id']}")
    assert polled.json()["protocol_id"] == published_protocol_id


async def test_a_training_run_has_no_protocol_id_when_it_is_first_accepted(client, csv_upload):
    """The 202 is built from the Run as enqueued, and at that moment the
    Protocol genuinely does not exist yet. Null here is honest, not missing."""
    dataset_id = await _create_dataset(client, csv_upload)
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text
    assert response.json()["protocol_id"] is None


async def test_polling_a_finished_training_run_yields_the_protocol_it_produced(client, csv_upload):
    """The transition the whole training screen depends on: submit, poll to
    `ready`, then follow `protocol_id` to the Scorecard. Before Run gained the
    column this was a dead end -- the id existed only inside a blob path."""
    dataset_id = await _create_dataset(client, csv_upload)
    run_id = (await _train(client, dataset_id)).json()["id"]

    polled = (await client.get(f"/api/v1/runs/{run_id}")).json()
    assert polled["status"] == "ready", polled
    assert polled["protocol_id"] is not None

    scorecard = await client.get(f"/api/v1/protocols/{polled['protocol_id']}/scorecard")
    assert scorecard.status_code == 200, scorecard.text


async def test_a_third_identical_request_after_two_failures_does_not_500(
    client, published_protocol_id, prediction_upload_ref
):
    """CRITICAL fix (Task 17 review): `find_by_cache_key` had no `.limit(1)`,
    so the moment two rows shared a cache_key, every subsequent lookup raised
    `MultipleResultsFound` -> a raw 500. Reachable by a user typo'ing
    `structure_column` and retrying the *identical* (still-typo'd) request --
    each FAILED attempt shares its cache_key with the last (a FAILED Run is
    never a cache hit, so a retry always creates a fresh row under the same
    key), so two retries is all it takes. Measured over HTTP by the reviewer:
    202, 202, then (before the fix) 500 on the third identical request.

    Deliberately reuses the *same* bad `structure_column` on every attempt --
    a request with a different one has a different cache_key (Task 17's own
    TDD caught that), so it would never collide and never reach the bug.
    """
    first = await _predict(
        client, published_protocol_id, prediction_upload_ref, structure_column="does-not-exist"
    )
    assert first.status_code == 202, first.text
    first_run = await client.get(f"/api/v1/runs/{first.json()['id']}")
    assert first_run.json()["status"] == "failed"

    second = await _predict(
        client, published_protocol_id, prediction_upload_ref, structure_column="does-not-exist"
    )
    assert second.status_code == 202, second.text
    second_run = await client.get(f"/api/v1/runs/{second.json()['id']}")
    assert second_run.json()["status"] == "failed"
    assert second.json()["id"] != first.json()["id"]  # a fresh row, same cache_key

    third = await _predict(
        client, published_protocol_id, prediction_upload_ref, structure_column="does-not-exist"
    )
    assert third.status_code == 202, third.text  # not 500


async def test_running_a_draft_protocol_is_a_409(
    client, trained_protocol_id, prediction_upload_ref
):
    response = await _predict(client, trained_protocol_id, prediction_upload_ref)
    assert response.status_code == 409, response.text


async def test_predicting_against_an_unknown_protocol_is_a_404(client, prediction_upload_ref):
    response = await _predict(
        client, "11111111-1111-1111-1111-111111111111", prediction_upload_ref
    )
    assert response.status_code == 404, response.text


async def test_predicting_with_an_unknown_upload_ref_is_a_404(client, published_protocol_id):
    response = await _predict(
        client, published_protocol_id, "11111111-1111-1111-1111-111111111111"
    )
    assert response.status_code == 404, response.text


async def test_a_missing_run_id_is_a_404(client):
    response = await client.get(f"/api/v1/runs/{uuid.uuid4()}")
    assert response.status_code == 404, response.text


async def test_workspace_id_in_the_body_is_rejected(
    client, published_protocol_id, prediction_upload_ref
):
    response = await _predict(
        client,
        published_protocol_id,
        prediction_upload_ref,
        workspace_id="00000000-0000-0000-0000-000000000001",
    )
    assert response.status_code == 422, response.text


async def test_viewer_cannot_submit_a_prediction(
    viewer_client, published_protocol_id, prediction_upload_ref
):
    response = await _predict(viewer_client, published_protocol_id, prediction_upload_ref)
    assert response.status_code == 403, response.text


async def test_viewer_can_poll_and_read_results(
    client, viewer_client, published_protocol_id, prediction_upload_ref
):
    """Reading is not gated behind editor -- only submitting and cancelling."""
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]

    assert (await viewer_client.get(f"/api/v1/runs/{run_id}")).status_code == 200
    assert (await viewer_client.get(f"/api/v1/runs/{run_id}/results")).status_code == 200


async def test_runs_are_scoped_to_the_callers_workspace(
    client, other_workspace_client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]

    assert (await other_workspace_client.get(f"/api/v1/runs/{run_id}")).status_code == 404
    assert (await other_workspace_client.get(f"/api/v1/runs/{run_id}/results")).status_code == 404
    assert (await other_workspace_client.post(f"/api/v1/runs/{run_id}/cancel")).status_code == 404
    assert (await other_workspace_client.get("/api/v1/runs")).json()["items"] == []


async def test_a_protocol_from_another_workspace_is_not_runnable(
    other_workspace_client, published_protocol_id
):
    """Design intent: a published Protocol is a shared asset within its own
    workspace, not across tenants -- resolved the same way every other
    Protocol read is (Task 16's `test_protocols_are_scoped_to_the_callers_workspace`).
    """
    upload = await other_workspace_client.post(
        "/api/v1/datasets/uploads", files={"file": ("data.csv", _PREDICTION_CSV, "text/csv")}
    )
    assert upload.status_code == 201, upload.text

    response = await _predict(
        other_workspace_client, published_protocol_id, upload.json()["upload_ref"]
    )
    assert response.status_code == 404, response.text


async def test_unauthenticated_request_is_rejected(anonymous_client):
    assert (await anonymous_client.post("/api/v1/runs", json={})).status_code == 401
    assert (await anonymous_client.get(f"/api/v1/runs/{uuid.uuid4()}")).status_code == 401


async def test_results_for_a_training_run_is_a_404(client, csv_upload):
    """`/results` is a prediction-shaped resource; a training Run's id has none."""
    dataset_id = await _create_dataset(client, csv_upload)
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text

    results = await client.get(f"/api/v1/runs/{response.json()['id']}/results")
    assert results.status_code == 404, results.text


async def test_results_for_a_run_with_no_results_yet_is_a_409(
    client, session_factory, workspace_id
):
    run = Run(
        kind=RunKind.PREDICTION,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k",
    )
    await SqlAlchemyRunRepository(session_factory).add(run)

    response = await client.get(f"/api/v1/runs/{run.id}/results")
    assert response.status_code == 409, response.text


async def test_cancel_flips_a_pending_run_to_cancelled(client, session_factory, workspace_id):
    run = Run(
        kind=RunKind.PREDICTION,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k2",
    )
    await SqlAlchemyRunRepository(session_factory).add(run)

    response = await client.post(f"/api/v1/runs/{run.id}/cancel")
    assert response.status_code == 204, response.text
    assert response.content == b""

    polled = await client.get(f"/api/v1/runs/{run.id}")
    assert polled.json()["status"] == "cancelled"


async def test_cancelling_an_already_ready_run_is_a_409(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]

    response = await client.post(f"/api/v1/runs/{run_id}/cancel")
    assert response.status_code == 409, response.text


async def test_viewer_cannot_cancel(viewer_client, client, session_factory, workspace_id):
    run = Run(
        kind=RunKind.PREDICTION,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k3",
    )
    await SqlAlchemyRunRepository(session_factory).add(run)

    response = await viewer_client.post(f"/api/v1/runs/{run.id}/cancel")
    assert response.status_code == 403, response.text


async def test_paging_through_more_results_than_the_limit_terminates(
    client, published_protocol_id, csv_upload
):
    csv = b"smiles\n" + b"\n".join(smiles.encode() for smiles in _STRUCTURES[:6]) + b"\n"
    upload_ref = await csv_upload(csv)
    submitted = await _predict(client, published_protocol_id, upload_ref)
    run_id = submitted.json()["id"]

    seen: list[str] = []
    cursor: str | None = None
    pages = 0
    while True:
        url = f"/api/v1/runs/{run_id}/results?limit=2"
        if cursor is not None:
            url += f"&cursor={cursor}"
        response = await client.get(url)
        assert response.status_code == 200, response.text
        page = response.json()
        pages += 1
        seen.extend(item["structure"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages < 10, f"pagination did not terminate: {pages} pages, saw {seen}"

    assert pages == 3
    assert len(seen) == 6
    assert len(set(seen)) == 6


async def test_listing_runs_returns_both_kinds_newest_first(
    client, published_protocol_id, prediction_upload_ref
):
    """Training a Protocol and then predicting with it leaves two Runs. Both
    are listed, and the prediction -- created second -- comes back first."""
    await _predict(client, published_protocol_id, prediction_upload_ref)

    items = (await client.get("/api/v1/runs")).json()["items"]
    assert [item["kind"] for item in items] == ["prediction", "training"]


async def test_listing_runs_filters_by_kind(client, published_protocol_id, prediction_upload_ref):
    """A prediction Run is what a user browses; a training Run belongs to its
    Protocol's history. Filtering server-side keeps a client from paging
    through the wrong kind to assemble one screen."""
    await _predict(client, published_protocol_id, prediction_upload_ref)

    predictions = (await client.get("/api/v1/runs?kind=prediction")).json()["items"]
    assert [item["kind"] for item in predictions] == ["prediction"]

    trainings = (await client.get("/api/v1/runs?kind=training")).json()["items"]
    assert [item["kind"] for item in trainings] == ["training"]


async def test_an_unknown_run_kind_is_a_422_not_a_silent_unfiltered_list(client):
    """Silently ignoring an unrecognised filter would hand back every Run as
    though the filter had matched them all."""
    response = await client.get("/api/v1/runs?kind=generation")
    assert response.status_code == 422, response.text


async def test_listing_runs_pages_with_an_opaque_cursor(
    client, published_protocol_id, prediction_upload_ref
):
    await _predict(client, published_protocol_id, prediction_upload_ref)

    first = (await client.get("/api/v1/runs?limit=1")).json()
    assert len(first["items"]) == 1
    assert first["next_cursor"] is not None

    second = (await client.get(f"/api/v1/runs?limit=1&cursor={first['next_cursor']}")).json()
    assert len(second["items"]) == 1
    assert second["items"][0]["id"] != first["items"][0]["id"]
    assert second["next_cursor"] is None


async def test_results_carry_their_row_id(client, published_protocol_id, prediction_upload_ref):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(f"/api/v1/runs/{submitted.json()['id']}/results")
    assert response.status_code == 200, response.text
    assert [item["row_id"] for item in response.json()["items"]] == [0, 1, 2]


async def test_results_can_be_sorted_by_a_readout(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]
    protocol = (await client.get(f"/api/v1/protocols/{published_protocol_id}")).json()
    readout = protocol["readouts"][0]["name"]

    ascending = await client.get(f"/api/v1/runs/{run_id}/results?sort_by={readout}&sort_dir=asc")
    descending = await client.get(f"/api/v1/runs/{run_id}/results?sort_by={readout}&sort_dir=desc")
    assert ascending.status_code == 200, ascending.text
    assert descending.status_code == 200, descending.text

    def values(response):
        return [item["readouts"][readout]["value"] for item in response.json()["items"]]

    assert values(ascending) == sorted(values(ascending))
    assert values(descending) == list(reversed(values(ascending)))
    # Reordering must not renumber: the same compound keeps the same handle,
    # because that handle is what POST /collections stores.
    assert {item["row_id"] for item in descending.json()["items"]} == {0, 1, 2}


async def test_results_can_be_filtered_by_a_range(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    run_id = submitted.json()["id"]
    unfiltered = await client.get(f"/api/v1/runs/{run_id}/results")
    assert len(unfiltered.json()["items"]) == 3

    # No compound can be more than perfectly applicable, so this floor empties
    # the view -- which must be a well-formed empty page, not a 4xx.
    response = await client.get(
        f"/api/v1/runs/{run_id}/results?filters=" + quote('{"applicability": {"min": 1.1}}')
    )
    assert response.status_code == 200, response.text
    assert response.json()["items"] == []
    assert response.json()["next_cursor"] is None


async def test_an_unknown_filter_column_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(
        f"/api/v1/runs/{submitted.json()['id']}/results?filters="
        + quote('{"not_a_column": {"min": 1}}')
    )
    assert response.status_code == 422, response.text


async def test_malformed_filter_json_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(f"/api/v1/runs/{submitted.json()['id']}/results?filters=not-json")
    assert response.status_code == 422, response.text


async def test_a_filter_with_no_bounds_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    response = await client.get(
        f"/api/v1/runs/{submitted.json()['id']}/results?filters=" + quote('{"applicability": {}}')
    )
    assert response.status_code == 422, response.text


async def test_an_unknown_sort_direction_is_a_422(
    client, published_protocol_id, prediction_upload_ref
):
    submitted = await _predict(client, published_protocol_id, prediction_upload_ref)
    protocol = (await client.get(f"/api/v1/protocols/{published_protocol_id}")).json()
    response = await client.get(
        f"/api/v1/runs/{submitted.json()['id']}/results"
        f"?sort_by={protocol['readouts'][0]['name']}&sort_dir=sideways"
    )
    assert response.status_code == 422, response.text

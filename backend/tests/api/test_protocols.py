"""End-to-end tests for the Protocol routes: train, list, read, publish and read
the Scorecard back.

Training runs through the real `InlineEnqueuer` (see the `app` fixture in
`tests/api/conftest.py`) -- by the time `POST /api/v1/protocols` returns, the
Run it names has already finished and the Protocol row it produced already
exists, even though the response body itself still shows the pre-enqueue
`pending` snapshot (see `TrainProtocol.__call__`: it returns the same in-memory
Run object it constructed, never the reloaded one). That is what makes it safe
for `trained_protocol_id` below to look the freshly trained Protocol up via
`GET /api/v1/protocols` immediately after the POST resolves, with no polling.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import polars as pl
import pytest
import pytest_asyncio

from daikonstudio.application.catalog.derive_readouts import derive_readouts
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.data.split import SplitStrategy
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)

# Twenty distinct compounds -- the same shape test_train_protocol.py uses and for
# the same reason: an 80/10/10 random split over four rows can leave a single-row
# test partition, which makes R2 (and this endpoint's residual list) degenerate
# and prints a warning. Twenty leaves a real 16/2/2 split.
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


def _csv() -> bytes:
    rows = "\n".join(f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_STRUCTURES))
    return f"smiles,y\n{rows}\n".encode()


async def _create_dataset(client, csv_upload) -> str:
    upload_ref = await csv_upload(_csv())
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


@pytest_asyncio.fixture
async def dataset_id(client, csv_upload) -> str:
    return await _create_dataset(client, csv_upload)


async def _train(client, dataset_id: str, **overrides: object) -> object:
    body: dict[str, object] = {
        "name": "solubility rf",
        "dataset_id": dataset_id,
        "engine_id": "ecfp4-randomforest",
        "conditions": {},
    }
    body.update(overrides)
    return await client.post("/api/v1/protocols", json=body)


@pytest_asyncio.fixture
async def trained_protocol_id(client, dataset_id) -> AsyncIterator[str]:
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text

    listing = await client.get("/api/v1/protocols")
    items = listing.json()["items"]
    assert len(items) == 1, items
    yield items[0]["id"]


async def test_training_request_returns_202_with_a_run(client, dataset_id):
    response = await _train(client, dataset_id)
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "pending"


async def test_the_trained_protocol_starts_as_an_unlocked_draft(client, trained_protocol_id):
    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}")).json()
    assert body["status"] == "draft"
    assert body["is_locked"] is False


async def test_a_readout_carries_the_datasets_unit_and_direction_over_http(
    client, trained_protocol_id
):
    """The whole point of deriving Readouts from the TargetSpec is that a
    predicted value arrives in the same unit, and pointing the same way, as a
    measured one. That pairing has to survive the HTTP boundary too -- it was
    dropped at four separate boundaries during the backend build, and a client
    rendering a bare float has no way to notice the fifth.

    The dataset these fixtures build declares `unit="logS", direction="high"`.
    """
    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}")).json()

    assert body["readouts"] == [
        {
            "name": "y",
            "type": "numeric",
            "unit": "logS",
            "direction": "high",
            "description": "Predicted y",
        }
    ]


async def test_publish_locks_the_protocol(client, trained_protocol_id):
    response = await client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    assert response.status_code == 204, response.text
    assert response.content == b""

    body = (await client.get(f"/api/v1/protocols/{trained_protocol_id}")).json()
    assert body["is_locked"] is True
    assert body["status"] == "published"
    assert body["published_at"] is not None


async def test_publishing_twice_returns_423_locked(client, trained_protocol_id):
    await client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    response = await client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    assert response.status_code == 423, response.text


async def test_scorecard_exposes_the_baseline_comparison(client, trained_protocol_id):
    card = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/scorecard")).json()
    assert card["baseline_engine_id"] == "ecfp4-randomforest"
    assert card["primary_metric"] in {"rmse", "mcc"}
    assert len(card["worst_rows"]) <= 20


async def test_scorecard_carries_the_targets_unit_direction_and_split_strategy(
    client, trained_protocol_id
):
    """I2 (whole-branch review, Important): `rmse: 0.61` reads identically
    whether the target was nM or uM, and a consumer had no way to tell which
    split produced a metric except a two-null inference
    (`random_split_metrics is None and random_split_unavailable is None`).
    The Dataset behind `trained_protocol_id` declares `unit="logS"`,
    `direction="high"`, and a RANDOM split -- all three must survive onto the
    Scorecard response, the fourth and last place a predicted number reaches
    a consumer without them (prediction results and both export formats
    already carry unit/direction)."""
    card = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/scorecard")).json()
    assert card["unit"] == "logS"
    assert card["direction"] == "high"
    assert card["split_strategy"] == "random"


async def test_scorecard_says_the_model_is_the_baseline_rather_than_faking_a_comparison(
    client, trained_protocol_id
):
    """ecfp4-randomforest with default conditions IS the baseline: the response
    must say so, not present the same numbers twice as an independent win."""
    card = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/scorecard")).json()
    assert card["baseline_is_self"] is True
    assert card["baseline_metrics"] == card["metrics"]


async def test_scorecard_reports_no_optimism_gap_for_an_already_random_split(
    client, trained_protocol_id
):
    """The Dataset behind `trained_protocol_id` is randomly split, so there is
    nothing to compare against -- `None`, not a fabricated zero-gap."""
    card = (await client.get(f"/api/v1/protocols/{trained_protocol_id}/scorecard")).json()
    assert card["random_split_metrics"] is None
    assert card["random_split_unavailable"] is None


async def test_a_missing_scorecard_id_is_a_404_not_a_500(client):
    response = await client.get("/api/v1/protocols/11111111-1111-1111-1111-111111111111/scorecard")
    assert response.status_code == 404, response.text


async def test_a_protocol_row_without_a_scorecard_blob_is_a_404_not_a_500(
    client, session_factory, workspace_id
):
    """Today's only creation path (training) writes the scorecard blob before the
    Protocol row -- see `train_protocol.py`'s "blobs first, Protocol row last"
    ordering -- so this cannot happen through the API yet. It is defended anyway:
    a future path that breaks that ordering, or an operator deleting a blob out
    from under a live row, must degrade to an honest 404, not a 500 mid-request.
    """
    protocol = InSilicoProtocol(
        workspace_id=workspace_id,
        name="orphaned",
        dataset_id=uuid.uuid4(),
        engine_id="ecfp4-randomforest",
        artifact_uri="file:///nowhere/model.joblib",
        readouts=derive_readouts(
            TargetSpec(column="y", kind=TargetKind.NUMERIC, unit=None, direction=Direction.HIGH),
            TaskType.REGRESSION,
        ),
        conditions={},
    )
    await SqlAlchemyProtocolRepository(session_factory).add(protocol)

    response = await client.get(f"/api/v1/protocols/{protocol.id}/scorecard")
    assert response.status_code == 404, response.text


async def test_publish_requires_the_editor_role(viewer_client, trained_protocol_id):
    response = await viewer_client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    assert response.status_code == 403, response.text


async def test_a_draft_protocol_is_visible_to_its_own_workspace(client, trained_protocol_id):
    """Drafts are not hidden from the workspace that trained them -- otherwise a
    freshly trained model would be invisible until someone else published it."""
    listed = (await client.get("/api/v1/protocols")).json()["items"]
    assert [item["status"] for item in listed] == ["draft"]


async def test_protocols_are_scoped_to_the_callers_workspace(
    client, other_workspace_client, trained_protocol_id
):
    assert (
        await other_workspace_client.get(f"/api/v1/protocols/{trained_protocol_id}")
    ).status_code == 404
    assert (await other_workspace_client.get("/api/v1/protocols")).json()["items"] == []
    assert (
        await other_workspace_client.post(f"/api/v1/protocols/{trained_protocol_id}/publish")
    ).status_code == 404


async def test_workspace_id_in_the_training_body_is_rejected(client, dataset_id):
    response = await _train(
        client, dataset_id, workspace_id="00000000-0000-0000-0000-000000000001"
    )
    assert response.status_code == 422, response.text


async def test_an_overlong_protocol_name_is_a_422_not_a_202_that_fails_later(client, dataset_id):
    """I3 (whole-branch review, Important): `name` has no `max_length` against
    `InSilicoProtocolModel.name`'s `String(256)` column. Worse than the same
    finding on the dataset/collection routes: the Protocol row is inserted by
    the *worker*, not this route, so an over-long name used to return 202 and
    let the request pay for three real model fits (the chosen engine, the
    mandatory baseline, and -- on a scaffold split -- the optimism-gap
    comparison) before dying on the insert, orphaning the artifact and
    scorecard-inputs blobs it had already written. Caught here instead, so
    the request never returns 202 at all."""
    response = await _train(client, dataset_id, name="x" * 257)
    assert response.status_code == 422, response.text


async def test_training_against_an_unknown_dataset_is_a_404(client):
    response = await _train(client, "11111111-1111-1111-1111-111111111111")
    assert response.status_code == 404, response.text


async def test_training_against_an_unknown_engine_is_a_404(client, dataset_id):
    """Important (Task 16 review): `engine_id` is a registry-membership check
    with exactly one possible answer -- unlike conditions, there is nothing
    the worker could resolve differently, so a bad id fails synchronously
    instead of returning 202 for a Run that provably cannot succeed."""
    response = await _train(client, dataset_id, engine_id="does-not-exist")
    assert response.status_code == 404, response.text


async def test_viewer_cannot_start_training(viewer_client, dataset_id):
    response = await _train(viewer_client, dataset_id)
    assert response.status_code == 403, response.text


async def test_scorecard_response_explains_an_optimism_gap_metric_that_is_undefined(
    client, csv_upload, monkeypatch: pytest.MonkeyPatch
):
    """CRITICAL fix (Task 16 review): `train_protocol.py` used to discard the
    reason a random-split metric was undefined, so the HTTP response either
    left it unexplained (when the Dataset's own split was fine) or -- worse --
    let a client attribute the wrong partition's reason to it via
    `metrics_undefined`. Rigs the random-split reshuffle to collapse its test
    rows onto one class while the scaffold split, trained for real, keeps
    both -- reproducing the reviewer's exact scenario end to end through the
    actual endpoint.
    """
    import daikonstudio.application.execution.train_protocol as module

    real_assign_split = module.assign_split

    def collapse_random_test_split_to_one_class(frame, structure_column, spec, normalizer):  # type: ignore[no-untyped-def]
        random_frame = real_assign_split(frame, structure_column, spec, normalizer)
        if spec.strategy is not SplitStrategy.RANDOM:
            return random_frame
        return random_frame.with_columns(
            pl.when(pl.col("split") == "test").then(0.0).otherwise(pl.col("y")).alias("y")
        )

    monkeypatch.setattr(module, "assign_split", collapse_random_test_split_to_one_class)

    values = [float(index % 2) for index in range(len(_STRUCTURES))]
    values[1] = 0.0  # CCN: seed=7's real scaffold split puts this in `test` -- balance it
    pairs = zip(_STRUCTURES, values, strict=True)
    rows = "\n".join(f"{smiles},{value}" for smiles, value in pairs)
    upload_ref = await csv_upload(f"smiles,y\n{rows}\n".encode())
    dataset_response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "binary-scaffold",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "target": {"column": "y", "kind": "binary"},
            "split": {"strategy": "scaffold", "seed": 7},
        },
    )
    assert dataset_response.status_code == 201, dataset_response.text

    train_response = await _train(client, dataset_response.json()["id"])
    assert train_response.status_code == 202, train_response.text
    protocol_id = (await client.get("/api/v1/protocols")).json()["items"][0]["id"]

    card = (await client.get(f"/api/v1/protocols/{protocol_id}/scorecard")).json()
    # The scaffold split's own test partition has both classes for real --
    # nothing to explain on that side.
    assert card["metrics_undefined"] is None
    assert all(value is not None for value in card["metrics"].values())
    # The random-split comparison collapsed to one class: undefined, and now
    # carrying its own reason rather than a bare null or the wrong partition's.
    assert card["random_split_metrics"] is not None
    assert all(value is None for value in card["random_split_metrics"].values())
    assert card["random_split_unavailable"] is None
    assert card["random_split_metrics_undefined"] is not None
    assert set(card["random_split_metrics_undefined"]) == set(card["random_split_metrics"])
    assert "test split" in card["random_split_metrics_undefined"]["mcc"]

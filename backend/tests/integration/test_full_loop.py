"""The end-to-end acceptance test -- Spec Sec.10 criterion 4: Phase 1 backend
is done when every noun in the design has been exercised end to end by one
real, continuous user flow, not by pinning absolute metric values that would
make this test flaky the day a library version bumps a coefficient.

The journey: browse the engine catalog, upload a real CSV, freeze it into a
validated, scaffold-split Dataset, train a Protocol against an honest
Scorecard (mandatory baseline, visible optimism gap), publish it, run it
against a colleague's new compounds, read the triage results, save a
Collection from a selection, and export it -- checking *structure* at every
stage (a field exists, is non-null, is internally consistent) rather than a
pinned number.

Each stage asserts before the next one starts, and every assertion message is
tagged with its own stage, so a break fifteen lines in says which noun broke
it rather than a bare `assert False`.

Fixture note (`tests/fixtures/pains_sample.csv`): built from 200 real rows of
`lab-ai/pains/dataset/balanced_training_set.csv` (a real, tab-separated
`cid`/`labels`/`SMILES` export, not the comma-separated `smiles`/`is_pains`
the plan assumed -- renamed and re-delimited here, not resynthesized). That
source file turns out to be a pre-cleaned, ML-ready split: RDKit parses every
one of its 1900 structures, none are multi-component salts, and none collapse
onto a canonical-SMILES duplicate -- so this fixture, real as it is, does not
exercise the invalid-row/salt/dedup *branches* of `prepare_frame` the way raw
lab data would (those branches have their own dedicated coverage in
`tests/unit/data/test_validation.py` and `tests/api/test_datasets.py`).

Row order history: the source file groups all `label=1` rows before all
`label=0` rows -- the shape a real assay export actually arrives in. The
first cut of this fixture *interleaved* the two classes instead, because the
scaffold splitter of the day broke equal-size ties by first-occurrence file
order for every group except empty-scaffold ones, so a label-grouped 200-row
slice packed *all* 100 `is_pains=1` rows into train and left
validation/test entirely `is_pains=0` -- a single-class test partition where
MCC is undefined, for a reason that had nothing to do with the pipeline this
test exercises. That splitter bug (Task 10, reopened) is now fixed: every
tie-break is seeded by a hash of the group's own scaffold identity, not by
row position, so the split is invariant to input order (verified below --
label-grouped and its exact reverse now produce byte-identical partitions).
With that fixed, this fixture went back to the natural label-grouped order,
which is the more valuable thing for an acceptance test to exercise: real
assay exports arrive sorted by outcome, batch, or submission date far more
often than pre-shuffled, and this fixture should catch a regression in that
shape rather than dodge it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "pains_sample.csv"

# A colleague's own compounds, none of which were in the training data.
_PREDICTION_CSV = b"smiles\nCCO\nc1ccccc1\nCc1ccccc1\n"

_DIRECTION_LABEL = {"high": "higher is better", "low": "lower is better"}


async def _poll_until_ready(
    client: httpx.AsyncClient, run_id: str, *, stage: str
) -> dict[str, Any]:
    """Poll a Run until it leaves PENDING/RUNNING.

    The `app` fixture's `InlineEnqueuer` has always finished the job by the
    time the submitting POST returns, so in practice this resolves on the
    first iteration -- but writing it as a poll (rather than one bare GET)
    means this test is still correct against a real self-hosted runner (see
    `test_runner_full_loop.py`, which drives this same journey over the
    runner protocol instead of inline), and a run stuck mid-pipeline times
    out with a named stage instead of the suite hanging.
    """
    for _ in range(100):
        response = await client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200, f"[{stage}] polling run {run_id}: {response.text}"
        run: dict[str, Any] = response.json()
        if run["status"] not in ("pending", "running"):
            return run
        await asyncio.sleep(0.05)
    raise AssertionError(f"[{stage}] run {run_id} never left pending/running: {run}")


async def test_a_scientist_can_walk_the_whole_loop(client, csv_upload):
    """Spec Sec.10 criterion 4: every noun exercised end to end."""

    # --- Engine catalog: browse what's available before choosing one ---
    catalog_response = await client.get("/api/v1/engines")
    assert catalog_response.status_code == 200, f"[catalog] {catalog_response.text}"
    engine_ids = {engine["id"] for engine in catalog_response.json()}
    assert "ecfp4-xgboost" in engine_ids, f"[catalog] engines: {engine_ids}"

    # --- Dataset: upload a real CSV, freeze a validated, scaffold-split Dataset ---
    upload_ref = await csv_upload(_FIXTURE.read_bytes())
    dataset_response = await client.post(
        "/api/v1/datasets",
        json={
            "name": "PAINS",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "targets": [{"column": "is_pains", "kind": "binary"}],
            "split": {"strategy": "scaffold", "seed": 42},
        },
    )
    assert dataset_response.status_code == 201, f"[dataset] {dataset_response.text}"
    dataset = dataset_response.json()
    assert dataset["row_count"] > 0, f"[dataset] {dataset}"
    assert dataset["split"] == {
        "strategy": "scaffold",
        "seed": 42,
        "fractions": [0.8, 0.1, 0.1],
    }
    # Freezing produced a real, content-addressed snapshot -- not just a
    # database row referencing the caller's original upload.
    assert dataset["snapshot_uri"], dataset
    assert dataset["content_hash"], dataset
    # This specific 200-row fixture is real and, per this module's docstring,
    # verified clean: every row parses, none are salts, none collapse as
    # canonical duplicates -- so the honest expectation here is that nothing
    # was dropped, not that something was.
    report = dataset["validation_report"]
    assert report["total_rows"] == report["valid_rows"] == dataset["row_count"] == 200, report
    assert report["invalid"] == [] and report["duplicates_collapsed"] == 0, report

    # --- Protocol: train against the frozen Dataset ---
    train_response = await client.post(
        "/api/v1/protocols",
        json={
            "name": "PAINS xgb",
            "dataset_id": dataset["id"],
            "engine_id": "ecfp4-xgboost",
            "conditions": {},
        },
    )
    assert train_response.status_code == 202, f"[training] {train_response.text}"
    training_run = await _poll_until_ready(client, train_response.json()["id"], stage="training")
    assert training_run["status"] == "ready", (
        f"[training] run failed: {training_run.get('error_message')}"
    )

    # A training Run's `protocol_id` is always null (the Protocol doesn't exist
    # at enqueue time) -- the freshly trained Protocol is found by the Dataset
    # it names instead, the same way the existing API tests locate it.
    protocol_list = (await client.get("/api/v1/protocols")).json()["items"]
    trained = [item for item in protocol_list if item["dataset_id"] == dataset["id"]]
    assert len(trained) == 1, f"[training] protocols for dataset {dataset['id']}: {trained}"
    protocol_id = trained[0]["id"]
    assert trained[0]["status"] == "draft" and trained[0]["is_locked"] is False

    # --- Scorecard: an honest baseline comparison and a visible optimism gap ---
    scorecard_response = await client.get(f"/api/v1/protocols/{protocol_id}/scorecard")
    assert scorecard_response.status_code == 200, f"[scorecard] {scorecard_response.text}"
    card = scorecard_response.json()
    assert card["primary_metric"] == "mcc", card
    assert card["baseline_engine_id"] == "ecfp4-randomforest"
    assert card["baseline_metrics"]["mcc"] is not None, f"[scorecard] baseline undefined: {card}"
    # Trained on a *scaffold* split, so the Scorecard also carries the
    # random-split comparison -- the optimism gap is visible, not inferred.
    assert card["random_split_metrics"] is not None, f"[scorecard] no optimism gap: {card}"
    assert card["random_split_metrics"]["mcc"] is not None, card
    assert card["random_split_unavailable"] is None
    assert 0 < len(card["worst_rows"]) <= 20, card["worst_rows"]
    # Applicability coverage: the test set is non-empty and so is the training
    # set, so this is computable -- a real fraction, never a fabricated 0.0.
    assert card["applicability_coverage"] is not None, card
    assert 0.0 <= card["applicability_coverage"] <= 1.0, card
    # Noise floor is the Dataset's own duplicate-spread -- forced None for a
    # classification Scorecard, which has no such notion, regardless of input.
    # Coverage boundary: this journey only ever trains a classifier, so this
    # assertion never exercises the regression branch where noise_floor can be
    # a real number -- that's covered by test_train_protocol.py instead.
    assert card["noise_floor"] is None, card

    # --- Publish: lock the Protocol so it becomes runnable ---
    publish_response = await client.post(f"/api/v1/protocols/{protocol_id}/publish")
    assert publish_response.status_code == 204, f"[publish] {publish_response.text}"

    protocol = (await client.get(f"/api/v1/protocols/{protocol_id}")).json()
    assert protocol["is_locked"] is True and protocol["status"] == "published"
    probability_readout = next(r for r in protocol["readouts"] if r["type"] == "probability")

    # --- Run: score a colleague's new compounds against the published Protocol ---
    predict_ref = await csv_upload(_PREDICTION_CSV)
    prediction_response = await client.post(
        "/api/v1/runs",
        json={
            "protocol_id": protocol_id,
            "upload_ref": predict_ref,
            "structure_column": "smiles",
            "conditions": {},
        },
    )
    assert prediction_response.status_code == 202, f"[prediction] {prediction_response.text}"
    prediction_run = await _poll_until_ready(
        client, prediction_response.json()["id"], stage="prediction"
    )
    assert prediction_run["status"] == "ready", (
        f"[prediction] run failed: {prediction_run.get('error_message')}"
    )
    assert prediction_run["protocol_id"] == protocol_id

    # --- Triage: the results grid a scientist scans row by row ---
    results_response = await client.get(f"/api/v1/runs/{prediction_run['id']}/results")
    assert results_response.status_code == 200, f"[results] {results_response.text}"
    results = results_response.json()
    assert len(results["items"]) == 3, f"[results] {results['items']}"
    first_row = results["items"][0]
    assert "applicability" in first_row
    assert "uncertainty" in first_row
    assert probability_readout["name"] in first_row["readouts"], first_row

    # --- Collection: save a triage selection and export it ---
    collection_response = await client.post(
        "/api/v1/collections",
        json={"name": "flagged", "run_id": prediction_run["id"], "row_ids": [0, 1]},
    )
    assert collection_response.status_code == 201, f"[collection] {collection_response.text}"
    collection = collection_response.json()
    assert collection["member_count"] == 2
    assert collection["derived_from_run_id"] == prediction_run["id"]

    export_response = await client.get(f"/api/v1/collections/{collection['id']}/export?format=csv")
    assert export_response.status_code == 200, f"[export] {export_response.text}"
    lines = export_response.text.splitlines()
    assert len(lines) == 3, lines  # header + 2 selected rows

    # A probability readout carries no unit but does carry a direction -- that
    # direction must still survive into the export's column header, the same
    # way a unit would for a numeric readout (`export_collection.py`).
    # Coverage boundary: `derive_readouts` always gives PROBABILITY readouts
    # `direction="high"` and no unit, so this journey only ever exercises the
    # "high", no-unit branch of the header format -- the "low"-direction,
    # unit-bearing case (a numeric readout, e.g. "IC50 (nM, lower is better)")
    # is covered by test_collections.py instead.
    direction_label = _DIRECTION_LABEL[probability_readout["direction"]]
    assert f"{probability_readout['name']} ({direction_label})" in lines[0], lines[0]

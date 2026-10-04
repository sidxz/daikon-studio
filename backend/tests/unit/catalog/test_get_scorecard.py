"""Regression coverage for Task 16 review Important 4: `GetScorecard` must
run `build_scorecard` off the event loop.

`build_scorecard` computes Murcko scaffolds and an O(test x train) Tanimoto
matrix -- measured at 0.86s for 16k train/2k test structures, which blocks
every other request on the process for the duration if run inline. Fakes
everything around `GetScorecard` (no real Postgres, blob store, or RDKit
needed) so the only thing under test is which thread `build_scorecard` runs on.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import replace
from typing import Any

from daikonstudio.application.catalog import get_scorecard as module
from daikonstudio.application.catalog.get_scorecard import GetScorecard, GetScorecardQuery
from daikonstudio.application.execution.train_protocol import ScorecardInputs, TargetInputs
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.execution.scorecard import Scorecard
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from tests.fakes.auth import FakeAuth


class _FakeProtocol:
    def __init__(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None:
        self.workspace_id = workspace_id
        self.id = protocol_id
        self.dataset_id = uuid.uuid4()
        self.readouts = (
            Readout(
                name="y", type=ReadoutType.NUMERIC, unit=None, direction=None, description="d"
            ),
        )


class _FakeProtocols:
    def __init__(self, protocol: _FakeProtocol) -> None:
        self._protocol = protocol

    async def get(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> _FakeProtocol | None:
        return self._protocol if protocol_id == self._protocol.id else None


class _FakeStore:
    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    def get_bytes(self, key: str) -> bytes:
        return self._blob


def _target(column: str, **overrides: Any) -> TargetInputs:
    fields: dict[str, Any] = dict(
        column=column,
        task="regression",
        metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.1},
        actual=[1.0, 2.0],
        predicted=[1.1, 1.9],
        prediction_kind="value",
        baseline_metrics={"rmse": 0.7, "mae": 0.6, "r2": 0.0},
        random_split_metrics=None,
        random_split_metrics_undefined=None,
        metrics_undefined=None,
        duplicate_spread=None,
        target_unit=None,
        target_direction=None,
    )
    fields.update(overrides)
    return TargetInputs(**fields)


def _inputs(protocol_id: uuid.UUID) -> ScorecardInputs:
    return ScorecardInputs(
        protocol_id=str(protocol_id),
        run_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        engine_id="ecfp4-randomforest",
        conditions={},
        structures=["CCO", "CCN"],
        train_structures=["CCC"],
        baseline_engine_id="ecfp4-randomforest",
        baseline_is_self=True,
        random_split_unavailable=None,
        split_strategy="random",
        targets=[_target("y", target_unit="nM", target_direction="low")],
    )


class _NoDatasets:
    async def get(self, workspace_id, dataset_id):
        return None


async def test_build_scorecard_runs_off_the_main_thread(monkeypatch) -> None:
    auth = FakeAuth()
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    inputs = _inputs(protocol.id)
    seen: list[threading.Thread] = []

    def fake_build_scorecard(**kwargs: object) -> Scorecard:
        seen.append(threading.current_thread())
        return Scorecard(
            target="y",
            joint_model=False,
            primary_metric="rmse",
            prediction_kind="value",
            metrics={},
            validation_metrics=None,
            metrics_undefined=None,
            engine_id="ecfp4-randomforest",
            conditions={},
            baseline_engine_id="x",
            baseline_conditions={},
            baseline_metrics={},
            baseline_is_self=True,
            random_split_metrics=None,
            random_split_unavailable=None,
            random_split_metrics_undefined=None,
            noise_floor=None,
            worst_rows=[],
            applicability_coverage=None,
            target_unit="nM",
            target_direction="low",
            split_strategy="random",
            parity=[],
            parity_sampled_from=None,
            residual_histogram=None,
            error_by_similarity=[],
            scaffold_errors=[],
            calibration=[],
        )

    monkeypatch.setattr(module, "build_scorecard", fake_build_scorecard)

    use_case = GetScorecard(
        _FakeProtocols(protocol),
        _FakeStore(inputs.to_json()),
        normalizer=RdkitStructureNormalizer(),
        datasets=_NoDatasets(),
    )
    result = await use_case(GetScorecardQuery(protocol_id=protocol.id), auth)

    assert len(result.unwrap()) == 1
    assert len(seen) == 1
    assert seen[0] is not threading.main_thread()


async def _scorecards_from_blob(
    workspace_id: uuid.UUID, protocol_id: uuid.UUID, blob: bytes
) -> list[Scorecard]:
    protocol = _FakeProtocol(workspace_id, protocol_id)
    use_case = GetScorecard(
        _FakeProtocols(protocol), _FakeStore(blob), RdkitStructureNormalizer(), _NoDatasets()
    )
    result = await use_case(
        GetScorecardQuery(protocol_id=protocol_id), auth=FakeAuth(workspace_id=workspace_id)
    )
    return result.unwrap()


async def _scorecards(
    workspace_id: uuid.UUID, protocol_id: uuid.UUID, inputs: ScorecardInputs
) -> list[Scorecard]:
    return await _scorecards_from_blob(workspace_id, protocol_id, inputs.to_json())


async def test_one_scorecard_per_target_and_a_degenerate_one_does_not_sink_its_siblings(
    monkeypatch,
) -> None:
    workspace_id, protocol_id = uuid.uuid4(), uuid.uuid4()
    inputs = replace(
        _inputs(protocol_id),
        targets=[
            _target(
                "active",
                task="binary_classification",
                prediction_kind="probability",
                metrics={"mcc": None, "balanced_accuracy": None, "auroc": None, "auprc": None},
                baseline_metrics={
                    "mcc": None,
                    "balanced_accuracy": None,
                    "auroc": None,
                    "auprc": None,
                },
                metrics_undefined={
                    "mcc": "Undefined: all test-set compounds have the same 'active' value."
                },
                actual=[0.0, 0.0],
                predicted=[0.2, 0.3],
            ),
            _target("solubility"),
        ],
    )
    cards = await _scorecards(workspace_id, protocol_id, inputs)
    assert [card.target for card in cards] == ["active", "solubility"]
    assert cards[0].metrics_undefined is not None
    assert cards[1].metrics["rmse"] == 0.5 and cards[1].metrics_undefined is None


async def test_the_test_set_chemistry_is_computed_once_for_every_target(monkeypatch) -> None:
    calls = 0
    real = module.held_out_chemistry

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "held_out_chemistry", counting)
    workspace_id, protocol_id = uuid.uuid4(), uuid.uuid4()
    inputs = replace(_inputs(protocol_id), targets=[_target("a"), _target("b"), _target("c")])
    assert len(await _scorecards(workspace_id, protocol_id, inputs)) == 3
    assert calls == 1


async def test_a_blob_written_before_targets_could_be_several_is_one_scorecard() -> None:
    workspace_id, protocol_id = uuid.uuid4(), uuid.uuid4()
    legacy = json.dumps(
        {
            "protocol_id": str(protocol_id),
            "run_id": "r",
            "dataset_id": "d",
            "engine_id": "ecfp4-xgboost",
            "task": "regression",
            "conditions": {},
            "metrics": {"rmse": 0.5},
            "actual": [1.0, 2.0],
            "predicted": [1.1, 1.9],
            "prediction_kind": "value",
            "structures": ["CCO", "CCN"],
            "train_structures": ["CCC"],
            "baseline_engine_id": "ecfp4-randomforest",
            "baseline_metrics": {"rmse": 0.7},
            "baseline_is_self": False,
            "random_split_metrics": None,
            "random_split_unavailable": None,
            "random_split_metrics_undefined": None,
            "metrics_undefined": None,
            "duplicate_spread": None,
            "target_unit": None,
            "target_direction": None,
            "split_strategy": "random",
        }
    ).encode()
    cards = await _scorecards_from_blob(workspace_id, protocol_id, legacy)
    assert [card.target for card in cards] == ["y"]  # named after the protocol's readout

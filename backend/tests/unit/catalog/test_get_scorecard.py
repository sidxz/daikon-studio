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

import pytest

from daikonstudio.application.catalog import get_scorecard as module
from daikonstudio.application.catalog.get_scorecard import (
    GetScorecard,
    GetScorecardQuery,
    replicate_summary,
)
from daikonstudio.application.execution.build_scorecard import HeldOutChemistry
from daikonstudio.application.execution.train_protocol import (
    ScorecardInputs,
    TargetInputs,
    scorecard_chemistry_key,
)
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.execution.scorecard import Scorecard
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from tests.fakes.auth import FakeAuth
from tests.fakes.protocol_access import FakeProtocolAccess


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
    """The inputs blob under its own key; anything else only once written."""

    def __init__(self, blob: bytes) -> None:
        self._blob = blob
        self.written: dict[str, bytes] = {}

    def get_bytes(self, key: str) -> bytes:
        if key.endswith("scorecard-inputs.json"):
            return self._blob
        if key in self.written:
            return self.written[key]
        raise FileNotFoundError(key)

    def put_bytes(self, key: str, data: bytes) -> str:
        self.written[key] = data
        return key


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


async def test_ranked_compounds_resolve_identifiers_even_outside_the_largest_errors(monkeypatch):
    auth = FakeAuth(workspace_role="admin")
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    structures = ["C" * (i + 1) for i in range(60)]
    inputs = replace(
        _inputs(protocol.id),
        structures=structures,
        targets=[
            _target(
                "y",
                predicted=list(range(60)),
                actual=[i + 1000 if 20 <= i < 40 else i for i in range(60)],
            )
        ],
    )
    store = _FakeStore(inputs.to_json())
    store.put_bytes(
        scorecard_chemistry_key(auth.workspace_id, protocol.id),
        HeldOutChemistry(similarities=[0.5] * 60, scaffolds=[""] * 60).to_json(),
    )

    class Datasets:
        async def get(self, workspace_id, dataset_id):
            return object()

    def identifiers(store, dataset, wanted):
        # Largest errors occupy only the middle third of this population.
        assert wanted == set(structures)
        return {structure: f"compound-{i}" for i, structure in enumerate(structures)}

    monkeypatch.setattr(module, "read_compound_ids", identifiers)
    service = GetScorecard(
        _FakeProtocols(protocol),
        store,
        RdkitStructureNormalizer(),
        Datasets(),
        FakeProtocolAccess(),
    )
    result = (await service(GetScorecardQuery(protocol_id=protocol.id), auth)).unwrap()[0]
    assert result.ranked_high[0].compound_id == "compound-59"
    assert result.ranked_low[0].compound_id == "compound-0"
    assert {r.compound_id for r in result.worst_rows} == {f"compound-{i}" for i in range(20, 40)}


async def test_build_scorecard_runs_off_the_main_thread(monkeypatch) -> None:
    auth = FakeAuth(workspace_role="admin")  # admins see every protocol
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
        access=FakeProtocolAccess(),
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
        _FakeProtocols(protocol),
        _FakeStore(blob),
        RdkitStructureNormalizer(),
        _NoDatasets(),
        FakeProtocolAccess(),
    )
    result = await use_case(
        GetScorecardQuery(protocol_id=protocol_id),
        auth=FakeAuth(workspace_id=workspace_id, workspace_role="admin"),
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


def _counting_chemistry(monkeypatch) -> list[int]:
    calls = [0]
    real = module.held_out_chemistry

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "held_out_chemistry", counting)
    return calls


async def test_chemistry_is_computed_once_stored_and_then_only_read(monkeypatch) -> None:
    """A Protocol trained before training stored its chemistry: the first view computes
    it, every later one reads it. Computing it on each view starved prod's API."""
    calls = _counting_chemistry(monkeypatch)
    auth = FakeAuth(workspace_role="admin")  # admins see every protocol
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    store = _FakeStore(_inputs(protocol.id).to_json())
    use_case = GetScorecard(
        _FakeProtocols(protocol),
        store,
        RdkitStructureNormalizer(),
        _NoDatasets(),
        FakeProtocolAccess(),
    )

    first = (await use_case(GetScorecardQuery(protocol_id=protocol.id), auth)).unwrap()
    second = (await use_case(GetScorecardQuery(protocol_id=protocol.id), auth)).unwrap()

    assert calls[0] == 1
    assert [key.endswith("scorecard-chemistry.json") for key in store.written] == [True]
    assert first == second


async def test_simultaneous_first_views_share_one_computation(monkeypatch) -> None:
    import asyncio

    calls = _counting_chemistry(monkeypatch)
    auth = FakeAuth(workspace_role="admin")  # admins see every protocol
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    store = _FakeStore(_inputs(protocol.id).to_json())
    use_case = GetScorecard(
        _FakeProtocols(protocol),
        store,
        RdkitStructureNormalizer(),
        _NoDatasets(),
        FakeProtocolAccess(),
    )

    results = await asyncio.gather(
        *(use_case(GetScorecardQuery(protocol_id=protocol.id), auth) for _ in range(3))
    )

    assert calls[0] == 1
    assert all(len(result.unwrap()) == 1 for result in results)


def test_summary_reports_mean_spread_and_count() -> None:
    summary = replicate_summary([0.40, 0.50, 0.60])

    assert summary.n == 3
    assert summary.mean == pytest.approx(0.50)
    assert summary.sd == pytest.approx(0.1)


def test_one_usable_draw_has_a_mean_and_no_spread() -> None:
    """`statistics.stdev` raises below two points, and a NaN here would reach the
    response, where NaN is not valid JSON."""
    summary = replicate_summary([0.42, None])

    assert summary.n == 1
    assert summary.mean == pytest.approx(0.42)
    assert summary.sd is None


def test_a_metric_undefined_in_every_draw_reports_nothing_measured() -> None:
    summary = replicate_summary([None, None, None])

    assert summary.n == 0
    assert summary.mean is None
    assert summary.sd is None


def test_identical_draws_have_a_measured_spread_of_zero() -> None:
    """A real measurement, and weak evidence. Pinned so it stays 0.0 rather than
    becoming None, which a reader would see as "not measured"."""
    summary = replicate_summary([0.5, 0.5])

    assert summary.n == 2
    assert summary.sd == 0.0


def test_no_draws_at_all_is_not_a_measurement() -> None:
    summary = replicate_summary([])

    assert summary.n == 0
    assert summary.mean is None
    assert summary.sd is None


def test_a_blob_written_before_draws_existed_still_loads() -> None:
    raw = json.loads(_inputs(uuid.uuid4()).to_json())
    # Exactly what an older deploy wrote: the three keys simply absent.
    del raw["replicate_seeds"]
    del raw["replicate_unavailable"]
    for target in raw["targets"]:
        del target["replicate_metrics"]

    inputs = ScorecardInputs.from_json(json.dumps(raw).encode())

    assert inputs.replicate_seeds is None
    assert inputs.replicate_unavailable is None
    assert all(target.replicate_metrics is None for target in inputs.targets)

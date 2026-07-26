"""Regression coverage for Task 16 review Important 4: `GetScorecard` must
run `build_scorecard` off the event loop.

`build_scorecard` computes Murcko scaffolds and an O(test x train) Tanimoto
matrix -- measured at 0.86s for 16k train/2k test structures, which blocks
every other request on the process for the duration if run inline. Fakes
everything around `GetScorecard` (no real Postgres, blob store, or RDKit
needed) so the only thing under test is which thread `build_scorecard` runs on.
"""

from __future__ import annotations

import threading
import uuid

from daikonstudio.application.catalog import get_scorecard as module
from daikonstudio.application.catalog.get_scorecard import GetScorecard, GetScorecardQuery
from daikonstudio.application.execution.train_protocol import ScorecardInputs
from daikonstudio.domain.execution.scorecard import Scorecard
from tests.fakes.auth import FakeAuth


class _FakeProtocol:
    def __init__(self, workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> None:
        self.workspace_id = workspace_id
        self.id = protocol_id


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


def _inputs(protocol_id: uuid.UUID) -> ScorecardInputs:
    return ScorecardInputs(
        protocol_id=str(protocol_id),
        run_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        engine_id="ecfp4-randomforest",
        task="regression",
        conditions={},
        metrics={"rmse": 0.5},
        actual=[1.0],
        predicted=[1.1],
        prediction_kind="value",
        structures=["CCO"],
        train_structures=["CCO"],
        baseline_engine_id="ecfp4-randomforest",
        baseline_metrics={"rmse": 0.5},
        baseline_is_self=True,
        random_split_metrics=None,
        random_split_unavailable=None,
        random_split_metrics_undefined=None,
        metrics_undefined=None,
        duplicate_spread=None,
    )


async def test_build_scorecard_runs_off_the_main_thread(monkeypatch) -> None:
    auth = FakeAuth()
    protocol = _FakeProtocol(auth.workspace_id, uuid.uuid4())
    inputs = _inputs(protocol.id)
    seen: list[threading.Thread] = []

    def fake_build_scorecard(**kwargs: object) -> Scorecard:
        seen.append(threading.current_thread())
        return Scorecard(
            primary_metric="rmse",
            prediction_kind="value",
            metrics={},
            metrics_undefined=None,
            baseline_engine_id="x",
            baseline_metrics={},
            baseline_is_self=True,
            random_split_metrics=None,
            random_split_unavailable=None,
            random_split_metrics_undefined=None,
            noise_floor=None,
            worst_rows=[],
            applicability_coverage=None,
        )

    monkeypatch.setattr(module, "build_scorecard", fake_build_scorecard)

    use_case = GetScorecard(
        _FakeProtocols(protocol),
        _FakeStore(inputs.to_json()),
        normalizer=object(),  # type: ignore[arg-type]
    )
    result = await use_case(GetScorecardQuery(protocol_id=protocol.id), auth)

    assert result.unwrap() is not None
    assert len(seen) == 1
    assert seen[0] is not threading.main_thread()

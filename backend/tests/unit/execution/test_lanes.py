"""Lane routing: an engine says what it needs, a deployment says where that runs.

These tests pin the seam, not the queue. What matters is that the lane on the
manifest is the lane the job is enqueued to -- for both run kinds -- and that the
default lane keeps arq's own queue name so adding lanes needs no drain-and-migrate.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    DEFAULT_LANE,
    EngineManifest,
    TaskType,
    lane_for,
)
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.application.execution.predict_with_protocol import (
    PredictWithProtocol,
    PredictWithProtocolCommand,
)
from daikonstudio.application.execution.train_protocol import TrainProtocol, TrainProtocolCommand
from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.domain.shared.errors import NotFoundError
from daikonstudio.infrastructure.worker import ArqEnqueuer, queue_for
from tests.fakes.auth import FakeAuth


class _StubEngine:
    def __init__(self, engine_id: str, lane: str, *, is_baseline: bool = False) -> None:
        self._manifest = EngineManifest(
            id=engine_id,
            version="1.0.0",
            name=engine_id,
            description="",
            tasks=(TaskType.REGRESSION,),
            lane=lane,
            is_baseline=is_baseline,
        )

    def manifest(self) -> EngineManifest:
        return self._manifest

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover - never called
        raise NotImplementedError

    def predict(self, ctx: PredictContext) -> pl.DataFrame:  # pragma: no cover
        raise NotImplementedError


class _RecordingPool:
    """Stands in for an ArqRedis pool; records the queue each job was routed to."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def enqueue_job(self, function: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"function": function, "args": args, "kwargs": kwargs})


def test_default_lane_keeps_arqs_own_queue_name() -> None:
    """Renaming the default queue would strand every job already sitting in it."""
    assert queue_for(DEFAULT_LANE) == "arq:queue"


def test_a_named_lane_gets_its_own_queue() -> None:
    assert queue_for("gpu") == "arq:queue:gpu"


async def test_arq_enqueuer_routes_to_the_lanes_queue() -> None:
    enqueuer = ArqEnqueuer("redis://unused")
    pool = _RecordingPool()
    enqueuer._pool = pool  # type: ignore[assignment]
    run_id = uuid.uuid4()

    await enqueuer.enqueue(run_id, lane="gpu")

    assert pool.calls == [
        {"function": "run_job", "args": (run_id,), "kwargs": {"_queue_name": "arq:queue:gpu"}}
    ]


async def test_arq_enqueuer_carries_only_the_run_id() -> None:
    """The load-bearing invariant: all job state lives on the Run row. A payload in
    the queue message is what makes the orchestrator un-swappable."""
    enqueuer = ArqEnqueuer("redis://unused")
    pool = _RecordingPool()
    enqueuer._pool = pool  # type: ignore[assignment]
    run_id = uuid.uuid4()

    await enqueuer.enqueue(run_id)

    assert pool.calls[0]["args"] == (run_id,)


def test_registry_exposes_the_lane_for_routing() -> None:
    registry = EngineRegistry({"heavy": _StubEngine("heavy", "gpu")})

    assert registry.get("heavy").manifest().lane == "gpu"


def test_manifest_lane_defaults_to_the_default_lane() -> None:
    """Existing engines must be untouched by this change."""
    manifest = EngineManifest(
        id="e", version="1", name="e", description="", tasks=(TaskType.REGRESSION,)
    )

    assert manifest.lane == DEFAULT_LANE


class _RecordingEnqueuer:
    def __init__(self) -> None:
        self.lanes: list[str] = []

    async def enqueue(self, run_id: uuid.UUID, lane: str = DEFAULT_LANE) -> None:
        self.lanes.append(lane)


class _StubRuns:
    def __init__(self) -> None:
        self.added: list[Any] = []

    async def add(self, run: Any) -> None:
        self.added.append(run)

    async def find_by_cache_key(self, workspace_id: uuid.UUID, cache_key: str) -> None:
        return None


class _StubStore:
    """Enough BlobStore for the upload-existence check and the input hash."""

    def exists(self, key: str) -> bool:
        return True

    def get_bytes(self, key: str) -> bytes:
        return b"smiles\nCCO\n"


@pytest.fixture
def training_setup() -> Any:
    """Builds TrainProtocol against stubs, with one engine declaring lane='gpu'."""
    auth = FakeAuth()
    dataset_id = uuid.uuid4()
    dataset = Dataset(
        id=dataset_id,
        workspace_id=auth.workspace_id,
        name="a dataset",
        structure_column="smiles",
        target=TargetSpec(column="y", kind=TargetKind.NUMERIC),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=7),
        content_hash="deadbeef",
        snapshot_uri="file:///snapshot.parquet",
        row_count=1,
        validation_report=ValidationReport(total_rows=1, valid_rows=1),
    )

    class _StubDatasets:
        async def get(self, ws: uuid.UUID, ds: uuid.UUID) -> Any:
            return dataset

    enqueuer = _RecordingEnqueuer()
    registry = EngineRegistry(
        {
            "heavy": _StubEngine("heavy", "gpu"),
            "plain": _StubEngine("plain", DEFAULT_LANE, is_baseline=True),
        }
    )
    use_case = TrainProtocol(_StubDatasets(), _StubRuns(), enqueuer, registry)  # type: ignore[arg-type]
    command = TrainProtocolCommand(
        name="run", dataset_id=dataset_id, engine_id="heavy", conditions={}
    )
    return enqueuer, command, auth, use_case


async def test_training_is_enqueued_to_its_engines_lane(training_setup: Any) -> None:
    """A GPU engine's training job must not land on the default queue, where a CPU
    worker would pick it up and fail on a missing CUDA install."""
    enqueuer, command, auth, use_case = training_setup

    result = await use_case(command, auth)

    assert result.unwrap() is not None
    assert enqueuer.lanes == ["gpu"]


async def test_an_absent_baseline_resolves_to_the_registry_default(training_setup: Any) -> None:
    _enqueuer, command, auth, use_case = training_setup

    run = (await use_case(command, auth)).unwrap()

    assert run.params["baseline_engine_id"] == "plain"


async def test_a_named_baseline_is_stored_resolved_never_as_none(training_setup: Any) -> None:
    """The concrete id, never None: params are write-once, so a Run storing None
    would be measured against whatever the registry flags at the moment a worker
    dequeues it -- which may not be what the user was shown."""
    _enqueuer, command, auth, use_case = training_setup

    run = (
        await use_case(
            replace(command, baseline_engine_id="plain", baseline_conditions={"k": 1}), auth
        )
    ).unwrap()

    assert run.params["baseline_engine_id"] == "plain"
    assert run.params["baseline_conditions"] == {"k": 1}


async def test_an_unknown_baseline_is_a_404_before_anything_is_enqueued(
    training_setup: Any,
) -> None:
    enqueuer, command, auth, use_case = training_setup

    result = await use_case(replace(command, baseline_engine_id="not-an-engine"), auth)

    assert isinstance(result.failure(), NotFoundError)
    assert enqueuer.lanes == []


async def test_a_gpu_baseline_pulls_a_default_lane_run_onto_the_gpu_queue(
    training_setup: Any,
) -> None:
    """The failure this prevents: a default-lane worker fits the chosen engine
    fine and then dies in the baseline's _require_chemprop()."""
    enqueuer, command, auth, use_case = training_setup

    result = await use_case(replace(command, engine_id="plain", baseline_engine_id="heavy"), auth)

    result.unwrap()
    assert enqueuer.lanes == ["gpu"]


async def test_the_baseline_pair_changes_the_cache_key(training_setup: Any) -> None:
    _enqueuer, command, auth, use_case = training_setup

    first = (await use_case(replace(command, baseline_engine_id="plain"), auth)).unwrap()
    second = (
        await use_case(
            replace(command, baseline_engine_id="plain", baseline_conditions={"k": 1}), auth
        )
    ).unwrap()

    assert first.cache_key != second.cache_key


@pytest.fixture
def prediction_setup() -> Any:
    """Builds PredictWithProtocol against stubs, with a published Protocol whose
    engine declares lane='gpu'."""
    auth = FakeAuth()
    protocol = InSilicoProtocol(
        workspace_id=auth.workspace_id,
        name="a published protocol",
        dataset_id=uuid.uuid4(),
        engine_id="heavy",
        artifact_uri="file:///artifact.joblib",
        readouts=(),
        conditions={},
        status=ProtocolStatus.PUBLISHED,
    )

    class _StubProtocols:
        async def get(self, ws: uuid.UUID, protocol_id: uuid.UUID) -> Any:
            return protocol

    enqueuer = _RecordingEnqueuer()
    registry = EngineRegistry({"heavy": _StubEngine("heavy", "gpu")})
    use_case = PredictWithProtocol(
        _StubProtocols(),  # type: ignore[arg-type]
        _StubRuns(),  # type: ignore[arg-type]
        _StubStore(),
        enqueuer,
        registry,
    )
    command = PredictWithProtocolCommand(
        protocol_id=protocol.id, upload_ref=str(uuid.uuid4()), structure_column="smiles"
    )
    return enqueuer, command, auth, use_case


async def test_prediction_is_enqueued_to_its_protocols_engine_lane(
    prediction_setup: Any,
) -> None:
    """A prediction Run carries no engine id of its own -- the lane comes from the
    Protocol's engine, or a published GPU model would be scored on a CPU worker."""
    enqueuer, command, auth, use_case = prediction_setup

    result = await use_case(command, auth)

    assert result.unwrap() is not None
    assert enqueuer.lanes == ["gpu"]


async def test_a_protocol_whose_engine_is_gone_never_creates_a_run(
    prediction_setup: Any,
) -> None:
    """Resolved before the Run row exists: an orphan PENDING Run no worker can serve
    is worse than a clean 404."""
    _, command, auth, use_case = prediction_setup
    use_case._engines = EngineRegistry({})

    result = await use_case(command, auth)

    assert isinstance(result.failure(), NotFoundError)
    assert use_case._runs.added == []


def _manifest(engine_id: str, lane: str) -> EngineManifest:
    return EngineManifest(
        id=engine_id,
        version="1.0.0",
        name=engine_id,
        description="",
        tasks=(TaskType.REGRESSION,),
        lane=lane,
    )


def test_lane_for_returns_default_when_every_engine_is_default_lane() -> None:
    assert lane_for(_manifest("a", "default"), _manifest("b", "default")) == "default"


def test_lane_for_returns_the_non_default_lane_whichever_side_declares_it() -> None:
    """A run fits the chosen engine *and* the baseline in one process, so it must
    land on a worker that can serve both. Choosing ecfp4-randomforest (default)
    with a chemprop-dmpnn baseline (gpu) previously routed to a default-lane
    worker, which fit the chosen engine and then died in _require_chemprop()."""
    assert lane_for(_manifest("chosen", "default"), _manifest("base", "gpu")) == "gpu"
    assert lane_for(_manifest("chosen", "gpu"), _manifest("base", "default")) == "gpu"


def test_lane_for_is_stable_when_both_declare_the_same_non_default_lane() -> None:
    assert lane_for(_manifest("a", "gpu"), _manifest("b", "gpu")) == "gpu"

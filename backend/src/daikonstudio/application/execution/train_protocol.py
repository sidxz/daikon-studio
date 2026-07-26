"""Training orchestration -- and with it, the honesty layer.

Training a model is the easy part. The failure mode of every no-code ML
product is a model that scores beautifully on a random split and then fails in
the lab, so this module refuses to produce a number on its own. Every training
request runs up to three fits:

1. **The chosen engine** on the Dataset's own split. This is the model that
   becomes a Protocol, and its number is the one being claimed.
2. **The baseline** (ECFP4 + RandomForest) on the identical frame and split.
   Not a flag, not a checkbox, not an option: in the Polaris ADMET benchmark
   fingerprint baselines placed around 20th of 66 teams, and a scientist whose
   sophisticated model cannot beat one needs to learn that on the first screen,
   not after ordering compounds.
3. **The chosen engine again, on a random split**, but only when the Dataset's
   split is a scaffold split. The difference between the two is the *optimism
   gap*: how much of the flattering number came from near-duplicate analogues
   straddling the split. Measured, not inferred. On a Dataset that is already
   randomly split there is nothing to compare against, so `random_split_metrics`
   is `None` rather than a repeat of the same figure.

Two properties are load-bearing and deliberately not left to an engine:

- **The task type comes from the Dataset's TargetSpec**, never from the values.
  A regression target whose values happen to all be 0.0 and 1.0 would otherwise
  silently train a classifier and report MCC for something the scientist
  declared continuous.
- **Conditions are validated before any compute.** An out-of-range hyperparameter
  fails the Run in milliseconds with a message naming the offending key, not
  after several minutes of fitting.

What this module does *not* do is build the Scorecard -- it produces
`ScorecardInputs`, the raw material, and Task 15 turns it into the thing a
human reads. The split is on purpose: this file owns "what was actually
measured", `build_scorecard` owns "how it is presented".

# ponytail: three fits per training request is free at ECFP4 speeds and will not be
# for GPU engines. When Phase 2 lands, make the random-split comparison opt-out for
# expensive engines. The baseline stays mandatory regardless.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_editor,
    require_same_workspace,
)
from daikonstudio.application.catalog.derive_readouts import derive_readouts
from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import TaskType, validate_conditions
from daikonstudio.application.engines.protocol import Engine
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind
from daikonstudio.domain.execution.run import Run, RunKind, compute_cache_key
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError


def artifact_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Where the chosen engine's fitted weights live. The single definition:
    training writes it, Task 17's prediction path reads it back."""
    return f"{workspace_id}/protocols/{protocol_id}/artifact/model.joblib"


def scorecard_inputs_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Where the raw material for the Scorecard lives.

    Addressed by *protocol*, not by run, so `GET /protocols/{id}/scorecard`
    (Task 16) can find it from the id in the URL without first locating the Run
    that produced it. The Run's `result_uri` points at this same object, and the
    document carries `protocol_id`, so the mapping is navigable in both
    directions with no extra column.
    """
    return f"{workspace_id}/protocols/{protocol_id}/scorecard-inputs.json"


@dataclass(frozen=True, kw_only=True)
class ScorecardInputs:
    """Everything measured during a training run, before anyone interprets it.

    This is the contract between Task 14 and Task 15: `build_scorecard` consumes
    `task`/`actual`/`predicted`/`structures`/`train_structures` plus the three
    comparison fields, and nothing here is a presentation decision.

    `baseline_is_self` and `random_split_unavailable` exist so the two honest
    "no comparison happened" cases stay distinguishable from a real comparison.
    Silence would let a Scorecard imply an independent baseline was beaten when
    the model *is* the baseline, or that there was no optimism gap when there
    was one and it could not be computed.
    """

    protocol_id: str
    run_id: str
    dataset_id: str
    engine_id: str
    task: str
    conditions: dict[str, Any]
    metrics: dict[str, float]
    actual: list[float]
    predicted: list[float]
    structures: list[str]
    train_structures: list[str]
    baseline_engine_id: str
    baseline_metrics: dict[str, float]
    baseline_is_self: bool
    random_split_metrics: dict[str, float] | None
    random_split_unavailable: str | None
    duplicate_spread: float | None

    def to_json(self) -> bytes:
        # `allow_nan` stays on: a classification metric is genuinely undefined on a
        # single-class test split, and NaN is the honest encoding of that. Python's
        # own decoder round-trips it; anything serving this over HTTP has to
        # decide how to render an undefined metric, which is a Task 15/16 call.
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, data: bytes) -> ScorecardInputs:
        return cls(**json.loads(data))


@dataclass(frozen=True, kw_only=True)
class TrainProtocolCommand:
    name: str
    dataset_id: uuid.UUID
    engine_id: str
    conditions: dict[str, object]

    def to_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_id": str(self.dataset_id),
            "engine_id": self.engine_id,
            "conditions": self.conditions,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> TrainProtocolCommand:
        return cls(
            name=params["name"],
            dataset_id=uuid.UUID(params["dataset_id"]),
            engine_id=params["engine_id"],
            conditions=params["conditions"],
        )


class TrainProtocol:
    """Creates the training Run and hands it to the queue. All of the actual
    work is `RunTraining`, below, running in the worker.

    Conditions are deliberately *not* validated here. An invalid hyperparameter
    fails the Run (visibly, on the row the client is already polling) rather
    than the request, so there is exactly one place a user looks for why a
    training attempt did not produce a model. The same goes for an unknown
    engine id.
    """

    def __init__(
        self,
        datasets: DatasetRepository,
        runs: RunRepository,
        enqueuer: JobEnqueuer,
    ) -> None:
        self._datasets = datasets
        self._runs = runs
        self._enqueuer = enqueuer

    async def __call__(
        self, command: TrainProtocolCommand, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        dataset = await self._datasets.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        # Belt and braces: the read above is already workspace-scoped in SQL, so
        # this cannot fire today. It stays because it is the guard that has to
        # hold if a future caller ever hands us a Dataset it fetched elsewhere.
        require_same_workspace(auth, dataset.workspace_id, entity_type="Dataset")

        run = Run(
            kind=RunKind.TRAINING,
            workspace_id=auth.workspace_id,
            requested_by=auth.user_id,
            # The Dataset's content hash pins the data *and* the split, so this key
            # identifies "this data, split this way, through this engine, with these
            # conditions". Nothing reuses a training Run in Phase 1 -- only
            # predictions are cached (Task 17) -- but the column is what makes
            # "have we already fitted exactly this?" answerable when that changes.
            cache_key=compute_cache_key(
                kind="training",
                content_hash=dataset.content_hash,
                engine_id=command.engine_id,
                conditions=sorted(command.conditions.items()),
            ),
            params=command.to_params(),
        )
        await self._runs.add(run)
        await self._enqueuer.enqueue(run.id)
        return Success(run)


class RunTraining:
    """The `RunKind.TRAINING` handler: three fits, one Protocol, one Scorecard.

    Raises rather than returning a Result, unlike a use case. Its caller is
    `run_job`, whose entire contract is to catch whatever a handler throws and
    record it on the Run -- so a failure here already has a home, and wrapping
    it in a Result would just mean unwrapping it one frame later.
    """

    def __init__(
        self,
        datasets: DatasetRepository,
        protocols: ProtocolRepository,
        runs: RunRepository,
        store: BlobStore,
        engines: EngineRegistry,
        normalizer: StructureNormalizer,
    ) -> None:
        self._datasets = datasets
        self._protocols = protocols
        self._runs = runs
        self._store = store
        self._engines = engines
        self._normalizer = normalizer

    async def __call__(self, run: Run) -> str:
        command = TrainProtocolCommand.from_params(run.params)
        dataset = await self._datasets.get(run.workspace_id, command.dataset_id)
        if dataset is None:
            raise NotFoundError("Dataset", str(command.dataset_id))

        engine = self._engines.get(command.engine_id)
        manifest = engine.manifest()
        # Before any compute, in this order: is the engine capable of this task,
        # and are the conditions legal? Both are milliseconds; a fit is minutes.
        task = _task_for(dataset)
        if task not in manifest.tasks:
            raise ValidationError(
                f"Engine '{manifest.id}' cannot train a {task.value} model; "
                f"it supports {', '.join(t.value for t in manifest.tasks)}"
            )
        conditions = validate_conditions(manifest, command.conditions)

        baseline = self._engines.baseline()
        baseline_manifest = baseline.manifest()
        baseline_conditions = validate_conditions(baseline_manifest, {})

        frame = pl.read_parquet(
            io.BytesIO(self._store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id)))
        )

        chosen = await self._fit(run, engine, dataset, task, conditions, frame, 0.33)

        # The baseline is unconditional -- with one exception that is *not* an
        # exception to the rule. When the chosen engine is the baseline engine on
        # the baseline's own default conditions, the second fit is the first fit:
        # same data, same split, same seed, same hyperparameters, bit-identical
        # result. Running it twice would burn minutes to rediscover a number we
        # are already holding. `baseline_is_self` travels with the metrics so the
        # Scorecard says "this model is the baseline" instead of presenting one
        # result twice as though a comparison had taken place.
        baseline_is_self = (
            manifest.id == baseline_manifest.id and conditions == baseline_conditions
        )
        if baseline_is_self:
            baseline_result = chosen
        else:
            baseline_result = await self._fit(
                run, baseline, dataset, task, baseline_conditions, frame, 0.66, "training baseline"
            )

        random_split_metrics, random_split_unavailable = await self._optimism_gap(
            run, engine, dataset, task, conditions, frame
        )

        train_rows = frame.filter(pl.col("split") == "train")
        test_rows = frame.filter(pl.col("split") == "test")
        # The engines score internally but return only metrics; the Scorecard also
        # needs the per-row predictions (worst residuals, applicability), so the
        # fitted artifact is replayed over the test split. A scoring pass, not a
        # fit -- cheap next to the three trainings above.
        predictions = await asyncio.to_thread(
            engine.predict,
            PredictContext(
                frame=test_rows,
                structure_column=dataset.structure_column,
                artifact=chosen.artifact,
                conditions=conditions,
            ),
        )

        protocol_id = uuid.uuid4()
        # Only the chosen engine's weights are persisted. The baseline's and the
        # random-split model's are dropped on the floor deliberately: an artifact
        # is only worth storing if something can run it, and the only runnable
        # thing here is the Protocol, of which there is exactly one. Those two
        # fits exist to produce numbers, and their numbers are kept. Writing
        # their weights would leave two blobs no row references -- and every one
        # of them is reproducible anyway from (content_hash, seed, engine
        # defaults), since nothing in this pipeline is unseeded.
        artifact_uri = self._store.put_bytes(
            artifact_key(run.workspace_id, protocol_id), chosen.artifact
        )
        protocol = InSilicoProtocol(
            id=protocol_id,
            workspace_id=run.workspace_id,
            name=command.name,
            dataset_id=dataset.id,
            engine_id=manifest.id,
            artifact_uri=artifact_uri,
            # Derived from the Dataset's TargetSpec, which is what makes a
            # predicted IC50 arrive in the same unit and direction as a measured
            # one. The Protocol is created in DRAFT; Task 16 publishes it.
            readouts=derive_readouts(dataset.target, task),
            conditions=conditions,
        )
        await self._protocols.add(protocol)

        inputs = ScorecardInputs(
            protocol_id=str(protocol_id),
            run_id=str(run.id),
            dataset_id=str(dataset.id),
            engine_id=manifest.id,
            task=task.value,
            conditions=conditions,
            metrics=chosen.metrics,
            actual=[float(value) for value in test_rows[dataset.target.column].to_list()],
            predicted=[float(value) for value in predictions["value"].to_list()],
            structures=[str(s) for s in test_rows[dataset.structure_column].to_list()],
            train_structures=[str(s) for s in train_rows[dataset.structure_column].to_list()],
            baseline_engine_id=baseline_manifest.id,
            baseline_metrics=baseline_result.metrics,
            baseline_is_self=baseline_is_self,
            random_split_metrics=random_split_metrics,
            random_split_unavailable=random_split_unavailable,
            duplicate_spread=dataset.validation_report.duplicate_spread,
        )
        return self._store.put_bytes(
            scorecard_inputs_key(run.workspace_id, protocol_id), inputs.to_json()
        )

    async def _optimism_gap(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        task: TaskType,
        conditions: dict[str, object],
        frame: pl.DataFrame,
    ) -> tuple[dict[str, float] | None, str | None]:
        """The chosen engine re-fitted on a random split of the same rows.

        Only for a scaffold split: on a Dataset that is already randomly split
        there is no second number to compare against, and reporting the same
        figure twice would invent a gap of zero where none was measured.
        """
        if dataset.split.strategy is not SplitStrategy.SCAFFOLD:
            return None, None
        # Outside the try on purpose: this writes to the Run row, and a failure
        # here is a persistence problem with the run itself, not a failure of the
        # comparison. Swallowing it would leave the aggregate's in-memory version
        # out of step with the row and turn a database problem into a missing
        # optimism gap.
        await self._progress(run, 0.9, "training random-split comparison")
        try:
            # Same seed and same fractions as the Dataset's own split, so the only
            # variable between the two numbers is the split *strategy* -- which is
            # the entire claim the optimism gap makes.
            random_frame = assign_split(
                frame,
                dataset.structure_column,
                SplitSpec(
                    strategy=SplitStrategy.RANDOM,
                    seed=dataset.split.seed,
                    fractions=dataset.split.fractions,
                ),
                self._normalizer,
            )
            result = await self._train_off_thread(engine, dataset, task, conditions, random_frame)
            return result.metrics, None
        except Exception as exc:
            # Deliberately broad, and deliberately not fatal. The scaffold number
            # and the baseline are the primary result and they are already in
            # hand; the optimism gap is the nice-to-have. Failing the whole run
            # here would throw away an honest, fully-computed model to protect a
            # comparison -- exactly backwards. Anything that can go wrong in this
            # leg (a split that cannot honour the fractions, a random partition
            # that leaves one class in the training rows, an engine that raises)
            # therefore degrades to a recorded reason rather than a failure.
            # Not silent: `random_split_unavailable` is what stops the Scorecard
            # showing an absent gap and a not-applicable gap identically.
            return None, repr(exc)

    async def _fit(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        task: TaskType,
        conditions: dict[str, object],
        frame: pl.DataFrame,
        fraction: float,
        phase: str | None = None,
    ) -> TrainResult:
        """Report the phase, then fit off the event loop.

        Progress is set *before* the work, naming what is about to run rather
        than what just finished: the row is the only channel the client has, and
        a phase that describes the completed step would leave the UI reading
        "training baseline" while the random-split fit is what is actually
        holding it up.
        """
        await self._progress(run, fraction, phase or f"training {engine.manifest().id}")
        return await self._train_off_thread(engine, dataset, task, conditions, frame)

    async def _train_off_thread(
        self,
        engine: Engine,
        dataset: Dataset,
        task: TaskType,
        conditions: dict[str, object],
        frame: pl.DataFrame,
    ) -> TrainResult:
        # train() is synchronous and CPU-bound by contract (see engines/protocol.py):
        # the worker offloads it so engine authors never have to think about threads.
        return await asyncio.to_thread(
            engine.train,
            TrainContext(
                frame=frame,
                task=task,
                structure_column=dataset.structure_column,
                target_column=dataset.target.column,
                conditions=conditions,
                seed=dataset.split.seed,
            ),
        )

    async def _progress(self, run: Run, fraction: float, phase: str) -> None:
        run.report_progress(fraction, phase=phase)
        await self._runs.update(run)


def _task_for(dataset: Dataset) -> TaskType:
    """The one place the task type is decided, and it reads the TargetSpec.

    Never the values. `TrainContext.task` is authoritative precisely so an
    engine cannot look at a column of 0.0s and 1.0s and decide for itself.
    """
    return (
        TaskType.BINARY_CLASSIFICATION
        if dataset.target.kind is TargetKind.BINARY
        else TaskType.REGRESSION
    )

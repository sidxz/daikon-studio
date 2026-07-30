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
import math
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
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
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

    This is the contract between Task 14 and Task 15. `build_scorecard` reads the
    metric dicts as measured -- it does not recompute them -- and uses
    `actual`/`predicted` for residuals and applicability only.

    **What `predicted` holds depends on the task, and `prediction_kind` says
    which so no consumer has to re-derive it:**

    - `prediction_kind == "value"` (regression): the predicted target, in the
      same unit as `actual`. `actual - predicted` is a residual.
    - `prediction_kind == "probability"` (binary classification): P(class=1), a
      float in [0, 1], while `actual` holds the 0/1 labels. Feeding this to a
      metric that expects hard labels raises; a consumer wanting labels must
      threshold it and own that choice explicitly.

    Three fields exist purely so an absent number cannot be mistaken for a
    different kind of absent number:

    - `baseline_is_self` -- the chosen engine *is* the baseline, so
      `baseline_metrics` is the same fit rather than an independent comparison.
    - `random_split_unavailable` -- there was an optimism gap to measure and the
      attempt failed, as distinct from `random_split_metrics is None` on a
      dataset that is already randomly split, where there is nothing to measure.
    - `metrics_undefined` -- why a metric came back undefined (`None`), keyed by
      metric name. Without it a Scorecard reading "your model -- versus baseline
      --" has no way to say why. It describes the Dataset's own test split, which
      `metrics` and `baseline_metrics` share; `random_split_metrics` is scored on
      a *different* partition, with its own possible reasons a metric there is
      undefined -- `random_split_metrics_undefined` is that explanation, kept
      separate rather than merged into `metrics_undefined` because the two
      partitions can disagree about which metrics are undefined and why. A
      consumer that explained a `random_split_metrics` null using
      `metrics_undefined` would (when both happen to be undefined) attribute the
      wrong partition's reason, and (when only the random split is undefined)
      find no explanation there at all -- a bare, unexplained null on the exact
      number an optimism-gap comparison exists to justify.

    `target_unit`/`target_direction` and `split_strategy` are the Dataset's own
    `TargetSpec.unit`/`.direction` and `SplitSpec.strategy.value` at the moment
    this Run trained -- carried through unchanged so `build_scorecard` (Task
    15 review, Important 2) can render a metric with the unit and direction
    that make it meaningful, and say which split strategy produced it, rather
    than a consumer inferring the strategy from `random_split_metrics`/
    `random_split_unavailable` both being `None`.
    """

    protocol_id: str
    run_id: str
    dataset_id: str
    engine_id: str
    task: str
    conditions: dict[str, Any]
    metrics: dict[str, float | None]
    actual: list[float]
    predicted: list[float]
    prediction_kind: str
    structures: list[str]
    train_structures: list[str]
    baseline_engine_id: str
    baseline_metrics: dict[str, float | None]
    baseline_is_self: bool
    random_split_metrics: dict[str, float | None] | None
    random_split_unavailable: str | None
    random_split_metrics_undefined: dict[str, str] | None
    metrics_undefined: dict[str, str] | None
    duplicate_spread: float | None
    target_unit: str | None
    target_direction: str | None
    split_strategy: str

    def to_json(self) -> bytes:
        # allow_nan=False on purpose. An undefined metric is real -- a single-class
        # test split makes every classification metric meaningless -- but `NaN` is
        # not JSON: strict parsers reject the document outright and jq quietly
        # turns it into null, so the "honest" encoding was only honest to Python.
        # Undefined metrics are already `None` by the time they get here (see
        # `_measured`), with the reason in `metrics_undefined`; this flag is what
        # stops a future edit silently reintroducing a bare NaN token.
        return json.dumps(asdict(self), allow_nan=False).encode()

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
    training attempt did not produce a model -- re-validating them here could
    even disagree with the worker, since resolving a condition's default is
    something only the engine's own manifest can do.

    `engine_id` does not share that argument: it is a registry membership
    check against a fixed, in-process set with exactly one possible answer, so
    there is nothing the worker could compute differently. Rejecting an
    unknown engine here means the request fails synchronously with a 404
    instead of returning a 202 for a Run that cannot possibly succeed.
    """

    def __init__(
        self,
        datasets: DatasetRepository,
        runs: RunRepository,
        enqueuer: JobEnqueuer,
        engines: EngineRegistry,
    ) -> None:
        self._datasets = datasets
        self._runs = runs
        self._enqueuer = enqueuer
        self._engines = engines

    async def __call__(
        self, command: TrainProtocolCommand, auth: AuthContext | None = None
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        try:
            self._engines.get(command.engine_id)
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", command.engine_id))

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
        _require_structure_column(dataset, frame)

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

        (
            random_split_metrics,
            random_split_unavailable,
            random_split_metrics_undefined,
        ) = await self._optimism_gap(run, engine, dataset, task, conditions, frame)

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
        metrics, undefined = _measured(chosen.metrics)
        baseline_metrics, baseline_undefined = _measured(baseline_result.metrics)
        inputs = ScorecardInputs(
            protocol_id=str(protocol_id),
            run_id=str(run.id),
            dataset_id=str(dataset.id),
            engine_id=manifest.id,
            task=task.value,
            conditions=conditions,
            metrics=metrics,
            actual=[float(value) for value in test_rows[dataset.target.column].to_list()],
            predicted=[float(value) for value in predictions["value"].to_list()],
            prediction_kind=("probability" if task is TaskType.BINARY_CLASSIFICATION else "value"),
            structures=[str(s) for s in test_rows[dataset.structure_column].to_list()],
            train_structures=[str(s) for s in train_rows[dataset.structure_column].to_list()],
            baseline_engine_id=baseline_manifest.id,
            baseline_metrics=baseline_metrics,
            baseline_is_self=baseline_is_self,
            random_split_metrics=random_split_metrics,
            random_split_unavailable=random_split_unavailable,
            random_split_metrics_undefined=random_split_metrics_undefined,
            metrics_undefined=_undefined_reasons(
                undefined | baseline_undefined, dataset, train_rows, test_rows
            ),
            duplicate_spread=dataset.validation_report.duplicate_spread,
            target_unit=dataset.target.unit,
            target_direction=(
                dataset.target.direction.value if dataset.target.direction is not None else None
            ),
            split_strategy=dataset.split.strategy.value,
        )

        # Blobs first, Protocol row last, and deliberately in that order. A
        # Protocol is the publishable, citable thing; a Protocol row with no
        # scorecard behind it is a model whose honesty data does not exist, which
        # Task 16 would happily list and then fail to serve -- precisely the
        # failure this codebase is built to prevent. Writing the row last means
        # every failure mode leaves *no* Protocol rather than a hollow one.
        #
        # The inverse cost is blobs with no row pointing at them when the insert
        # fails. That is the cheap direction: they sit inside the workspace's own
        # prefix under an id nothing references, exactly like the orphan snapshot
        # `create_dataset.py` already documents, and the same sweep reclaims both.
        # One artifact, not three. The baseline's and the random-split model's
        # weights are never written: an artifact is only worth storing if something
        # can run it, the only runnable thing is the Protocol, and there is exactly
        # one. Those two fits exist to produce numbers, and their numbers are kept
        # above -- the fits themselves are reproducible from (content_hash, seed,
        # engine defaults), since nothing in this pipeline is unseeded.
        artifact_uri = self._store.put_bytes(
            artifact_key(run.workspace_id, protocol_id), chosen.artifact
        )
        result_uri = self._store.put_bytes(
            scorecard_inputs_key(run.workspace_id, protocol_id), inputs.to_json()
        )
        await self._protocols.add(
            InSilicoProtocol(
                id=protocol_id,
                workspace_id=run.workspace_id,
                name=command.name,
                dataset_id=dataset.id,
                engine_id=manifest.id,
                artifact_uri=artifact_uri,
                # Derived from the Dataset's TargetSpec, which is what makes a
                # predicted IC50 arrive in the same unit and direction as a
                # measured one. Created in DRAFT; Task 16 publishes it.
                readouts=derive_readouts(dataset.target, task),
                conditions=conditions,
            )
        )
        # After the Protocol row exists, so the link is never dangling. The
        # worker's own `succeed()` + `update()` is what persists it -- both this
        # and `result_uri` ride out on that one write.
        run.link_protocol(protocol_id)
        return result_uri

    async def _optimism_gap(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        task: TaskType,
        conditions: dict[str, object],
        frame: pl.DataFrame,
    ) -> tuple[dict[str, float | None] | None, str | None, dict[str, str] | None]:
        """The chosen engine re-fitted on a random split of the same rows.

        Only for a scaffold split: on a Dataset that is already randomly split
        there is no second number to compare against, and reporting the same
        figure twice would invent a gap of zero where none was measured.

        Returns `(metrics, unavailable_reason, metrics_undefined)`. The third
        element is this leg's own answer to the same question `metrics_undefined`
        answers for the Dataset's own split -- computed from the *random*
        partition, not the scaffold one, because the two can disagree about
        which metrics are undefined and why: a class that survives the
        scaffold split's test rows can still collapse to one class under a
        random reshuffle, or vice versa. Reusing the scaffold split's reasons
        here would misattribute them to a different partition; leaving this
        undefined-but-unexplained would be a bare null on the exact number the
        optimism gap exists to justify. Both are the failure this field
        prevents.
        """
        if dataset.split.strategy is not SplitStrategy.SCAFFOLD:
            return None, None, None
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
            metrics, undefined = _measured(result.metrics)
            reasons = _undefined_reasons(
                undefined,
                dataset,
                random_frame.filter(pl.col("split") == "train"),
                random_frame.filter(pl.col("split") == "test"),
            )
            return metrics, None, reasons
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
            return None, repr(exc), None

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


def _measured(metrics: dict[str, float]) -> tuple[dict[str, float | None], set[str]]:
    """Split an engine's metrics into JSON-encodable values and undefined names.

    An engine reports an undefined metric as NaN (see `engines/_scoring.py`,
    which returns NaN for all four classification metrics rather than letting
    balanced accuracy quietly collapse to plain accuracy). NaN is the right
    thing to *mean* and the wrong thing to *store*: it is not JSON. `None` is,
    and the reason travels alongside it.
    """
    undefined = {name for name, value in metrics.items() if math.isnan(value)}
    return {name: None if name in undefined else value for name, value in metrics.items()}, (
        undefined
    )


def _undefined_reasons(
    undefined: set[str], dataset: Dataset, train_rows: pl.DataFrame, test_rows: pl.DataFrame
) -> dict[str, str] | None:
    """Why those metrics are undefined, in words a scientist can act on.

    Derived from the split that produced them rather than guessed: the reason a
    classification metric has no value is almost always that one side of the
    split holds a single class, and which side it is changes what the scientist
    should do about it.
    """
    if not undefined:
        return None
    column = dataset.target.column
    if test_rows[column].n_unique() < 2:
        reason = (
            f"every row in the test split has the same '{column}' value, so this "
            "metric has no defined value -- add positives (or negatives) to the "
            "dataset, or split it differently"
        )
    elif train_rows[column].n_unique() < 2:
        reason = (
            f"every row in the training split has the same '{column}' value, so the "
            "model only ever learned one class and this metric has no defined value"
        )
    else:
        # Not a case this function can explain from the split alone. Say that,
        # rather than attribute it to a cause that was ruled out two lines up.
        reason = "the engine reported this metric as undefined"
    return dict.fromkeys(sorted(undefined), reason)


# The `server_default` migration 005 backfilled onto Datasets created before the
# column existed. Those rows have no true answer, so the sentinel is deliberately
# not a plausible column name -- and training refuses it by name below.
_LEGACY_STRUCTURE_COLUMN = "unknown"


def _require_structure_column(dataset: Dataset, frame: pl.DataFrame) -> None:
    """Fail with a cause a human can act on, not a polars `ColumnNotFoundError`.

    Covers both the pre-migration sentinel and any other drift between what the
    Dataset records and what its snapshot actually contains -- one guard, because
    every such Dataset is untrainable for the same reason and reaching RDKit with
    the wrong column produces a message naming neither the Dataset nor the cause.
    """
    is_legacy = dataset.structure_column == _LEGACY_STRUCTURE_COLUMN
    if not is_legacy and dataset.structure_column in frame.columns:
        return
    # Everything actionable goes in the message, not `detail`: the worker records
    # `repr(exc)` on the Run, and a DomainError's repr carries only its args -- so
    # anything parked in `detail` would be invisible exactly where a scientist
    # looks for why their training run failed.
    cause = (
        "this Dataset predates the structure_column migration, so which column "
        "holds its structures was never recorded -- re-upload it to train on it"
        if is_legacy
        else f"its snapshot holds: {', '.join(frame.columns)}"
    )
    raise ValidationError(
        f"Dataset '{dataset.id}' records its structures in column "
        f"'{dataset.structure_column}', which cannot be used: {cause}"
    )


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

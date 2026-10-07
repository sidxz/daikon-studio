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
3. **The chosen engine again, on a random split**, whenever the Dataset's own
   split is a grouped one (scaffold, identity or position). The difference
   between the two is the *optimism gap*: how much of the flattering number came
   from near-duplicate analogues, homologues or repeat mutations at one site
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
# once an engine's own fit is slow -- which is a property of the fit, not of the lane
# it runs on. Make the random-split comparison opt-out on measured fit cost when one
# engine actually crosses that line. The baseline stays mandatory regardless.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import lzma
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_editor,
    require_same_workspace,
)
from daikonstudio.application.catalog.chemical_space import write_chemical_space
from daikonstudio.application.catalog.derive_readouts import derive_readouts
from daikonstudio.application.data.assign_split import assign_split
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.engines.checkpoints import (
    DEFAULT_INTERVAL_SECONDS,
    RESULT_FORMAT,
    TRAINING_STATE,
    TRAINING_STATE_SCOPE,
    Checkpoints,
    checkpoint_root,
    unpack_result,
)
from daikonstudio.application.engines.context import (
    MIN_CUTOFF_CLASS_COUNT,
    EpochPoint,
    PredictContext,
    ProgressReporter,
    RunInterrupted,
    TrainContext,
    TrainResult,
)
from daikonstudio.application.engines.manifest import (
    ENSEMBLE_SIZE,
    EngineManifest,
    TaskType,
    lane_for,
    reset_inapplicable_conditions,
    validate_conditions,
)
from daikonstudio.application.engines.protocol import Engine
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
from daikonstudio.application.execution.build_scorecard import (
    held_out_chemistry,
    primary_metric_for,
)
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.failure_message import user_facing_error
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.chemical_space_layout import (
    ChemicalSpaceLayout,
    TooFewCompounds,
)
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.catalog.readout import ReadoutType
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.execution.run import (
    Run,
    RunKind,
    RunStatus,
    TargetHeadline,
    compute_cache_key,
)
from daikonstudio.domain.shared.errors import (
    DomainError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def artifact_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Where the chosen engine's fitted weights live. The single definition:
    training writes it, Task 17's prediction path reads it back."""
    return f"{workspace_id}/protocols/{protocol_id}/artifact/model.joblib"


# The xz container's own magic bytes. Pickles start with b"\x80", torch checkpoints and
# the FanOut container with b"PK", so the formats never collide.
_XZ_MAGIC = b"\xfd7zXZ\x00"


def pack_artifact(artifact: bytes) -> bytes:
    """What is stored for a trained model: the engine's bytes, xz-compressed.

    A runner uploads the artifact in one request under `runner_upload_max_bytes`, and a
    random forest's pickle grows with its training rows -- about 80 bytes per tree node,
    90 MB for 500 trees on 8,000 compounds, several GB for a four-target fan-out on a few
    hundred thousand. Preset 1 measured 8.3x on that forest in about a second; higher
    presets cost far more time for little more.
    """
    return lzma.compress(artifact, preset=1)


def unpack_artifact(stored: bytes) -> bytes:
    """The engine's own bytes back. Artifacts stored before compression existed are
    read as they are."""
    return lzma.decompress(stored) if stored.startswith(_XZ_MAGIC) else stored


def scorecard_inputs_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """Where the raw material for the Scorecard lives.

    Addressed by *protocol*, not by run, so `GET /protocols/{id}/scorecard`
    (Task 16) can find it from the id in the URL without first locating the Run
    that produced it. The Run's `result_uri` points at this same object, and the
    document carries `protocol_id`, so the mapping is navigable in both
    directions with no extra column.
    """
    return f"{workspace_id}/protocols/{protocol_id}/scorecard-inputs.json"


def scorecard_chemistry_key(workspace_id: uuid.UUID, protocol_id: uuid.UUID) -> str:
    """The test set's `HeldOutChemistry`, beside the inputs it is computed from.

    Computed once, where the training ran, so the API never does it on a page load:
    at 404k compounds it is a 40k x 323k Tanimoto search, which held the API's GIL
    until /ready stopped answering and the healthcheck killed it, on every reload
    (prod, 2026-10-05). A Protocol trained before this existed has none; the
    Scorecard computes it once on first view and stores it here.
    """
    return f"{workspace_id}/protocols/{protocol_id}/scorecard-chemistry.json"


# How often the reporter is allowed a database round-trip. Cancellation latency is
# bounded by this; against a fit measured in minutes that is immaterial, and it keeps a
# 500-epoch run from writing 500 rows.
_PROGRESS_INTERVAL_SECONDS = 10.0
# How long the training thread will wait for the event loop to service one checkpoint.
# Generous: the loop is otherwise idle while the fit runs.
_CHECKPOINT_TIMEOUT_SECONDS = 30.0
# Progress spans per leg, so a long fit's own per-epoch progress has somewhere to move
# instead of the bar sitting at a single number for its whole duration.
_CHOSEN_SPAN = (0.0, 0.6)
_BASELINE_SPAN = (0.6, 0.7)
_RANDOM_SPLIT_SPAN = (0.7, 0.95)
# Prefixed to an engine's own progress text in the stages that are not the chosen
# model's fit: with the baseline the same engine as the model, "Training Chemprop
# D-MPNN" alone cannot say which of the three fits is running.
_STAGE_LABELS = {"baseline": "Baseline", "random-split": "Random-split comparison"}


def _staged(scope: str, phase: str) -> str:
    stage = _STAGE_LABELS.get(scope)
    return phase if stage is None else f"{stage}: {phase}"


class _EpochBuffer:
    """One stage's finished epochs, between progress writes. `record` runs on the
    worker thread, `take` on the event loop; the swap in `take` cannot lose a point,
    since an append racing it lands in the list `take` already holds."""

    def __init__(self, fit: str) -> None:
        self._fit = fit
        self._points: list[EpochPoint] = []

    def record(self, point: EpochPoint) -> None:
        self._points.append(replace(point, fit=self._fit))

    def take(self) -> list[EpochPoint]:
        points, self._points = self._points, []
        return points


@dataclass(frozen=True, kw_only=True)
class TargetInputs:
    """One target's share of `ScorecardInputs`: everything measured about it.

    **What `predicted` holds depends on the task, and `prediction_kind` says
    which so no consumer has to re-derive it:**

    - `prediction_kind == "value"` (regression): the predicted target, in the
      same unit as `actual`. `actual - predicted` is a residual.
    - `prediction_kind == "probability"` (binary classification): P(class=1), a
      float in [0, 1], while `actual` holds the 0/1 labels. Feeding this to a
      metric that expects hard labels raises; a consumer wanting labels must
      threshold it and own that choice explicitly.

    `metrics_undefined` -- why a metric came back undefined (`None`), keyed by
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

    `validation_metrics` is the chosen engine scored on the *validation*
    partition by the identical code that produced `metrics` from the test one. It
    is the number a scientist is meant to tune conditions against, and it exists
    because until it did there was none: the validation split was assigned and
    never read, so the only feedback available when choosing between two sets of
    conditions was the test score -- and a test set consulted once per retrain is
    no longer held out. `None` when the split declared a zero validation
    fraction, which is a legitimate choice and not a measurement failure.

    `target_unit`/`target_direction` are the Dataset's own `TargetSpec.unit`/
    `.direction` at the moment this Run trained -- carried through unchanged so
    `build_scorecard` (Task 15 review, Important 2) can render a metric with the
    unit and direction that make it meaningful.
    """

    column: str
    task: str
    metrics: dict[str, float | None]
    # Defaulted for the same reason `baseline_conditions` below is: a required
    # field here would make every Scorecard blob written before validation scoring
    # existed unreadable. `None` therefore means either "this run predates the
    # field" or "the split had no validation partition" -- both are honestly
    # rendered as "not measured".
    validation_metrics: dict[str, float | None] | None = None
    actual: list[float]
    predicted: list[float]
    prediction_kind: str
    baseline_metrics: dict[str, float | None]
    random_split_metrics: dict[str, float | None] | None
    random_split_metrics_undefined: dict[str, str] | None
    metrics_undefined: dict[str, str] | None
    duplicate_spread: float | None
    target_unit: str | None
    target_direction: str | None
    # The decision cutoffs MCC and balanced accuracy were measured at, for the model and
    # its baseline. Defaulted for the reason `validation_metrics` is: a blob written
    # before cutoffs could be tuned has none, and `None` means 0.5.
    cutoff: float | None = None
    baseline_cutoff: float | None = None
    # Why a requested tuning did not happen; `None` when it did or was not requested.
    cutoff_note: str | None = None


# What a blob written before targets could be several stored at its top level.
_PER_TARGET = tuple(f.name for f in fields(TargetInputs) if f.name != "column")


@dataclass(frozen=True, kw_only=True)
class ScorecardInputs:
    """Everything measured during a training run, before anyone interprets it.

    This is the contract between Task 14 and Task 15. `build_scorecard` reads the
    metric dicts as measured -- it does not recompute them -- and uses
    `actual`/`predicted` for residuals and applicability only.

    Run-level facts appear once; everything measured per target is in `targets`,
    in the Dataset's order. `structures` and `train_structures` are run-level on
    purpose: every target shares one split and one set of test rows, and at 324k
    compounds those two lists are most of the blob.

    Two fields exist purely so an absent number cannot be mistaken for a
    different kind of absent number:

    - `baseline_is_self` -- the chosen engine *is* the baseline, so each target's
      `baseline_metrics` is the same fit rather than an independent comparison.
    - `random_split_unavailable` -- there was an optimism gap to measure and the
      attempt failed, as distinct from `random_split_metrics is None` on a
      dataset that is already randomly split, where there is nothing to measure.

    `split_strategy` is the Dataset's own `SplitSpec.strategy.value` at the moment
    this Run trained -- carried through unchanged so `build_scorecard` can say
    which split strategy produced a metric, rather than a consumer inferring it
    from `random_split_metrics`/`random_split_unavailable` both being `None`.
    """

    protocol_id: str
    run_id: str
    dataset_id: str
    engine_id: str
    conditions: dict[str, Any]
    structures: list[str]
    train_structures: list[str]
    baseline_engine_id: str
    # Defaulted because `from_json` is `cls(**json.loads(data))`: a required
    # field here makes every Scorecard blob written before the baseline became
    # choosable unreadable. Also load-bearing for display -- when the baseline
    # is the *same* engine with different settings (pretrained vs not), the two
    # engine ids are identical and this is the only thing distinguishing them.
    baseline_conditions: dict[str, Any] = field(default_factory=dict)
    baseline_is_self: bool
    random_split_unavailable: str | None
    split_strategy: str
    # True when one model learned every target; False for one model per target --
    # and for every Protocol trained before targets could be several, which had one
    # target and one model either way. Recorded here rather than read off the
    # engine's manifest later, so the Scorecard says what happened, not what the
    # engine would do today.
    joint_model: bool = False
    targets: list[TargetInputs]

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
    def from_json(cls, data: bytes, *, legacy_column: str = "") -> ScorecardInputs:
        """`legacy_column` names the one target of a blob written before targets
        could be several, which stored its per-target fields at the top level and
        never recorded the column. Only `GetScorecard` renders that name, so only it
        needs to pass one."""
        raw = json.loads(data)
        if "targets" not in raw:
            raw["targets"] = [
                {"column": legacy_column, **{k: raw.pop(k) for k in _PER_TARGET if k in raw}}
            ]
        raw["targets"] = [TargetInputs(**target) for target in raw["targets"]]
        return cls(**raw)


@dataclass(frozen=True, kw_only=True)
class TrainProtocolCommand:
    name: str
    dataset_id: uuid.UUID
    engine_id: str
    conditions: dict[str, object]
    # What this Run is measured against. `None` means "whatever the registry
    # flags as the default baseline", and survives only until `TrainProtocol`
    # resolves it -- what lands in `run.params` is always a concrete id, so a
    # queued Run cannot be silently retargeted by a registry change before a
    # worker picks it up. It stays `None` when read back off a row written
    # before the baseline was choosable.
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, object] = field(default_factory=dict)
    # The sweep's own name, repeated on every member. A label, not a grouping:
    # the id is a column precisely because `params` is write-once, and this
    # rides along so the sweeps list can title a group without a second table
    # or a second query. `None` on a solo run.
    sweep_name: str | None = None
    # Choose each binary target's decision cutoff on the validation partition. The same
    # request reaches the model, its baseline and the random-split fit.
    tune_cutoffs: bool = False

    def to_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_id": str(self.dataset_id),
            "engine_id": self.engine_id,
            "conditions": self.conditions,
            "baseline_engine_id": self.baseline_engine_id,
            "baseline_conditions": self.baseline_conditions,
            "sweep_name": self.sweep_name,
            "tune_cutoffs": self.tune_cutoffs,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> TrainProtocolCommand:
        return cls(
            name=params["name"],
            dataset_id=uuid.UUID(params["dataset_id"]),
            engine_id=params["engine_id"],
            conditions=params["conditions"],
            baseline_engine_id=params.get("baseline_engine_id"),
            baseline_conditions=params.get("baseline_conditions") or {},
            sweep_name=params.get("sweep_name"),
            tune_cutoffs=params.get("tune_cutoffs", False),
        )


def joint_kind_error(manifest: EngineManifest, dataset: Dataset) -> ValidationError | None:
    """Why a joint engine cannot train on this Dataset, or None when it can.

    An engine that declares `supports_multitask` learns every target in one model
    with one loss, and adding a squared error to a cross-entropy needs a relative
    weighting nobody can set honestly. So a Dataset that mixes continuous value and
    active or inactive targets trains only on engines that fit one model per target.
    Checked at enqueue (`TrainProtocol`, `SubmitSweep`), which refuse before a Run
    exists, and again by the worker for a Run enqueued before the rule existed.
    """
    kinds = {target.kind for target in dataset.targets}
    if not manifest.supports_multitask or len(kinds) < 2:
        return None
    numeric = [t.column for t in dataset.targets if t.kind is TargetKind.NUMERIC]
    binary = [t.column for t in dataset.targets if t.kind is TargetKind.BINARY]
    return ValidationError(
        f"{manifest.name} trains one joint model, so every target must be the same kind. "
        f"This dataset has continuous value targets ({', '.join(numeric)}) and active or "
        f"inactive targets ({', '.join(binary)}).",
        detail=(
            "Choose an engine that trains one model per target, or a dataset whose "
            "targets are all one kind."
        ),
    )


def deadline_scale(
    manifest: EngineManifest,
    dataset: Dataset,
    conditions: dict[str, object],
    baseline: EngineManifest,
    baseline_conditions: dict[str, object],
) -> int:
    """How many times over its lane's deadline a training Run may take: one lane budget
    per model fitted inside it.

    The chosen engine fits in each of its legs -- the model, and on any grouped split
    the random-split comparison too -- and the baseline fits once. In each, a fan-out
    engine fits once per target and a joint engine once regardless, and an ensemble
    fits once per member. A chemprop ensemble of four on a scaffold split is eight
    chemprop fits plus the baseline: the prod run that hit an 8 h limit at 80% needed
    about 11.5 h.

    ponytail: a cheap baseline costs a whole lane budget here, so the limit is loose
    rather than tight. It is a ceiling for a hung fit, not an estimate.
    """
    # Phrased as "is this a grouped strategy", i.e. anything but RANDOM, rather than as
    # a list of members: `_optimism_gap` runs the second leg for every non-random
    # strategy, and a new member added to that side but forgotten here would get half
    # the budget it needs and die on the lane deadline at ~80% of a long run.
    legs = 1 if dataset.split.strategy is SplitStrategy.RANDOM else 2
    return legs * _fits(manifest, dataset, conditions) + _fits(
        baseline, dataset, baseline_conditions
    )


def _fits(manifest: EngineManifest, dataset: Dataset, conditions: dict[str, object]) -> int:
    per_target = 1 if manifest.supports_multitask else len(dataset.targets)
    return per_target * _ensemble_size(manifest, conditions)


def _ensemble_size(manifest: EngineManifest, conditions: dict[str, object]) -> int:
    """1 for an engine without the setting, and for conditions the worker will refuse:
    they are validated there, where a refusal fails the Run visibly (see `TrainProtocol`)."""
    if all(spec.key != ENSEMBLE_SIZE for spec in manifest.conditions):
        return 1
    try:
        resolved: dict[str, Any] = validate_conditions(manifest, conditions)
    except ValueError:
        return 1
    return int(resolved[ENSEMBLE_SIZE])


def training_lane(engines: EngineRegistry, engine_id: str, baseline_engine_id: str | None) -> str:
    """The lane a training Run needs: the chosen engine and its baseline fit inside
    one job, so the queue has to serve both. Shared by enqueue (`TrainProtocol`)
    and re-enqueue (`RetryRun`), so the rule has exactly one home -- a retry that
    resolved the lane from the engine alone would strand a run whose baseline is
    the gpu-lane one on the default lane."""
    engine = engines.get(engine_id)
    baseline = engines.get(baseline_engine_id) if baseline_engine_id else engines.baseline()
    return lane_for(engine.manifest(), baseline.manifest())


class TrainProtocol:
    """Creates the training Run and hands it to the queue. All of the actual
    work is `RunTraining`, below, running in the worker.

    Conditions are deliberately *not* validated here. An invalid hyperparameter
    fails the Run (visibly, on the row the client is already polling) rather
    than the request, so there is exactly one place a user looks for why a
    training attempt did not produce a model -- re-validating them here could
    even disagree with the worker, since resolving a condition's default is
    something only the engine's own manifest can do. The one thing done to them
    here is not validation: a setting that cannot apply to this dataset's tasks is
    put back to its default, so the cache key, the stored params and the worker's
    self-baseline check all see the same, honest conditions.

    `engine_id` does not share that argument: it is a registry membership
    check against a fixed, in-process set with exactly one possible answer, so
    there is nothing the worker could compute differently. Rejecting an
    unknown engine here means the request fails synchronously with a 404
    instead of returning a 202 for a Run that cannot possibly succeed.

    `sweep_id` is the one thing a caller may add to an otherwise identical
    request. It is stamped on the row rather than folded into `params` because
    `params` is write-once and unindexed: a mistyped grouping could never be
    corrected, and the cancel cascade and the sweeps list both filter on it.
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
        self,
        command: TrainProtocolCommand,
        auth: AuthContext | None = None,
        *,
        sweep_id: uuid.UUID | None = None,
    ) -> Result[Run, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        try:
            engine = self._engines.get(command.engine_id)
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", command.engine_id))

        # Same argument as `engine_id` above: a registry membership check with
        # exactly one possible answer, so an unknown baseline is a synchronous
        # 404 rather than a 202 for a Run that cannot succeed. The *conditions*
        # stay unvalidated here, for the reason in this class's docstring.
        try:
            baseline = (
                self._engines.get(command.baseline_engine_id)
                if command.baseline_engine_id
                else self._engines.baseline()
            )
        except UnknownEngineError:
            return Failure(NotFoundError("Baseline engine", command.baseline_engine_id))

        dataset = await self._datasets.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        # Belt and braces: the read above is already workspace-scoped in SQL, so
        # this cannot fire today. It stays because it is the guard that has to
        # hold if a future caller ever hands us a Dataset it fetched elsewhere.
        require_same_workspace(auth, dataset.workspace_id, entity_type="Dataset")

        for candidate in (engine, baseline):
            refused = joint_kind_error(candidate.manifest(), dataset)
            if refused is not None:
                return Failure(refused)

        # Pin the resolved id into what gets persisted. `params` is write-once,
        # so a Run storing `None` would be measured against whatever the registry
        # flags at the moment a worker dequeues it -- which may not be what the
        # user was shown when they submitted. The same goes for a setting this dataset
        # cannot use: reset here, before anything is keyed or persisted.
        tasks = {_task_for(target) for target in dataset.targets}
        command = replace(
            command,
            baseline_engine_id=baseline.manifest().id,
            conditions=reset_inapplicable_conditions(engine.manifest(), command.conditions, tasks),
            baseline_conditions=reset_inapplicable_conditions(
                baseline.manifest(), command.baseline_conditions, tasks
            ),
        )

        run = Run(
            kind=RunKind.TRAINING,
            workspace_id=auth.workspace_id,
            requested_by=auth.user_id,
            # The Dataset's content hash pins the data *and* the split, so this key
            # identifies "this data, split this way, through this engine, with these
            # conditions, measured against this baseline". Nothing reuses a training
            # Run today -- only predictions are cached -- but fit-result caching is a
            # live deferred item, and a key omitting the baseline would let it serve
            # a run whose comparison was against a different model.
            cache_key=compute_cache_key(
                kind="training",
                content_hash=dataset.content_hash,
                engine_id=command.engine_id,
                conditions=sorted(command.conditions.items()),
                baseline_engine_id=command.baseline_engine_id,
                baseline_conditions=sorted(command.baseline_conditions.items()),
                # Only when on: a key hashes whatever parts it is given, so an absent
                # part leaves every key computed before this option existed unchanged.
                **({"tune_cutoffs": True} if command.tune_cutoffs else {}),
            ),
            params={
                **command.to_params(),
                "deadline_scale": deadline_scale(
                    engine.manifest(),
                    dataset,
                    command.conditions,
                    baseline.manifest(),
                    command.baseline_conditions,
                ),
            },
            # Which sweep asked for this run, or None for a solo request. The
            # only difference between the two, deliberately: a sweep child is
            # the same object, with the same cache key, baseline resolution and
            # lane, produced by this same code path.
            sweep_id=sweep_id,
        )
        await self._runs.add(run)
        # Both engines fit inside this one Run, so the queue has to serve both.
        await self._enqueuer.enqueue(
            run.id,
            lane=training_lane(self._engines, command.engine_id, command.baseline_engine_id),
        )
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
        deadline_seconds: float | None = None,
        layout: ChemicalSpaceLayout | None = None,
        checkpoint_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._datasets = datasets
        self._protocols = protocols
        self._runs = runs
        self._store = store
        self._engines = engines
        self._normalizer = normalizer
        self._deadline_seconds = deadline_seconds
        self._layout = layout
        self._checkpoint_interval_seconds = checkpoint_interval_seconds
        self._checkpoints: Checkpoints | None = None
        self._deadline_at: float | None = None
        self._tune_cutoffs = False

    async def __call__(self, run: Run) -> str:
        self._deadline_at = (
            time.monotonic() + self._deadline_seconds
            if self._deadline_seconds is not None
            else None
        )
        command = TrainProtocolCommand.from_params(run.params)
        # The same request reaches all three fits: a tuned model measured against an
        # untuned baseline would flatter the model by construction.
        self._tune_cutoffs = command.tune_cutoffs
        dataset = await self._datasets.get(run.workspace_id, command.dataset_id)
        if dataset is None:
            raise NotFoundError("Dataset", str(command.dataset_id))

        engine = self._engines.get(command.engine_id)
        manifest = engine.manifest()
        # Before any compute, in this order: is the engine capable of this task,
        # and are the conditions legal? Both are milliseconds; a fit is minutes.
        targets = {target.column: _task_for(target) for target in dataset.targets}
        _check_capable(manifest, dataset)
        conditions = validate_conditions(manifest, command.conditions)

        # Resolved at enqueue for new Runs; `None` only on a row written before
        # the baseline became choosable, where the registry default is correct.
        baseline = (
            self._engines.get(command.baseline_engine_id)
            if command.baseline_engine_id
            else self._engines.baseline()
        )
        baseline_manifest = baseline.manifest()
        _check_capable(baseline_manifest, dataset, baseline=True)
        baseline_conditions = validate_conditions(baseline_manifest, command.baseline_conditions)

        frame = pl.read_parquet(
            io.BytesIO(self._store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id)))
        )
        _require_structure_column(dataset, frame)

        # The run's saved progress. A retry of this same run (same id) finds what an
        # earlier attempt saved and skips it; success clears it below.
        self._checkpoints = Checkpoints(
            self._store,
            checkpoint_root(run.workspace_id, dataset.id, run.id),
            interval_seconds=self._checkpoint_interval_seconds,
        )

        chosen = await self._fit(
            run, engine, dataset, targets, conditions, frame, _CHOSEN_SPAN, scope="model"
        )

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
                run,
                baseline,
                dataset,
                targets,
                baseline_conditions,
                frame,
                _BASELINE_SPAN,
                "Training baseline model",
                scope="baseline",
            )

        random_split, random_split_unavailable = await self._optimism_gap(
            run, engine, dataset, targets, conditions, frame
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
                target_columns=dataset.target_columns,
            ),
        )

        protocol_id = uuid.uuid4()
        per_target: list[TargetInputs] = []
        headlines: list[TargetHeadline] = []
        for target in dataset.targets:
            task = targets[target.column]
            metrics, undefined = _measured(chosen.metrics[target.column])
            baseline_metrics, baseline_undefined = _measured(
                baseline_result.metrics[target.column]
            )
            gap = random_split.get(target.column) if random_split is not None else None
            # `predict` returns one row per (compound, target); this target's rows, in
            # test-set order, line up with `actual` below.
            predicted = predictions.filter(pl.col("target") == target.column).sort("row_id")
            cutoff = (chosen.cutoffs or {}).get(target.column)
            per_target.append(
                TargetInputs(
                    column=target.column,
                    task=task.value,
                    metrics=metrics,
                    validation_metrics=(
                        _measured(chosen.validation_metrics[target.column])[0]
                        if chosen.validation_metrics is not None
                        else None
                    ),
                    actual=[float(value) for value in test_rows[target.column].to_list()],
                    predicted=[float(value) for value in predicted["value"].to_list()],
                    prediction_kind=(
                        "probability" if task is TaskType.BINARY_CLASSIFICATION else "value"
                    ),
                    baseline_metrics=baseline_metrics,
                    random_split_metrics=gap[0] if gap is not None else None,
                    random_split_metrics_undefined=gap[1] if gap is not None else None,
                    metrics_undefined=_undefined_reasons(
                        undefined | baseline_undefined, target.column, train_rows, test_rows
                    ),
                    duplicate_spread=dataset.validation_report.duplicate_spread.get(target.column),
                    target_unit=target.unit,
                    target_direction=(
                        target.direction.value if target.direction is not None else None
                    ),
                    cutoff=cutoff,
                    baseline_cutoff=(baseline_result.cutoffs or {}).get(target.column),
                    cutoff_note=(
                        _cutoff_note(target.column, cutoff, frame)
                        if self._tune_cutoffs and task is TaskType.BINARY_CLASSIFICATION
                        else None
                    ),
                )
            )
            primary = primary_metric_for(task)
            headlines.append(
                TargetHeadline(
                    column=target.column,
                    primary_metric=primary,
                    value=metrics.get(primary),
                    baseline_value=baseline_metrics.get(primary),
                )
            )
        inputs = ScorecardInputs(
            protocol_id=str(protocol_id),
            run_id=str(run.id),
            dataset_id=str(dataset.id),
            engine_id=manifest.id,
            conditions=conditions,
            structures=[str(s) for s in test_rows[dataset.structure_column].to_list()],
            train_structures=[str(s) for s in train_rows[dataset.structure_column].to_list()],
            baseline_engine_id=baseline_manifest.id,
            baseline_conditions=baseline_conditions,
            baseline_is_self=baseline_is_self,
            random_split_unavailable=random_split_unavailable,
            split_strategy=dataset.split.strategy.value,
            joint_model=manifest.supports_multitask,
            targets=per_target,
        )

        # A CLASS readout is named after its target column. The cutoff rides on it so
        # prediction labels at the same operating point the Scorecard measured.
        readouts = tuple(
            replace(readout, threshold=(chosen.cutoffs or {}).get(readout.name))
            if readout.type is ReadoutType.CLASS
            else readout
            for readout in derive_readouts(dataset.targets)
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
            artifact_key(run.workspace_id, protocol_id), pack_artifact(chosen.artifact)
        )
        result_uri = self._store.put_bytes(
            scorecard_inputs_key(run.workspace_id, protocol_id), inputs.to_json()
        )
        chemistry = await asyncio.to_thread(
            held_out_chemistry,
            inputs.structures,
            inputs.train_structures,
            self._normalizer,
            dataset.validation_report.structure_kind,
        )
        self._store.put_bytes(
            scorecard_chemistry_key(run.workspace_id, protocol_id), chemistry.to_json()
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
                readouts=readouts,
                conditions=conditions,
                # The person who asked for the training, not the runner that ran it.
                created_by=run.requested_by,
            )
        )
        # After the Protocol row exists, so the link is never dangling. The
        # worker's own `succeed()` + `update()` is what persists it -- both this
        # and `result_uri` ride out on that one write.
        run.link_protocol(protocol_id)

        # Denormalised for ranking. Both dicts are already measured above, so
        # this is a lookup, not a computation -- and it rides out on the same
        # `succeed()` + `update()` write that persists `result_uri`, so a run
        # can never be READY with no metric on it.
        run.record_metrics(headlines)

        # The Protocol exists and the run is about to be READY: nothing here will be
        # resumed again. Best effort -- `clear` logs and swallows a failure.
        if self._checkpoints is not None:
            await asyncio.to_thread(self._checkpoints.clear)

        await self._map_chemical_space(run, protocol_id, frame, dataset)
        return result_uri

    async def _optimism_gap(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        targets: dict[str, TaskType],
        conditions: dict[str, object],
        frame: pl.DataFrame,
    ) -> tuple[
        dict[str, tuple[dict[str, float | None], dict[str, str] | None]] | None, str | None
    ]:
        """The chosen engine re-fitted on a random split of the same rows.

        Skipped only for a Dataset that is *already* randomly split: there is no
        second number to compare against, and reporting the same figure twice
        would invent a gap of zero where none was measured. Every other strategy
        -- scaffold, identity, position -- gets the comparison, and the condition
        below is written as "is it random" rather than "is it scaffold" for a
        reason beyond tidiness: the Scorecard renders a missing gap as "not
        applicable: trained on a random split", so skipping it for an
        identity-split or position-split Protocol would print a plainly false
        statement about how the model was trained.

        Returns `(per target: (metrics, metrics_undefined), unavailable_reason)`. The
        second element of each pair is this leg's own answer to the same question
        `metrics_undefined` answers for the Dataset's own split -- computed from the
        *random* partition, not the grouped one, because the two can disagree about
        which metrics are undefined and why: a class that survives the
        grouped split's test rows can still collapse to one class under a
        random reshuffle, or vice versa. Reusing the grouped split's reasons
        here would misattribute them to a different partition; leaving this
        undefined-but-unexplained would be a bare null on the exact number the
        optimism gap exists to justify. Both are the failure this field
        prevents.
        """
        if dataset.split.strategy is SplitStrategy.RANDOM:
            return None, None
        # Outside the try on purpose: this writes to the Run row, and a failure
        # here is a persistence problem with the run itself, not a failure of the
        # comparison. Swallowing it would leave the aggregate's in-memory version
        # out of step with the row and turn a database problem into a missing
        # optimism gap.
        await self._progress(
            run, _RANDOM_SPLIT_SPAN[0], "Training on a random split for comparison"
        )
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
                # No clusterer: this split is always RANDOM, which never groups by
                # homology, so threading one here would be a dead parameter.
                self._normalizer,
            )
            result = await self._train_off_thread(
                run,
                engine,
                dataset,
                targets,
                conditions,
                random_frame,
                _RANDOM_SPLIT_SPAN,
                scope="random-split",
            )
            train_rows = random_frame.filter(pl.col("split") == "train")
            test_rows = random_frame.filter(pl.col("split") == "test")
            gap: dict[str, tuple[dict[str, float | None], dict[str, str] | None]] = {}
            for column in targets:
                metrics, undefined = _measured(result.metrics[column])
                gap[column] = (
                    metrics,
                    _undefined_reasons(undefined, column, train_rows, test_rows),
                )
            return gap, None
        except RunInterrupted:
            # Not degradable, unlike every other failure in this leg. A cancellation or
            # a deadline means stop, and recording it as an unavailable comparison would
            # let the run succeed after the user asked it not to.
            raise
        except Exception as exc:
            # Deliberately broad, and deliberately not fatal. The grouped-split
            # number and the baseline are the primary result and they are already in
            # hand; the optimism gap is the nice-to-have. Failing the whole run
            # here would throw away an honest, fully-computed model to protect a
            # comparison -- exactly backwards. Anything that can go wrong in this
            # leg (a split that cannot honour the fractions, a random partition
            # that leaves one class in the training rows, an engine that raises)
            # therefore degrades to a recorded reason rather than a failure.
            # Not silent: `random_split_unavailable` is what stops the Scorecard
            # showing an absent gap and a not-applicable gap identically.
            return None, user_facing_error(exc)

    async def _fit(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        targets: dict[str, TaskType],
        conditions: dict[str, object],
        frame: pl.DataFrame,
        span: tuple[float, float],
        phase: str | None = None,
        *,
        scope: str,
    ) -> TrainResult:
        """Report the phase, then fit off the event loop.

        Progress is set *before* the work, naming what is about to run rather
        than what just finished: the row is the only channel the client has, and
        a phase that describes the completed step would leave the UI reading
        "training baseline" while the random-split fit is what is actually
        holding it up. Inside the span, an engine that calls `ctx.report` moves
        the bar itself.
        """
        resolved_phase = phase or f"Training {engine.manifest().name}"
        await self._progress(run, span[0], resolved_phase)
        return await self._train_off_thread(
            run, engine, dataset, targets, conditions, frame, span, scope=scope
        )

    async def _train_off_thread(
        self,
        run: Run,
        engine: Engine,
        dataset: Dataset,
        targets: dict[str, TaskType],
        conditions: dict[str, object],
        frame: pl.DataFrame,
        span: tuple[float, float],
        *,
        scope: str,
    ) -> TrainResult:
        # train() is synchronous and blocking by contract, whichever device it resolves
        # to (see engines/protocol.py): the worker offloads it so engine authors never
        # have to think about threads.
        # `run` is threaded through only so the reporter can reach the row -- the
        # engine never sees it.
        manifest = engine.manifest()
        stage = (
            self._checkpoints.scoped(
                scope,
                engine=manifest.id,
                engine_version=manifest.version,
                result_format=RESULT_FORMAT,
            )
            if self._checkpoints is not None
            else None
        )
        if stage is not None:
            saved = await asyncio.to_thread(stage.load, "result")
            if saved is not None:
                try:
                    restored = unpack_result(saved)
                except Exception:
                    logger.warning(
                        "The saved %s fit did not unpack; refitting it.", scope, exc_info=True
                    )
                else:
                    await self._progress(
                        run,
                        span[1],
                        _staged(scope, f"Restored the {manifest.name} fit from saved progress"),
                    )
                    return restored
        epochs = _EpochBuffer(scope)
        try:
            result = await asyncio.to_thread(
                engine.train,
                TrainContext(
                    frame=frame,
                    targets=targets,
                    structure_column=dataset.structure_column,
                    conditions=conditions,
                    seed=dataset.split.seed,
                    tune_cutoffs=self._tune_cutoffs,
                    checkpoints=stage,
                    report=self._reporter(run, span, scope, epochs=epochs),
                    record_epoch=epochs.record,
                ),
            )
        finally:
            # What the fit recorded since the last progress write, stopped or not: the
            # charts then end where the fit did.
            await self._flush_epochs(run, epochs)
        if stage is not None:
            # Off the event loop: packing is minutes of compression for a large model, and
            # on a runner the save is an HTTP upload; the loop also carries the heartbeat.
            await asyncio.to_thread(_save_result, stage, result)
        return result

    def _reporter(
        self,
        run: Run,
        span: tuple[float, float],
        scope: str = "model",
        *,
        epochs: _EpochBuffer | None = None,
    ) -> ProgressReporter:
        """A callback the engine invokes from the worker thread.

        Three things happen per call, in this order and for this reason:

        1. The deadline is checked. It needs no I/O, so it runs on every call -- which
           is what makes an overrunning fit stop promptly rather than at the next
           throttled checkpoint.
        2. The database round-trip is throttled to `_PROGRESS_INTERVAL_SECONDS`.
        3. The Run's status is re-read and progress written. Reading is the point: a
           cancellation happens in the API process against a different row instance
           entirely, so this worker's in-memory aggregate would never see it.

        `asyncio.run_coroutine_threadsafe` is how a synchronous engine reaches the
        event loop that owns the repositories. Blocking on the result is deliberate:
        the engine must not proceed past a checkpoint that says the run was cancelled.
        """
        loop = asyncio.get_running_loop()
        low, high = span
        last_written = 0.0

        def report(fraction: float, phase: str) -> None:
            nonlocal last_written
            self._check_deadline()
            now = time.monotonic()
            if now - last_written < _PROGRESS_INTERVAL_SECONDS:
                return
            last_written = now
            clamped = min(max(fraction, 0.0), 1.0)
            future = asyncio.run_coroutine_threadsafe(
                self._checkpoint(run, low + (high - low) * clamped, _staged(scope, phase), epochs),
                loop,
            )
            try:
                wanted = future.result(timeout=_CHECKPOINT_TIMEOUT_SECONDS)
            except (TimeoutError, ServiceUnavailableError):
                # The API answered too slowly or not at all -- busy, or mid-deploy. One
                # progress write is not worth a fit that may have run for hours (prod,
                # 2026-10-04: a 26-minute fit died here). Drop it and carry on; the next
                # write, a throttle interval later, tries again and still sees a cancel.
                # A refusal (another runner owns the run, a version conflict) still ends
                # the fit: those are not this kind of error.
                future.cancel()
                logger.warning(
                    "Progress write for run %s did not complete; training continues",
                    run.id,
                    exc_info=True,
                )
                return
            if not wanted:
                raise RunInterrupted("the run was cancelled", cancelled=True)

        return report

    async def _checkpoint(
        self, run: Run, fraction: float, phase: str, epochs: _EpochBuffer | None = None
    ) -> bool:
        """Write progress, and the epochs finished since the last write; report whether
        the run is still wanted."""
        current = await self._runs.get_by_id(run.id)
        if current is None or current.status is not RunStatus.RUNNING:
            return False
        # The server's version, not this copy's: a write that timed out after the server
        # applied it leaves this copy a version behind, and every later write would then
        # fail its check and end the fit.
        run.version = current.version
        run.report_progress(fraction, phase=phase)
        await self._runs.update(run)
        if epochs is not None:
            await self._flush_epochs(run, epochs)
        return True

    async def _flush_epochs(self, run: Run, epochs: _EpochBuffer) -> None:
        points = epochs.take()
        if not points:
            return
        try:
            await self._runs.append_epochs(run.id, points)
        except Exception:
            # A chart is not worth a training run: the points are dropped, the fit goes on.
            logger.warning(
                "Saving %d training epochs for run %s failed", len(points), run.id, exc_info=True
            )

    def _check_deadline(self) -> None:
        """Raise past the soft deadline. Called on every `ctx.report` and between
        fits -- the latter is the only check the tree and GP engines ever reach,
        since none of them reports progress during a fit."""
        if self._deadline_at is not None and time.monotonic() > self._deadline_at:
            saved = (
                " Resume continues from its last saved progress."
                if self._checkpoints is not None
                else ""
            )
            raise RunInterrupted(
                f"The run exceeded its {self._deadline_seconds:.0f} s time limit and was "
                f"stopped.{saved} An administrator can raise the limit "
                "(STUDIO_WORKER_JOB_TIMEOUT, or STUDIO_WORKER_JOB_TIMEOUT_BY_LANE for one lane).",
                cancelled=False,
            )

    async def _map_chemical_space(
        self, run: Run, protocol_id: uuid.UUID, frame: pl.DataFrame, dataset: Dataset
    ) -> None:
        """Draw the protocol's chemical-space map. Best-effort, and after the Protocol
        row exists, so nothing here may fail or interrupt the run: a protocol without
        a map is whole, a run marked FAILED beside a live protocol is not.

        Hence no `_check_deadline` (it raises): a deadline that has already passed --
        a tree or GP fit that overran it still finishes READY -- skips the map, and
        `make backfill-maps` draws it later.
        """
        if self._layout is None:
            return
        if self._deadline_at is not None and time.monotonic() > self._deadline_at:
            logger.info(
                "Skipped the chemical-space map for protocol %s: the run is past its "
                "deadline. `make backfill-maps` will draw it.",
                protocol_id,
            )
            return
        run.report_progress(0.97, phase="Mapping chemical space")
        try:
            await self._runs.update(run)
            await asyncio.to_thread(
                write_chemical_space,
                self._store,
                run.workspace_id,
                protocol_id,
                frame,
                dataset.structure_column,
                dataset.split.seed,
                self._layout,
            )
        except TooFewCompounds:
            pass  # a map of four compounds says nothing; the page says "no map"
        except Exception:
            logger.exception("Chemical-space map failed for protocol %s", protocol_id)

    async def _progress(self, run: Run, fraction: float, phase: str) -> None:
        self._check_deadline()
        run.report_progress(fraction, phase=phase)
        await self._runs.update(run)


def _save_result(stage: Checkpoints, result: TrainResult) -> None:
    """Save a finished fit, then free the training state it no longer needs: a neural
    fit's in-progress state is hundreds of megabytes that nothing will resume from now.
    Kept when the save failed, since the next attempt would then resume from it. A
    runner cannot delete one blob, so `discard` empties it."""
    if stage.save_result(result):
        stage.scoped(TRAINING_STATE_SCOPE).discard(TRAINING_STATE)


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
    undefined: set[str], column: str, train_rows: pl.DataFrame, test_rows: pl.DataFrame
) -> dict[str, str] | None:
    """Why those metrics are undefined, in words a scientist can act on.

    Derived from the split that produced them rather than guessed: the reason a
    classification metric has no value is almost always that one side of the
    split holds a single class, and which side it is changes what the scientist
    should do about it.
    """
    if not undefined:
        return None
    if test_rows[column].n_unique() < 2:
        reason = (
            f"Undefined: all test-set compounds have the same '{column}' value. Add "
            "compounds of the missing class, or use a different split."
        )
    elif train_rows[column].n_unique() < 2:
        reason = (
            f"Undefined: all training-set compounds have the same '{column}' value, so "
            "the model learned only one class."
        )
    else:
        # Not a case this function can explain from the split alone. Say that,
        # rather than attribute it to a cause that was ruled out two lines up.
        reason = "Undefined: the engine returned no value for this metric."
    return dict.fromkeys(sorted(undefined), reason)


def _cutoff_note(column: str, cutoff: float | None, frame: pl.DataFrame) -> str | None:
    """Why a requested cutoff tuning did not happen, or None when it did.

    Called only for a binary target on a run that asked for tuning. The engines leave a
    cutoff untuned in exactly two ways, and the Scorecard has to say which: validation
    held too few compounds of a class (the count is read off the same split the engine
    saw), or the model's validation probabilities gave no cutoff a reason to be chosen.
    """
    if cutoff is not None:
        return None
    validation = frame.filter(pl.col("split") == "validation")
    if validation.height == 0:
        return "Not tuned: this split has no validation set, so the cutoff stays at 0.5."
    labels = validation[column].drop_nulls()
    positives = int((labels == 1).sum())
    negatives = labels.len() - positives
    if positives < MIN_CUTOFF_CLASS_COUNT or negatives < MIN_CUTOFF_CLASS_COUNT:
        return (
            f"Not tuned: the validation set has {positives} active and {negatives} inactive "
            f"compounds for '{column}'; at least {MIN_CUTOFF_CLASS_COUNT} of each are "
            "needed, so the cutoff stays at 0.5."
        )
    return (
        "Not tuned: the model's validation predictions could not support a cutoff, so it "
        "stays at 0.5."
    )


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
    # Everything actionable goes in the message: the worker records
    # `user_facing_error(exc)` on the Run, which renders a DomainError as
    # `message (detail)` -- the message leads, so it is what a scientist reads
    # first when their training run failed.
    if is_legacy:
        raise ValidationError(
            "This dataset was created before its structure column was recorded and "
            "cannot be trained on. Upload it again."
        )
    raise ValidationError(
        f"The structure column '{dataset.structure_column}' is missing from this "
        f"dataset's stored data. Available columns: {', '.join(frame.columns)}."
    )


def _task_label(task: TaskType) -> str:
    return task.value.replace("_", " ")


def _task_for(target: TargetSpec) -> TaskType:
    """The one place a task is decided, and it reads the target's spec -- never its
    values. `TrainContext.targets` is authoritative precisely so an engine cannot
    look at a column of 0.0s and 1.0s and decide for itself."""
    return (
        TaskType.BINARY_CLASSIFICATION if target.kind is TargetKind.BINARY else TaskType.REGRESSION
    )


def _check_capable(manifest: EngineManifest, dataset: Dataset, *, baseline: bool = False) -> None:
    """Before any compute: can this engine train every target here? Milliseconds,
    against a fit measured in minutes.

    `baseline` only changes the wording: the baseline is not the engine the scientist
    chose, so the message has to name it as the baseline."""
    subject = f"The baseline engine ({manifest.name})" if baseline else manifest.name
    for task in dict.fromkeys(_task_for(target) for target in dataset.targets):
        if task not in manifest.tasks:
            raise ValidationError(
                f"{subject} does not support {_task_label(task)}. "
                f"Supported tasks: {', '.join(map(_task_label, manifest.tasks))}."
            )
    kind = dataset.validation_report.structure_kind
    if kind not in manifest.structure_kinds:
        reads = " or ".join(_structure_kind_label(k) for k in manifest.structure_kinds)
        raise ValidationError(
            f"{subject} reads {reads}, but this dataset's structure column holds "
            f"{_structure_kind_label(kind)}. Choose an engine built for this kind of "
            "data. Nothing is caught later if this is allowed through: the model would "
            "train, score and report as though the numbers meant something."
        )
    refused = joint_kind_error(manifest, dataset)
    if refused is not None:
        raise refused


def _structure_kind_label(kind: StructureKind) -> str:
    return "amino-acid sequences" if kind is StructureKind.SEQUENCE else "small molecules"

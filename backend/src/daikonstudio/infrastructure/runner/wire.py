"""Wire envelopes for the self-hosted runner protocol.

Pydantic models that mirror domain aggregates and value objects field-for-
field, so a `Run`/`Dataset`/`InSilicoProtocol` can cross an HTTP boundary and
come back unchanged. Shared by the server-side protocol routes (Task 7) and
the runner-side HTTP port clients (Task 8) -- one definition, both directions.

Domain-only imports (plus pydantic): this module sits in `infrastructure` so
both consumers may import it, but it must never import `interface` or
`application` -- the runner client process has neither.

Enums cross the wire as their plain `str` value, not as the enum type itself:
`from_domain` writes `.value`, `to_domain` casts back via the enum
constructor (`RunKind(...)`, `TargetKind(...)`, ...). That keeps the wire
schema a stable string contract independent of the domain enum classes.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ConflictRow, InvalidRow, ValidationReport
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus

_FORBID = ConfigDict(extra="forbid")


class ReadoutWire(BaseModel):
    """Mirrors `Readout` (`domain/catalog/readout.py`)."""

    model_config = _FORBID

    name: str
    type: str
    unit: str | None
    direction: str | None
    description: str
    threshold: float | None = None

    @classmethod
    def from_domain(cls, readout: Readout) -> ReadoutWire:
        return cls(
            name=readout.name,
            type=readout.type.value,
            unit=readout.unit,
            direction=readout.direction,
            description=readout.description,
            threshold=readout.threshold,
        )

    def to_domain(self) -> Readout:
        return Readout(
            name=self.name,
            type=ReadoutType(self.type),
            unit=self.unit,
            direction=self.direction,
            description=self.description,
            threshold=self.threshold,
        )


class TargetSpecWire(BaseModel):
    """Mirrors `TargetSpec` (`domain/data/target.py`)."""

    model_config = _FORBID

    column: str
    kind: str
    unit: str | None = None
    direction: str | None = None

    @classmethod
    def from_domain(cls, target: TargetSpec) -> TargetSpecWire:
        return cls(
            column=target.column,
            kind=target.kind.value,
            unit=target.unit,
            direction=target.direction.value if target.direction else None,
        )

    def to_domain(self) -> TargetSpec:
        return TargetSpec(
            column=self.column,
            kind=TargetKind(self.kind),
            unit=self.unit,
            direction=Direction(self.direction) if self.direction else None,
        )


class SplitSpecWire(BaseModel):
    """Mirrors `SplitSpec` (`domain/data/split.py`)."""

    model_config = _FORBID

    strategy: str
    seed: int
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)

    @classmethod
    def from_domain(cls, split: SplitSpec) -> SplitSpecWire:
        return cls(strategy=split.strategy.value, seed=split.seed, fractions=split.fractions)

    def to_domain(self) -> SplitSpec:
        return SplitSpec(
            strategy=SplitStrategy(self.strategy), seed=self.seed, fractions=self.fractions
        )


class ValidationReportWire(BaseModel):
    """Mirrors `ValidationReport` (`domain/data/validation.py`).

    `invalid`/`conflicting` type straight to the domain's own `InvalidRow`/
    `ConflictRow` frozen dataclasses -- pydantic validates and serializes
    plain dataclasses natively, so a dedicated wire mirror for each would be
    pure duplication of fields that carry no enum and need no cast.
    """

    model_config = _FORBID

    total_rows: int
    valid_rows: int
    invalid: list[InvalidRow] = []
    conflicting: list[ConflictRow] = []
    duplicates_collapsed: int = 0
    salts_flagged: int = 0
    duplicate_spread: dict[str, float] = {}

    @classmethod
    def from_domain(cls, report: ValidationReport) -> ValidationReportWire:
        return cls(
            total_rows=report.total_rows,
            valid_rows=report.valid_rows,
            invalid=list(report.invalid),
            conflicting=list(report.conflicting),
            duplicates_collapsed=report.duplicates_collapsed,
            salts_flagged=report.salts_flagged,
            duplicate_spread=report.duplicate_spread,
        )

    def to_domain(self) -> ValidationReport:
        return ValidationReport(
            total_rows=self.total_rows,
            valid_rows=self.valid_rows,
            invalid=list(self.invalid),
            conflicting=list(self.conflicting),
            duplicates_collapsed=self.duplicates_collapsed,
            salts_flagged=self.salts_flagged,
            duplicate_spread=self.duplicate_spread,
        )


class RunEnvelope(BaseModel):
    """Mirrors every `Run.__init__` kwarg (`domain/execution/run.py`)."""

    model_config = _FORBID

    id: uuid.UUID
    kind: str
    workspace_id: uuid.UUID
    requested_by: uuid.UUID
    cache_key: str
    params: dict[str, Any]
    protocol_id: uuid.UUID | None
    status: str
    progress: float
    phase: str | None
    result_uri: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    version: int
    sweep_id: uuid.UUID | None = None
    metrics: dict[str, Any] | None = None

    @classmethod
    def from_domain(cls, run: Run) -> RunEnvelope:
        return cls(
            id=run.id,
            kind=run.kind.value,
            workspace_id=run.workspace_id,
            requested_by=run.requested_by,
            cache_key=run.cache_key,
            params=dict(run.params),
            protocol_id=run.protocol_id,
            status=run.status.value,
            progress=run.progress,
            phase=run.phase,
            result_uri=run.result_uri,
            error_message=run.error_message,
            created_at=run.created_at,
            updated_at=run.updated_at,
            version=run.version,
            sweep_id=run.sweep_id,
            metrics=run.metrics,
        )

    def to_domain(self) -> Run:
        return Run(
            id=self.id,
            kind=RunKind(self.kind),
            workspace_id=self.workspace_id,
            requested_by=self.requested_by,
            cache_key=self.cache_key,
            params=self.params,
            protocol_id=self.protocol_id,
            status=RunStatus(self.status),
            progress=self.progress,
            phase=self.phase,
            result_uri=self.result_uri,
            error_message=self.error_message,
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
            sweep_id=self.sweep_id,
            metrics=self.metrics,
        )


class TargetHeadlineWire(BaseModel):
    model_config = _FORBID

    # Bounded like the rest of the shape: a column name is at most as long as the
    # dataset column it came from, and a metric name is one of a short fixed set.
    column: str = Field(max_length=1024)
    primary_metric: str = Field(max_length=64)
    value: float | None
    baseline_value: float | None


class RunMetricsWire(BaseModel):
    """Mirrors exactly what `Run.record_metrics` writes (`domain/execution/run.py`):
    one headline per target -- the only shape a runner-reported training `metrics`
    payload may take.

    A free `dict[str, Any]` here would let a self-hosted runner store
    arbitrary JSON verbatim in the `runs.metrics` column, inside the same
    security boundary the module docstring calls out. Pinning the shape
    means a malformed payload is a 422 at the edge, not a write.
    """

    model_config = _FORBID

    targets: list[TargetHeadlineWire] = Field(max_length=4096)


class PredictionCountsWire(BaseModel):
    """The other shape `runs.metrics` takes: `Run.record_prediction_counts` on a
    prediction run. Same reasoning as `RunMetricsWire` -- two closed shapes, so
    a runner still cannot store arbitrary JSON through this endpoint."""

    model_config = _FORBID

    uploaded_rows: int
    scored_rows: int


class RunUpdateEnvelope(BaseModel):
    """Only the mutable fields a handler writes back while a run is in
    flight -- not a full `Run` mirror, so no `to_domain`. `expected_version`
    carries the optimistic-concurrency check the repository's `update()`
    already enforces (see `infrastructure/persistence/sqlalchemy/execution/
    repository.py`).

    `status` and `expected_version` are the only required fields.  Every
    other field defaults to `None` and is applied only when the caller
    actually sent it (`interface/routes/runner_api.py` checks
    `model_fields_set`) -- a required-and-nullable field here would mean any
    update that legitimately omits `phase`/`progress`/etc. (e.g. a bare
    heartbeat-style status flip) silently nulls out whatever the previous
    update had reported, which is what the security review's Important 1
    finding caught in practice (progress 0.9 -> 0.0, phase "fit" -> None).
    """

    model_config = _FORBID

    status: str
    expected_version: int
    progress: float | None = None
    phase: str | None = None
    result_uri: str | None = None
    error_message: str | None = None
    protocol_id: uuid.UUID | None = None
    metrics: RunMetricsWire | PredictionCountsWire | None = None


class DatasetEnvelope(BaseModel):
    """Mirrors every `Dataset.__init__` kwarg (`domain/data/dataset.py`)."""

    model_config = _FORBID

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    structure_column: str
    targets: list[TargetSpecWire]
    split: SplitSpecWire
    content_hash: str
    snapshot_uri: str
    row_count: int
    validation_report: ValidationReportWire
    created_at: datetime
    updated_at: datetime
    version: int

    @classmethod
    def from_domain(cls, dataset: Dataset) -> DatasetEnvelope:
        return cls(
            id=dataset.id,
            workspace_id=dataset.workspace_id,
            name=dataset.name,
            structure_column=dataset.structure_column,
            targets=[TargetSpecWire.from_domain(t) for t in dataset.targets],
            split=SplitSpecWire.from_domain(dataset.split),
            content_hash=dataset.content_hash,
            snapshot_uri=dataset.snapshot_uri,
            row_count=dataset.row_count,
            validation_report=ValidationReportWire.from_domain(dataset.validation_report),
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            version=dataset.version,
        )

    def to_domain(self) -> Dataset:
        return Dataset(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            structure_column=self.structure_column,
            targets=tuple(t.to_domain() for t in self.targets),
            split=self.split.to_domain(),
            content_hash=self.content_hash,
            snapshot_uri=self.snapshot_uri,
            row_count=self.row_count,
            validation_report=self.validation_report.to_domain(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
        )


class ProtocolEnvelope(BaseModel):
    """Mirrors every `InSilicoProtocol.__init__` kwarg
    (`domain/catalog/protocol.py`)."""

    model_config = _FORBID

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    dataset_id: uuid.UUID
    engine_id: str
    artifact_uri: str
    readouts: tuple[ReadoutWire, ...]
    conditions: dict[str, Any]
    status: str
    published_at: datetime | None
    parent_protocol_id: uuid.UUID | None
    protocol_version: int
    created_at: datetime
    updated_at: datetime
    version: int

    @classmethod
    def from_domain(cls, protocol: InSilicoProtocol) -> ProtocolEnvelope:
        return cls(
            id=protocol.id,
            workspace_id=protocol.workspace_id,
            name=protocol.name,
            dataset_id=protocol.dataset_id,
            engine_id=protocol.engine_id,
            artifact_uri=protocol.artifact_uri,
            readouts=tuple(ReadoutWire.from_domain(readout) for readout in protocol.readouts),
            conditions=dict(protocol.conditions),
            status=protocol.status.value,
            published_at=protocol.published_at,
            parent_protocol_id=protocol.parent_protocol_id,
            protocol_version=protocol.protocol_version,
            created_at=protocol.created_at,
            updated_at=protocol.updated_at,
            version=protocol.version,
        )

    def to_domain(self) -> InSilicoProtocol:
        return InSilicoProtocol(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            dataset_id=self.dataset_id,
            engine_id=self.engine_id,
            artifact_uri=self.artifact_uri,
            readouts=tuple(readout.to_domain() for readout in self.readouts),
            conditions=self.conditions,
            status=ProtocolStatus(self.status),
            published_at=self.published_at,
            parent_protocol_id=self.parent_protocol_id,
            protocol_version=self.protocol_version,
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=self.version,
        )


class ClaimResponse(BaseModel):
    """What a runner gets back for claiming a queued run: the run itself,
    plus two DIFFERENT numbers a runner must not conflate (Important 1+2,
    final review, after they had been):

    - `deadline_seconds`: the JOB's own soft deadline -- `worker_job_timeout`,
      what `TrainContext.report` checks a fit's elapsed time against
      (`application/execution/train_protocol.py`) and raises `RunInterrupted`
      past. A run past this point is considered to have overrun on its own
      merits, not to have gone silent.
    - `lease_seconds`: how long the runner's CLAIM on this run can go
      unrenewed before the server treats it as abandoned and requeues it
      (`RunQueue.claim_next`'s `lease_seconds`, `settings.runner_lease_seconds`).
      Renewed by any authenticated call against a claimed run (`claimed_run`'s
      own docstring) -- the agent's heartbeat (`infrastructure/runner/agent.py`)
      exists so that renewal keeps happening even when the engine executing
      the job never calls back into `ctx` on its own.
    """

    model_config = _FORBID

    run: RunEnvelope
    deadline_seconds: int
    lease_seconds: int


class BlobPutResponse(BaseModel):
    model_config = _FORBID

    uri: str


class RunUpdateResponse(BaseModel):
    model_config = _FORBID

    version: int

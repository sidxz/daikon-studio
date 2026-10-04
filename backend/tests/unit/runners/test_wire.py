"""Roundtrip coverage for the runner protocol wire envelopes.

`Entity.__eq__` only compares `id` (see `domain/shared/entity.py`), so a
roundtrip that silently dropped a field would still pass a plain `==`
assertion. Every aggregate-backed envelope is therefore compared via
`__dict__` instead; the frozen-dataclass value objects (`TargetSpec`,
`SplitSpec`, `ValidationReport`'s rows, `Readout`) get a real `==` from
`@dataclass`, so those compare directly wherever they show up nested inside
that `__dict__`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from daikonstudio.domain.catalog.protocol import InSilicoProtocol, ProtocolStatus
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ConflictRow, InvalidRow, ValidationReport
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.infrastructure.runner.wire import (
    BlobPutResponse,
    ClaimResponse,
    DatasetEnvelope,
    ProtocolEnvelope,
    RunEnvelope,
    RunUpdateEnvelope,
    RunUpdateResponse,
)

_NOW = datetime(2026, 8, 5, 12, 30, tzinfo=UTC)
_EARLIER = datetime(2026, 8, 1, 9, 0, tzinfo=UTC)


def _run() -> Run:
    return Run(
        id=uuid.uuid4(),
        kind=RunKind.TRAINING,
        workspace_id=uuid.uuid4(),
        requested_by=uuid.uuid4(),
        cache_key="a" * 64,
        params={"engine_id": "xgboost", "conditions": {"assay": "IC50"}, "seed": 7},
        protocol_id=uuid.uuid4(),
        status=RunStatus.RUNNING,
        progress=0.42,
        phase="featurizing",
        result_uri="s3://bucket/run/result.parquet",
        error_message=None,
        created_at=_EARLIER,
        updated_at=_NOW,
        version=3,
    )


def _dataset() -> Dataset:
    return Dataset(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        name="herg-binders",
        structure_column="smiles",
        # Two targets, so the round trip also pins that their order survives the wire.
        targets=(
            TargetSpec(column="ic50", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW),
            TargetSpec(column="active", kind=TargetKind.BINARY),
        ),
        split=SplitSpec(strategy=SplitStrategy.SCAFFOLD, seed=42, fractions=(0.7, 0.15, 0.15)),
        content_hash="b" * 64,
        snapshot_uri="s3://bucket/dataset/snapshot.parquet",
        row_count=1234,
        validation_report=ValidationReport(
            total_rows=1250,
            valid_rows=1234,
            invalid=[
                InvalidRow(row_number=3, value="not-a-smiles", reason="unparseable structure"),
                InvalidRow(row_number=91, value="", reason="missing target value"),
            ],
            conflicting=[
                ConflictRow(structure="CCO", column="active", values=[1, 0], row_numbers=[12, 88]),
            ],
            duplicates_collapsed=9,
            salts_flagged=2,
            duplicate_spread={"ic50": 0.038},
        ),
        created_at=_EARLIER,
        updated_at=_NOW,
        version=1,
    )


def _protocol() -> InSilicoProtocol:
    return InSilicoProtocol(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        name="herg-binders-v1",
        dataset_id=uuid.uuid4(),
        engine_id="chemeleon",
        artifact_uri="s3://bucket/protocol/artifact.bin",
        readouts=(
            Readout(
                name="ic50",
                type=ReadoutType.NUMERIC,
                unit="nM",
                direction="low",
                description="Predicted IC50",
            ),
            Readout(
                name="active",
                type=ReadoutType.PROBABILITY,
                unit=None,
                direction=None,
                description="Probability of activity",
            ),
            # A tuned cutoff must survive the wire, or a runner would label at 0.5.
            Readout(
                name="active_class",
                type=ReadoutType.CLASS,
                unit=None,
                direction=None,
                description="Predicted active class",
                threshold=0.3,
            ),
        ),
        conditions={"assay": "IC50", "temperature_c": 25, "replicates": 3},
        status=ProtocolStatus.PUBLISHED,
        published_at=_NOW,
        parent_protocol_id=uuid.uuid4(),
        protocol_version=2,
        created_at=_EARLIER,
        updated_at=_NOW,
        version=4,
    )


def test_run_envelope_roundtrip() -> None:
    run = _run()
    result = RunEnvelope.from_domain(run).to_domain()
    assert result.__dict__ == run.__dict__


def test_run_envelope_json_roundtrip() -> None:
    run = _run()
    envelope = RunEnvelope.from_domain(run)
    result = RunEnvelope.model_validate_json(envelope.model_dump_json()).to_domain()
    assert result.__dict__ == run.__dict__


def test_dataset_envelope_roundtrip() -> None:
    dataset = _dataset()
    result = DatasetEnvelope.from_domain(dataset).to_domain()
    assert result.__dict__ == dataset.__dict__


def test_dataset_envelope_json_roundtrip() -> None:
    dataset = _dataset()
    envelope = DatasetEnvelope.from_domain(dataset)
    result = DatasetEnvelope.model_validate_json(envelope.model_dump_json()).to_domain()
    assert result.__dict__ == dataset.__dict__


def test_protocol_envelope_roundtrip() -> None:
    protocol = _protocol()
    result = ProtocolEnvelope.from_domain(protocol).to_domain()
    assert result.__dict__ == protocol.__dict__
    assert result.conditions == protocol.conditions


def test_protocol_envelope_json_roundtrip() -> None:
    protocol = _protocol()
    envelope = ProtocolEnvelope.from_domain(protocol)
    result = ProtocolEnvelope.model_validate_json(envelope.model_dump_json()).to_domain()
    assert result.__dict__ == protocol.__dict__
    assert result.conditions == protocol.conditions


def test_a_readouts_threshold_survives_the_wire_and_its_absence_stays_absent() -> None:
    tuned, untuned = _protocol().readouts[2], _protocol().readouts[1]
    assert (tuned.threshold, untuned.threshold) == (0.3, None)

    envelope = ProtocolEnvelope.from_domain(_protocol())
    thresholds = [readout.threshold for readout in envelope.readouts]
    assert thresholds == [None, None, 0.3]
    assert [r.threshold for r in envelope.to_domain().readouts] == [None, None, 0.3]


def test_bare_dtos_json_roundtrip() -> None:
    """The four envelopes with no `to_domain` -- unlike the aggregate
    mirrors above, plain pydantic equality (field-by-field) is exactly
    right here, since none of these carry an `Entity` with an id-only
    `__eq__`."""
    instances = [
        RunUpdateEnvelope(
            status="running",
            progress=0.75,
            phase="training",
            result_uri=None,
            error_message=None,
            protocol_id=uuid.uuid4(),
            expected_version=2,
        ),
        ClaimResponse(run=RunEnvelope.from_domain(_run()), deadline_seconds=300, lease_seconds=60),
        BlobPutResponse(uri="s3://bucket/blob/artifact.bin"),
        RunUpdateResponse(version=5),
    ]
    for instance in instances:
        assert type(instance).model_validate_json(instance.model_dump_json()) == instance


def test_run_update_metrics_accepts_both_closed_shapes_and_nothing_else():
    """`runs.metrics` holds a training run's headline (`record_metrics`) or a
    prediction run's counts (`record_prediction_counts`). A runner may report
    either; a free-form payload is still a 422 at the edge, never a write."""
    import pytest
    from pydantic import ValidationError as PydanticValidationError

    from daikonstudio.infrastructure.runner.wire import RunUpdateEnvelope

    training = RunUpdateEnvelope(
        status="ready",
        expected_version=1,
        metrics={
            "targets": [
                {"column": "y", "primary_metric": "mcc", "value": 0.6, "baseline_value": 0.5}
            ]
        },
    )
    assert training.metrics is not None
    assert training.metrics.model_dump() == {
        "targets": [{"column": "y", "primary_metric": "mcc", "value": 0.6, "baseline_value": 0.5}]
    }

    prediction = RunUpdateEnvelope(
        status="ready", expected_version=1, metrics={"uploaded_rows": 4, "scored_rows": 3}
    )
    assert prediction.metrics is not None
    assert prediction.metrics.model_dump() == {"uploaded_rows": 4, "scored_rows": 3}

    with pytest.raises(PydanticValidationError):
        RunUpdateEnvelope(status="ready", expected_version=1, metrics={"anything": "goes"})

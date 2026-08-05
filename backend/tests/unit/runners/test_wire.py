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
    DatasetEnvelope,
    ProtocolEnvelope,
    RunEnvelope,
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
        target=TargetSpec(
            column="ic50", kind=TargetKind.NUMERIC, unit="nM", direction=Direction.LOW
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
                ConflictRow(structure="CCO", values=[1, 0], row_numbers=[12, 88]),
            ],
            duplicates_collapsed=9,
            salts_flagged=2,
            duplicate_spread=0.038,
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

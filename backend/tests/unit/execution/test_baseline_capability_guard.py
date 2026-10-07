"""The baseline task-capability guard, unreachable through the real registry:
all three shipped engines declare both tasks (see `default_registry`'s own
docstring and `tests/integration/test_train_protocol.py`'s coverage), so
reaching it needs a registry holding a stub that declares only one.

A focused unit test rather than an integration one -- `RunTraining.__call__`
raises this before the parquet read (`self._store.get_bytes(...)`), so
nothing past the Dataset lookup and the two engines' manifests is ever
touched, and nothing here needs a database, a blob store, or a worker
dispatcher to prove it.
"""

from __future__ import annotations

import uuid

import polars as pl
import pytest

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import EngineManifest, TaskType
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.application.execution.train_protocol import RunTraining, TrainProtocolCommand
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.shared.errors import ValidationError


class _StubEngine:
    def __init__(self, engine_id: str, tasks: tuple[TaskType, ...]) -> None:
        self._manifest = EngineManifest(
            id=engine_id, version="1.0.0", name=engine_id, description="", tasks=tasks
        )

    def manifest(self) -> EngineManifest:
        return self._manifest

    def train(self, ctx: TrainContext) -> TrainResult:  # pragma: no cover - guard fires first
        raise NotImplementedError

    def predict(self, ctx: PredictContext) -> pl.DataFrame:  # pragma: no cover - guard fires first
        raise NotImplementedError


class _StubDatasets:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    async def get(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> Dataset | None:
        return self._dataset


def _binary_dataset() -> Dataset:
    return Dataset(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        name="a dataset",
        structure_column="smiles",
        targets=(TargetSpec(column="y", kind=TargetKind.BINARY),),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=7),
        content_hash="deadbeef",
        snapshot_uri="file:///snapshot.parquet",
        row_count=1,
        validation_report=ValidationReport(total_rows=1, valid_rows=1),
    )


async def test_a_baseline_that_cannot_serve_the_task_fails_before_any_fit() -> None:
    """The failure must name the *baseline* engine, not the chosen one:
    reusing the chosen engine's message would send a scientist to inspect the
    model they picked, which is fine, over a baseline they never chose."""
    dataset = _binary_dataset()
    chosen = _StubEngine("chosen", (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION))
    regression_only_baseline = _StubEngine("regression-only", (TaskType.REGRESSION,))
    registry = EngineRegistry({"chosen": chosen, "regression-only": regression_only_baseline})

    command = TrainProtocolCommand(
        name="doomed",
        dataset_id=dataset.id,
        engine_id="chosen",
        conditions={},
        baseline_engine_id="regression-only",
    )
    run = Run(
        kind=RunKind.TRAINING,
        workspace_id=dataset.workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k",
        params=command.to_params(),
    )
    # Only `datasets` and `engines` are real collaborators: the guard fires
    # before anything else this constructor takes is ever touched.
    training = RunTraining(_StubDatasets(dataset), None, None, None, registry, None)  # type: ignore[arg-type]

    # Raising anything other than ValidationError here (a bare
    # NotImplementedError from `_StubEngine.train`/`.predict`, say) would mean
    # the guard did not stop the run before a fit was attempted.
    with pytest.raises(ValidationError) as raised:
        await training(run)

    message = str(raised.value)
    assert "regression-only" in message
    assert "binary classification" in message
    assert "chosen" not in message


def test_an_engine_is_refused_on_a_structure_kind_it_cannot_read() -> None:
    """The silent-nonsense guard, and the reason it has to exist at all.

    A wrong-kind mismatch is not a crash. ESM-2 handed canonical SMILES still returns a
    1280-column matrix, XGBoost still fits it, every metric still computes and the
    Scorecard still renders -- on embeddings of text the protein model has never seen the
    like of. Nothing downstream can notice, so this is the only place it can be stopped.
    """
    from daikonstudio.application.execution.train_protocol import _check_capable
    from daikonstudio.domain.data.structure_kind import StructureKind

    molecules = EngineManifest(
        id="ecfp4-xgboost",
        version="1.0.0",
        name="ECFP4 + XGBoost",
        description="",
        tasks=(TaskType.REGRESSION,),
    )
    sequences = EngineManifest(
        id="esm2-xgboost",
        version="1.0.0",
        name="ESM-2 embeddings + XGBoost",
        description="",
        tasks=(TaskType.REGRESSION,),
        structure_kinds=(StructureKind.SEQUENCE,),
    )

    def dataset_of(kind: StructureKind) -> Dataset:
        return Dataset(
            workspace_id=uuid.uuid4(),
            name="d",
            structure_column="structure",
            targets=(TargetSpec(column="y", kind=TargetKind.NUMERIC, unit=None),),
            split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=0),
            content_hash="h",
            snapshot_uri="s",
            row_count=1,
            validation_report=ValidationReport(total_rows=1, valid_rows=1, structure_kind=kind),
        )

    # The matching pairs are allowed.
    _check_capable(molecules, dataset_of(StructureKind.MOLECULE))
    _check_capable(sequences, dataset_of(StructureKind.SEQUENCE))

    with pytest.raises(ValidationError) as sequence_engine_on_molecules:
        _check_capable(sequences, dataset_of(StructureKind.MOLECULE))
    assert "amino-acid sequences" in str(sequence_engine_on_molecules.value)
    assert "small molecules" in str(sequence_engine_on_molecules.value)

    with pytest.raises(ValidationError) as molecule_engine_on_sequences:
        _check_capable(molecules, dataset_of(StructureKind.SEQUENCE))
    assert "small molecules" in str(molecule_engine_on_sequences.value)

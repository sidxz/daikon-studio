"""Training orchestration end to end: a CSV becomes a DRAFT Protocol plus the
raw material for an honest Scorecard.

These tests exercise the *honesty layer*, not the plumbing. Each one pins a
property the product would be worthless without:

- the baseline is trained on every run, whether or not anyone asked for it;
- the optimism gap is measured, not inferred, and only where it means something;
- the task type comes from the Dataset's TargetSpec and never from the values;
- an invalid condition fails the run before a single model is fitted;
- only the chosen engine's weights are kept -- the other two fits leave no blobs.

Everything runs through the real path: real RDKit, real sklearn/XGBoost fits,
real Postgres, real Parquet snapshots on a temp blob store, and the real
`run_job` dispatcher via `InlineEnqueuer`. Nothing about training is mocked,
because a mocked baseline is exactly the failure this file exists to prevent.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.application.data.create_dataset import (
    CreateDataset,
    CreateDatasetCommand,
    StoreUpload,
)
from daikonstudio.application.execution.train_protocol import (
    ScorecardInputs,
    TrainProtocol,
    TrainProtocolCommand,
    artifact_key,
    scorecard_inputs_key,
)
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.execution.run import Run, RunStatus
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.models import DatasetModel
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
from daikonstudio.infrastructure.worker import InlineEnqueuer
from tests.fakes.auth import FakeAuth

# Twenty compounds: one scaffold family of two (benzene/toluene), five acyclic
# rows with no scaffold at all, and thirteen distinct ring systems. Sized and
# shaped so an 80/10/10 split leaves 16/2/2 under *both* strategies -- a
# single-row test partition would make R2 undefined and print a warning, and a
# congeneric series would make the scaffold split raise (Task 10's behaviour,
# covered there).
_STRUCTURES = (
    "CCO",
    "CCN",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "C1CCOC1",
    "C1CCCC1",
    "C1CC1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",
    "C1CCC2CCCCC2C1",
    "c1cnc2ccccc2c1",
    "O=C1CCCCC1",
)


def _csv(values: tuple[float, ...] | None = None) -> bytes:
    numbers = values or tuple(1.0 + 0.37 * index for index in range(len(_STRUCTURES)))
    rows = "\n".join(
        f"{smiles},{value}" for smiles, value in zip(_STRUCTURES, numbers, strict=True)
    )
    return f"smiles,y\n{rows}\n".encode()


class Studio:
    """The whole backend, wired for one test: upload -> dataset -> train."""

    def __init__(self, sessions: async_sessionmaker, blobs: Path) -> None:
        self.auth = FakeAuth()
        self.blobs = blobs
        self.sessions = sessions
        self.store = FsspecBlobStore(f"file://{blobs}")
        self.normalizer = RdkitStructureNormalizer()
        self.datasets = SqlAlchemyDatasetRepository(sessions)
        self.protocols = SqlAlchemyProtocolRepository(sessions)
        self.runs = SqlAlchemyRunRepository(sessions)
        self._upload = StoreUpload(self.store)
        self._create = CreateDataset(self.datasets, self.store, self.normalizer)
        self._train = TrainProtocol(self.datasets, self.runs, InlineEnqueuer(sessions, self.store))

    async def dataset(
        self,
        *,
        strategy: SplitStrategy = SplitStrategy.RANDOM,
        kind: TargetKind = TargetKind.NUMERIC,
        values: tuple[float, ...] | None = None,
        unit: str | None = "logS",
    ) -> Dataset:
        upload_ref = (await self._upload(_csv(values), self.auth)).unwrap()
        command = CreateDatasetCommand(
            name=f"dataset-{strategy.value}-{kind.value}",
            upload_ref=str(upload_ref),
            structure_column="smiles",
            target=TargetSpec(column="y", kind=kind, unit=unit, direction=Direction.HIGH),
            split=SplitSpec(strategy=strategy, seed=7),
        )
        return (await self._create(command, self.auth)).unwrap()

    async def train(
        self, *, dataset_id: uuid.UUID, engine_id: str, conditions: dict[str, object]
    ) -> Run:
        command = TrainProtocolCommand(
            name="a trained model",
            dataset_id=dataset_id,
            engine_id=engine_id,
            conditions=conditions,
        )
        return (await self._train(command, self.auth)).unwrap()

    async def wait(self, run: Run) -> Run:
        """`InlineEnqueuer` has already executed the job by the time `train()`
        returns, so this is a reload rather than a poll -- but the tests read
        the same either way if the enqueuer is ever swapped for the arq one."""
        return await self.reload(run)

    async def reload(self, run: Run) -> Run:
        reloaded = await self.runs.get(self.auth.workspace_id, run.id)
        assert reloaded is not None
        return reloaded

    async def scorecard_for(self, run: Run) -> ScorecardInputs:
        reloaded = await self.reload(run)
        assert reloaded.result_uri is not None, reloaded.error_message
        key = reloaded.result_uri.removeprefix(f"file://{self.blobs}/")
        return ScorecardInputs.from_json(self.store.get_bytes(key))

    async def protocol_for(self, run: Run) -> InSilicoProtocol:
        inputs = await self.scorecard_for(run)
        protocol = await self.protocols.get(self.auth.workspace_id, uuid.UUID(inputs.protocol_id))
        assert protocol is not None
        return protocol


@pytest_asyncio.fixture
async def studio(_migrated_engine: AsyncEngine, tmp_path: Path) -> AsyncIterator[Studio]:
    """One connection + one outer transaction per test, rolled back at teardown --
    the same recipe as the other integration fixtures of this name."""
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield Studio(
            async_sessionmaker(
                bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
            ),
            tmp_path,
        )
        await connection.rollback()


async def test_training_produces_a_draft_protocol_with_derived_readouts(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    protocol = await studio.protocol_for(run)
    assert protocol.status.value == "draft"
    assert protocol.readouts[0].unit == dataset.target.unit
    assert protocol.engine_id == "ecfp4-xgboost"
    assert protocol.dataset_id == dataset.id
    assert (await studio.reload(run)).status is RunStatus.READY


async def test_training_always_also_trains_the_baseline(studio: Studio) -> None:
    """The baseline is mandatory, not a checkbox."""
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.baseline_metrics is not None
    assert scorecard.baseline_engine_id == "ecfp4-randomforest"
    assert scorecard.baseline_is_self is False
    # Two genuinely different fits, not the same numbers copied twice.
    assert scorecard.baseline_metrics != scorecard.metrics


async def test_a_scaffold_split_also_reports_the_random_split_number(studio: Studio) -> None:
    """The optimism gap must be visible rather than inferred."""
    dataset = await studio.dataset(strategy=SplitStrategy.SCAFFOLD)
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.random_split_metrics is not None
    assert scorecard.random_split_unavailable is None
    assert set(scorecard.random_split_metrics) == set(scorecard.metrics)


async def test_a_random_split_reports_no_optimism_gap(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.random_split_metrics is None
    assert scorecard.random_split_unavailable is None


async def test_invalid_conditions_fail_the_run_with_a_useful_message(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={"max_depth": 999}
    )
    await studio.wait(run)

    failed = await studio.reload(run)
    assert failed.status.value == "failed"
    assert failed.error_message is not None
    assert "max_depth" in failed.error_message
    # Failed before any compute: no protocol, no artifact, no scorecard.
    assert not (studio.blobs / str(studio.auth.workspace_id) / "protocols").exists()


async def test_choosing_the_baseline_engine_says_so_instead_of_faking_a_comparison(
    studio: Studio,
) -> None:
    """Training ECFP4+RandomForest against ECFP4+RandomForest is one fit, not two.

    The Scorecard must not present the identical numbers twice as though an
    independent baseline had been beaten (or lost to) -- `baseline_is_self`
    is what lets Task 15 say "this model is the baseline" out loud.
    """
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.baseline_is_self is True
    assert scorecard.baseline_engine_id == "ecfp4-randomforest"
    assert scorecard.baseline_metrics == scorecard.metrics


async def test_non_default_conditions_still_earn_a_real_baseline(studio: Studio) -> None:
    """Same engine, different hyperparameters: the canonical baseline is a
    different fit and has to actually run."""
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id,
        engine_id="ecfp4-randomforest",
        conditions={"n_estimators": 40},
    )
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.baseline_is_self is False
    assert scorecard.conditions["n_estimators"] == 40


async def test_a_failed_optimism_gap_does_not_destroy_the_honest_result(
    studio: Studio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scaffold number is the primary result; the random-split number is a
    nice-to-have. Losing the second must not take the first down with it."""
    import daikonstudio.application.execution.train_protocol as module

    real = module.assign_split

    def explode(frame, structure_column, spec, normalizer):  # type: ignore[no-untyped-def]
        if spec.strategy is SplitStrategy.RANDOM:
            raise ValueError("no random split for you")
        return real(frame, structure_column, spec, normalizer)

    monkeypatch.setattr(module, "assign_split", explode)

    dataset = await studio.dataset(strategy=SplitStrategy.SCAFFOLD)
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    assert (await studio.reload(run)).status is RunStatus.READY
    scorecard = await studio.scorecard_for(run)
    assert scorecard.random_split_metrics is None
    assert scorecard.random_split_unavailable is not None
    assert "no random split for you" in scorecard.random_split_unavailable
    assert scorecard.metrics and scorecard.baseline_metrics


async def test_the_task_comes_from_the_target_spec_not_from_the_values(
    studio: Studio,
) -> None:
    """A numeric target that happens to hold only 0.0 and 1.0 is still regression.

    Inferring from the values would silently train a classifier and report MCC
    for something the scientist declared continuous.
    """
    values = tuple(float(index % 2) for index in range(len(_STRUCTURES)))
    dataset = await studio.dataset(kind=TargetKind.NUMERIC, values=values)
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.task == "regression"
    assert set(scorecard.metrics) == {"rmse", "mae", "r2"}
    protocol = await studio.protocol_for(run)
    assert [readout.type.value for readout in protocol.readouts] == ["numeric"]


async def test_a_binary_target_trains_a_classifier_and_derives_two_readouts(
    studio: Studio,
) -> None:
    values = tuple(float(index % 2) for index in range(len(_STRUCTURES)))
    dataset = await studio.dataset(kind=TargetKind.BINARY, values=values, unit=None)
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.task == "binary_classification"
    assert set(scorecard.metrics) == {"mcc", "balanced_accuracy", "auroc", "auprc"}
    assert "accuracy" not in scorecard.metrics
    protocol = await studio.protocol_for(run)
    assert [readout.type.value for readout in protocol.readouts] == ["probability", "class"]


async def test_only_the_chosen_engine_leaves_an_artifact_behind(studio: Studio) -> None:
    """Three fits, one artifact. The baseline and the random-split model exist to
    produce numbers, not to be run -- persisting their weights would orphan two
    blobs no row references."""
    dataset = await studio.dataset(strategy=SplitStrategy.SCAFFOLD)
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    protocol = await studio.protocol_for(run)
    artifacts = sorted(
        (studio.blobs / str(studio.auth.workspace_id) / "protocols").rglob("*.joblib")
    )
    assert len(artifacts) == 1
    assert protocol.artifact_uri.endswith(artifact_key(studio.auth.workspace_id, protocol.id))
    assert studio.store.exists(scorecard_inputs_key(studio.auth.workspace_id, protocol.id))


async def test_scorecard_inputs_carry_the_test_set_predictions(studio: Studio) -> None:
    """Task 15 builds the Scorecard from these; if they don't line up, worst_rows
    and applicability coverage are meaningless."""
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    card = await studio.scorecard_for(run)
    assert len(card.actual) == len(card.predicted) == len(card.structures) == 2
    assert len(card.train_structures) == 16
    assert not set(card.structures) & set(card.train_structures)
    assert card.duplicate_spread is None


async def test_progress_is_reported_through_the_named_phases(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    done = await studio.reload(run)
    assert done.progress == 1.0
    assert done.phase == "training baseline"


async def test_training_another_workspaces_dataset_is_a_not_found(studio: Studio) -> None:
    dataset = await studio.dataset()
    studio.auth.workspace_id = uuid.uuid4()

    result = await studio._train(
        TrainProtocolCommand(
            name="theft",
            dataset_id=dataset.id,
            engine_id="ecfp4-xgboost",
            conditions={},
        ),
        studio.auth,
    )

    assert result.failure().__class__.__name__ == "NotFoundError"


async def test_an_unknown_engine_fails_the_run(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="nope", conditions={})
    await studio.wait(run)

    failed = await studio.reload(run)
    assert failed.status is RunStatus.FAILED
    assert failed.error_message is not None
    assert "nope" in failed.error_message


async def test_a_failed_scorecard_write_leaves_no_protocol_behind(studio: Studio) -> None:
    """A Protocol row with no scorecard behind it is a publishable model with no
    honesty data -- Task 16 would list it and then have nothing to serve. The row
    is written last precisely so no partial failure can produce one."""
    real_put = studio.store.put_bytes

    def fail_on_scorecard(key: str, data: bytes) -> str:
        if key.endswith("scorecard-inputs.json"):
            raise OSError("blob store went away")
        return real_put(key, data)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(studio.store, "put_bytes", fail_on_scorecard)
    try:
        dataset = await studio.dataset()
        run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    finally:
        monkeypatch.undo()

    failed = await studio.reload(run)
    assert failed.status is RunStatus.FAILED
    assert "blob store went away" in (failed.error_message or "")
    assert await studio.protocols.list(studio.auth.workspace_id) == []


async def test_the_stored_scorecard_is_valid_json_even_when_a_metric_is_undefined(
    studio: Studio,
) -> None:
    """A single-class test split makes every classification metric meaningless.
    That is real and must be recorded -- but as `null` plus a reason, not as the
    bare `NaN` token, which is not JSON: strict parsers reject the whole document
    and jq silently turns it into null with no explanation attached."""
    dataset = await studio.dataset(
        kind=TargetKind.BINARY, values=(0.0,) * len(_STRUCTURES), unit=None
    )
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    protocol = await studio.protocol_for(run)
    raw = studio.store.get_bytes(scorecard_inputs_key(studio.auth.workspace_id, protocol.id))
    assert b"NaN" not in raw

    # `parse_constant` is only called for NaN/Infinity/-Infinity, so this is a
    # decoder that rejects exactly the non-standard tokens a strict parser would.
    def reject(token: str) -> object:
        raise AssertionError(f"non-standard JSON token in stored scorecard: {token}")

    document = json.loads(raw, parse_constant=reject)

    assert document["metrics"]["mcc"] is None
    card = await studio.scorecard_for(run)
    assert card.metrics["mcc"] is None
    assert card.metrics_undefined is not None
    assert set(card.metrics_undefined) == {"auprc", "auroc", "balanced_accuracy", "mcc"}
    assert "test split" in card.metrics_undefined["mcc"]


async def test_a_defined_metric_carries_no_undefined_reason(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    card = await studio.scorecard_for(run)
    assert card.metrics_undefined is None
    assert all(value is not None for value in card.metrics.values())


async def test_predictions_say_what_they_are_rather_than_leaving_it_to_be_inferred(
    studio: Studio,
) -> None:
    """`predicted` is the target value for regression and P(class=1) for
    classification. Task 15 must not have to re-derive that from the task."""
    regression = await studio.dataset()
    run = await studio.train(dataset_id=regression.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)
    assert (await studio.scorecard_for(run)).prediction_kind == "value"

    values = tuple(float(index % 2) for index in range(len(_STRUCTURES)))
    classification = await studio.dataset(kind=TargetKind.BINARY, values=values, unit=None)
    run = await studio.train(
        dataset_id=classification.id, engine_id="ecfp4-randomforest", conditions={}
    )
    await studio.wait(run)
    card = await studio.scorecard_for(run)
    assert card.prediction_kind == "probability"
    assert all(0.0 <= value <= 1.0 for value in card.predicted)


async def test_a_dataset_predating_the_structure_column_migration_fails_readably(
    studio: Studio,
) -> None:
    """Migration 005 backfills `'unknown'`, which is deliberately not a plausible
    column name. Training must say so, rather than reaching polars and dying with
    a ColumnNotFoundError that names neither the Dataset nor the real cause."""
    dataset = await studio.dataset()
    async with studio.sessions() as session:
        await session.execute(
            update(DatasetModel)
            .where(DatasetModel.id == dataset.id)
            .values(structure_column="unknown")
        )
        await session.commit()

    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    failed = await studio.reload(run)
    assert failed.status is RunStatus.FAILED
    assert failed.error_message is not None
    assert str(dataset.id) in failed.error_message
    assert "predates the structure_column migration" in failed.error_message


def test_default_registry_has_exactly_one_baseline() -> None:
    """`EngineRegistry.baseline()` raises on zero or two -- this is the assertion
    that the shipped registry is the one it is happy with."""
    assert default_registry().baseline().manifest().id == "ecfp4-randomforest"

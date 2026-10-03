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

import polars as pl
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
from daikonstudio.infrastructure.jobs import InlineEnqueuer
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


def _alternating_values() -> tuple[float, ...]:
    """0.0/1.0 in runs of two (`0,0,1,1,0,0,1,1,...`), not a plain `index % 2`
    alternation.

    `Studio.dataset()`'s default RANDOM split (seed=7) puts indices 9 and 11
    of `_STRUCTURES` together in its 2-row test partition (and, symmetrically,
    both in the same 2-row half of most even/odd splits) -- under a plain
    `index % 2` alternation those two indices share the same parity, so the
    test partition ends up holding only one value.

    That alone no longer matters for a BINARY target: `create_dataset.py`'s
    I1 guard (whole-branch review, narrowed on re-review) only checks a
    BINARY target's *train* partition, not its test partition -- a
    single-class test split is already reported honestly via
    `metrics_undefined` rather than refused outright, and rejecting it too
    would block a perfectly trainable model (a seed sweep against balanced
    and imbalanced binary datasets found this firing on 9-15 of 25 seeds,
    always on `test`, never `train`). It still matters for a NUMERIC target,
    whose test partition *is* checked (`r2_score` returns a misleadingly
    real-looking `0.0` on a constant test target, with nothing to flag it as
    undefined) -- this fixture's remaining callers are exactly the ones that
    still need both values present in that 2-row partition. This period-4
    pattern still yields only 0.0/1.0, the property every caller of this
    fixture actually cares about, while keeping both values present in every
    2+-row partition that split produces.
    """
    return tuple(float((index // 2) % 2) for index in range(len(_STRUCTURES)))


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
        self._train = TrainProtocol(
            self.datasets,
            self.runs,
            InlineEnqueuer(sessions, self.store),
            default_registry(),
        )

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
        self,
        *,
        dataset_id: uuid.UUID,
        engine_id: str,
        conditions: dict[str, object],
        baseline_engine_id: str | None = None,
        baseline_conditions: dict[str, object] | None = None,
    ) -> Run:
        command = TrainProtocolCommand(
            name="a trained model",
            dataset_id=dataset_id,
            engine_id=engine_id,
            conditions=conditions,
            baseline_engine_id=baseline_engine_id,
            baseline_conditions=baseline_conditions or {},
        )
        return (await self._train(command, self.auth)).unwrap()

    async def wait(self, run: Run) -> Run:
        """`InlineEnqueuer` has already executed the job by the time `train()`
        returns, so this is a reload rather than a poll -- but the tests read
        the same either way if the enqueuer is ever swapped for the database-
        backed one."""
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


async def test_a_finished_training_run_names_the_protocol_it_produced(studio: Studio) -> None:
    """A client holding a run id must be able to reach the Scorecard its own
    work just built. Before `Run.protocol_id` the only route was parsing the id
    back out of the blob path in `result_uri`, which ties every client to the
    storage layout.

    Asserted against `protocol_for`, which reads the id out of the scorecard
    blob -- so this pins the column and the blob to the same Protocol rather
    than merely checking the column is non-null.
    """
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)

    protocol = await studio.protocol_for(run)
    assert (await studio.reload(run)).protocol_id == protocol.id


async def test_a_failed_training_run_links_no_protocol(studio: Studio) -> None:
    """The link is written after the Protocol row exists, so a run that dies
    before that point leaves no dangling reference for a client to chase."""
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={"max_depth": 999}
    )
    await studio.wait(run)

    reloaded = await studio.reload(run)
    assert reloaded.status is RunStatus.FAILED
    assert reloaded.protocol_id is None


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
    assert scorecard.random_split_metrics_undefined is None
    assert set(scorecard.random_split_metrics) == set(scorecard.metrics)


async def test_a_random_split_reports_no_optimism_gap(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    assert scorecard.random_split_metrics is None
    assert scorecard.random_split_unavailable is None
    assert scorecard.random_split_metrics_undefined is None


async def test_a_metric_undefined_only_on_the_random_split_carries_its_own_reason(
    studio: Studio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL fix (Task 16 review): a metric can be undefined on the
    random-split comparison for a reason that has nothing to do with the
    Dataset's own (scaffold) split. Before the fix, that reason was computed
    and then discarded -- the random-split nulls either had no explanation at
    all (when the scaffold split's own metrics were defined, so
    `metrics_undefined` was `None`), or inherited `metrics_undefined`'s
    reason, which describes the *wrong* partition.

    Rigs the random-split reshuffle to collapse its test rows onto a single
    class, regardless of which structures land there, while the scaffold
    split -- trained for real, unpatched -- keeps both classes on both sides
    (seed=7's real assignment puts `CCN` and `CCCCO` in its test partition;
    `values[1]` is set so those two disagree).
    """
    import daikonstudio.application.execution.train_protocol as module

    real_assign_split = module.assign_split

    def collapse_random_test_split_to_one_class(frame, structure_column, spec, normalizer):  # type: ignore[no-untyped-def]
        random_frame = real_assign_split(frame, structure_column, spec, normalizer)
        if spec.strategy is not SplitStrategy.RANDOM:
            return random_frame
        return random_frame.with_columns(
            pl.when(pl.col("split") == "test").then(0.0).otherwise(pl.col("y")).alias("y")
        )

    monkeypatch.setattr(module, "assign_split", collapse_random_test_split_to_one_class)

    values = [float(index % 2) for index in range(len(_STRUCTURES))]
    values[1] = 0.0  # CCN: scaffold split (seed=7) puts this in `test` -- balance it
    dataset = await studio.dataset(
        strategy=SplitStrategy.SCAFFOLD, kind=TargetKind.BINARY, values=tuple(values), unit=None
    )
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    scorecard = await studio.scorecard_for(run)
    # The scaffold split's own test partition has both classes for real -- its
    # metrics are defined, and `metrics_undefined` correctly says there is
    # nothing to explain on that side.
    assert scorecard.metrics_undefined is None
    assert all(value is not None for value in scorecard.metrics.values())
    # The random-split comparison collapsed to one class -- its metrics are
    # undefined, and that must carry its *own* reason: not `None` (a bare,
    # unexplained null on the number the optimism gap exists to justify), and
    # not `metrics_undefined` (which would name the scaffold split's test set,
    # a different partition that was never single-class here).
    assert scorecard.random_split_metrics is not None
    assert all(value is None for value in scorecard.random_split_metrics.values())
    assert scorecard.random_split_unavailable is None  # it WAS computed, just undefined
    assert scorecard.random_split_metrics_undefined is not None
    assert set(scorecard.random_split_metrics_undefined) == set(scorecard.random_split_metrics)
    assert "test-set" in scorecard.random_split_metrics_undefined["mcc"]


async def test_invalid_conditions_fail_the_run_with_a_useful_message(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={"max_depth": 999}
    )
    await studio.wait(run)

    failed = await studio.reload(run)
    assert failed.status.value == "failed"
    assert failed.error_message is not None
    assert "Maximum tree depth" in failed.error_message
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


async def test_a_chosen_baseline_is_the_one_that_gets_fit(studio: Studio) -> None:
    """The Scorecard names the baseline that actually ran, not the flagged default."""
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id,
        engine_id="ecfp4-randomforest",
        conditions={"n_estimators": 20},
        baseline_engine_id="ecfp4-xgboost",
        baseline_conditions={"n_estimators": 20},
    )
    await studio.wait(run)

    inputs = await studio.scorecard_for(run)
    assert inputs.baseline_engine_id == "ecfp4-xgboost"
    # validate_conditions fills the rest of the manifest's defaults.
    assert inputs.baseline_conditions["n_estimators"] == 20
    assert inputs.baseline_is_self is False


async def test_same_engine_different_conditions_is_not_a_self_comparison(
    studio: Studio,
) -> None:
    """The case that makes 'pretrained vs not' work: one engine id, two settings.
    If this collapsed into baseline_is_self the second fit would never run and
    the Scorecard would present one result twice as though it were a comparison."""
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id,
        engine_id="ecfp4-randomforest",
        conditions={"n_estimators": 20},
        baseline_engine_id="ecfp4-randomforest",
        baseline_conditions={"n_estimators": 200},
    )
    await studio.wait(run)

    inputs = await studio.scorecard_for(run)
    assert inputs.baseline_is_self is False
    assert inputs.baseline_conditions["n_estimators"] == 200


def _scorecard_inputs_fixture() -> ScorecardInputs:
    """One valid, fully-populated `ScorecardInputs` -- the seed for the
    read-old-blobs test below, which deletes a key from its JSON and checks
    the default fills in."""
    return ScorecardInputs(
        protocol_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        engine_id="ecfp4-randomforest",
        task="regression",
        conditions={"n_estimators": 200},
        metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        actual=[1.0, 2.0],
        predicted=[1.1, 1.9],
        prediction_kind="value",
        structures=["CCO", "CCN"],
        train_structures=["CCCO"],
        baseline_engine_id="ecfp4-randomforest",
        baseline_conditions={"n_estimators": 200},
        baseline_metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9},
        baseline_is_self=True,
        random_split_metrics=None,
        random_split_unavailable=None,
        random_split_metrics_undefined=None,
        metrics_undefined=None,
        duplicate_spread=None,
        target_unit="logS",
        target_direction="high",
        split_strategy="random",
    )


def test_scorecard_inputs_reads_a_blob_written_before_baseline_conditions_existed() -> None:
    """`from_json` is `cls(**json.loads(data))`. Without a default, every
    Scorecard blob written before this change becomes unreadable."""
    dataset = _scorecard_inputs_fixture()  # build one valid instance at module scope
    legacy = json.loads(dataset.to_json())
    del legacy["baseline_conditions"]

    restored = ScorecardInputs.from_json(json.dumps(legacy).encode())

    assert restored.baseline_conditions == {}


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
    values = _alternating_values()
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
    # Restored to the plain alternation (I1 re-review): the narrowed guard
    # (`create_dataset.py`'s `_degenerate_partition`) no longer checks a
    # BINARY target's *test* partition -- only its train partition, which
    # this pattern never made single-class -- so `_alternating_values()`'s
    # workaround is unnecessary here.
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
    assert done.phase == "Training baseline model"


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


async def test_an_unknown_engine_is_rejected_before_a_run_is_ever_created(
    studio: Studio,
) -> None:
    """Task 16 review (Important 3): unlike an invalid *condition* -- which can
    only be checked once the worker resolves the engine's own manifest -- an
    unknown `engine_id` is a registry-membership check with exactly one
    possible answer, so `TrainProtocol` rejects it synchronously rather than
    creating a Run that is certain to fail."""
    dataset = await studio.dataset()

    result = await studio._train(
        TrainProtocolCommand(
            name="doomed",
            dataset_id=dataset.id,
            engine_id="nope",
            conditions={},
        ),
        studio.auth,
    )

    assert result.failure().__class__.__name__ == "NotFoundError"
    assert result.failure().entity_id == "nope"


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
    # An infrastructure error reaches the run as its class, not its text: an
    # OSError's text is where blob paths live (see failure_message.py).
    assert "OSError" in (failed.error_message or "")
    assert await studio.protocols.list(studio.auth.workspace_id) == []


async def test_the_stored_scorecard_is_valid_json_even_when_a_metric_is_undefined(
    studio: Studio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-class split makes every classification metric meaningless.
    That is real and must be recorded -- but as `null` plus a reason, not as
    the bare `NaN` token, which is not JSON: strict parsers reject the whole
    document and jq silently turns it into null with no explanation attached.

    This used to build the Dataset itself from an all-single-class CSV, so
    both its train and test partitions were degenerate by construction. I1
    (whole-branch review) closed exactly that door: `create_dataset.py` now
    rejects a Dataset whose train or test partition is single-class before
    training ever runs, so that CSV would now fail at `studio.dataset()`
    itself rather than reach a NaN. The one split that guard cannot see is
    the optimism gap's internal random reshuffle (`RunTraining`'s own
    `assign_split` call, for a SCAFFOLD-split Dataset only) -- not a new
    Dataset, so never routed through `CreateDataset` -- which is what this
    now collapses instead, the same rig
    `test_a_metric_undefined_only_on_the_random_split_carries_its_own_reason`
    uses to reach an undefined metric legitimately.
    """
    import daikonstudio.application.execution.train_protocol as module

    real_assign_split = module.assign_split

    def collapse_random_test_split_to_one_class(frame, structure_column, spec, normalizer):  # type: ignore[no-untyped-def]
        random_frame = real_assign_split(frame, structure_column, spec, normalizer)
        if spec.strategy is not SplitStrategy.RANDOM:
            return random_frame
        return random_frame.with_columns(
            pl.when(pl.col("split") == "test").then(0.0).otherwise(pl.col("y")).alias("y")
        )

    monkeypatch.setattr(module, "assign_split", collapse_random_test_split_to_one_class)

    dataset = await studio.dataset(
        strategy=SplitStrategy.SCAFFOLD,
        kind=TargetKind.BINARY,
        values=_alternating_values(),
        unit=None,
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

    assert document["random_split_metrics"]["mcc"] is None
    card = await studio.scorecard_for(run)
    assert card.random_split_metrics["mcc"] is None
    assert card.random_split_metrics_undefined is not None
    assert set(card.random_split_metrics_undefined) == {
        "auprc",
        "auroc",
        "balanced_accuracy",
        "mcc",
    }
    assert "test-set" in card.random_split_metrics_undefined["mcc"]


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

    # Restored to the plain alternation (I1 re-review) -- see the comment on
    # `test_a_binary_target_trains_a_classifier_and_derives_two_readouts`.
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
    assert "created before its structure column was recorded" in failed.error_message


def test_default_registry_has_exactly_one_baseline() -> None:
    """`EngineRegistry.baseline()` raises on zero or two -- this is the assertion
    that the shipped registry is the one it is happy with."""
    assert default_registry().baseline().manifest().id == "ecfp4-randomforest"


async def test_training_records_its_headline_metric(studio: Studio) -> None:
    """The number a sweep ranks on lands on the row, not only in the blob."""
    # `_alternating_values()`, not a plain alternation: RANDOM split (the
    # default here) needs both classes present in its 2-row test partition or
    # mcc comes back undefined (`None`) -- see that fixture's own docstring.
    dataset = await studio.dataset(kind=TargetKind.BINARY, values=_alternating_values(), unit=None)
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)

    trained_run = await studio.reload(run)
    assert trained_run.metrics is not None
    assert trained_run.metrics["primary_metric"] == "mcc"
    assert isinstance(trained_run.metrics["value"], float)
    assert isinstance(trained_run.metrics["baseline_value"], float)

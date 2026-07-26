"""Prediction runs: a colleague's own compounds through someone else's
published Protocol, with content-addressed result caching.

Everything here runs through the real path: real RDKit, real sklearn/XGBoost
fits and predictions, real Postgres, real Parquet on a temp blob store, and
the real `run_job` dispatcher via `InlineEnqueuer` -- the same recipe
`test_train_protocol.py` uses, extended with the prediction side.

Three properties this file exists to pin, named directly after the task's
landmines and decisions:

- a FAILED or CANCELLED Run must never be served back as a cache hit, only a
  READY one -- `find_by_cache_key` returns the most recent row regardless of
  status, so the filtering has to happen on this side of that call;
- only a *published* Protocol can be run -- a draft raises `ConflictError`
  before a Run is ever created;
- the cache key folds in the protocol version, so publishing a new version
  invalidates prior predictions rather than silently serving stale ones.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import polars as pl
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.application.catalog.publish_protocol import (
    PublishProtocol,
    PublishProtocolCommand,
)
from daikonstudio.application.data.create_dataset import (
    CreateDataset,
    CreateDatasetCommand,
    StoreUpload,
    upload_key,
)
from daikonstudio.application.execution.predict_with_protocol import (
    PredictWithProtocol,
    PredictWithProtocolCommand,
    predictions_key,
)
from daikonstudio.application.execution.train_protocol import (
    ScorecardInputs,
    TrainProtocol,
    TrainProtocolCommand,
)
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import (
    RESERVED_TARGET_COLUMNS,
    Direction,
    TargetKind,
    TargetSpec,
)
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, compute_cache_key
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
from daikonstudio.infrastructure.worker import InlineEnqueuer
from tests.fakes.auth import FakeAuth

# Twenty compounds to train on -- the same shape `test_train_protocol.py` uses
# and for the same reason: a real 16/2/2 split under either strategy.
_TRAIN_STRUCTURES = (
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

# A colleague's own compounds -- some near the training set, one far from it
# (a fluorinated aromatic, absent from the training scaffolds above), and one
# outright invalid, to exercise canonicalization and applicability together.
_QUERY_CSV = b"smiles\nCCO\nc1ccccc1\nFc1ccc(F)cc1\nnot-a-molecule\n"


def _train_csv() -> bytes:
    rows = "\n".join(
        f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_TRAIN_STRUCTURES)
    )
    return f"smiles,y\n{rows}\n".encode()


class Studio:
    """The whole backend, wired for one test: train, publish, then predict."""

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
        self._create_dataset = CreateDataset(self.datasets, self.store, self.normalizer)
        enqueuer = InlineEnqueuer(sessions, self.store)
        self._train = TrainProtocol(self.datasets, self.runs, enqueuer, default_registry())
        self._predict = PredictWithProtocol(self.protocols, self.runs, self.store, enqueuer)
        self._publish = PublishProtocol(self.protocols)

    async def upload(self, data: bytes) -> str:
        return str((await self._upload(data, self.auth)).unwrap())

    async def dataset(
        self, *, strategy: SplitStrategy = SplitStrategy.RANDOM, seed: int = 7
    ) -> Dataset:
        upload_ref = await self.upload(_train_csv())
        command = CreateDatasetCommand(
            name=f"dataset-{strategy.value}",
            upload_ref=upload_ref,
            structure_column="smiles",
            target=TargetSpec(
                column="y", kind=TargetKind.NUMERIC, unit="logS", direction=Direction.HIGH
            ),
            split=SplitSpec(strategy=strategy, seed=seed),
        )
        return (await self._create_dataset(command, self.auth)).unwrap()

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
        """`InlineEnqueuer` has already executed the job by the time the
        enqueuing call returns, so this is a reload rather than a poll."""
        return await self.reload(run)

    async def reload(self, run: Run) -> Run:
        reloaded = await self.runs.get(self.auth.workspace_id, run.id)
        assert reloaded is not None
        return reloaded

    async def protocol_for(self, run: Run) -> InSilicoProtocol:
        reloaded = await self.reload(run)
        assert reloaded.result_uri is not None, reloaded.error_message
        key = reloaded.result_uri.removeprefix(f"file://{self.blobs}/")
        inputs = ScorecardInputs.from_json(self.store.get_bytes(key))
        protocol = await self.protocols.get(self.auth.workspace_id, uuid.UUID(inputs.protocol_id))
        assert protocol is not None
        return protocol

    async def trained_and_published_protocol(
        self, *, strategy: SplitStrategy = SplitStrategy.RANDOM, engine_id: str = "ecfp4-xgboost"
    ) -> InSilicoProtocol:
        dataset = await self.dataset(strategy=strategy)
        run = await self.train(dataset_id=dataset.id, engine_id=engine_id, conditions={})
        await self.wait(run)
        protocol = await self.protocol_for(run)
        return (
            await self._publish(PublishProtocolCommand(protocol_id=protocol.id), self.auth)
        ).unwrap()

    async def publish_new_version(self, protocol: InSilicoProtocol) -> InSilicoProtocol:
        """A fresh version chained to `protocol`, published immediately.

        Reuses the same artifact rather than retraining: this exercises the
        version-invalidates-cache property, not the training pipeline again.
        """
        child = protocol.new_version(artifact_uri=protocol.artifact_uri)
        await self.protocols.add(child)
        return (
            await self._publish(PublishProtocolCommand(protocol_id=child.id), self.auth)
        ).unwrap()

    async def predict_raw(
        self,
        protocol_id: uuid.UUID,
        upload_ref: str,
        *,
        structure_column: str = "smiles",
        conditions: dict[str, object] | None = None,
    ):
        command = PredictWithProtocolCommand(
            protocol_id=protocol_id,
            upload_ref=upload_ref,
            structure_column=structure_column,
            conditions=conditions or {},
        )
        return await self._predict(command, self.auth)

    async def predict(
        self,
        protocol_id: uuid.UUID,
        upload_ref: str,
        *,
        structure_column: str = "smiles",
        conditions: dict[str, object] | None = None,
    ) -> Run:
        result = await self.predict_raw(
            protocol_id, upload_ref, structure_column=structure_column, conditions=conditions
        )
        return result.unwrap()

    def results_frame(self, run: Run) -> pl.DataFrame:
        key = predictions_key(self.auth.workspace_id, run.id)
        return pl.read_parquet(io.BytesIO(self.store.get_bytes(key)))


@pytest_asyncio.fixture
async def studio(_migrated_engine: AsyncEngine, tmp_path: Path) -> AsyncIterator[Studio]:
    async with _migrated_engine.connect() as connection:
        await connection.begin()
        yield Studio(
            async_sessionmaker(
                bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
            ),
            tmp_path,
        )
        await connection.rollback()


@pytest_asyncio.fixture
async def published_protocol(studio: Studio) -> InSilicoProtocol:
    return await studio.trained_and_published_protocol()


@pytest_asyncio.fixture
async def draft_protocol(studio: Studio) -> InSilicoProtocol:
    dataset = await studio.dataset()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    await studio.wait(run)
    return await studio.protocol_for(run)


@pytest_asyncio.fixture
async def upload_ref(studio: Studio) -> str:
    return await studio.upload(_QUERY_CSV)


async def test_identical_inputs_reuse_the_cached_run(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    first = await studio.predict(published_protocol.id, upload_ref)
    await studio.wait(first)

    second = await studio.predict(published_protocol.id, upload_ref)

    assert second.id == first.id
    assert second.result_uri == (await studio.reload(first)).result_uri
    assert second.status is RunStatus.READY  # never re-queued


async def test_different_conditions_produce_a_different_cache_key(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    first = await studio.predict(published_protocol.id, upload_ref, conditions={"threshold": 0.5})
    second = await studio.predict(published_protocol.id, upload_ref, conditions={"threshold": 0.7})
    assert first.cache_key != second.cache_key


async def test_a_different_structure_column_produces_a_different_cache_key(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    """Which column is read as structures changes what gets predicted just as
    much as the bytes or the conditions do -- caught during Task 17's own TDD
    when the original cache key omitted it and two requests differing only in
    `structure_column` collided."""
    first = await studio.predict(published_protocol.id, upload_ref, structure_column="smiles")
    second = await studio.predict_raw(
        published_protocol.id, upload_ref, structure_column="does-not-exist"
    )
    assert first.cache_key != second.unwrap().cache_key


async def test_a_new_protocol_version_invalidates_the_cache(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    """The cache key differing is necessary but not sufficient: Task 17
    review, Important 3 caught that the v2 run this produces reached `failed`
    with `FileNotFoundError` -- `RunPrediction` was re-deriving the artifact's
    and scorecard's storage keys from `protocol.id`, which for a versioned
    Protocol (no retraining has happened) does not match where those blobs
    actually live. Asserting `READY` here, not just the key inequality, is
    what would have caught that.
    """
    first = await studio.predict(published_protocol.id, upload_ref)
    first = await studio.wait(first)
    assert first.status is RunStatus.READY

    v2 = await studio.publish_new_version(published_protocol)
    second = await studio.predict(v2.id, upload_ref)
    second = await studio.wait(second)

    assert first.cache_key != second.cache_key
    assert second.status is RunStatus.READY, second.error_message


async def test_running_an_unpublished_protocol_is_rejected(
    studio: Studio, draft_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    result = await studio.predict_raw(draft_protocol.id, upload_ref)
    assert result.failure().__class__.__name__ == "ConflictError"


async def test_a_failed_run_is_not_served_back_as_a_cache_hit(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    """Landmine 1: `find_by_cache_key` returns the most recent row for a
    `(workspace_id, cache_key)` pair regardless of status. A FAILED Run under
    the *same* cache_key a fresh identical request would compute must not be
    served back as though it held a result.

    Precomputes the exact cache_key `PredictWithProtocol` itself would derive
    (the same technique the cancelled-run test below uses) so the planted
    FAILED row is a genuine collision on that key, not merely a different
    request that happens to have failed -- reusing a different
    `structure_column` here would only prove that field is in the key
    (already covered by `test_a_different_structure_column_produces_a_different_cache_key`),
    not that a same-key FAILED row falls through.
    """
    input_hash = hashlib.sha256(
        studio.store.get_bytes(upload_key(studio.auth.workspace_id, uuid.UUID(upload_ref)))
    ).hexdigest()
    cache_key = compute_cache_key(
        kind="prediction",
        protocol_id=str(published_protocol.id),
        protocol_version=published_protocol.protocol_version,
        input_hash=input_hash,
        structure_column="smiles",
        conditions={},
    )
    command = PredictWithProtocolCommand(
        protocol_id=published_protocol.id, upload_ref=upload_ref, structure_column="smiles"
    )
    failed = Run(
        kind=RunKind.PREDICTION,
        workspace_id=studio.auth.workspace_id,
        requested_by=studio.auth.user_id,
        cache_key=cache_key,
        params=command.to_params(),
    )
    await studio.runs.add(failed)
    failed.start()
    failed.fail("engine exploded")
    await studio.runs.update(failed)

    retried = await studio.predict(published_protocol.id, upload_ref)
    retried = await studio.wait(retried)

    assert retried.cache_key == cache_key  # genuine collision, not a different request
    assert retried.id != failed.id
    assert retried.status is RunStatus.READY

    # CRITICAL fix (Task 17 review): two rows now share this cache_key (the
    # FAILED one and the READY retry) -- `find_by_cache_key` must still
    # resolve to exactly one (the newest) rather than raising
    # `MultipleResultsFound`, which escaped as a raw 500 before `.limit(1)`
    # was added to the repository query.
    third = await studio.predict(published_protocol.id, upload_ref)
    assert third.id == retried.id
    assert third.status is RunStatus.READY


async def test_a_cancelled_run_is_not_served_back_as_a_cache_hit(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    """Landmine 1 + Decision 3: a cancelled Run keeps its cache_key, but a
    cancelled Run was never allowed to finish -- there is no result behind it
    to serve. The next identical request must start a fresh Run rather than
    reuse the cancelled one.

    Precomputes the exact cache_key `PredictWithProtocol` itself would derive
    for this (protocol, upload, conditions) so the pre-existing Run this test
    plants is a genuine collision, not a coincidence.
    """
    input_hash = hashlib.sha256(
        studio.store.get_bytes(upload_key(studio.auth.workspace_id, uuid.UUID(upload_ref)))
    ).hexdigest()
    cache_key = compute_cache_key(
        kind="prediction",
        protocol_id=str(published_protocol.id),
        protocol_version=published_protocol.protocol_version,
        input_hash=input_hash,
        structure_column="smiles",
        conditions={},
    )
    command = PredictWithProtocolCommand(
        protocol_id=published_protocol.id, upload_ref=upload_ref, structure_column="smiles"
    )
    pending = Run(
        kind=RunKind.PREDICTION,
        workspace_id=studio.auth.workspace_id,
        requested_by=studio.auth.user_id,
        cache_key=cache_key,
        params=command.to_params(),
    )
    await studio.runs.add(pending)
    pending.cancel()
    await studio.runs.update(pending)

    fresh = await studio.predict(published_protocol.id, upload_ref)
    fresh = await studio.wait(fresh)

    assert fresh.cache_key == cache_key  # genuine collision, not a different request
    assert fresh.id != pending.id
    assert fresh.status is RunStatus.READY


async def test_predictions_carry_structure_readouts_uncertainty_and_applicability(
    studio: Studio, published_protocol: InSilicoProtocol, upload_ref: str
) -> None:
    run = await studio.predict(published_protocol.id, upload_ref)
    run = await studio.wait(run)
    assert run.status is RunStatus.READY

    frame = studio.results_frame(run)
    # The invalid "not-a-molecule" row is dropped; the other three canonicalize.
    assert frame.height == 3
    readout_name = published_protocol.readouts[0].name
    assert readout_name in frame.columns
    assert "uncertainty" in frame.columns
    assert "applicability" in frame.columns
    # Whole-branch review follow-up (C1 staleness): every column this run
    # actually wrote, besides the dynamic readout name, must be one of the
    # names `create_dataset.py` reserves against a TargetSpec -- checked
    # against the real output of `RunPrediction`, not a hand-maintained
    # mirror of what it is believed to write.
    assert set(frame.columns) - {readout_name} <= RESERVED_TARGET_COLUMNS
    # ecfp4-xgboost: no ensemble spread to report -- never a fabricated number.
    assert frame["uncertainty"].is_null().all()
    # CCO and c1ccccc1 are training compounds themselves (similarity 1.0); the
    # fluorinated aromatic is comparatively far from anything trained on.
    applicability = dict(
        zip(frame["structure"].to_list(), frame["applicability"].to_list(), strict=True)
    )
    assert applicability["CCO"] == 1.0
    assert applicability["c1ccccc1"] == 1.0


async def test_a_classification_protocol_predicts_both_probability_and_class(
    studio: Studio, upload_ref: str
) -> None:
    # Plain alternation, restored (I1 re-review): the default RANDOM split
    # (seed=7) below puts indices 9 and 11 of `_TRAIN_STRUCTURES` together in
    # its 2-row test partition, single-class under this pattern -- but
    # `create_dataset.py`'s narrowed guard no longer checks a BINARY target's
    # test partition (only train, which this pattern never made
    # single-class; see `_scoring.py`'s NaN branch and
    # `_undefined_reasons` for why a single-class test split is already
    # handled honestly rather than refused outright).
    values = tuple(float(index % 2) for index in range(len(_TRAIN_STRUCTURES)))
    rows = "\n".join(f"{s},{v}" for s, v in zip(_TRAIN_STRUCTURES, values, strict=True))
    upload_ref_dataset = await studio.upload(f"smiles,y\n{rows}\n".encode())
    dataset = (
        await studio._create_dataset(
            CreateDatasetCommand(
                name="binary",
                upload_ref=upload_ref_dataset,
                structure_column="smiles",
                target=TargetSpec(column="y", kind=TargetKind.BINARY),
                split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=7),
            ),
            studio.auth,
        )
    ).unwrap()
    run = await studio.train(dataset_id=dataset.id, engine_id="ecfp4-randomforest", conditions={})
    await studio.wait(run)
    protocol = await studio.protocol_for(run)
    published = (
        await studio._publish(PublishProtocolCommand(protocol_id=protocol.id), studio.auth)
    ).unwrap()

    prediction_run = await studio.predict(published.id, upload_ref)
    prediction_run = await studio.wait(prediction_run)
    assert prediction_run.status is RunStatus.READY

    frame = studio.results_frame(prediction_run)
    probability_readout, class_readout = published.readouts
    assert probability_readout.name in frame.columns
    assert class_readout.name in frame.columns
    assert all(0.0 <= v <= 1.0 for v in frame[probability_readout.name].to_list())
    assert set(frame[class_readout.name].to_list()) <= {0.0, 1.0}
    assert set(frame.columns) - {probability_readout.name, class_readout.name} <= (
        RESERVED_TARGET_COLUMNS
    )

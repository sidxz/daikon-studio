"""Sweeps: N training runs submitted as one group, ranked together.

Against real Postgres and the real repository -- the columns exist to be
queried and grouped, and an ORM round-trip that never touches SQL would
not prove either.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from returns.result import Failure
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from daikonstudio.application.data.create_dataset import (
    CreateDataset,
    CreateDatasetCommand,
    StoreUpload,
)
from daikonstudio.application.engines.registry import UnknownEngineError
from daikonstudio.application.execution.sweeps import (
    CancelSweep,
    CancelSweepCommand,
    GetSweep,
    GetSweepQuery,
    ListSweeps,
    ListSweepsQuery,
    SubmitSweep,
    SubmitSweepCommand,
    SweepConfig,
)
from daikonstudio.application.execution.train_protocol import TrainProtocol
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus, TargetHeadline
from daikonstudio.domain.shared.errors import NotFoundError
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.jobs import DbEnqueuer
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.models import RunModel
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.queue import SqlAlchemyRunQueue
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
from tests.fakes.auth import FakeAuth


@pytest_asyncio.fixture
async def sessions(_migrated_engine: AsyncEngine):
    return async_sessionmaker(_migrated_engine, expire_on_commit=False)


# Twenty compounds -- the same shape `test_train_protocol.py` uses, for the
# same reason: a real 16/2/2 split under either strategy. Nothing here trains
# on the resulting Dataset (the `submit_sweep` fixture below wires `DbEnqueuer`,
# not `InlineEnqueuer`), but `CreateDataset` itself still rejects a degenerate
# partition, so the fixture has to be one it accepts.
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


def _csv() -> bytes:
    rows = "\n".join(f"{smiles},{1.0 + 0.37 * index}" for index, smiles in enumerate(_STRUCTURES))
    return f"smiles,y\n{rows}\n".encode()


def _mixed_csv() -> bytes:
    rows = "\n".join(
        f"{smiles},{1.0 + 0.37 * index},{(index // 2) % 2}"
        for index, smiles in enumerate(_STRUCTURES)
    )
    return f"smiles,y,active\n{rows}\n".encode()


@pytest.fixture
def auth() -> FakeAuth:
    return FakeAuth()


@pytest_asyncio.fixture
async def runs_repository(sessions: async_sessionmaker) -> SqlAlchemyRunRepository:
    return SqlAlchemyRunRepository(sessions)


@pytest_asyncio.fixture
async def dataset(sessions: async_sessionmaker, tmp_path: Path, auth: FakeAuth) -> Dataset:
    """A real, trainable Dataset -- built through the real `CreateDataset` use
    case rather than inserted directly, so `SubmitSweep`'s own `dataset_id`
    lookup has something real to find."""
    store = FsspecBlobStore(f"file://{tmp_path}")
    datasets = SqlAlchemyDatasetRepository(sessions)
    upload = StoreUpload(store)
    create = CreateDataset(datasets, store, RdkitStructureNormalizer())
    upload_ref = (await upload(_csv(), auth)).unwrap()
    command = CreateDatasetCommand(
        name="sweep dataset",
        upload_ref=str(upload_ref),
        structure_column="smiles",
        targets=(
            TargetSpec(column="y", kind=TargetKind.NUMERIC, unit="logS", direction=Direction.HIGH),
        ),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=7),
    )
    return (await create(command, auth)).unwrap()


@pytest_asyncio.fixture
async def mixed_dataset(sessions: async_sessionmaker, tmp_path: Path, auth: FakeAuth) -> Dataset:
    """Two targets of different kinds: the shape a joint engine must refuse."""
    store = FsspecBlobStore(f"file://{tmp_path}")
    datasets = SqlAlchemyDatasetRepository(sessions)
    upload = StoreUpload(store)
    create = CreateDataset(datasets, store, RdkitStructureNormalizer())
    upload_ref = (await upload(_mixed_csv(), auth)).unwrap()
    command = CreateDatasetCommand(
        name="mixed sweep dataset",
        upload_ref=str(upload_ref),
        structure_column="smiles",
        targets=(
            TargetSpec(column="y", kind=TargetKind.NUMERIC),
            TargetSpec(column="active", kind=TargetKind.BINARY),
        ),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=7),
    )
    return (await create(command, auth)).unwrap()


@pytest_asyncio.fixture
async def submit_sweep(
    sessions: async_sessionmaker, runs_repository: SqlAlchemyRunRepository
) -> SubmitSweep:
    """Built from the real `TrainProtocol` -- a fake here would prove nothing,
    since the whole design claim is that a sweep child and a solo run are the
    same object from the same code path. Wired with `DbEnqueuer` rather than
    `InlineEnqueuer` (`STUDIO_INLINE_JOBS` off): the runs only need to be
    created and land `pending` with a lane, not actually trained -- three real
    fits per config, three configs per test, would make this file minutes
    long.
    """
    datasets = SqlAlchemyDatasetRepository(sessions)
    engines = default_registry()
    queue = SqlAlchemyRunQueue(sessions)
    train = TrainProtocol(datasets, runs_repository, DbEnqueuer(queue), engines)
    return SubmitSweep(datasets, engines, train)


@pytest_asyncio.fixture
async def cancel_sweep(runs_repository: SqlAlchemyRunRepository) -> CancelSweep:
    return CancelSweep(runs_repository)


@pytest_asyncio.fixture
async def get_sweep(runs_repository: SqlAlchemyRunRepository) -> GetSweep:
    return GetSweep(runs_repository)


@pytest_asyncio.fixture
async def list_sweeps(runs_repository: SqlAlchemyRunRepository) -> ListSweeps:
    return ListSweeps(runs_repository)


def _run(*, workspace_id: uuid.UUID, sweep_id: uuid.UUID | None = None, name: str = "s") -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k" * 64,
        params={"name": name, "sweep_name": "BBBP comparison", "dataset_id": str(uuid.uuid4())},
        sweep_id=sweep_id,
    )


@pytest.mark.asyncio
async def test_sweep_id_and_metrics_round_trip(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    sweep_id = uuid.uuid4()
    run = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    await repository.add(run)

    run.start()
    run.record_metrics(
        [TargetHeadline(column="y", primary_metric="mcc", value=0.603, baseline_value=0.632)]
    )
    await repository.update(run)

    stored = await repository.get(workspace_id, run.id)
    assert stored is not None
    assert stored.sweep_id == sweep_id
    assert stored.metrics == {
        "targets": [
            {"column": "y", "primary_metric": "mcc", "value": 0.603, "baseline_value": 0.632}
        ]
    }


@pytest.mark.asyncio
async def test_solo_run_has_no_sweep_id(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    run = _run(workspace_id=workspace_id)
    await repository.add(run)

    stored = await repository.get(workspace_id, run.id)
    assert stored is not None
    assert stored.sweep_id is None
    assert stored.metrics is None


@pytest.mark.asyncio
async def test_list_by_sweep_returns_only_that_sweep(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    mine, theirs = uuid.uuid4(), uuid.uuid4()
    for _ in range(3):
        await repository.add(_run(workspace_id=workspace_id, sweep_id=mine))
    await repository.add(_run(workspace_id=workspace_id, sweep_id=theirs))
    await repository.add(_run(workspace_id=workspace_id))

    runs = await repository.list_by_sweep(workspace_id, mine)

    assert len(runs) == 3
    assert {run.sweep_id for run in runs} == {mine}


@pytest.mark.asyncio
async def test_list_by_sweep_is_workspace_scoped(sessions) -> None:
    """The filter is in the SQL, not applied after the fetch."""
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    await repository.add(_run(workspace_id=uuid.uuid4(), sweep_id=sweep_id))

    assert await repository.list_by_sweep(uuid.uuid4(), sweep_id) == []


@pytest.mark.asyncio
async def test_sweep_summaries_counts_by_status(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    sweep_id = uuid.uuid4()
    pending = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    finished = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    await repository.add(pending)
    await repository.add(finished)
    finished.start()
    finished.succeed("file:///tmp/x.json")
    await repository.update(finished)

    summaries = await repository.sweep_summaries(workspace_id)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.sweep_id == sweep_id
    assert summary.name == "BBBP comparison"
    assert summary.total == 2
    assert summary.by_status == {"pending": 1, "ready": 1}


@pytest.mark.asyncio
async def test_sweep_summaries_ignores_solo_runs(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    await repository.add(_run(workspace_id=workspace_id))

    assert await repository.sweep_summaries(workspace_id) == []


@pytest.mark.asyncio
async def test_list_sweeps_clamps_a_nonpositive_limit(sessions, list_sweeps, auth) -> None:
    """`limit=0` would otherwise render an empty page, and a negative value
    reaches Postgres as `LIMIT -1`, which it rejects with a 500 -- this
    codebase's convention is a clamped value instead, same as `ListRuns`
    (`application/pagination.py`'s `clamp_limit`)."""
    repository = SqlAlchemyRunRepository(sessions)
    for _ in range(2):
        await repository.add(_run(workspace_id=auth.workspace_id, sweep_id=uuid.uuid4()))

    zero = (await list_sweeps(ListSweepsQuery(limit=0), auth=auth)).unwrap()
    negative = (await list_sweeps(ListSweepsQuery(limit=-5), auth=auth)).unwrap()

    assert len(zero) == 1
    assert len(negative) == 1


@pytest.mark.asyncio
async def test_submit_creates_one_run_per_config_sharing_a_sweep_id(
    submit_sweep, dataset, auth
) -> None:
    result = await submit_sweep(
        SubmitSweepCommand(
            name="BBBP comparison",
            dataset_id=dataset.id,
            configs=[
                SweepConfig(engine_id="ecfp4-randomforest", conditions={"n_estimators": 100}),
                SweepConfig(engine_id="ecfp4-randomforest", conditions={"n_estimators": 300}),
                SweepConfig(engine_id="ecfp4-xgboost", conditions={}),
            ],
        ),
        auth=auth,
    )

    sweep = result.unwrap()
    assert len(sweep.runs) == 3
    assert {run.sweep_id for run in sweep.runs} == {sweep.sweep_id}
    assert [run.params["name"] for run in sweep.runs] == [
        "BBBP comparison #1",
        "BBBP comparison #2",
        "BBBP comparison #3",
    ]
    assert all(run.params["sweep_name"] == "BBBP comparison" for run in sweep.runs)


@pytest.mark.asyncio
async def test_an_unknown_engine_in_the_last_config_creates_no_runs(
    submit_sweep, dataset, auth, runs_repository
) -> None:
    """Pre-flight, not fail-halfway. A partially-submitted sweep is
    indistinguishable from a complete one, and nobody asked for it."""
    result = await submit_sweep(
        SubmitSweepCommand(
            name="doomed",
            dataset_id=dataset.id,
            configs=[
                SweepConfig(engine_id="ecfp4-randomforest", conditions={}),
                SweepConfig(engine_id="no-such-engine", conditions={}),
            ],
        ),
        auth=auth,
    )

    assert isinstance(result, Failure)
    assert await runs_repository.sweep_summaries(auth.workspace_id) == []


@pytest.mark.asyncio
async def test_a_registry_with_no_baseline_refuses_the_sweep_instead_of_raising(
    submit_sweep, dataset, auth, runs_repository, monkeypatch
) -> None:
    def no_baseline():
        raise UnknownEngineError("No baseline engine is configured on this server.")

    monkeypatch.setattr(submit_sweep._engines, "baseline", no_baseline)

    result = await submit_sweep(
        SubmitSweepCommand(
            name="no baseline",
            dataset_id=dataset.id,
            configs=[SweepConfig(engine_id="ecfp4-randomforest", conditions={})],
        ),
        auth=auth,
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), NotFoundError)
    assert await runs_repository.sweep_summaries(auth.workspace_id) == []


@pytest.mark.asyncio
async def test_an_empty_config_list_is_rejected(submit_sweep, dataset, auth) -> None:
    result = await submit_sweep(
        SubmitSweepCommand(name="empty", dataset_id=dataset.id, configs=[]),
        auth=auth,
    )
    assert isinstance(result, Failure)


@pytest.mark.asyncio
async def test_cancel_sweep_cancels_only_the_non_terminal_runs(
    sessions, cancel_sweep, auth
) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    queued = _run(workspace_id=auth.workspace_id, sweep_id=sweep_id)
    done = _run(workspace_id=auth.workspace_id, sweep_id=sweep_id)
    await repository.add(queued)
    await repository.add(done)
    done.start()
    done.succeed("file:///tmp/x.json")
    await repository.update(done)

    cancelled = (await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)).unwrap()

    assert cancelled == 1
    members = await repository.list_by_sweep(auth.workspace_id, sweep_id)
    runs = {run.id: run.status for run in members}
    assert runs[queued.id] is RunStatus.CANCELLED
    assert runs[done.id] is RunStatus.READY


@pytest.mark.asyncio
async def test_cancel_sweep_is_idempotent(sessions, cancel_sweep, auth) -> None:
    """A second cancel is not an error. The first one already stopped the
    work, and a 409 here would make the retry of a dropped request look like
    a failure."""
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    await repository.add(_run(workspace_id=auth.workspace_id, sweep_id=sweep_id))

    first = (await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)).unwrap()
    second = (await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)).unwrap()

    assert (first, second) == (1, 0)


@pytest.mark.asyncio
async def test_cancel_sweep_survives_a_version_race_on_one_member(sessions, auth) -> None:
    """The real failure mode: a running member's own worker writes a
    checkpoint in the window between `CancelSweep`'s read (`list_by_sweep`)
    and its write (`update`), so `update` loses the optimistic-concurrency
    race with a `ConcurrencyConflictError` -- even though `run.cancel()`
    itself never raised. Before the fix, that exception escaped
    `CancelSweep.__call__` unhandled and the walk stopped there, leaving
    every member after the raced one (in submission order) still running.

    Forced here by bumping one member's `version` in the database directly,
    inside a `list_by_sweep` override, landing squarely in that window --
    the honest way to reproduce the race without depending on a real fit
    finishing at the wrong instant (Task 10 tried that and couldn't).
    """
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    members = [_run(workspace_id=auth.workspace_id, sweep_id=sweep_id) for _ in range(3)]
    for member in members:
        await repository.add(member)
    raced = members[1]  # not first, not last: proves the walk doesn't stop here

    class _RepositoryRacedOnRead:
        """Delegates everything to the real repository except the one
        `list_by_sweep` call `CancelSweep` makes: right after it reads, bump
        `raced`'s version in the database -- simulating a worker's checkpoint
        landing before `CancelSweep`'s own `update(raced)` gets there."""

        def __init__(self, inner: SqlAlchemyRunRepository) -> None:
            self._inner = inner

        async def list_by_sweep(self, workspace_id: uuid.UUID, sweep_id: uuid.UUID):
            runs = await self._inner.list_by_sweep(workspace_id, sweep_id)
            async with sessions() as session:
                await session.execute(
                    sa_update(RunModel)
                    .where(RunModel.id == raced.id)
                    .values(version=RunModel.version + 1)
                )
                await session.commit()
            return runs

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    cancel_sweep = CancelSweep(_RepositoryRacedOnRead(repository))

    result = await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)

    # No exception escaped -- `.unwrap()` on a Failure would raise, so
    # reaching this line at all is part of what this test proves.
    cancelled = result.unwrap()
    assert cancelled == 3

    final = await repository.list_by_sweep(auth.workspace_id, sweep_id)
    assert {run.id: run.status for run in final} == {
        member.id: RunStatus.CANCELLED for member in members
    }


@pytest.mark.asyncio
async def test_cancel_unknown_sweep_is_not_found(cancel_sweep, auth) -> None:
    result = await cancel_sweep(CancelSweepCommand(sweep_id=uuid.uuid4()), auth=auth)
    assert isinstance(result, Failure)


@pytest.mark.asyncio
async def test_get_sweep_returns_members_in_submission_order(sessions, get_sweep, auth) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    for index in range(3):
        await repository.add(
            _run(workspace_id=auth.workspace_id, sweep_id=sweep_id, name=f"s #{index + 1}")
        )

    runs = (await get_sweep(GetSweepQuery(sweep_id=sweep_id), auth=auth)).unwrap()

    assert [run.params["name"] for run in runs] == ["s #1", "s #2", "s #3"]


@pytest.mark.asyncio
async def test_a_joint_engine_in_the_last_config_on_a_mixed_dataset_creates_no_runs(
    submit_sweep, mixed_dataset, auth, runs_repository
) -> None:
    result = await submit_sweep(
        SubmitSweepCommand(
            name="doomed",
            dataset_id=mixed_dataset.id,
            configs=[
                SweepConfig(engine_id="ecfp4-randomforest", conditions={}),
                SweepConfig(engine_id="chemprop-dmpnn", conditions={}),
            ],
        ),
        auth=auth,
    )
    assert isinstance(result, Failure)
    assert await runs_repository.sweep_summaries(auth.workspace_id) == []

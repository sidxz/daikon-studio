"""Lagom composition root -- the one place a use case is handed its collaborators.

Everything is lazy: building the container touches no database, no filesystem and
no network, so `create_app()` can build it at import time and a test can swap a
binding before the first request. Overriding for tests is `Container(parent)` plus
a `define` on the child -- lagom refuses a second `define` on the same container,
which is what keeps production wiring from being quietly reassigned.
"""

from __future__ import annotations

import httpx
from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.application.catalog.access_controlled_repository import (
    AccessControlledProtocolRepository,
)
from daikonstudio.application.catalog.delete_protocol import DeleteProtocol
from daikonstudio.application.catalog.get_chemical_space import (
    GetProtocolChemicalSpace,
    GetProtocolChemicalSpaceCompounds,
    GetRunChemicalSpace,
    GetRunChemicalSpaceCompounds,
)
from daikonstudio.application.catalog.get_scorecard import GetScorecard
from daikonstudio.application.catalog.list_protocols import GetProtocol, ListProtocols
from daikonstudio.application.catalog.publish_protocol import PublishProtocol
from daikonstudio.application.data.build_dataset import GetDatasetBuild, StartDatasetBuild
from daikonstudio.application.data.create_collection import CreateCollection, GetCollection
from daikonstudio.application.data.create_dataset import CreateDataset, StoreUpload
from daikonstudio.application.data.delete_dataset import DeleteDataset
from daikonstudio.application.data.export_collection import ExportCollection
from daikonstudio.application.data.get_dataset import GetDataset
from daikonstudio.application.data.get_dataset_compounds import GetDatasetCompounds
from daikonstudio.application.data.get_dataset_profile import GetDatasetProfile
from daikonstudio.application.data.import_chemcellar_run import ImportChemCellarRun
from daikonstudio.application.data.list_collections import ListCollections
from daikonstudio.application.data.list_datasets import ListDatasets
from daikonstudio.application.data.set_dataset_id_column import (
    GetDatasetColumns,
    SetDatasetIdColumn,
)
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.application.execution.claim_run import ClaimRun
from daikonstudio.application.execution.discard_abandoned_progress import (
    DiscardAbandonedProgress,
)
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.list_runs import ListRuns
from daikonstudio.application.execution.predict_with_protocol import (
    CancelRun,
    GetPredictionResults,
    GetRun,
    GetRunEpochs,
    PredictWithProtocol,
)
from daikonstudio.application.execution.retry_run import RetryRun
from daikonstudio.application.execution.sweeps import (
    CancelSweep,
    GetSweep,
    ListSweeps,
    SubmitSweep,
)
from daikonstudio.application.execution.train_protocol import TrainProtocol
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.chemcellar import ChemCellar
from daikonstudio.application.ports.dataset_build_repository import DatasetBuildRepository
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.application.runners.manage import CreateRunner, ListRunners, RevokeRunner
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.chemcellar.client import HttpChemCellar
from daikonstudio.infrastructure.duar.auth import get_duar
from daikonstudio.infrastructure.duar.protocol_access import DuarProtocolAccess
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.jobs import DbEnqueuer, InlineEnqueuer
from daikonstudio.infrastructure.persistence.session import create_session_factory
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.build_repository import (
    SqlAlchemyDatasetBuildRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.collection_repository import (
    SqlAlchemyCollectionRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.queue import SqlAlchemyRunQueue
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.runners.repository import (
    SqlAlchemyRunnerRepository,
)
from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
from daikonstudio.settings import Settings


def create_container(settings: Settings | None = None) -> Container:
    resolved = settings or Settings()
    container = Container()

    container.define(
        async_sessionmaker,
        Singleton(lambda: create_session_factory(resolved.database_url)),
    )
    container.define(
        BlobStore,  # type: ignore[type-abstract]
        Singleton(lambda: FsspecBlobStore(resolved.blob_base_url, resolved.blob_storage_options)),
    )
    container.define(
        StructureNormalizer,  # type: ignore[type-abstract]
        Singleton(RdkitStructureNormalizer),
    )
    # One pooled client for the process's lifetime. It is not closed explicitly,
    # which is harmless at exit.
    container.define(
        ChemCellar,  # type: ignore[type-abstract]
        Singleton(lambda: HttpChemCellar(httpx.AsyncClient(), resolved.chemcellar_api_url)),
    )
    # Engines hold no per-run state (infrastructure/engines/registry.py), so one
    # shared registry is safe -- unlike async_sessionmaker, nothing here is ever
    # overridden per test/request, so caching carries none of the JobEnqueuer
    # risk below.
    container.define(EngineRegistry, Singleton(default_registry))

    def _datasets(c: Container) -> SqlAlchemyDatasetRepository:
        return SqlAlchemyDatasetRepository(c[async_sessionmaker])

    # Lazy: resolved on first use. API tests define a FakeProtocolAccess in a child container.
    container.define(ProtocolAccess, Singleton(lambda: DuarProtocolAccess(get_duar())))  # type: ignore[type-abstract]

    def _protocols(c: Container) -> ProtocolRepository:
        return AccessControlledProtocolRepository(
            SqlAlchemyProtocolRepository(c[async_sessionmaker]),
            c[ProtocolAccess],  # type: ignore[type-abstract]
        )

    def _runs(c: Container) -> SqlAlchemyRunRepository:
        return SqlAlchemyRunRepository(c[async_sessionmaker])

    # Bound as container keys (not just the private helpers above) so the
    # runner-protocol routes -- which resolve every collaborator through the
    # container rather than constructing one in a route, like every other
    # route -- can depend on `RunRepository`/`DatasetRepository`/
    # `ProtocolRepository` directly (`interface/routes/runner_api.py`).
    container.define(RunRepository, lambda c: _runs(c))  # type: ignore[type-abstract]
    container.define(DatasetRepository, lambda c: _datasets(c))  # type: ignore[type-abstract]
    container.define(ProtocolRepository, lambda c: _protocols(c))  # type: ignore[type-abstract]

    def _collections(c: Container) -> SqlAlchemyCollectionRepository:
        return SqlAlchemyCollectionRepository(c[async_sessionmaker])

    container.define(StoreUpload, lambda c: StoreUpload(c[BlobStore]))
    container.define(
        ImportChemCellarRun,
        lambda c: ImportChemCellarRun(c[ChemCellar], c[BlobStore]),
    )
    container.define(
        CreateDataset,
        lambda c: CreateDataset(
            _datasets(c),
            c[BlobStore],
            c[StructureNormalizer],
        ),
    )
    container.define(
        DatasetBuildRepository,  # type: ignore[type-abstract]
        lambda c: SqlAlchemyDatasetBuildRepository(c[async_sessionmaker]),
    )
    container.define(
        StartDatasetBuild,
        lambda c: StartDatasetBuild(c[DatasetBuildRepository], c[CreateDataset]),
    )
    container.define(GetDatasetBuild, lambda c: GetDatasetBuild(c[DatasetBuildRepository]))
    container.define(GetDataset, lambda c: GetDataset(_datasets(c)))
    container.define(ListDatasets, lambda c: ListDatasets(_datasets(c)))
    container.define(
        GetDatasetProfile,
        lambda c: GetDatasetProfile(_datasets(c), c[BlobStore], c[StructureNormalizer]),
    )
    container.define(
        DeleteDataset,
        lambda c: DeleteDataset(_datasets(c), _protocols(c), _runs(c), c[BlobStore]),
    )
    container.define(SetDatasetIdColumn, lambda c: SetDatasetIdColumn(_datasets(c), c[BlobStore]))
    container.define(GetDatasetColumns, lambda c: GetDatasetColumns(_datasets(c), c[BlobStore]))
    container.define(
        GetDatasetCompounds, lambda c: GetDatasetCompounds(_datasets(c), c[BlobStore])
    )

    container.define(
        RunQueue,  # type: ignore[type-abstract]
        lambda c: SqlAlchemyRunQueue(c[async_sessionmaker]),
    )
    # Plain factory, NOT Singleton: both branches depend on c[async_sessionmaker],
    # which tests override per test -- see the InlineEnqueuer caching incident
    # documented in the JobEnqueuer comment this replaces.
    #
    # Chosen once per resolution, from STUDIO_INLINE_JOBS: tests and local dev run
    # the job in-process (no runner needed at all), a real deployment leaves it on
    # the row for a self-hosted runner to claim. See `infrastructure/jobs.py`'s
    # module docstring for both implementations.
    container.define(
        JobEnqueuer,  # type: ignore[type-abstract]
        lambda c: (
            InlineEnqueuer(c[async_sessionmaker], c[BlobStore], c[ProtocolAccess])
            if resolved.inline_jobs
            else DbEnqueuer(c[RunQueue])
        ),
    )

    container.define(
        RunnerRepository,  # type: ignore[type-abstract]
        lambda c: SqlAlchemyRunnerRepository(c[async_sessionmaker]),
    )
    # Resolved settings, for interface dependencies that need config values
    # directly rather than through a use case (e.g. lease-extension seconds).
    container.define(Settings, Singleton(lambda: resolved))

    container.define(CreateRunner, lambda c: CreateRunner(c[RunnerRepository]))
    container.define(
        ListRunners,
        lambda c: ListRunners(
            c[RunnerRepository],
            c[RunQueue],
            online_threshold_seconds=resolved.runner_online_threshold_seconds,
            busy_threshold_seconds=resolved.runner_lease_seconds,
        ),
    )
    container.define(RevokeRunner, lambda c: RevokeRunner(c[RunnerRepository]))
    container.define(
        ClaimRun,
        lambda c: ClaimRun(
            c[RunQueue],
            _runs(c),
            lease_seconds=resolved.runner_lease_seconds,
            max_active_per_workspace=resolved.workspace_max_active_runs,
            max_attempts=resolved.runner_max_attempts,
            deadline_seconds=resolved.worker_job_timeout,
            deadline_by_lane=resolved.worker_job_timeout_by_lane,
        ),
    )

    container.define(
        TrainProtocol,
        lambda c: TrainProtocol(_datasets(c), _runs(c), c[JobEnqueuer], c[EngineRegistry]),
    )
    container.define(
        SubmitSweep,
        lambda c: SubmitSweep(_datasets(c), c[EngineRegistry], c[TrainProtocol]),
    )
    container.define(ListSweeps, lambda c: ListSweeps(_runs(c), c[ProtocolAccess]))
    container.define(GetSweep, lambda c: GetSweep(_runs(c), c[ProtocolAccess]))
    container.define(CancelSweep, lambda c: CancelSweep(_runs(c), c[ProtocolAccess]))
    container.define(PublishProtocol, lambda c: PublishProtocol(_protocols(c), c[ProtocolAccess]))
    container.define(
        DeleteProtocol,
        lambda c: DeleteProtocol(_protocols(c), _runs(c), c[BlobStore], c[ProtocolAccess]),
    )
    container.define(
        GetScorecard,
        lambda c: GetScorecard(
            _protocols(c), c[BlobStore], c[StructureNormalizer], _datasets(c), c[ProtocolAccess]
        ),
    )
    container.define(
        GetProtocolChemicalSpace,
        lambda c: GetProtocolChemicalSpace(_protocols(c), c[BlobStore], c[ProtocolAccess]),
    )
    container.define(
        GetProtocolChemicalSpaceCompounds,
        lambda c: GetProtocolChemicalSpaceCompounds(
            _protocols(c), c[BlobStore], _datasets(c), c[ProtocolAccess]
        ),
    )
    container.define(
        GetRunChemicalSpace,
        lambda c: GetRunChemicalSpace(_runs(c), _protocols(c), c[BlobStore], c[ProtocolAccess]),
    )
    container.define(
        GetRunChemicalSpaceCompounds,
        lambda c: GetRunChemicalSpaceCompounds(
            _runs(c), _protocols(c), c[BlobStore], c[ProtocolAccess]
        ),
    )
    container.define(ListProtocols, lambda c: ListProtocols(_protocols(c), c[ProtocolAccess]))
    container.define(GetProtocol, lambda c: GetProtocol(_protocols(c), c[ProtocolAccess]))

    container.define(
        PredictWithProtocol,
        lambda c: PredictWithProtocol(
            _protocols(c),
            _runs(c),
            c[BlobStore],
            c[JobEnqueuer],
            c[EngineRegistry],
            c[ProtocolAccess],
        ),
    )
    container.define(GetRun, lambda c: GetRun(_runs(c), c[ProtocolAccess]))
    container.define(GetRunEpochs, lambda c: GetRunEpochs(_runs(c), c[ProtocolAccess]))
    container.define(ListRuns, lambda c: ListRuns(_runs(c), c[ProtocolAccess]))
    container.define(CancelRun, lambda c: CancelRun(_runs(c), c[ProtocolAccess]))
    container.define(
        DiscardAbandonedProgress, lambda c: DiscardAbandonedProgress(_runs(c), c[BlobStore])
    )
    container.define(
        RetryRun,
        lambda c: RetryRun(
            _runs(c),
            _protocols(c),
            c[JobEnqueuer],
            c[EngineRegistry],
            c[BlobStore],
            c[ProtocolAccess],
        ),
    )
    container.define(
        GetPredictionResults,
        lambda c: GetPredictionResults(_runs(c), _protocols(c), c[BlobStore], c[ProtocolAccess]),
    )

    container.define(
        CreateCollection,
        lambda c: CreateCollection(
            _collections(c), _runs(c), _protocols(c), c[BlobStore], c[ProtocolAccess]
        ),
    )
    container.define(GetCollection, lambda c: GetCollection(_collections(c)))
    container.define(ListCollections, lambda c: ListCollections(_collections(c)))
    container.define(
        ExportCollection,
        lambda c: ExportCollection(
            _collections(c), _runs(c), _protocols(c), c[BlobStore], c[ProtocolAccess]
        ),
    )

    return container

"""Lagom composition root -- the one place a use case is handed its collaborators.

Everything is lazy: building the container touches no database, no filesystem and
no network, so `create_app()` can build it at import time and a test can swap a
binding before the first request. Overriding for tests is `Container(parent)` plus
a `define` on the child -- lagom refuses a second `define` on the same container,
which is what keeps production wiring from being quietly reassigned.
"""

from __future__ import annotations

from lagom import Container, Singleton
from sqlalchemy.ext.asyncio import async_sessionmaker

from daikonstudio.application.catalog.get_scorecard import GetScorecard
from daikonstudio.application.catalog.list_protocols import GetProtocol, ListProtocols
from daikonstudio.application.catalog.publish_protocol import PublishProtocol
from daikonstudio.application.data.create_collection import CreateCollection, GetCollection
from daikonstudio.application.data.create_dataset import CreateDataset, StoreUpload
from daikonstudio.application.data.export_collection import ExportCollection
from daikonstudio.application.data.get_dataset import GetDataset
from daikonstudio.application.data.get_dataset_compounds import GetDatasetCompounds
from daikonstudio.application.data.get_dataset_profile import GetDatasetProfile
from daikonstudio.application.data.list_collections import ListCollections
from daikonstudio.application.data.list_datasets import ListDatasets
from daikonstudio.application.engines.registry import EngineRegistry
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.list_runs import ListRuns
from daikonstudio.application.execution.predict_with_protocol import (
    CancelRun,
    GetPredictionResults,
    GetRun,
    PredictWithProtocol,
)
from daikonstudio.application.execution.train_protocol import TrainProtocol
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.run_queue import RunQueue
from daikonstudio.application.ports.runner_repository import RunnerRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.application.runners.manage import CreateRunner, ListRunners, RevokeRunner
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.engines.registry import default_registry
from daikonstudio.infrastructure.jobs import DbEnqueuer, InlineEnqueuer
from daikonstudio.infrastructure.persistence.session import create_session_factory
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
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
    # Engines hold no per-run state (infrastructure/engines/registry.py), so one
    # shared registry is safe -- unlike async_sessionmaker, nothing here is ever
    # overridden per test/request, so caching carries none of the JobEnqueuer
    # risk below.
    container.define(EngineRegistry, Singleton(default_registry))

    def _datasets(c: Container) -> SqlAlchemyDatasetRepository:
        return SqlAlchemyDatasetRepository(c[async_sessionmaker])

    def _protocols(c: Container) -> SqlAlchemyProtocolRepository:
        return SqlAlchemyProtocolRepository(c[async_sessionmaker])

    def _runs(c: Container) -> SqlAlchemyRunRepository:
        return SqlAlchemyRunRepository(c[async_sessionmaker])

    def _collections(c: Container) -> SqlAlchemyCollectionRepository:
        return SqlAlchemyCollectionRepository(c[async_sessionmaker])

    container.define(StoreUpload, lambda c: StoreUpload(c[BlobStore]))
    container.define(
        CreateDataset,
        lambda c: CreateDataset(
            _datasets(c),
            c[BlobStore],
            c[StructureNormalizer],
        ),
    )
    container.define(GetDataset, lambda c: GetDataset(_datasets(c)))
    container.define(ListDatasets, lambda c: ListDatasets(_datasets(c)))
    container.define(
        GetDatasetProfile,
        lambda c: GetDatasetProfile(_datasets(c), c[BlobStore], c[StructureNormalizer]),
    )
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
            InlineEnqueuer(c[async_sessionmaker], c[BlobStore])
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
        ),
    )
    container.define(RevokeRunner, lambda c: RevokeRunner(c[RunnerRepository]))

    container.define(
        TrainProtocol,
        lambda c: TrainProtocol(_datasets(c), _runs(c), c[JobEnqueuer], c[EngineRegistry]),
    )
    container.define(PublishProtocol, lambda c: PublishProtocol(_protocols(c)))
    container.define(
        GetScorecard,
        lambda c: GetScorecard(_protocols(c), c[BlobStore], c[StructureNormalizer]),
    )
    container.define(ListProtocols, lambda c: ListProtocols(_protocols(c)))
    container.define(GetProtocol, lambda c: GetProtocol(_protocols(c)))

    container.define(
        PredictWithProtocol,
        lambda c: PredictWithProtocol(
            _protocols(c), _runs(c), c[BlobStore], c[JobEnqueuer], c[EngineRegistry]
        ),
    )
    container.define(GetRun, lambda c: GetRun(_runs(c)))
    container.define(ListRuns, lambda c: ListRuns(_runs(c)))
    container.define(CancelRun, lambda c: CancelRun(_runs(c)))
    container.define(
        GetPredictionResults,
        lambda c: GetPredictionResults(_runs(c), _protocols(c), c[BlobStore]),
    )

    container.define(
        CreateCollection,
        lambda c: CreateCollection(_collections(c), _runs(c), _protocols(c), c[BlobStore]),
    )
    container.define(GetCollection, lambda c: GetCollection(_collections(c)))
    container.define(ListCollections, lambda c: ListCollections(_collections(c)))
    container.define(
        ExportCollection,
        lambda c: ExportCollection(_collections(c), _runs(c), _protocols(c), c[BlobStore]),
    )

    return container

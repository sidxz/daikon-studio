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
from daikonstudio.application.data.create_dataset import CreateDataset, StoreUpload
from daikonstudio.application.data.get_dataset import GetDataset
from daikonstudio.application.data.list_datasets import ListDatasets
from daikonstudio.application.execution.enqueue import JobEnqueuer
from daikonstudio.application.execution.train_protocol import TrainProtocol
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.persistence.session import create_session_factory
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
from daikonstudio.infrastructure.worker import ArqEnqueuer, InlineEnqueuer
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
        Singleton(lambda: FsspecBlobStore(resolved.blob_base_url)),
    )
    container.define(
        StructureNormalizer,  # type: ignore[type-abstract]
        Singleton(RdkitStructureNormalizer),
    )

    def _datasets(c: Container) -> SqlAlchemyDatasetRepository:
        return SqlAlchemyDatasetRepository(c[async_sessionmaker])

    def _protocols(c: Container) -> SqlAlchemyProtocolRepository:
        return SqlAlchemyProtocolRepository(c[async_sessionmaker])

    def _runs(c: Container) -> SqlAlchemyRunRepository:
        return SqlAlchemyRunRepository(c[async_sessionmaker])

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

    def _build_enqueuer(c: Container) -> JobEnqueuer:
        # Chosen once, at container-build time, from STUDIO_INLINE_JOBS: tests and
        # local dev run the job in-process (no Valkey needed at all), a real
        # deployment pushes it to Redis for the arq worker to pick up. See
        # `infrastructure/worker.py`'s module docstring for both implementations.
        if resolved.inline_jobs:
            return InlineEnqueuer(c[async_sessionmaker], c[BlobStore])  # type: ignore[type-abstract]
        return ArqEnqueuer(resolved.redis_url)

    container.define(JobEnqueuer, Singleton(_build_enqueuer))  # type: ignore[type-abstract]

    container.define(
        TrainProtocol, lambda c: TrainProtocol(_datasets(c), _runs(c), c[JobEnqueuer])
    )
    container.define(PublishProtocol, lambda c: PublishProtocol(_protocols(c)))
    container.define(
        GetScorecard,
        lambda c: GetScorecard(_protocols(c), c[BlobStore], c[StructureNormalizer]),
    )
    container.define(ListProtocols, lambda c: ListProtocols(_protocols(c)))
    container.define(GetProtocol, lambda c: GetProtocol(_protocols(c)))

    return container

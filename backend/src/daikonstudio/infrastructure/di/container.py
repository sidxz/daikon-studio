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

from daikonstudio.application.data.create_dataset import CreateDataset, StoreUpload
from daikonstudio.application.data.get_dataset import GetDataset
from daikonstudio.application.data.list_datasets import ListDatasets
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.persistence.session import create_session_factory
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
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
        Singleton(lambda: FsspecBlobStore(resolved.blob_base_url)),
    )
    container.define(
        StructureNormalizer,  # type: ignore[type-abstract]
        Singleton(RdkitStructureNormalizer),
    )

    def _datasets(c: Container) -> SqlAlchemyDatasetRepository:
        return SqlAlchemyDatasetRepository(c[async_sessionmaker])

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

    return container

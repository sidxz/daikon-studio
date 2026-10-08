"""Build chemical-space maps, and run neighbours, for everything that predates them.

    make backfill-maps          # or: python -m daikonstudio.infrastructure.backfill_maps

New protocols map themselves during training and new runs keep their neighbours
during prediction; this command covers the protocols and runs written before
that. It walks every workspace, so it reads the tables directly rather than
through the workspace-scoped repositories' listings.

Idempotent: a map whose stored version is current, and a run whose neighbours
already exist, are counted and skipped, so re-running after a partial failure
only does the remaining work. One line per item; exit 1 if anything failed.
"""

from __future__ import annotations

import asyncio
import io
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daikonstudio.application.catalog.chemical_space import (
    MAP_VERSION,
    NEIGHBOURS,
    neighbours_key,
    neighbours_parquet,
    read_meta,
    write_chemical_space,
)
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.execution.train_protocol import (
    ScorecardInputs,
    scorecard_inputs_key,
)
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.chemical_space_layout import (
    ChemicalSpaceLayout,
    TooFewCompounds,
)
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.execution.run import RunKind, RunStatus
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.models import (
    InSilicoProtocolModel,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.catalog.repository import (
    SqlAlchemyProtocolRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.data.repository import (
    SqlAlchemyDatasetRepository,
)
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.models import RunModel


@dataclass
class BackfillReport:
    maps_built: int = 0
    maps_current: int = 0
    # Sequence datasets, which have no chemical space to map. Counted apart from
    # failures so a run over a mixed workspace does not read as partly broken.
    maps_skipped: int = 0
    neighbours_built: int = 0
    neighbours_current: int = 0
    failures: list[str] = field(default_factory=list)


async def backfill(
    session_factory: async_sessionmaker[AsyncSession],
    store: BlobStore,
    layout: ChemicalSpaceLayout,
    normalizer: StructureNormalizer,
    log: Callable[[str], None] = print,
) -> BackfillReport:
    report = BackfillReport()
    protocols = SqlAlchemyProtocolRepository(session_factory)
    datasets = SqlAlchemyDatasetRepository(session_factory)

    async with session_factory() as session:
        listed = (
            await session.execute(
                select(InSilicoProtocolModel.id, InSilicoProtocolModel.workspace_id).order_by(
                    InSilicoProtocolModel.created_at
                )
            )
        ).all()

    for protocol_id, workspace_id in listed:
        protocol = await protocols.get(workspace_id, protocol_id)
        if protocol is None:
            continue
        label = f"protocol {protocol_id} ({protocol.name})"
        try:
            meta = read_meta(store, workspace_id, protocol_id)
            if meta is not None and int(meta.get("version", 0)) >= MAP_VERSION:
                report.maps_current += 1
            else:
                dataset = await datasets.get(workspace_id, protocol.dataset_id)
                if dataset is None:
                    raise LookupError(f"dataset {protocol.dataset_id} not found")
                if dataset.validation_report.structure_kind is StructureKind.SEQUENCE:
                    # Same refusal as the training path: the layout does not fail on
                    # sequences, it returns a convincing cloud of nothing. Counted as
                    # skipped rather than failed -- there is nothing wrong here to fix.
                    report.maps_skipped += 1
                    log(f"skipped {label}: sequence dataset has no chemical space")
                    continue
                frame = pl.read_parquet(
                    io.BytesIO(store.get_bytes(snapshot_key(workspace_id, dataset.id)))
                )
                try:
                    await asyncio.to_thread(
                        write_chemical_space,
                        store,
                        workspace_id,
                        protocol_id,
                        frame,
                        dataset.structure_column,
                        dataset.split.seed,
                        layout,
                    )
                    report.maps_built += 1
                    log(f"mapped {label}: {frame.height} compounds")
                except TooFewCompounds:
                    report.maps_current += 1
                    log(f"skipped {label}: too few compounds to map")
        except Exception as error:  # one bad protocol must not stop the rest
            report.failures.append(f"{label}: {error}")
            log(f"FAILED {label}: {error}")
            continue

        await _backfill_runs(
            session_factory, store, normalizer, workspace_id, protocol_id, report, log
        )
    return report


async def _backfill_runs(
    session_factory: async_sessionmaker[AsyncSession],
    store: BlobStore,
    normalizer: StructureNormalizer,
    workspace_id: uuid.UUID,
    protocol_id: uuid.UUID,
    report: BackfillReport,
    log: Callable[[str], None],
) -> None:
    async with session_factory() as session:
        runs = (
            await session.execute(
                select(RunModel.id, RunModel.result_uri).where(
                    RunModel.workspace_id == workspace_id,
                    RunModel.protocol_id == protocol_id,
                    RunModel.kind == RunKind.PREDICTION.value,
                    RunModel.status == RunStatus.READY.value,
                )
            )
        ).all()
    if not runs:
        return

    train_structures: list[str] | None = None
    for run_id, result_uri in runs:
        if store.exists(neighbours_key(workspace_id, run_id)):
            report.neighbours_current += 1
            continue
        try:
            if train_structures is None:
                train_structures = ScorecardInputs.from_json(
                    store.get_bytes(scorecard_inputs_key(workspace_id, protocol_id))
                ).train_structures
            if not train_structures or result_uri is None:
                report.neighbours_current += 1
                continue
            structures = [
                str(s)
                for s in pl.read_parquet(io.BytesIO(store.get_bytes(result_uri)))[
                    "structure"
                ].to_list()
            ]
            indices, similarities = await asyncio.to_thread(
                normalizer.nearest_neighbours_tanimoto, structures, train_structures, NEIGHBOURS
            )
            store.put_bytes(
                neighbours_key(workspace_id, run_id), neighbours_parquet(indices, similarities)
            )
            report.neighbours_built += 1
            log(f"placed run {run_id}: {len(structures)} compounds")
        except Exception as error:
            report.failures.append(f"run {run_id}: {error}")
            log(f"FAILED run {run_id}: {error}")


async def main() -> None:
    from daikonstudio.infrastructure.chem.chemical_space import UmapLayout
    from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
    from daikonstudio.infrastructure.persistence.session import create_session_factory
    from daikonstudio.infrastructure.storage.fsspec_blob_store import FsspecBlobStore
    from daikonstudio.settings import Settings

    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    store = FsspecBlobStore(settings.blob_base_url, settings.blob_storage_options)
    report = await backfill(session_factory, store, UmapLayout(), RdkitStructureNormalizer())
    print(
        f"maps: {report.maps_built} built, {report.maps_current} current, "
        f"{report.maps_skipped} skipped (sequences); "
        f"run neighbours: {report.neighbours_built} built, {report.neighbours_current} current; "
        f"{len(report.failures)} failed"
    )
    if report.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

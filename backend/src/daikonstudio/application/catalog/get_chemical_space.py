"""Read side of the chemical-space map.

The protocol's points, a run's compounds placed among them, and single-compound
lookups for hover. Placement happens here, not in the browser, so the client draws
coordinates and never needs the neighbour arithmetic. A map that was never built
(older protocols before the backfill, datasets too small to map, a failed layout)
is a normal state, reported as `missing`, never an error.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from dataclasses import dataclass, replace

import numpy as np
import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.catalog.chemical_space import (
    PARTITION_CODES,
    place,
    read_meta,
    read_neighbours,
    read_points,
)
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.data.compound_ids import read_compound_ids
from daikonstudio.application.execution.build_scorecard import _APPLICABILITY_THRESHOLD
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.execution.run import Run, RunKind, RunStatus
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)

#: Hover asks for one compound at a time; 50 leaves room for a neighbour list.
MAX_LOOKUPS = 50
_PARTITION_NAMES = {code: name for name, code in PARTITION_CODES.items()}


def _round(values: list[float]) -> list[float]:
    """Four decimals of the unit square is a fraction of a pixel on any screen."""
    return [round(float(v), 4) for v in values]


@dataclass(frozen=True, kw_only=True)
class ChemicalSpaceView:
    status: str
    method: str | None = None
    params: dict[str, object] | None = None
    counts: dict[str, int] | None = None
    x: list[float] | None = None
    y: list[float] | None = None
    partition: list[int] | None = None


@dataclass(frozen=True, kw_only=True)
class MapCompound:
    index: int
    structure: str
    partition: str
    # Looked up from the dataset's identifier column when asked for, if it has one.
    compound_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class RunChemicalSpaceView:
    status: str
    x: list[float] | None = None
    y: list[float] | None = None
    row_id: list[int] | None = None
    applicability: list[float | None] | None = None
    neighbors: list[list[int]] | None = None
    total: int | None = None
    in_domain: int | None = None
    nearest_min: float | None = None
    nearest_max: float | None = None
    threshold: float = _APPLICABILITY_THRESHOLD


@dataclass(frozen=True, kw_only=True)
class RunMapCompound:
    row_id: int
    structure: str
    compound_id: str | None
    values: dict[str, float | None]
    applicability: float | None


@dataclass(frozen=True, kw_only=True)
class GetProtocolChemicalSpaceQuery:
    protocol_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class GetProtocolChemicalSpaceCompoundsQuery:
    protocol_id: uuid.UUID
    indices: list[int]


@dataclass(frozen=True, kw_only=True)
class GetRunChemicalSpaceQuery:
    run_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class GetRunChemicalSpaceCompoundsQuery:
    run_id: uuid.UUID
    rows: list[int]


def _too_many(count: int) -> ValidationError | None:
    if count > MAX_LOOKUPS:
        return ValidationError(f"Ask for at most {MAX_LOOKUPS} compounds at a time.")
    return None


class GetProtocolChemicalSpace:
    def __init__(
        self,
        protocols: ProtocolRepository,
        store: BlobStore,
        access: ProtocolAccess,
    ) -> None:
        self._access = access
        self._protocols = protocols
        self._store = store

    async def __call__(
        self, query: GetProtocolChemicalSpaceQuery, auth: AuthContext | None = None
    ) -> Result[ChemicalSpaceView, DomainError]:
        require_authenticated(auth)
        assert auth is not None
        protocol = await visible_protocol(
            self._protocols, self._access, auth, auth.workspace_id, query.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(query.protocol_id)))
        meta = read_meta(self._store, protocol.workspace_id, protocol.id)
        if meta is None:
            return Success(ChemicalSpaceView(status="missing"))
        try:
            points = read_points(self._store, protocol.workspace_id, protocol.id)
        except FileNotFoundError:
            return Success(ChemicalSpaceView(status="missing"))
        return Success(
            ChemicalSpaceView(
                status="ready",
                method=str(meta.get("method")),
                params=dict(meta.get("params", {})),
                counts={str(k): int(v) for k, v in dict(meta.get("counts", {})).items()},
                x=_round(points["x"].to_list()),
                y=_round(points["y"].to_list()),
                partition=[int(p) for p in points["partition"].to_list()],
            )
        )


class GetProtocolChemicalSpaceCompounds:
    def __init__(
        self,
        protocols: ProtocolRepository,
        store: BlobStore,
        datasets: DatasetRepository,
        access: ProtocolAccess,
    ) -> None:
        self._access = access
        self._protocols = protocols
        self._store = store
        self._datasets = datasets

    async def __call__(
        self, query: GetProtocolChemicalSpaceCompoundsQuery, auth: AuthContext | None = None
    ) -> Result[list[MapCompound], DomainError]:
        require_authenticated(auth)
        assert auth is not None
        if error := _too_many(len(query.indices)):
            return Failure(error)
        protocol = await visible_protocol(
            self._protocols, self._access, auth, auth.workspace_id, query.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(query.protocol_id)))
        try:
            points = read_points(self._store, protocol.workspace_id, protocol.id)
        except FileNotFoundError:
            return Success([])
        structures = points["structure"].to_list()
        partitions = points["partition"].to_list()
        found = [
            MapCompound(
                index=i,
                structure=str(structures[i]),
                partition=_PARTITION_NAMES[int(partitions[i])],
            )
            for i in query.indices
            if 0 <= i < len(structures)
        ]
        if found:
            dataset = await self._datasets.get(protocol.workspace_id, protocol.dataset_id)
            if dataset is not None:
                try:
                    ids = await asyncio.to_thread(
                        read_compound_ids,
                        self._store,
                        dataset,
                        [item.structure for item in found],
                    )
                except FileNotFoundError:
                    ids = None
                if ids:
                    found = [replace(item, compound_id=ids.get(item.structure)) for item in found]
        return Success(found)


async def _ready_prediction(
    runs: RunRepository,
    protocols: ProtocolRepository,
    auth: AuthContext,
    run_id: uuid.UUID,
    access: ProtocolAccess,
) -> tuple[Run, InSilicoProtocol] | DomainError:
    """The same guards as the results endpoint, so the two agree about which runs exist."""
    run = await runs.get(auth.workspace_id, run_id)
    if run is None or run.kind is not RunKind.PREDICTION:
        return NotFoundError("Prediction run", str(run_id))
    if run.status is not RunStatus.READY or run.result_uri is None:
        return ConflictError(
            f"The map is available only for completed runs; this run is {run.status.label}."
        )
    protocol_id = uuid.UUID(run.params["protocol_id"])
    protocol = await visible_protocol(protocols, access, auth, auth.workspace_id, protocol_id)
    if protocol is None:
        return NotFoundError("Protocol", str(protocol_id))
    return run, protocol


class GetRunChemicalSpace:
    def __init__(
        self,
        runs: RunRepository,
        protocols: ProtocolRepository,
        store: BlobStore,
        access: ProtocolAccess,
    ) -> None:
        self._access = access
        self._runs = runs
        self._protocols = protocols
        self._store = store

    async def __call__(
        self, query: GetRunChemicalSpaceQuery, auth: AuthContext | None = None
    ) -> Result[RunChemicalSpaceView, DomainError]:
        require_authenticated(auth)
        assert auth is not None
        found = await _ready_prediction(
            self._runs, self._protocols, auth, query.run_id, self._access
        )
        if isinstance(found, DomainError):
            return Failure(found)
        run, protocol = found

        neighbours = read_neighbours(self._store, run.workspace_id, run.id)
        if (
            read_meta(self._store, protocol.workspace_id, protocol.id) is None
            or neighbours is None
        ):
            return Success(RunChemicalSpaceView(status="missing"))
        assert run.result_uri is not None  # `_ready_prediction` refuses a run without one
        points = read_points(self._store, protocol.workspace_id, protocol.id)
        results = pl.read_parquet(io.BytesIO(self._store.get_bytes(run.result_uri)))

        # Neighbour indices count training compounds in snapshot order; these are
        # the map rows holding them, in that same order.
        train_rows = np.flatnonzero(points["partition"].to_numpy() == PARTITION_CODES["train"])
        train_xy = points.select("x", "y").to_numpy()[train_rows]
        indices = np.array(neighbours["neighbor_index"].to_list(), dtype=np.int64)
        similarities = np.array(neighbours["neighbor_similarity"].to_list(), dtype=float)
        xy = place(train_xy, indices, similarities)

        applicability = [
            None if v is None else float(v) for v in results["applicability"].to_list()
        ]
        measured = [v for v in applicability if v is not None]
        return Success(
            RunChemicalSpaceView(
                status="ready",
                x=_round(xy[:, 0].tolist()),
                y=_round(xy[:, 1].tolist()),
                row_id=list(range(results.height)),
                applicability=applicability,
                neighbors=train_rows[indices].tolist(),
                total=results.height,
                in_domain=sum(1 for v in measured if v >= _APPLICABILITY_THRESHOLD),
                nearest_min=round(min(measured), 2) if measured else None,
                nearest_max=round(max(measured), 2) if measured else None,
            )
        )


class GetRunChemicalSpaceCompounds:
    def __init__(
        self,
        runs: RunRepository,
        protocols: ProtocolRepository,
        store: BlobStore,
        access: ProtocolAccess,
    ) -> None:
        self._access = access
        self._runs = runs
        self._protocols = protocols
        self._store = store

    async def __call__(
        self, query: GetRunChemicalSpaceCompoundsQuery, auth: AuthContext | None = None
    ) -> Result[list[RunMapCompound], DomainError]:
        require_authenticated(auth)
        assert auth is not None
        if error := _too_many(len(query.rows)):
            return Failure(error)
        found = await _ready_prediction(
            self._runs, self._protocols, auth, query.run_id, self._access
        )
        if isinstance(found, DomainError):
            return Failure(found)
        run, protocol = found
        assert run.result_uri is not None
        results = pl.read_parquet(io.BytesIO(self._store.get_bytes(run.result_uri)))
        readouts = [r.name for r in protocol.readouts if r.name in results.columns]
        has_ids = "compound_id" in results.columns
        items = []
        for row in query.rows:
            if not 0 <= row < results.height:
                continue
            record = results.row(row, named=True)
            items.append(
                RunMapCompound(
                    row_id=row,
                    structure=str(record["structure"]),
                    compound_id=record["compound_id"] if has_ids else None,
                    values={
                        name: None if record[name] is None else float(record[name])
                        for name in readouts
                    },
                    applicability=record["applicability"],
                )
            )
        return Success(items)

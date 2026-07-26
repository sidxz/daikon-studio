"""Rendering a Collection's snapshot as a file a chemist can act on.

CSV and SDF are read differently, so provenance is stamped differently in
each -- inferable from the file's own content, never only from a filename
or a Content-Type header a client can drop:

- CSV: `generation_method` is its own column, holding the value on every
  row. A spreadsheet shows it as plainly as any other column; a unit lives
  in the column header (`"IC50 (nM)"`) since a CSV cell is expected to hold
  a bare, sortable, plottable number, not a number wearing a unit as text.
- SDF: `generation_method` is its own SD tag on every molecule block --
  the SDF equivalent of a column, so a chemistry tool renders it as a
  visible field the same way it would for a measured value's own tags. A
  readout's own tag carries its value *and* its unit together
  (`"7.24 nM"`, not a bare `"7.24"`), because that is the one thing an SD
  tag's value is expected to be self-describing about: SDF has no header
  row to hang a unit on the way CSV does.

Both branches read `protocol.readouts` fresh via the Collection's
`derived_from_run_id -> Run.params["protocol_id"]`, the same path
`GetPredictionResults` already uses, rather than the Collection snapshotting
its own copy of readout names/units/directions. That is safe only because a
published Protocol's readouts are immutable for the rest of that Protocol's
life (`domain/catalog/protocol.py`): there is no code path in this system
that deletes or edits a Protocol once published, so re-reading it here can
never disagree with what the Collection's numbers actually mean.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from enum import StrEnum

import polars as pl
from rdkit import Chem
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.collection_repository import CollectionRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.catalog.readout import Readout
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


class ExportFormat(StrEnum):
    CSV = "csv"
    SDF = "sdf"


@dataclass(frozen=True, kw_only=True)
class ExportCollectionQuery:
    collection_id: uuid.UUID
    format: ExportFormat


@dataclass(frozen=True, kw_only=True)
class CollectionExport:
    content: bytes
    media_type: str
    filename: str


class ExportCollection:
    def __init__(
        self,
        collections: CollectionRepository,
        runs: RunRepository,
        protocols: ProtocolRepository,
        store: BlobStore,
    ) -> None:
        self._collections = collections
        self._runs = runs
        self._protocols = protocols
        self._store = store

    async def __call__(
        self, query: ExportCollectionQuery, auth: AuthContext | None = None
    ) -> Result[CollectionExport, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        collection = await self._collections.get(auth.workspace_id, query.collection_id)
        if collection is None:
            return Failure(NotFoundError("Collection", str(query.collection_id)))

        run = await self._runs.get(auth.workspace_id, collection.derived_from_run_id)
        if run is None:
            return Failure(NotFoundError("Run", str(collection.derived_from_run_id)))
        protocol_id = uuid.UUID(run.params["protocol_id"])
        protocol = await self._protocols.get(auth.workspace_id, protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(protocol_id)))

        try:
            raw = self._store.get_bytes(collection.snapshot_uri)
        except FileNotFoundError:
            return Failure(NotFoundError("Collection snapshot", str(collection.id)))
        frame = pl.read_parquet(io.BytesIO(raw))

        generation_method = collection.provenance.generation_method.value
        if query.format is ExportFormat.CSV:
            content = _render_csv(frame, protocol.readouts, generation_method)
            media_type = "text/csv"
        else:
            content = _render_sdf(frame, protocol.readouts, generation_method)
            media_type = "chemical/x-mdl-sdfile"

        return Success(
            CollectionExport(
                content=content,
                media_type=media_type,
                filename=f"{collection.id}.{query.format.value}",
            )
        )


def _render_csv(
    frame: pl.DataFrame, readouts: tuple[Readout, ...], generation_method: str
) -> bytes:
    rename = {"structure": "smiles"}
    for readout in readouts:
        label = f"{readout.name} ({readout.unit})" if readout.unit else readout.name
        rename[readout.name] = label
    out = frame.rename(rename).with_columns(pl.lit(generation_method).alias("generation_method"))
    buffer = io.BytesIO()
    out.write_csv(buffer)
    return buffer.getvalue()


def _render_sdf(
    frame: pl.DataFrame, readouts: tuple[Readout, ...], generation_method: str
) -> bytes:
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    for row in frame.to_dicts():
        mol = Chem.MolFromSmiles(row["structure"])
        if mol is None:
            # The structure was already canonicalized once by RunPrediction
            # before it was ever written to the run's results Parquet -- this
            # is unreachable in practice, but skipping (not raising) means one
            # corrupt row cannot take down an export of every other compound
            # a scientist is trying to hand to a chemist.
            continue
        mol.SetProp("_Name", row["structure"])
        for readout in readouts:
            value = row[readout.name]
            label = f"{value} {readout.unit}" if readout.unit else str(value)
            mol.SetProp(readout.name, label)
        mol.SetProp("generation_method", generation_method)
        writer.write(mol)
    writer.close()
    return buffer.getvalue().encode()

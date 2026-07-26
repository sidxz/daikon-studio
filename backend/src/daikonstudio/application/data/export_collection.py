"""Rendering a Collection's snapshot as a file a chemist can act on.

CSV and SDF are read differently, so provenance is stamped differently in
each -- inferable from the file's own content, never only from a filename
or a Content-Type header a client can drop:

- CSV: `generation_method` is its own column, holding the value on every
  row. A spreadsheet shows it as plainly as any other column; a unit (and
  direction) lives in the column header (`"IC50 (nM, lower is better)"`)
  since a CSV cell is expected to hold a bare, sortable, plottable number,
  not a number wearing a unit as text.
- SDF: `generation_method` is its own SD tag on every molecule block --
  the SDF equivalent of a column, so a chemistry tool renders it as a
  visible field the same way it would for a measured value's own tags. A
  readout's own tag carries its value, unit *and* direction together
  (`"7.24 nM (lower is better)"`, not a bare `"7.24"`), because that is the
  one thing an SD tag's value is expected to be self-describing about: SDF
  has no header row to hang a unit or direction on the way CSV does.

Direction travels alongside unit in both formats for the same reason
`derive_readouts.py` derives it in the first place: "the same unit and the
same *direction* as a measured one" is the whole point, and a chemist
reading a bare `IC50 (nM): 6.47` next to a measured column has no way to
tell from the file whether lower is better.

Both branches read `protocol.readouts` fresh via the Collection's
`derived_from_run_id -> Run.params["protocol_id"]`, the same path
`GetPredictionResults` already uses, rather than the Collection snapshotting
its own copy of readout names/units/directions. That is safe only because a
published Protocol's readouts are immutable for the rest of that Protocol's
life (`domain/catalog/protocol.py`): there is no code path in this system
that deletes or edits a Protocol once published, so re-reading it here can
never disagree with what the Collection's numbers actually mean -- see the
`ponytail:` comment on the Protocol lookup below for the ceiling on that.

Rather than a fixed list of "reserved" names, the guard against a naming
collision is generic: before either render runs, this module computes the
*actual* final column labels (CSV) or SD tag names (SDF) that rendering
would produce for this Protocol's readouts, and rejects if any two of them
would land on the same name. This is deliberately not a fixed set of
strings to avoid, because each format's own mechanics create more than one
way to collide:

- CSV always renames `"structure"` to `"smiles"` and always appends
  `"generation_method"`. A readout renders to the bare string `"smiles"`
  whenever it has *no* unit and *no* direction -- not exotic; a
  classification CLASS readout with no direction set produces exactly
  that -- and colliding with the appended `"smiles"` column silently drops
  either the structure or the readout's value. A readout literally named
  `generation_method` (again with no unit/direction) collides with the
  appended provenance column the same way.
- SDF never renames anything -- a readout's SD tag is always its bare
  `readout.name`, unit or no unit -- so the only collision surface there is
  a readout named `generation_method`, clobbered by the provenance tag
  `SetProp`'d right after it.

Checking the *rendered* labels for uniqueness, rather than hardcoding the
names above, is what makes this catch both today's known cases and any
future one either mechanism introduces, without anyone having to remember
to extend a list. It also keeps the genuinely safe case safe: a readout
named `smiles` that *does* carry a unit renders as `"smiles (nM, lower is
better)"` in CSV, which does not collide with the bare `"smiles"` column
and exports normally.
"""

from __future__ import annotations

import io
import uuid
from collections import Counter
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
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError

_DIRECTION_LABEL = {"high": "higher is better", "low": "lower is better"}


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
        # ponytail: re-resolves the Protocol live on every export rather than
        # the Collection snapshotting its own copy of readout metadata (see
        # this module's docstring for why that's safe today). Revisit if a
        # future task ever lets a published Protocol be deleted or edited.
        protocol = await self._protocols.get(auth.workspace_id, protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(protocol_id)))

        labels = _final_labels(protocol.readouts, query.format)
        duplicates = sorted(name for name, count in Counter(labels).items() if count > 1)
        if duplicates:
            return Failure(
                ValidationError(
                    f"Cannot export as {query.format.value}: rendering these readouts would "
                    f"produce duplicate column/tag name(s): {duplicates}",
                    detail=(
                        "Rename the colliding readout, or give it a unit/direction so it "
                        "renders to a distinct name."
                    ),
                )
            )

        try:
            raw = self._store.get_bytes(collection.snapshot_uri)
        except FileNotFoundError:
            return Failure(NotFoundError("Collection snapshot", str(collection.id)))
        # ponytail: reads and holds the entire snapshot Parquet in memory --
        # `GetPredictionResults` carries this exact note for the same
        # `pl.read_parquet` shape. Fine at today's per-Collection sizes (a
        # scientist's triage selection, not a whole run); upgrade path if a
        # Collection's row count ever grows large is `scan_parquet`.
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


def _unit_and_direction(readout: Readout) -> str | None:
    """`"nM, lower is better"` / `"nM"` / `"lower is better"` / `None` --
    whichever of unit and direction the readout actually carries, joined for
    a column header's parenthetical."""
    direction_label = _DIRECTION_LABEL.get(readout.direction or "")
    parts = [part for part in (readout.unit, direction_label) if part]
    return ", ".join(parts) if parts else None


def _csv_rename(readouts: tuple[Readout, ...]) -> dict[str, str]:
    """The exact `polars.DataFrame.rename` mapping `_render_csv` applies --
    factored out so the collision guard in `ExportCollection.__call__` checks
    the *same* computed labels the renderer actually produces, rather than a
    second, driftable copy of this logic."""
    rename = {"structure": "smiles"}
    for readout in readouts:
        label = _unit_and_direction(readout)
        rename[readout.name] = f"{readout.name} ({label})" if label else readout.name
    return rename


def _final_labels(readouts: tuple[Readout, ...], export_format: ExportFormat) -> list[str]:
    """The complete set of column names (CSV) or SD tag names (SDF) this
    Protocol's readouts would render to, including the `generation_method`
    column/tag this module always appends -- what `ExportCollection.__call__`
    checks for duplicates before either render runs.

    `"generation_method"` is also one of `domain.data.target.RESERVED_TARGET_COLUMNS`
    (C1, whole-branch review) -- a TargetSpec can no longer be named that, so
    this branch of the collision this guard checks for is unreachable through
    `CreateDataset` today. The guard itself stays: it is generic (computed
    labels, not a fixed list), so it still catches the *other* collision this
    module's mechanics create (a readout rendering to the bare `"smiles"`
    string), which is not one of the reserved names.
    """
    if export_format is ExportFormat.CSV:
        return [*_csv_rename(readouts).values(), "generation_method"]
    return [*(readout.name for readout in readouts), "generation_method"]


def _render_csv(
    frame: pl.DataFrame, readouts: tuple[Readout, ...], generation_method: str
) -> bytes:
    out = frame.rename(_csv_rename(readouts)).with_columns(
        pl.lit(generation_method).alias("generation_method")
    )
    buffer = io.BytesIO()
    out.write_csv(buffer)
    return buffer.getvalue()


def _value_label(value: object, readout: Readout) -> str:
    """`"7.24 nM (lower is better)"` -- the bare value, its unit, and its
    direction, all in one self-describing string. SDF has no header row to
    hang a unit or direction on the way CSV does, so the *value* itself has
    to carry them."""
    text = f"{value} {readout.unit}" if readout.unit else str(value)
    direction_label = _DIRECTION_LABEL.get(readout.direction or "")
    return f"{text} ({direction_label})" if direction_label else text


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
            mol.SetProp(readout.name, _value_label(row[readout.name], readout))
        mol.SetProp("generation_method", generation_method)
        writer.write(mol)
    writer.close()
    return buffer.getvalue().encode()

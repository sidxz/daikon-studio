"""A ChemCellar run's compounds, stored as an upload the prediction flow already reads.

An ordinary upload (a CSV at `upload_key`) keeps everything downstream unchanged:
`PredictWithProtocol`, its cache, the worker and the export all read that file and
nothing else. The one addition is a small record beside it naming the ChemCellar run.
`PredictWithProtocol` copies that record onto the Run, so the run's page can say where
its compounds came from. The record is written here, on the server, from ChemCellar's
own answer. It never comes from the browser, so it cannot become attached to another file.

Rows are sorted, so an unchanged run imports to the same bytes and a second prediction
on it is a cache hit.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.data.create_dataset import upload_key, upload_source_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.chemcellar import ChemCellar
from daikonstudio.domain.shared.errors import DomainError, ValidationError

COLUMNS = ("smiles", "compound_id", "name")
# The frontend preview's PREVIEW_SAMPLE_SIZE.
SAMPLE_SIZE = 6


@dataclass(frozen=True, kw_only=True)
class ChemCellarImport:
    upload_ref: uuid.UUID
    compound_count: int
    without_structure: int
    sample: tuple[str, ...]
    source: dict[str, Any]


class ImportChemCellarRun:
    def __init__(self, chemcellar: ChemCellar, store: BlobStore) -> None:
        self._chemcellar = chemcellar
        self._store = store

    async def __call__(
        self,
        run_id: uuid.UUID,
        *,
        forwarded_headers: Mapping[str, str],
        auth: AuthContext | None = None,
    ) -> Result[ChemCellarImport, DomainError]:
        require_authenticated(auth)
        require_editor(auth)  # It writes an upload, as StoreUpload does.
        assert auth is not None

        found = await self._chemcellar.run_compounds(run_id, forwarded_headers=forwarded_headers)
        drawn = sorted(
            (c for c in found.compounds if c.smiles and c.smiles.strip()),
            key=lambda c: (c.registration_number or "", str(c.molecule_id)),
        )
        if not found.compounds:
            return Failure(ValidationError("This ChemCellar run has no compounds yet."))
        if not drawn:
            return Failure(
                ValidationError("This ChemCellar run has no compounds with a disclosed structure.")
            )

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(
            ((c.smiles or "").strip(), c.registration_number or "", c.name or "") for c in drawn
        )
        source: dict[str, Any] = {
            "app": "chemcellar",
            "run_id": str(found.run_id),
            "protocol_id": str(found.protocol_id),
            "protocol_name": found.protocol_name,
            "run_date": found.run_date.isoformat(),
            "compounds_without_structure": len(found.compounds) - len(drawn),
        }

        upload_ref = uuid.uuid4()
        await asyncio.to_thread(
            self._store.put_bytes,
            upload_key(auth.workspace_id, upload_ref),
            buffer.getvalue().encode(),
        )
        await asyncio.to_thread(
            self._store.put_bytes,
            upload_source_key(auth.workspace_id, upload_ref),
            json.dumps(source).encode(),
        )
        return Success(
            ChemCellarImport(
                upload_ref=upload_ref,
                compound_count=len(drawn),
                without_structure=len(found.compounds) - len(drawn),
                sample=tuple((c.smiles or "").strip() for c in drawn[:SAMPLE_SIZE]),
                source=source,
            )
        )

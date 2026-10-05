"""Browsing and importing ChemCellar runs, on the requesting user's behalf.

The caller's own Duar headers are forwarded to ChemCellar, which authorizes the user
itself (see `application/ports/chemcellar.py`). Browsing goes straight to the port;
importing goes through `ImportChemCellarRun`, which writes an upload.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from daikonstudio.application.data.import_chemcellar_run import ImportChemCellarRun
from daikonstudio.application.ports.chemcellar import ChemCellar
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.interface.routes.runs import RunSourceResponse

router = APIRouter(prefix="/api/v1/chemcellar", tags=["chemcellar"])

_FORWARDED_HEADERS = ("authorization", "x-authz-token")

ChemCellarDep = Annotated[ChemCellar, Depends(use_case(ChemCellar))]
ImportChemCellarRunDep = Annotated[ImportChemCellarRun, Depends(use_case(ImportChemCellarRun))]


def _forwarded(request: Request) -> dict[str, str]:
    """The caller's own Duar headers, so ChemCellar authorizes the user, not Studio."""
    return {name: value for name in _FORWARDED_HEADERS if (value := request.headers.get(name))}


class ChemCellarProtocolResponse(BaseModel):
    id: uuid.UUID
    name: str


class ChemCellarRunResponse(BaseModel):
    id: uuid.UUID
    protocol_id: uuid.UUID
    run_date: date
    status: str
    # Compounds with at least one readout: ChemCellar's own count for the run.
    measured_count: int
    plate_count: int
    plate_barcodes: list[str]


class ChemCellarImportResponse(BaseModel):
    upload_ref: uuid.UUID
    compound_count: int
    # Compounds in the run whose structure ChemCellar does not disclose; not included.
    without_structure: int
    sample: list[str]
    source: RunSourceResponse


@router.get("/protocols", response_model=list[ChemCellarProtocolResponse])
async def list_chemcellar_protocols(
    request: Request, auth: AuthDep, chemcellar: ChemCellarDep
) -> list[ChemCellarProtocolResponse]:
    protocols = await chemcellar.list_protocols(forwarded_headers=_forwarded(request))
    return [ChemCellarProtocolResponse(id=p.id, name=p.name) for p in protocols]


@router.get("/protocols/{protocol_id}/runs", response_model=list[ChemCellarRunResponse])
async def list_chemcellar_runs(
    protocol_id: uuid.UUID, request: Request, auth: AuthDep, chemcellar: ChemCellarDep
) -> list[ChemCellarRunResponse]:
    runs = await chemcellar.list_runs(protocol_id, forwarded_headers=_forwarded(request))
    return [
        ChemCellarRunResponse(
            id=r.id,
            protocol_id=r.protocol_id,
            run_date=r.run_date,
            status=r.status,
            measured_count=r.measured_count,
            plate_count=r.plate_count,
            plate_barcodes=list(r.plate_barcodes),
        )
        for r in runs
    ]


@router.post("/runs/{run_id}/import", response_model=ChemCellarImportResponse, status_code=201)
async def import_chemcellar_run(
    run_id: uuid.UUID, request: Request, auth: AuthDep, service: ImportChemCellarRunDep
) -> ChemCellarImportResponse:
    imported = result_to_response(
        await service(run_id, forwarded_headers=_forwarded(request), auth=auth)
    )
    return ChemCellarImportResponse(
        upload_ref=imported.upload_ref,
        compound_count=imported.compound_count,
        without_structure=imported.without_structure,
        sample=list(imported.sample),
        source=RunSourceResponse.model_validate(imported.source),
    )

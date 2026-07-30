"""Run endpoints: submit a prediction, poll it, read its results, cancel it.

`RunResponse` is defined here rather than duplicated: `protocols.py`'s own
training endpoint returns the same shape (a freshly created Run) and imports
it from here, so there is exactly one definition of what a Run looks like
over HTTP.

Every request body here is `extra="forbid"`, the same reason `datasets.py`
and `protocols.py` give: `workspace_id` is refused, not silently dropped.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict

from daikonstudio.application.execution.predict_with_protocol import (
    CancelRun,
    CancelRunCommand,
    GetPredictionResults,
    GetPredictionResultsQuery,
    GetRun,
    GetRunQuery,
    PredictedReadout,
    PredictionRow,
    PredictWithProtocol,
    PredictWithProtocolCommand,
)
from daikonstudio.domain.execution.run import Run
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.interface.pagination import PaginatedResponse

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])

PredictWithProtocolDep = Annotated[PredictWithProtocol, Depends(use_case(PredictWithProtocol))]
GetRunDep = Annotated[GetRun, Depends(use_case(GetRun))]
CancelRunDep = Annotated[CancelRun, Depends(use_case(CancelRun))]
GetPredictionResultsDep = Annotated[GetPredictionResults, Depends(use_case(GetPredictionResults))]


class PredictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: uuid.UUID
    upload_ref: str
    structure_column: str
    conditions: dict[str, Any] = {}


class RunResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    kind: str
    status: str
    progress: float
    phase: str | None
    result_uri: str | None
    error_message: str | None
    # Set at enqueue time for a prediction Run (you pick the Protocol to run),
    # and at completion for a training Run (the Protocol does not exist until
    # the work finishes). So it is null for a training Run only while that run
    # is still pending or running, or if it failed.
    #
    # Without it a client holding only a run id has no way to fetch the
    # Protocol's readout metadata -- the unit and direction that make a
    # prediction comparable to a measurement (Task 17 review, Important 5) --
    # and no way at all to reach the Scorecard a training run just built.
    protocol_id: uuid.UUID | None
    created_at: datetime

    @classmethod
    def from_domain(cls, run: Run) -> RunResponse:
        return cls(
            id=run.id,
            workspace_id=run.workspace_id,
            kind=run.kind.value,
            status=run.status.value,
            progress=run.progress,
            phase=run.phase,
            result_uri=run.result_uri,
            error_message=run.error_message,
            protocol_id=run.protocol_id,
            created_at=run.created_at,
        )


class PredictedReadoutResponse(BaseModel):
    value: float
    unit: str | None
    direction: str | None

    @classmethod
    def from_domain(cls, readout: PredictedReadout) -> PredictedReadoutResponse:
        return cls(value=readout.value, unit=readout.unit, direction=readout.direction)


class PredictionResponse(BaseModel):
    """One scored compound. `readouts` holds one entry per Readout the
    Protocol declares (a numeric value for regression; probability and class
    for classification), each carrying its own unit and direction so a
    prediction can be lined up against a measurement without a second call --
    see `PredictionRow`'s own docstring for why `uncertainty` and
    `applicability` are shaped the way they are."""

    structure: str
    readouts: dict[str, PredictedReadoutResponse]
    uncertainty: float | None
    applicability: float | None

    @classmethod
    def from_domain(cls, row: PredictionRow) -> PredictionResponse:
        return cls(
            structure=row.structure,
            readouts={
                name: PredictedReadoutResponse.from_domain(readout)
                for name, readout in row.readouts.items()
            },
            uncertainty=row.uncertainty,
            applicability=row.applicability,
        )


@router.post("", response_model=RunResponse, status_code=202)
async def create_run(
    body: PredictBody, auth: AuthDep, service: PredictWithProtocolDep
) -> RunResponse:
    command = PredictWithProtocolCommand(
        protocol_id=body.protocol_id,
        upload_ref=body.upload_ref,
        structure_column=body.structure_column,
        conditions=body.conditions,
    )
    return RunResponse.from_domain(result_to_response(await service(command, auth=auth)))


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: uuid.UUID, auth: AuthDep, service: GetRunDep) -> RunResponse:
    run = result_to_response(await service(GetRunQuery(run_id=run_id), auth=auth))
    return RunResponse.from_domain(run)


@router.get("/{run_id}/results", response_model=PaginatedResponse[PredictionResponse])
async def get_run_results(
    run_id: uuid.UUID,
    auth: AuthDep,
    service: GetPredictionResultsDep,
    cursor: str | None = None,
    limit: int | None = None,
) -> PaginatedResponse[PredictionResponse]:
    page = result_to_response(
        await service(
            GetPredictionResultsQuery(run_id=run_id, cursor=cursor, limit=limit), auth=auth
        )
    )
    return PaginatedResponse(
        items=[PredictionResponse.from_domain(row) for row in page.items],
        next_cursor=page.next_cursor,
    )


@router.post("/{run_id}/cancel", status_code=204)
async def cancel_run(run_id: uuid.UUID, auth: AuthDep, service: CancelRunDep) -> Response:
    result_to_response(await service(CancelRunCommand(run_id=run_id), auth=auth))
    return Response(status_code=204)

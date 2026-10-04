"""Run endpoints: submit a prediction, poll it, read its results, cancel it.

`RunResponse` is defined here rather than duplicated: `protocols.py`'s own
training endpoint returns the same shape (a freshly created Run) and imports
it from here, so there is exactly one definition of what a Run looks like
over HTTP.

Every request body here is `extra="forbid"`, the same reason `datasets.py`
and `protocols.py` give: `workspace_id` is refused, not silently dropped.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, StrictBool

from daikonstudio.application.catalog.get_chemical_space import (
    MAX_LOOKUPS,
    GetRunChemicalSpace,
    GetRunChemicalSpaceCompounds,
    GetRunChemicalSpaceCompoundsQuery,
    GetRunChemicalSpaceQuery,
)
from daikonstudio.application.execution.list_runs import ListRuns, ListRunsQuery
from daikonstudio.application.execution.predict_with_protocol import (
    CancelRun,
    CancelRunCommand,
    GetPredictionResultRanges,
    GetPredictionResultRangesQuery,
    GetPredictionResults,
    GetPredictionResultsQuery,
    GetRun,
    GetRunQuery,
    PredictedReadout,
    PredictionRow,
    PredictWithProtocol,
    PredictWithProtocolCommand,
)
from daikonstudio.application.execution.result_view import RangeFilter, SortSpec
from daikonstudio.application.execution.retry_run import RetryRun, RetryRunCommand
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.interface.pagination import PaginatedResponse

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])

PredictWithProtocolDep = Annotated[PredictWithProtocol, Depends(use_case(PredictWithProtocol))]
GetRunDep = Annotated[GetRun, Depends(use_case(GetRun))]
CancelRunDep = Annotated[CancelRun, Depends(use_case(CancelRun))]
GetPredictionResultsDep = Annotated[GetPredictionResults, Depends(use_case(GetPredictionResults))]
GetPredictionResultRangesDep = Annotated[
    GetPredictionResultRanges, Depends(use_case(GetPredictionResultRanges))
]
ListRunsDep = Annotated[ListRuns, Depends(use_case(ListRuns))]
RetryRunDep = Annotated[RetryRun, Depends(use_case(RetryRun))]


class PredictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: uuid.UUID
    upload_ref: str
    structure_column: str
    conditions: dict[str, Any] = {}
    # Optional: the uploaded column that names each compound, carried into the
    # results as `compound_id` so predictions can be joined back to the file.
    id_column: str | None = None


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
    # Outcome numbers denormalised onto the row. A training run: the headline
    # metric, its value and the baseline's (`Run.record_metrics`). A prediction
    # run: `uploaded_rows` and `scored_rows`, whose difference is the structures
    # that did not parse (`Run.record_prediction_counts`). Null until READY.
    metrics: dict[str, Any] | None
    # The runner lane a queued run waits on; null before it is enqueued (and in
    # inline mode, which has no queue). Together with GET /runners this is what
    # lets a client say "no runner serves the gpu lane right now".
    lane: str | None
    # The name the scientist typed for a training Run, read from its write-once
    # `params`. It lets the Protocols page label a run that has no Protocol yet;
    # a prediction Run carries none.
    name: str | None
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
            metrics=run.metrics,
            lane=run.lane,
            name=run.params.get("name"),
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
    `applicability` are shaped the way they are. `uncertainty` has one entry per
    target, keyed by its column."""

    row_id: int
    structure: str
    readouts: dict[str, PredictedReadoutResponse]
    uncertainty: dict[str, float | None]
    applicability: float | None
    # The row's 1-based position in the uploaded file, and the value of the
    # upload's identifier column when the request named one. Both null on runs
    # scored before these were recorded.
    input_row: int | None
    compound_id: str | None

    @classmethod
    def from_domain(cls, row: PredictionRow) -> PredictionResponse:
        return cls(
            row_id=row.row_id,
            structure=row.structure,
            readouts={
                name: PredictedReadoutResponse.from_domain(readout)
                for name, readout in row.readouts.items()
            },
            uncertainty=row.uncertainty,
            applicability=row.applicability,
            input_row=row.input_row,
            compound_id=row.compound_id,
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
        id_column=body.id_column,
    )
    return RunResponse.from_domain(result_to_response(await service(command, auth=auth)))


@router.get("", response_model=PaginatedResponse[RunResponse])
async def list_runs(
    auth: AuthDep,
    service: ListRunsDep,
    kind: RunKind | None = None,
    protocol_id: uuid.UUID | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> PaginatedResponse[RunResponse]:
    """Declared before `/{run_id}` so the empty path is not swallowed by the
    id route -- Starlette matches in declaration order."""
    # `limit` is clamped inside the use case, not here: a worker calling it
    # directly must get the same ceiling as an HTTP caller.
    page = result_to_response(
        await service(
            ListRunsQuery(kind=kind, protocol_id=protocol_id, cursor=cursor, limit=limit),
            auth=auth,
        )
    )
    return PaginatedResponse(
        items=[RunResponse.from_domain(run) for run in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: uuid.UUID, auth: AuthDep, service: GetRunDep) -> RunResponse:
    run = result_to_response(await service(GetRunQuery(run_id=run_id), auth=auth))
    return RunResponse.from_domain(run)


def _parse_filters(raw: str | None) -> tuple[RangeFilter, ...]:
    """`{"applicability": {"min": 0.5}}` -> RangeFilters.

    A JSON object in one query parameter rather than repeated scalar params
    (`applicability_min=...`): filterable columns are a Protocol's own readout
    names, so a fixed parameter list cannot name them without being invented
    per Protocol.
    """
    if raw is None:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise _invalid_filters(f"not valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise _invalid_filters("must be a JSON object")

    filters: list[RangeFilter] = []
    for column, bounds in parsed.items():
        if not isinstance(bounds, dict):
            raise _invalid_filters(f"'{column}' must be an object with min and/or max")
        minimum, maximum = bounds.get("min"), bounds.get("max")
        if minimum is None and maximum is None:
            raise _invalid_filters(f"'{column}' needs at least one of min or max")
        try:
            filters.append(
                RangeFilter(
                    column=column,
                    minimum=None if minimum is None else float(minimum),
                    maximum=None if maximum is None else float(maximum),
                )
            )
        except (TypeError, ValueError) as error:
            raise _invalid_filters(f"'{column}': min and max must be numbers") from error
    return tuple(filters)


def _invalid_filters(message: str) -> RequestValidationError:
    return RequestValidationError(
        [{"loc": ("query", "filters"), "msg": message, "type": "value_error"}]
    )


GetRunChemicalSpaceDep = Annotated[GetRunChemicalSpace, Depends(use_case(GetRunChemicalSpace))]
GetRunChemicalSpaceCompoundsDep = Annotated[
    GetRunChemicalSpaceCompounds, Depends(use_case(GetRunChemicalSpaceCompounds))
]


class RunMapPointsResponse(BaseModel):
    """One entry per scored compound, in results-file order (`row_id`). `x`/`y` are
    the similarity-weighted centre of `neighbors`, which index the protocol map's
    points; `applicability` is the true Tanimoto to the nearest of them."""

    x: list[float]
    y: list[float]
    row_id: list[int]
    applicability: list[float | None]
    neighbors: list[list[int]]


class RunMapSummaryResponse(BaseModel):
    total: int
    in_domain: int
    threshold: float
    nearest_min: float | None
    nearest_max: float | None


class RunChemicalSpaceResponse(BaseModel):
    status: Literal["ready", "missing"]
    points: RunMapPointsResponse | None = None
    summary: RunMapSummaryResponse | None = None


class RunMapCompoundResponse(BaseModel):
    row_id: int
    structure: str
    compound_id: str | None
    values: dict[str, float | None]
    applicability: float | None


@router.get("/{run_id}/chemical-space", response_model=RunChemicalSpaceResponse)
async def get_run_chemical_space(
    run_id: uuid.UUID, auth: AuthDep, service: GetRunChemicalSpaceDep
) -> RunChemicalSpaceResponse:
    view = result_to_response(await service(GetRunChemicalSpaceQuery(run_id=run_id), auth=auth))
    if view.status != "ready":
        return RunChemicalSpaceResponse(status="missing")
    assert view.x is not None and view.y is not None and view.row_id is not None
    assert view.applicability is not None and view.neighbors is not None
    assert view.total is not None and view.in_domain is not None
    return RunChemicalSpaceResponse(
        status="ready",
        points=RunMapPointsResponse(
            x=view.x,
            y=view.y,
            row_id=view.row_id,
            applicability=view.applicability,
            neighbors=view.neighbors,
        ),
        summary=RunMapSummaryResponse(
            total=view.total,
            in_domain=view.in_domain,
            threshold=view.threshold,
            nearest_min=view.nearest_min,
            nearest_max=view.nearest_max,
        ),
    )


@router.get("/{run_id}/chemical-space/compounds", response_model=list[RunMapCompoundResponse])
async def get_run_chemical_space_compounds(
    run_id: uuid.UUID,
    auth: AuthDep,
    service: GetRunChemicalSpaceCompoundsDep,
    rows: Annotated[list[int], Query(max_length=MAX_LOOKUPS)],
) -> list[RunMapCompoundResponse]:
    items = result_to_response(
        await service(GetRunChemicalSpaceCompoundsQuery(run_id=run_id, rows=rows), auth=auth)
    )
    return [
        RunMapCompoundResponse(
            row_id=i.row_id,
            structure=i.structure,
            compound_id=i.compound_id,
            values=i.values,
            applicability=i.applicability,
        )
        for i in items
    ]


@router.get("/{run_id}/results", response_model=PaginatedResponse[PredictionResponse])
async def get_run_results(
    run_id: uuid.UUID,
    auth: AuthDep,
    service: GetPredictionResultsDep,
    cursor: str | None = None,
    limit: int | None = None,
    sort_by: str | None = None,
    sort_dir: Literal["asc", "desc"] = "asc",
    filters: str | None = None,
) -> PaginatedResponse[PredictionResponse]:
    """`sort_by` and `filters` name columns a Protocol declares, so which names
    are legal is decided in the use case -- this function only parses the wire
    format. `sort_dir` is a Literal, so FastAPI rejects anything else itself."""
    page = result_to_response(
        await service(
            GetPredictionResultsQuery(
                run_id=run_id,
                cursor=cursor,
                limit=limit,
                sort=None
                if sort_by is None
                else SortSpec(column=sort_by, descending=sort_dir == "desc"),
                filters=_parse_filters(filters),
            ),
            auth=auth,
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


class RetryRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Start over instead of resuming: discards the run's saved training progress.
    fresh: StrictBool = False


@router.post("/{run_id}/retry", status_code=204)
async def retry_run(
    run_id: uuid.UUID, auth: AuthDep, service: RetryRunDep, body: RetryRunBody | None = None
) -> Response:
    """Re-execute a failed or cancelled run in place, resuming a training run from its
    saved progress unless `fresh` is set; 409 for any other status."""
    fresh = body.fresh if body is not None else False
    result_to_response(await service(RetryRunCommand(run_id=run_id, fresh=fresh), auth=auth))
    return Response(status_code=204)


class ColumnRangeResponse(BaseModel):
    min: float | None
    max: float | None


@router.get("/{run_id}/results/ranges", response_model=dict[str, ColumnRangeResponse])
async def get_run_result_ranges(
    run_id: uuid.UUID, auth: AuthDep, service: GetPredictionResultRangesDep
) -> dict[str, ColumnRangeResponse]:
    """Each numeric results column's range across the whole Run, keyed by the same
    column names `results` sorts and filters by: the readouts, the uncertainty
    columns and `applicability`. Unfiltered, so a scale does not move with a filter."""
    ranges = result_to_response(
        await service(GetPredictionResultRangesQuery(run_id=run_id), auth=auth)
    )
    return {
        column: ColumnRangeResponse(min=bounds.minimum, max=bounds.maximum)
        for column, bounds in ranges.items()
    }

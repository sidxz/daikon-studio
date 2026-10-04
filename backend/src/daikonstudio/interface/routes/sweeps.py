"""Sweep endpoints: submit N configs, list the groups, read one, cancel it.

`SweepRunResponse` is the one place `Run.params` is projected onto the wire.
That is deliberate and scoped: a ranked comparison is unreadable without the
engine and conditions each row was produced from, and every alternative --
re-fetching a Protocol per row, or reading N Scorecards -- costs a request per
run to recover something the row already holds.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from daikonstudio.application.execution.sweeps import (
    MAX_CONFIGS,
    CancelSweep,
    CancelSweepCommand,
    GetSweep,
    GetSweepQuery,
    ListSweeps,
    ListSweepsQuery,
    SubmitSweep,
    SubmitSweepCommand,
    SweepConfig,
)
from daikonstudio.application.ports.run_repository import SweepSummary
from daikonstudio.domain.execution.run import Run
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/sweeps", tags=["sweeps"])

SubmitSweepDep = Annotated[SubmitSweep, Depends(use_case(SubmitSweep))]
ListSweepsDep = Annotated[ListSweeps, Depends(use_case(ListSweeps))]
GetSweepDep = Annotated[GetSweep, Depends(use_case(GetSweep))]
CancelSweepDep = Annotated[CancelSweep, Depends(use_case(CancelSweep))]


class SweepConfigBody(BaseModel):
    engine_id: str
    conditions: dict[str, Any] = Field(default_factory=dict)


class SubmitSweepBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_id: uuid.UUID
    # min_length=1 here as well as in the use case: an empty sweep is a 422
    # about the request's shape, and saying so at the edge means the client
    # gets a field-level error instead of a domain message.
    configs: list[SweepConfigBody] = Field(min_length=1, max_length=MAX_CONFIGS)
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, Any] = Field(default_factory=dict)
    # Applies to every config in the sweep: configs measured at different cutoffs are
    # not a comparison.
    tune_cutoffs: bool = False


class SweepRunResponse(BaseModel):
    id: uuid.UUID
    sweep_id: uuid.UUID | None
    name: str
    engine_id: str
    conditions: dict[str, Any]
    status: str
    progress: float
    phase: str | None
    error_message: str | None
    protocol_id: uuid.UUID | None
    # None until this run reaches `ready`. The ranked table shows such a row as
    # unranked rather than as zero -- see `Run.record_metrics`.
    metrics: dict[str, Any] | None
    created_at: datetime

    @classmethod
    def from_domain(cls, run: Run) -> SweepRunResponse:
        return cls(
            id=run.id,
            sweep_id=run.sweep_id,
            name=run.params.get("name", ""),
            engine_id=run.params.get("engine_id", ""),
            conditions=run.params.get("conditions", {}),
            status=run.status.value,
            progress=run.progress,
            phase=run.phase,
            error_message=run.error_message,
            protocol_id=run.protocol_id,
            metrics=run.metrics,
            created_at=run.created_at,
        )


class SweepResponse(BaseModel):
    sweep_id: uuid.UUID
    name: str | None
    dataset_id: uuid.UUID | None
    created_at: datetime
    total: int
    by_status: dict[str, int]

    @classmethod
    def from_domain(cls, summary: SweepSummary) -> SweepResponse:
        return cls(
            sweep_id=summary.sweep_id,
            name=summary.name,
            dataset_id=summary.dataset_id,
            created_at=summary.created_at,
            total=summary.total,
            by_status=summary.by_status,
        )


class SweepDetailResponse(BaseModel):
    sweep_id: uuid.UUID
    name: str | None
    dataset_id: uuid.UUID | None
    runs: list[SweepRunResponse]

    @classmethod
    def from_runs(cls, sweep_id: uuid.UUID, runs: list[Run]) -> SweepDetailResponse:
        # Every member carries the same two, written once at submission, so the
        # first one answers for the group without a second query. Both routes
        # that call this (submit, get) only ever hold a non-empty list --
        # `SubmitSweep` fails before creating any run, and `GetSweep` returns
        # `NotFoundError` for an empty sweep -- but the empty case is still
        # handled here rather than left to raise, so this stays safe even if a
        # future caller's guarantee is weaker than today's two.
        if not runs:
            return cls(sweep_id=sweep_id, name=None, dataset_id=None, runs=[])
        first = runs[0].params
        dataset_id = first.get("dataset_id")
        return cls(
            sweep_id=sweep_id,
            name=first.get("sweep_name"),
            dataset_id=uuid.UUID(dataset_id) if dataset_id else None,
            runs=[SweepRunResponse.from_domain(run) for run in runs],
        )


class SweepListResponse(BaseModel):
    items: list[SweepResponse]


@router.post("", response_model=SweepDetailResponse, status_code=202)
async def submit_sweep(
    body: SubmitSweepBody, auth: AuthDep, service: SubmitSweepDep
) -> SweepDetailResponse:
    command = SubmitSweepCommand(
        name=body.name,
        dataset_id=body.dataset_id,
        configs=[
            SweepConfig(engine_id=config.engine_id, conditions=config.conditions)
            for config in body.configs
        ],
        baseline_engine_id=body.baseline_engine_id,
        baseline_conditions=body.baseline_conditions,
        tune_cutoffs=body.tune_cutoffs,
    )
    sweep = result_to_response(await service(command, auth=auth))
    return SweepDetailResponse.from_runs(sweep.sweep_id, sweep.runs)


@router.get("", response_model=SweepListResponse)
async def list_sweeps(auth: AuthDep, service: ListSweepsDep, limit: int = 50) -> SweepListResponse:
    # `limit` is clamped inside the use case, not here -- same pattern as
    # `list_runs`, so a zero or negative value never reaches Postgres as
    # `LIMIT -1`.
    summaries = result_to_response(await service(ListSweepsQuery(limit=limit), auth=auth))
    return SweepListResponse(items=[SweepResponse.from_domain(s) for s in summaries])


@router.get("/{sweep_id}", response_model=SweepDetailResponse)
async def get_sweep(
    sweep_id: uuid.UUID, auth: AuthDep, service: GetSweepDep
) -> SweepDetailResponse:
    runs = result_to_response(await service(GetSweepQuery(sweep_id=sweep_id), auth=auth))
    return SweepDetailResponse.from_runs(sweep_id, runs)


@router.post("/{sweep_id}/cancel", status_code=204)
async def cancel_sweep(sweep_id: uuid.UUID, auth: AuthDep, service: CancelSweepDep) -> Response:
    result_to_response(await service(CancelSweepCommand(sweep_id=sweep_id), auth=auth))
    return Response(status_code=204)

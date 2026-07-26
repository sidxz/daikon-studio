"""Protocol endpoints: train, list, read, publish, and read the Scorecard.

Training returns 202 with the *Run*, not the Protocol -- the Protocol does not
exist yet when the request returns (see `TrainProtocol.__call__`: it enqueues
the job and hands back the Run it just created). Everything else here reads or
mutates a Protocol by id.

Conditions are accepted as a bare `dict[str, Any]` and forwarded unvalidated:
`TrainProtocol` (Task 14) deliberately validates them against the chosen
engine's manifest inside the worker, not at the API boundary, so an invalid
hyperparameter fails the Run itself -- visibly, on the row the client is
already polling -- rather than the request. Re-validating here would just be
the same check running twice on two different inputs (the route sees the raw
body; the worker sees whatever the engine's own defaults filled in), which
could disagree.

Every request body here is `extra="forbid"`, for the same reason
`datasets.py` documents: `workspace_id` is refused, not silently dropped, so
no client comes to believe supplying one had any effect.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict

from daikonstudio.application.catalog.get_scorecard import GetScorecard, GetScorecardQuery
from daikonstudio.application.catalog.list_protocols import (
    GetProtocol,
    GetProtocolQuery,
    ListProtocols,
    ListProtocolsQuery,
)
from daikonstudio.application.catalog.publish_protocol import (
    PublishProtocol,
    PublishProtocolCommand,
)
from daikonstudio.application.execution.train_protocol import TrainProtocol, TrainProtocolCommand
from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.catalog.readout import Readout
from daikonstudio.domain.execution.run import Run
from daikonstudio.domain.execution.scorecard import Scorecard, WorstRow
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.interface.pagination import PaginatedResponse

router = APIRouter(prefix="/api/v1/protocols", tags=["protocols"])

TrainProtocolDep = Annotated[TrainProtocol, Depends(use_case(TrainProtocol))]
ListProtocolsDep = Annotated[ListProtocols, Depends(use_case(ListProtocols))]
GetProtocolDep = Annotated[GetProtocol, Depends(use_case(GetProtocol))]
GetScorecardDep = Annotated[GetScorecard, Depends(use_case(GetScorecard))]
PublishProtocolDep = Annotated[PublishProtocol, Depends(use_case(PublishProtocol))]


class TrainProtocolBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_id: uuid.UUID
    engine_id: str
    conditions: dict[str, Any]


def _readout_to_dict(readout: Readout) -> dict[str, Any]:
    return {
        "name": readout.name,
        "type": readout.type.value,
        "unit": readout.unit,
        "direction": readout.direction,
        "description": readout.description,
    }


class RunResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    kind: str
    status: str
    progress: float
    phase: str | None
    result_uri: str | None
    error_message: str | None
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
            created_at=run.created_at,
        )


class ProtocolResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    dataset_id: uuid.UUID
    engine_id: str
    artifact_uri: str
    readouts: list[dict[str, Any]]
    conditions: dict[str, Any]
    status: str
    is_locked: bool
    published_at: datetime | None
    parent_protocol_id: uuid.UUID | None
    protocol_version: int
    created_at: datetime

    @classmethod
    def from_domain(cls, protocol: InSilicoProtocol) -> ProtocolResponse:
        return cls(
            id=protocol.id,
            workspace_id=protocol.workspace_id,
            name=protocol.name,
            dataset_id=protocol.dataset_id,
            engine_id=protocol.engine_id,
            artifact_uri=protocol.artifact_uri,
            readouts=[_readout_to_dict(readout) for readout in protocol.readouts],
            conditions=dict(protocol.conditions),
            status=protocol.status.value,
            is_locked=protocol.is_locked,
            published_at=protocol.published_at,
            parent_protocol_id=protocol.parent_protocol_id,
            protocol_version=protocol.protocol_version,
            created_at=protocol.created_at,
        )


class WorstRowResponse(BaseModel):
    structure: str
    actual: float
    predicted: float
    residual: float
    scaffold: str
    similarity: float | None

    @classmethod
    def from_domain(cls, row: WorstRow) -> WorstRowResponse:
        return cls(
            structure=row.structure,
            actual=row.actual,
            predicted=row.predicted,
            residual=row.residual,
            scaffold=row.scaffold,
            similarity=row.similarity,
        )


class ScorecardResponse(BaseModel):
    """Every field here is load-bearing for honest rendering -- see
    `domain/execution/scorecard.py`'s docstring for what each one means and
    why it exists. In particular: `baseline_is_self` true means the chosen
    engine *is* the baseline (render "this model is the baseline", not a
    head-to-head that never happened); `metrics_undefined` is why a metric in
    `metrics` reads `null` instead of a number; `random_split_unavailable`
    distinguishes "not applicable" (`None`/`None`) from "could not be
    computed" (`None`/a reason) for `random_split_metrics`.
    """

    primary_metric: str
    prediction_kind: str
    metrics: dict[str, float | None]
    metrics_undefined: dict[str, str] | None
    baseline_engine_id: str
    baseline_metrics: dict[str, float | None]
    baseline_is_self: bool
    random_split_metrics: dict[str, float | None] | None
    random_split_unavailable: str | None
    noise_floor: float | None
    worst_rows: list[WorstRowResponse]
    applicability_coverage: float | None

    @classmethod
    def from_domain(cls, card: Scorecard) -> ScorecardResponse:
        return cls(
            primary_metric=card.primary_metric,
            prediction_kind=card.prediction_kind,
            metrics=card.metrics,
            metrics_undefined=card.metrics_undefined,
            baseline_engine_id=card.baseline_engine_id,
            baseline_metrics=card.baseline_metrics,
            baseline_is_self=card.baseline_is_self,
            random_split_metrics=card.random_split_metrics,
            random_split_unavailable=card.random_split_unavailable,
            noise_floor=card.noise_floor,
            worst_rows=[WorstRowResponse.from_domain(row) for row in card.worst_rows],
            applicability_coverage=card.applicability_coverage,
        )


@router.post("", response_model=RunResponse, status_code=202)
async def train_protocol(
    body: TrainProtocolBody, auth: AuthDep, service: TrainProtocolDep
) -> RunResponse:
    command = TrainProtocolCommand(
        name=body.name,
        dataset_id=body.dataset_id,
        engine_id=body.engine_id,
        conditions=body.conditions,
    )
    return RunResponse.from_domain(result_to_response(await service(command, auth=auth)))


@router.get("", response_model=PaginatedResponse[ProtocolResponse])
async def list_protocols(
    auth: AuthDep,
    service: ListProtocolsDep,
    cursor: str | None = None,
    limit: int | None = None,
) -> PaginatedResponse[ProtocolResponse]:
    page = result_to_response(
        await service(ListProtocolsQuery(cursor=cursor, limit=limit), auth=auth)
    )
    return PaginatedResponse(
        items=[ProtocolResponse.from_domain(protocol) for protocol in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/{protocol_id}", response_model=ProtocolResponse)
async def get_protocol(
    protocol_id: uuid.UUID, auth: AuthDep, service: GetProtocolDep
) -> ProtocolResponse:
    protocol = result_to_response(
        await service(GetProtocolQuery(protocol_id=protocol_id), auth=auth)
    )
    return ProtocolResponse.from_domain(protocol)


@router.get("/{protocol_id}/scorecard", response_model=ScorecardResponse)
async def get_scorecard(
    protocol_id: uuid.UUID, auth: AuthDep, service: GetScorecardDep
) -> ScorecardResponse:
    card = result_to_response(await service(GetScorecardQuery(protocol_id=protocol_id), auth=auth))
    return ScorecardResponse.from_domain(card)


@router.post("/{protocol_id}/publish", status_code=204)
async def publish_protocol(
    protocol_id: uuid.UUID, auth: AuthDep, service: PublishProtocolDep
) -> Response:
    result_to_response(await service(PublishProtocolCommand(protocol_id=protocol_id), auth=auth))
    return Response(status_code=204)

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
from pydantic import BaseModel, ConfigDict, Field

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
from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.data.target import Direction
from daikonstudio.domain.execution.scorecard import Scorecard, WorstRow
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.interface.pagination import PaginatedResponse
from daikonstudio.interface.routes.runs import RunResponse

router = APIRouter(prefix="/api/v1/protocols", tags=["protocols"])

TrainProtocolDep = Annotated[TrainProtocol, Depends(use_case(TrainProtocol))]
ListProtocolsDep = Annotated[ListProtocols, Depends(use_case(ListProtocols))]
GetProtocolDep = Annotated[GetProtocol, Depends(use_case(GetProtocol))]
GetScorecardDep = Annotated[GetScorecard, Depends(use_case(GetScorecard))]
PublishProtocolDep = Annotated[PublishProtocol, Depends(use_case(PublishProtocol))]


class TrainProtocolBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # `max_length` matches `InSilicoProtocolModel.name`'s `String(256)` column.
    # Worse than the other three fields this same review finding covers: the
    # worker, not this route, is what inserts the Protocol row -- so an
    # over-long name here would return 202, run all three fits (the chosen
    # engine, the mandatory baseline, and -- on a scaffold split -- the
    # optimism-gap comparison), and only then fail on the insert, having
    # already paid for compute the request should never have accepted
    # (whole-branch review, Important 3).
    name: str = Field(max_length=256)
    dataset_id: uuid.UUID
    engine_id: str
    conditions: dict[str, Any]
    # Optional: absent means the registry's flagged default baseline. A
    # comparison always happens -- this chooses which one, it does not skip it.
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, Any] = Field(default_factory=dict)


class ReadoutResponse(BaseModel):
    """One declared output of a trained Protocol.

    Typed rather than a bare `dict` because `unit` and `direction` are the
    whole point of a Readout: they are what let a predicted IC50 be lined up
    against a measured one instead of being a bare float. That pairing was
    dropped at four separate boundaries during the backend build; handing the
    client `dict[str, Any]` here would have invited a fifth.

    `direction` is narrowed to the `Direction` enum even though `Readout` types
    it as `str`. Every value `derive_readouts` can produce is a `Direction`
    member or `None`, so the narrower contract is honest and gives a client a
    closed union instead of an open string.
    """

    name: str
    type: ReadoutType
    unit: str | None
    direction: Direction | None
    description: str

    @classmethod
    def from_domain(cls, readout: Readout) -> ReadoutResponse:
        return cls(
            name=readout.name,
            type=readout.type,
            unit=readout.unit,
            direction=Direction(readout.direction) if readout.direction else None,
            description=readout.description,
        )


class ProtocolResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    dataset_id: uuid.UUID
    engine_id: str
    artifact_uri: str
    readouts: list[ReadoutResponse]
    # Stays an open map: the keys are whatever conditions the chosen Engine
    # declares, which is exactly the thing the self-describing catalogue makes
    # discoverable at runtime rather than fixing in a schema.
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
            readouts=[ReadoutResponse.from_domain(readout) for readout in protocol.readouts],
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
    computed" (`None`/a reason) for `random_split_metrics`;
    `random_split_metrics_undefined` is `metrics_undefined`'s own counterpart
    for `random_split_metrics` -- a *different* partition that can disagree
    with the Dataset's own split about which metrics are undefined and why,
    so a renderer must not reuse `metrics_undefined` to explain a
    `random_split_metrics` null.

    `unit`/`direction` are the target's own -- what `metrics`, `worst_rows`'
    `actual`/`predicted`/`residual`, and `noise_floor` are all measured in, and
    which way is better. `split_strategy` (`"random"` or `"scaffold"`) is
    which split produced `metrics`/`baseline_metrics`/`worst_rows` -- read this
    instead of inferring it from `random_split_metrics`/
    `random_split_unavailable` both being `None`.
    """

    primary_metric: str
    prediction_kind: str
    metrics: dict[str, float | None]
    metrics_undefined: dict[str, str] | None
    engine_id: str
    conditions: dict[str, Any]
    baseline_engine_id: str
    baseline_conditions: dict[str, Any]
    baseline_metrics: dict[str, float | None]
    baseline_is_self: bool
    random_split_metrics: dict[str, float | None] | None
    random_split_unavailable: str | None
    random_split_metrics_undefined: dict[str, str] | None
    noise_floor: float | None
    worst_rows: list[WorstRowResponse]
    applicability_coverage: float | None
    unit: str | None
    direction: str | None
    split_strategy: str

    @classmethod
    def from_domain(cls, card: Scorecard) -> ScorecardResponse:
        return cls(
            primary_metric=card.primary_metric,
            prediction_kind=card.prediction_kind,
            metrics=card.metrics,
            metrics_undefined=card.metrics_undefined,
            engine_id=card.engine_id,
            conditions=card.conditions,
            baseline_engine_id=card.baseline_engine_id,
            baseline_conditions=card.baseline_conditions,
            baseline_metrics=card.baseline_metrics,
            baseline_is_self=card.baseline_is_self,
            random_split_metrics=card.random_split_metrics,
            random_split_unavailable=card.random_split_unavailable,
            random_split_metrics_undefined=card.random_split_metrics_undefined,
            noise_floor=card.noise_floor,
            worst_rows=[WorstRowResponse.from_domain(row) for row in card.worst_rows],
            applicability_coverage=card.applicability_coverage,
            unit=card.target_unit,
            direction=card.target_direction,
            split_strategy=card.split_strategy,
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
        baseline_engine_id=body.baseline_engine_id,
        baseline_conditions=body.baseline_conditions,
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

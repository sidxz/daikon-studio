"""Runner endpoints: register, list, revoke self-hosted runners.

Instance-level, not workspace-scoped -- same reason `domain/runners/runner.py`
gives: a runner serves lanes, and lanes cross workspaces. Every request body
here is `extra="forbid"`, the same convention every other route module
follows. A plain list, no pagination: single-digit runners.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field

from daikonstudio.application.runners.manage import (
    CreatedRunner,
    CreateRunner,
    CreateRunnerCommand,
    ListRunners,
    RevokeRunner,
    RevokeRunnerCommand,
    RunnerStatusView,
)
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies._core import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/runners", tags=["runners"])

CreateRunnerDep = Annotated[CreateRunner, Depends(use_case(CreateRunner))]
ListRunnersDep = Annotated[ListRunners, Depends(use_case(ListRunners))]
RevokeRunnerDep = Annotated[RevokeRunner, Depends(use_case(RevokeRunner))]


class CreateRunnerBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Matches `RunnerModel.name`'s String(128); without the bound a longer name
    # was an asyncpg DataError, surfacing as a 500 instead of a 422.
    name: str = Field(max_length=128)
    lanes: list[str]


class CreatedRunnerResponse(BaseModel):
    id: uuid.UUID
    name: str
    lanes: list[str]
    token: str
    created_at: datetime

    @classmethod
    def from_domain(cls, created: CreatedRunner) -> CreatedRunnerResponse:
        return cls(
            id=created.runner.id,
            name=created.runner.name,
            lanes=list(created.runner.lanes),
            token=created.token,
            created_at=created.runner.created_at,
        )


class RunnerResponse(BaseModel):
    id: uuid.UUID
    name: str
    lanes: list[str]
    online: bool
    last_seen_at: datetime | None
    revoked: bool
    current_run_id: uuid.UUID | None
    created_at: datetime

    @classmethod
    def from_domain(cls, view: RunnerStatusView) -> RunnerResponse:
        runner = view.runner
        return cls(
            id=runner.id,
            name=runner.name,
            lanes=list(runner.lanes),
            online=view.online,
            last_seen_at=runner.last_seen_at,
            revoked=runner.is_revoked,
            current_run_id=view.current_run_id,
            created_at=runner.created_at,
        )


@router.post("", response_model=CreatedRunnerResponse, status_code=201)
async def create_runner(
    body: CreateRunnerBody, auth: AuthDep, service: CreateRunnerDep
) -> CreatedRunnerResponse:
    command = CreateRunnerCommand(name=body.name, lanes=tuple(body.lanes))
    created = result_to_response(await service(command, auth=auth))
    return CreatedRunnerResponse.from_domain(created)


@router.get("", response_model=list[RunnerResponse])
async def list_runners(auth: AuthDep, service: ListRunnersDep) -> list[RunnerResponse]:
    views = result_to_response(await service(auth=auth))
    return [RunnerResponse.from_domain(view) for view in views]


@router.post("/{runner_id}/revoke", status_code=204)
async def revoke_runner(runner_id: uuid.UUID, auth: AuthDep, service: RevokeRunnerDep) -> Response:
    result_to_response(await service(RevokeRunnerCommand(runner_id=runner_id), auth=auth))
    return Response(status_code=204)

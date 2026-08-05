"""The machine-facing runner protocol: `/api/v1/runner/*`.

No `AuthDep` anywhere in this module -- Sentinel does not run on this prefix
at all (see `app.py`), and every route instead depends on `RunnerDep` /
`ClaimedRunRead` / `ClaimedRunWrite` from `interface/dependencies/runner_auth.py`.
That is the entire security boundary this module sits behind, so every route
here resolves its collaborators through the container like every other route
-- never constructs a repository or the blob store directly.

The update route (`POST /runs/{id}`) deliberately does not go through
`Run`'s own state-machine methods (`start`/`report_progress`/`succeed`/
`fail`): a self-hosted runner is trusted to report whatever status it is
actually in, not to have that status re-derived by re-running the
aggregate's own transition rules against it. What *is* enforced here is the
one rule that matters for the security boundary -- a runner can never report
`cancelled` (that is a client-initiated stop, `POST /api/v1/runs/{id}/cancel`,
`interface/routes/runs.py`) -- and the optimistic-concurrency version check.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from daikonstudio.application.execution.claim_run import ClaimRun
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.execution.run import RunStatus
from daikonstudio.domain.shared.errors import (
    AuthorizationError,
    ConcurrencyConflictError,
    NotFoundError,
    ValidationError,
)
from daikonstudio.infrastructure.runner.wire import (
    BlobPutResponse,
    ClaimResponse,
    DatasetEnvelope,
    ProtocolEnvelope,
    RunEnvelope,
    RunUpdateEnvelope,
    RunUpdateResponse,
)
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies.runner_auth import (
    ClaimedRunRead,
    ClaimedRunWrite,
    RunnerDep,
)
from daikonstudio.interface.error_handlers import result_to_response
from daikonstudio.settings import Settings

router = APIRouter(prefix="/api/v1/runner", tags=["runner-protocol"])

ClaimRunDep = Annotated[ClaimRun, Depends(use_case(ClaimRun))]
RunRepositoryDep = Annotated[RunRepository, Depends(use_case(RunRepository))]
DatasetRepositoryDep = Annotated[DatasetRepository, Depends(use_case(DatasetRepository))]
ProtocolRepositoryDep = Annotated[ProtocolRepository, Depends(use_case(ProtocolRepository))]
BlobStoreDep = Annotated[BlobStore, Depends(use_case(BlobStore))]
SettingsDep = Annotated[Settings, Depends(use_case(Settings))]

# Runners never cancel -- see the module docstring.
_MUTABLE_STATUSES = {"running", "ready", "failed"}


@router.post("/claim")
async def claim(runner: RunnerDep, service: ClaimRunDep) -> Response:
    claimed = result_to_response(await service(runner=runner))
    if claimed is None:
        return Response(status_code=204)
    run, deadline_seconds = claimed
    envelope = ClaimResponse(run=RunEnvelope.from_domain(run), deadline_seconds=deadline_seconds)
    return JSONResponse(content=envelope.model_dump(mode="json"))


@router.get("/runs/{run_id}", response_model=RunEnvelope)
async def get_run(run: ClaimedRunRead) -> RunEnvelope:
    return RunEnvelope.from_domain(run)


@router.post("/runs/{run_id}", response_model=RunUpdateResponse)
async def update_run(
    run: ClaimedRunWrite, body: RunUpdateEnvelope, runs: RunRepositoryDep
) -> RunUpdateResponse:
    if body.status not in _MUTABLE_STATUSES:
        raise ValidationError(
            f"status must be one of {sorted(_MUTABLE_STATUSES)}, got {body.status!r}"
        )
    if body.expected_version != run.version:
        raise ConcurrencyConflictError("Run", str(run.id))

    # Only the mutable fields -- never params/kind/workspace_id, which stay
    # exactly what the use case that created the row put there.
    run.status = RunStatus(body.status)
    run.progress = body.progress
    run.phase = body.phase
    run.result_uri = body.result_uri
    run.error_message = body.error_message
    run.protocol_id = body.protocol_id
    run.version = body.expected_version
    await runs.update(run)  # raises ConcurrencyConflictError (-> 409) on a lost race
    return RunUpdateResponse(version=run.version)


@router.get("/runs/{run_id}/dataset", response_model=DatasetEnvelope)
async def get_dataset(run: ClaimedRunRead, datasets: DatasetRepositoryDep) -> DatasetEnvelope:
    raw_dataset_id = run.params.get("dataset_id")
    if raw_dataset_id is None:
        raise NotFoundError("Dataset")
    dataset = await datasets.get(run.workspace_id, uuid.UUID(str(raw_dataset_id)))
    if dataset is None:
        raise NotFoundError("Dataset", str(raw_dataset_id))
    return DatasetEnvelope.from_domain(dataset)


@router.get("/runs/{run_id}/protocol", response_model=ProtocolEnvelope)
async def get_protocol(run: ClaimedRunRead, protocols: ProtocolRepositoryDep) -> ProtocolEnvelope:
    if run.protocol_id is None:
        raise NotFoundError("Protocol")
    protocol = await protocols.get(run.workspace_id, run.protocol_id)
    if protocol is None:
        raise NotFoundError("Protocol", str(run.protocol_id))
    return ProtocolEnvelope.from_domain(protocol)


@router.post("/runs/{run_id}/protocol", status_code=201)
async def create_protocol(
    run: ClaimedRunWrite, body: ProtocolEnvelope, protocols: ProtocolRepositoryDep
) -> Response:
    if body.workspace_id != run.workspace_id:
        raise ValidationError("protocol workspace_id must match the run's workspace_id")
    await protocols.add(body.to_domain())
    return Response(status_code=201)


def _guard_workspace_prefix(workspace_id: uuid.UUID, key: str) -> None:
    if not key.startswith(f"{workspace_id}/"):
        raise AuthorizationError(f"Blob key '{key}' is outside this run's workspace")


@router.get("/runs/{run_id}/blobs/{key:path}")
async def get_blob(run: ClaimedRunRead, key: str, store: BlobStoreDep) -> Response:
    _guard_workspace_prefix(run.workspace_id, key)
    try:
        data = store.get_bytes(key)
    except (FileNotFoundError, OSError) as error:
        raise NotFoundError("Blob", key) from error
    return Response(content=data, media_type="application/octet-stream")


@router.put("/runs/{run_id}/blobs/{key:path}", response_model=BlobPutResponse)
async def put_blob(
    run: ClaimedRunWrite,
    key: str,
    request: Request,
    store: BlobStoreDep,
    settings: SettingsDep,
) -> BlobPutResponse:
    _guard_workspace_prefix(run.workspace_id, key)

    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > settings.runner_upload_max_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds runner_upload_max_bytes")

    # The `BlobStore` port is bytes-in, bytes-out (`application/ports/blob_store.py`),
    # so buffering the whole body in memory is inherent to this endpoint -- the
    # Content-Length check above is a fast-fail for an honest client, this is the
    # real check for one that lies about (or omits) that header.
    body = await request.body()
    if len(body) > settings.runner_upload_max_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds runner_upload_max_bytes")

    return BlobPutResponse(uri=store.put_bytes(key, body))

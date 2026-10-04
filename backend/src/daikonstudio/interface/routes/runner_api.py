"""The machine-facing runner protocol: `/api/v1/runner/*`.

No `AuthDep` anywhere in this module -- Duar does not run on this prefix
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

import posixpath
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from daikonstudio.application.engines.checkpoints import checkpoint_root
from daikonstudio.application.execution.claim_run import ClaimRun
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.catalog.protocol import ProtocolStatus
from daikonstudio.domain.execution.run import RunKind, RunStatus
from daikonstudio.domain.shared.errors import (
    AuthorizationError,
    ConcurrencyConflictError,
    ConflictError,
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


@router.post("/claim", response_model=ClaimResponse)
async def claim(runner: RunnerDep, service: ClaimRunDep) -> Response:
    claimed = result_to_response(await service(runner=runner))
    if claimed is None:
        return Response(status_code=204)
    run, deadline_seconds, lease_seconds = claimed
    envelope = ClaimResponse(
        run=RunEnvelope.from_domain(run),
        deadline_seconds=deadline_seconds,
        lease_seconds=lease_seconds,
    )
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
    # exactly what the use case that created the row put there. And only the
    # fields the caller actually sent: `RunUpdateEnvelope`'s optional fields
    # default to None, so a bare status-only heartbeat must not null out
    # whatever a previous update reported (security review, Important 1).
    fields = body.model_fields_set
    run.status = RunStatus(body.status)
    if "progress" in fields and body.progress is not None:
        # Unlike phase/result_uri/error_message, `Run.progress` is a plain
        # `float` (not nullable) -- an explicit `"progress": null` is nonsense
        # for a fraction-complete value, so treat it the same as omitted
        # rather than accept a value the domain type cannot hold.
        run.progress = body.progress
    if "phase" in fields:
        run.phase = body.phase
    if "result_uri" in fields:
        run.result_uri = body.result_uri
    if "error_message" in fields:
        run.error_message = body.error_message
    if "metrics" in fields:
        # `Run.metrics` is the plain dict the repository/wire round-trip
        # already expects (`_to_model`, `RunEnvelope.metrics`) -- `body.metrics`
        # is the typed `RunMetricsWire` pydantic validated it into, not that
        # dict itself.
        run.metrics = body.metrics.model_dump() if body.metrics is not None else None
    if "protocol_id" in fields:
        if body.protocol_id is not None:
            # write-once, same as every other caller of link_protocol: raises
            # ConflictError (-> 409) if this run is already linked elsewhere.
            run.link_protocol(body.protocol_id)
        elif run.protocol_id is not None:
            raise ConflictError(
                f"Run '{run.id}' is already linked to protocol '{run.protocol_id}'; "
                "cannot unlink it"
            )
    run.version = body.expected_version
    run.updated_at = datetime.now(UTC)
    await runs.update(run)  # raises ConcurrencyConflictError (-> 409) on a lost race
    return RunUpdateResponse(version=run.version)


@router.get("/runs/{run_id}/dataset", response_model=DatasetEnvelope)
async def get_dataset(run: ClaimedRunRead, datasets: DatasetRepositoryDep) -> DatasetEnvelope:
    raw_dataset_id = run.params.get("dataset_id")
    if raw_dataset_id is None:
        raise NotFoundError("Dataset")
    try:
        dataset_id = uuid.UUID(str(raw_dataset_id))
    except ValueError as error:
        raise NotFoundError("Dataset", str(raw_dataset_id)) from error
    dataset = await datasets.get(run.workspace_id, dataset_id)
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
    protocol = body.to_domain()
    # Publishing is an editor-role, human-facing action (`PublishProtocol`,
    # `interface/routes/protocols.py`) that also goes through
    # `InSilicoProtocol.publish()`'s own invariants. A runner reporting the
    # protocol it just trained must never be able to hand back an envelope
    # that skips straight to published -- force draft server-side regardless
    # of what the body carried (security review, Important 2).
    protocol.status = ProtocolStatus.DRAFT
    protocol.published_at = None
    # The creator, for the delete permission, comes from the run, never from the
    # envelope: a runner token must not be able to name who may delete what.
    protocol.created_by = run.requested_by
    # A duplicate id raises ConflictError (-> 409) from the repository itself
    # -- same pattern as `SqlAlchemyRunnerRepository.add` -- rather than a raw
    # IntegrityError surfacing here as a 500 with SQL in the traceback.
    await protocols.add(protocol)
    return Response(status_code=201)


def _normalize_blob_key(key: str, blob_base_url: str) -> str:
    """A key a runner sends is sometimes the FULL STORE URI `FsspecBlobStore.
    put_bytes` returned from an earlier write (e.g. `InSilicoProtocol.
    artifact_uri`, which `RunPrediction` deliberately reads back BY URI --
    see `predict_with_protocol.py`'s own comment -- rather than re-deriving a
    key from ids that may not match a versioned protocol's own), not a bare
    workspace-relative key. Strip the configured base when it's present so
    `_guard_workspace_prefix` below always sees a plain `{workspace_id}/...`
    key, whichever form arrived -- and so the SAME normalized string is what
    gets checked and what gets used for the store call, which is what keeps
    this from becoming a check-one-thing-use-another gap (final review,
    Critical 1).
    """
    prefix = f"{blob_base_url.rstrip('/')}/"
    return key[len(prefix) :] if key.startswith(prefix) else key


def _guard_workspace_prefix(workspace_id: uuid.UUID, key: str) -> None:
    """`key.startswith(prefix)` alone is not confinement -- `FsspecBlobStore._path`
    string-concatenates the key onto the base path and the underlying
    filesystem happily resolves `..` in it, so `{ws}/../../pwned.txt` both
    starts with `{ws}/` AND escapes the workspace (and the blob root
    entirely). Security review, Critical 1 -- proven end-to-end with
    percent-encoded `..` segments in the URL, which arrive here already
    decoded (ASGI's `scope["path"]` is decoded before routing).

    Reject on three independent grounds: a rooted key, a literal `..`/`.`
    path segment, and (belt-and-braces) a `posixpath.normpath` of the key
    landing outside the prefix -- any one of these tripping is enough to
    refuse, so no single encoding trick can satisfy all three at once.

    Callers pass this the output of `_normalize_blob_key`, never the raw
    path param -- a full store URI legitimately does not start with
    `{workspace_id}/` and must not be rejected on that basis alone.
    """
    prefix = f"{workspace_id}/"
    segments = key.split("/")
    confined = (
        not key.startswith("/")
        and ".." not in segments
        and "." not in segments
        and key.startswith(prefix)
        and posixpath.normpath(key).startswith(prefix)
    )
    if not confined:
        raise AuthorizationError(f"Blob key '{key}' is outside this run's workspace")


@router.get("/runs/{run_id}/blobs/{key:path}")
async def get_blob(
    run: ClaimedRunRead, key: str, store: BlobStoreDep, settings: SettingsDep
) -> Response:
    key = _normalize_blob_key(key, settings.blob_base_url)
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
    key = _normalize_blob_key(key, settings.blob_base_url)
    _guard_workspace_prefix(run.workspace_id, key)

    # `.isdigit()` rather than a bare `int(...)`: a garbage or negative
    # Content-Length (a client can send anything) must not raise and 500 --
    # treat anything that isn't a plain non-negative integer as absent and
    # fall through to the streaming check below, which is authoritative
    # either way.
    content_length = request.headers.get("content-length")
    if (
        content_length is not None
        and content_length.isdigit()
        and int(content_length) > settings.runner_upload_max_bytes
    ):
        raise HTTPException(status_code=413, detail="Upload exceeds runner_upload_max_bytes")

    # The `BlobStore` port is bytes-in, bytes-out (`application/ports/blob_store.py`),
    # so buffering the whole body in memory is inherent to this endpoint -- but
    # buffering it *unbounded* is not: a chunked request carries no Content-Length
    # at all, so the fast-fail above never fires for one, and `await request.body()`
    # would read the whole thing before any check ran. Read the stream instead and
    # abort the moment the running total crosses the cap.
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > settings.runner_upload_max_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds runner_upload_max_bytes")
        chunks.append(chunk)
    body = b"".join(chunks)

    return BlobPutResponse(uri=store.put_bytes(key, body))


@router.delete("/runs/{run_id}/checkpoints", status_code=204)
async def delete_checkpoints(run: ClaimedRunWrite, store: BlobStoreDep) -> Response:
    """Delete this run's saved training progress, and nothing else: the folder is
    derived from the run itself, never from the request."""
    dataset_id = run.params.get("dataset_id")
    if run.kind is RunKind.TRAINING and dataset_id:
        store.delete_prefix(checkpoint_root(run.workspace_id, uuid.UUID(str(dataset_id)), run.id))
    return Response(status_code=204)

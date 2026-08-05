"""HTTP implementations of the four ports `jobs.run_job` reads off `ctx`
(`runs`, `datasets`, `protocols`, `store`) -- the self-hosted-runner twin of
`build_sqlalchemy_ctx`. A runner claims a run_id from a studio it talks to
only over HTTP, so `run_job` (which never imports SqlAlchemy, only the port
Protocols) executes unmodified against these instead: `build_http_ctx` is a
drop-in replacement for `build_sqlalchemy_ctx`.

Every class here implements only the subset of its port that `run_job`'s call
graph actually reaches (see each port's own docstring for which methods that
is) -- everything else raises `NotImplementedError("api-side only")`, since a
runner process has no route for it and calling it would be a bug in the
handler, not a missing feature here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import httpx

from daikonstudio.domain.catalog.protocol import InSilicoProtocol
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.domain.shared.errors import ConcurrencyConflictError
from daikonstudio.infrastructure.runner.wire import (
    BlobPutResponse,
    DatasetEnvelope,
    ProtocolEnvelope,
    RunEnvelope,
    RunUpdateEnvelope,
    RunUpdateResponse,
)

_NOT_IMPLEMENTED = "api-side only"


class RunnerApiClient:
    """One claimed run's view of the studio. Bound to a run_id because every
    protocol URL is run-scoped -- constructing it per job is the design.

    Two transports for one reason: `BlobStore` (`application/ports/blob_store.py`)
    is a sync Protocol -- an engine's fit loop calls `get_bytes`/`put_bytes`
    directly, with no `await` -- so blob traffic goes over a sync `httpx.Client`
    while everything else (the run/dataset/protocol repositories, all called
    from `async def` handlers) goes over an `httpx.AsyncClient`. Both point at
    the same base URL and carry the same bearer token.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        run_id: uuid.UUID,
        *,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.run_id = run_id
        runner_base_url = f"{base_url}/api/v1/runner"
        headers = {"Authorization": f"Bearer {token}"}
        self._api = httpx.AsyncClient(
            base_url=runner_base_url, headers=headers, timeout=60.0, transport=async_transport
        )
        self._blobs = httpx.Client(
            base_url=runner_base_url, headers=headers, timeout=60.0, transport=sync_transport
        )

    async def aclose(self) -> None:
        await self._api.aclose()
        self._blobs.close()


class HttpRunRepository:
    """Implements the two `RunRepository` methods `run_job` calls: `get_by_id`
    (`_load`) and `update` (`_save`). Everything else on the port is a
    human/application-facing method (`add`, `get`, `find_by_cache_key`,
    `list`) with no counterpart on the runner protocol."""

    def __init__(self, client: RunnerApiClient) -> None:
        self._client = client

    async def get_by_id(self, run_id: uuid.UUID) -> Run | None:
        response = await self._client._api.get(f"/runs/{run_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return RunEnvelope.model_validate(response.json()).to_domain()

    async def update(self, run: Run) -> None:
        """POST the run's mutable fields, plus `expected_version` for the
        optimistic-concurrency check the server enforces the same way
        `SqlAlchemyRunRepository.update` does.

        `protocol_id` is included only when `run.protocol_id` is actually set:
        `Run.link_protocol` is write-once, so a Run that hasn't linked one yet
        always has it `None` here -- sending that `None` explicitly would tell
        the server to *unlink* an already-linked run (409), instead of the
        intended "I have nothing to say about protocol_id" omission. Every
        other field is unconditional because a `Run`'s in-memory state after
        `start`/`report_progress`/`succeed`/`fail` already carries whatever
        the job means to persist, unlike a partial heartbeat PATCH.

        Constructing the envelope with exactly these kwargs (rather than
        every field) is what makes `model_dump(exclude_unset=True)` below
        actually omit `protocol_id` on the wire when it wasn't passed --
        `model_fields_set` only tracks fields the constructor call named.
        """
        fields: dict[str, Any] = {
            "status": run.status.value,
            "expected_version": run.version,
            "progress": run.progress,
            "phase": run.phase,
            "result_uri": run.result_uri,
            "error_message": run.error_message,
        }
        if run.protocol_id is not None:
            fields["protocol_id"] = run.protocol_id
        # Conditional for the same reason `protocol_id` is: this is set once,
        # at the end of a training run, and every progress checkpoint before
        # that has it None. Sending that None explicitly would tell the server
        # to clear it on the very next heartbeat.
        if run.metrics is not None:
            fields["metrics"] = run.metrics
        envelope = RunUpdateEnvelope(**fields)

        response = await self._client._api.post(
            f"/runs/{run.id}", json=envelope.model_dump(mode="json", exclude_unset=True)
        )
        if response.status_code == 409:
            raise ConcurrencyConflictError("Run", str(run.id))
        response.raise_for_status()
        run.version = RunUpdateResponse.model_validate(response.json()).version

    async def add(self, run: Run) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def get(self, workspace_id: uuid.UUID, run_id: uuid.UUID) -> Run | None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def find_by_cache_key(self, workspace_id: uuid.UUID, cache_key: str) -> Run | None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        kind: RunKind | None = None,
        protocol_id: uuid.UUID | None = None,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[Run]:
        raise NotImplementedError(_NOT_IMPLEMENTED)


class HttpDatasetRepository:
    """Implements only `get`, over the run-scoped `GET /runs/{run_id}/dataset`
    -- the server derives the dataset from the claimed run's own `params`, so
    the `dataset_id` argument is accepted (to satisfy the `DatasetRepository`
    Protocol) but not sent."""

    def __init__(self, client: RunnerApiClient) -> None:
        self._client = client

    async def get(self, workspace_id: uuid.UUID, dataset_id: uuid.UUID) -> Dataset | None:
        response = await self._client._api.get(f"/runs/{self._client.run_id}/dataset")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return DatasetEnvelope.model_validate(response.json()).to_domain()

    async def add(self, dataset: Dataset) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def find_by_content_hash(
        self, workspace_id: uuid.UUID, content_hash: str
    ) -> Dataset | None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[Dataset]:
        raise NotImplementedError(_NOT_IMPLEMENTED)


class HttpProtocolRepository:
    """Implements `get` and `add`, over the run-scoped
    `GET`/`POST /runs/{run_id}/protocol` -- the two calls `RunTraining` makes:
    reading the protocol a prediction run targets, and reporting the one a
    training run just produced. `workspace_id`/`protocol_id` on `get` are
    accepted for the same Protocol-conformance reason as `HttpDatasetRepository.get`
    but not sent -- the server resolves both from the claimed run."""

    def __init__(self, client: RunnerApiClient) -> None:
        self._client = client

    async def get(
        self, workspace_id: uuid.UUID, protocol_id: uuid.UUID
    ) -> InSilicoProtocol | None:
        response = await self._client._api.get(f"/runs/{self._client.run_id}/protocol")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return ProtocolEnvelope.model_validate(response.json()).to_domain()

    async def add(self, protocol: InSilicoProtocol) -> None:
        envelope = ProtocolEnvelope.from_domain(protocol)
        response = await self._client._api.post(
            f"/runs/{self._client.run_id}/protocol", json=envelope.model_dump(mode="json")
        )
        response.raise_for_status()

    async def update(self, protocol: InSilicoProtocol) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def list(
        self,
        workspace_id: uuid.UUID,
        *,
        cursor: tuple[datetime, uuid.UUID] | None = None,
        limit: int = 50,
    ) -> list[InSilicoProtocol]:
        raise NotImplementedError(_NOT_IMPLEMENTED)


class HttpBlobStore:
    """Sync, like the `BlobStore` port itself -- see `RunnerApiClient`'s
    docstring for why this rides its own `httpx.Client` rather than the
    async one. Only `get_bytes`/`put_bytes` are implemented: those are the
    only two calls anywhere in the job path (`exists`/`delete` have no
    caller between `run_job` and the engines it dispatches to)."""

    def __init__(self, client: RunnerApiClient) -> None:
        self._client = client

    def get_bytes(self, key: str) -> bytes:
        response = self._client._blobs.get(f"/runs/{self._client.run_id}/blobs/{key}")
        response.raise_for_status()
        return response.content

    def put_bytes(self, key: str, data: bytes) -> str:
        response = self._client._blobs.put(
            f"/runs/{self._client.run_id}/blobs/{key}", content=data
        )
        response.raise_for_status()
        return BlobPutResponse.model_validate(response.json()).uri

    def exists(self, key: str) -> bool:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def delete(self, key: str) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)


def build_http_ctx(
    base_url: str,
    token: str,
    run_id: uuid.UUID,
    *,
    deadline_seconds: int | None,
    **transports: Any,
) -> dict[str, Any]:
    """The runner-side twin of `infrastructure.jobs.build_sqlalchemy_ctx`:
    assembles the same `ctx` shape `run_job` expects, backed by HTTP instead
    of a `sessionmaker`. `**transports` forwards `async_transport`/
    `sync_transport` test seams straight through to `RunnerApiClient`.

    `"_client"` is carried on the ctx (not read by `run_job` itself) purely so
    the caller that built this ctx can `aclose()` it when the job is done --
    the four ports above only ever see it, never own its lifecycle.
    """
    client = RunnerApiClient(base_url, token, run_id, **transports)
    return {
        "runs": HttpRunRepository(client),
        "datasets": HttpDatasetRepository(client),
        "protocols": HttpProtocolRepository(client),
        "store": HttpBlobStore(client),
        "job_deadline_seconds": deadline_seconds,
        "_client": client,
    }

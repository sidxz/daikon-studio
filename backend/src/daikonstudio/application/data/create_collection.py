"""Turning a triage selection into a durable Collection.

`row_ids` are the same plain integer offsets `GetPredictionResults` pages
over -- positions into the run's results Parquet in file-write order, not a
separate id space. A client that paged through `/runs/{id}/results` already
has these from each row's position (offset + index within the page).

Three ways a selection can be wrong, and none of them are silently repaired:
empty (no rows chosen -- a saved Collection with zero members is not a
triage decision, it's a mistake), duplicated (a client bug or a double
click, and de-duplicating would leave a `member_count` the caller can no
longer predict from the length of what they sent), and out-of-range
(stale paging state, or an id from a different run entirely). Any of these
silently accepted would let a scientist believe they saved compounds they
did not -- so all three are `ValidationError` (422), and out-of-range is
checked against the *run's own* row count, never clamped or dropped.

`CreateCollection` also copies the selected rows into the Collection's own
snapshot blob rather than recording only which rows of the run's blob were
chosen -- see `domain/data/collection.py`'s docstring for why.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated, require_editor
from daikonstudio.application.execution.predict_with_protocol import predictions_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.collection_repository import CollectionRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.run_repository import RunRepository
from daikonstudio.domain.data.collection import Collection
from daikonstudio.domain.execution.run import RunKind, RunStatus
from daikonstudio.domain.shared.errors import (
    ConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)
from daikonstudio.domain.shared.provenance import (
    GenerationMethod,
    Provenance,
    ProvenanceSourceType,
)


def collection_snapshot_key(workspace_id: uuid.UUID, collection_id: uuid.UUID) -> str:
    """Where one Collection's own copy of its selected rows lives -- a
    dedicated key per Collection, never the Run's own `predictions_key`, so
    the Collection's lifetime is never coupled to the Run's blob (see
    `domain/data/collection.py`)."""
    return f"{workspace_id}/collections/{collection_id}/snapshot.parquet"


@dataclass(frozen=True, kw_only=True)
class CreateCollectionCommand:
    name: str
    run_id: uuid.UUID
    row_ids: tuple[int, ...]


class CreateCollection:
    def __init__(
        self,
        collections: CollectionRepository,
        runs: RunRepository,
        protocols: ProtocolRepository,
        store: BlobStore,
    ) -> None:
        self._collections = collections
        self._runs = runs
        self._protocols = protocols
        self._store = store

    async def __call__(
        self, command: CreateCollectionCommand, auth: AuthContext | None = None
    ) -> Result[Collection, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        run = await self._runs.get(auth.workspace_id, command.run_id)
        if run is None or run.kind is not RunKind.PREDICTION:
            # A training Run has no per-row results to triage -- same "this
            # resource doesn't have what you're asking for" 404 `GetPredictionResults`
            # gives a training run's `/results`, not a 409 that implies waiting helps.
            return Failure(NotFoundError("Run", str(command.run_id)))
        if run.status is not RunStatus.READY:
            return Failure(
                ConflictError(
                    f"Run '{run.id}' is not ready (status: '{run.status.value}'); "
                    "only a ready run's results can be saved into a Collection",
                    detail=run.error_message if run.status is RunStatus.FAILED else None,
                )
            )

        if not command.row_ids:
            return Failure(ValidationError("row_ids must not be empty"))
        if any(row_id < 0 for row_id in command.row_ids):
            return Failure(ValidationError("row_ids must not be negative"))
        if len(set(command.row_ids)) != len(command.row_ids):
            return Failure(ValidationError("row_ids must not contain duplicates"))

        protocol_id = uuid.UUID(run.params["protocol_id"])
        protocol = await self._protocols.get(auth.workspace_id, protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(protocol_id)))

        try:
            raw = self._store.get_bytes(predictions_key(run.workspace_id, run.id))
        except FileNotFoundError:
            return Failure(NotFoundError("Prediction results", str(run.id)))
        frame = pl.read_parquet(io.BytesIO(raw))

        out_of_range = sorted(row_id for row_id in command.row_ids if row_id >= frame.height)
        if out_of_range:
            return Failure(
                ValidationError(
                    f"row_ids out of range for this run's results: {out_of_range}",
                    detail=f"This run has {frame.height} result row(s): 0-{frame.height - 1}.",
                )
            )

        # Fancy row selection in the caller's own order (a ranked "top N"),
        # not re-sorted -- `row_ids` are already validated as non-negative,
        # in-range and duplicate-free above, so this is a well-defined gather.
        selected = frame[list(command.row_ids)]

        collection_id = uuid.uuid4()
        buffer = io.BytesIO()
        selected.write_parquet(buffer)
        snapshot_uri = self._store.put_bytes(
            collection_snapshot_key(auth.workspace_id, collection_id), buffer.getvalue()
        )

        collection = Collection(
            id=collection_id,
            workspace_id=auth.workspace_id,
            name=command.name,
            derived_from_run_id=run.id,
            member_count=selected.height,
            snapshot_uri=snapshot_uri,
            provenance=Provenance(
                source_type=ProvenanceSourceType.INTERNAL,
                generation_method=GenerationMethod.AI_PREDICTED,
                note=f"Predicted by protocol {protocol.id} (v{protocol.protocol_version})",
            ),
        )
        await self._collections.add(collection)
        return Success(collection)


@dataclass(frozen=True, kw_only=True)
class GetCollectionQuery:
    collection_id: uuid.UUID


class GetCollection:
    def __init__(self, collections: CollectionRepository) -> None:
        self._collections = collections

    async def __call__(
        self, query: GetCollectionQuery, auth: AuthContext | None = None
    ) -> Result[Collection, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        collection = await self._collections.get(auth.workspace_id, query.collection_id)
        if collection is None:
            return Failure(NotFoundError("Collection", str(query.collection_id)))
        return Success(collection)

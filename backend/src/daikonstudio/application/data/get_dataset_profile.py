"""Read a Dataset's profile, computing it the first time it is asked for.

Computed lazily and cached, rather than written during `CreateDataset`, for
three reasons that all point the same way:

- A Dataset is immutable and content-addressed, so a cached profile is a pure
  function of data that cannot change. There is nothing to invalidate, which is
  the property that makes a write-once cache safe rather than a staleness bug
  waiting to happen.
- The pass is several seconds of RDKit on a large file (descriptors, scaffolds, a
  Tanimoto matrix and a pair scan). `CreateDataset` already does its chemistry
  inside the HTTP request; adding this to it would make every upload pay for a
  page most uploads never open.
- Every Dataset frozen before this existed gets a profile the first time someone
  looks at one, with no migration and no backfill.

The first computation for a large dataset takes minutes (four for 400k
compounds), so it runs in the background instead of inside the request: the
first request starts it and answers `ProfileComputing`, every later request --
a reload, a second tab, a second reader -- joins the same computation, and once
the result is saved every request reads it. One viewer waits once, and never
starts a second copy of the work by reloading.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.data.build_profile import build_profile
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.dataset import Dataset
from daikonstudio.domain.data.profile import (
    PROFILE_VERSION,
    DatasetProfile,
    profile_from_dict,
    profile_to_dict,
)
from daikonstudio.domain.shared.errors import ConflictError, DomainError, NotFoundError

logger = logging.getLogger(__name__)


def profile_key(workspace_id: uuid.UUID | str, dataset_id: uuid.UUID | str) -> str:
    return f"{workspace_id}/datasets/{dataset_id}/profile.json"


@dataclass(frozen=True, kw_only=True)
class GetDatasetProfileQuery:
    dataset_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class ProfileComputing:
    """The answer while a profile is being computed: when it started, and how big it is."""

    started_at: datetime
    compounds: int


_Key = tuple[uuid.UUID, uuid.UUID]

#: The computation running for each dataset, so a second request joins it instead of
#: starting another. Module-level because a use case is built per request.
#: ponytail: per process. The API runs one; with several workers each could compute
#: once -- move this to a marker in the blob store (or a DB row) then.
_RUNNING: dict[_Key, tuple[datetime, asyncio.Task[None]]] = {}
#: A failure is reported to the next request, once; the request after that retries.
#: The cause goes to the log, not the reader: raw exception text names internals.
_FAILED: set[_Key] = set()


class GetDatasetProfile:
    def __init__(
        self,
        repository: DatasetRepository,
        store: BlobStore,
        normalizer: StructureNormalizer,
    ) -> None:
        self._repository = repository
        self._store = store
        self._normalizer = normalizer

    async def __call__(
        self, query: GetDatasetProfileQuery, auth: AuthContext | None = None
    ) -> Result[DatasetProfile | ProfileComputing, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        dataset = await self._repository.get(auth.workspace_id, query.dataset_id)
        if dataset is None:
            # 404 rather than 403 for another tenant's id, matching `GetDataset`:
            # a distinguishable "exists but not yours" is itself a disclosure.
            return Failure(NotFoundError("Dataset", str(query.dataset_id)))

        key = profile_key(dataset.workspace_id, dataset.id)
        try:
            cached = self._store.get_bytes(key)
        except FileNotFoundError:
            cached = None
        if cached is not None:
            try:
                stored = json.loads(cached)
                # The version gate catches the dangerous case: a profile whose
                # *shape* still parses but whose numbers were computed by an
                # older, differently-behaved `build_profile`. Without it that
                # blob is served forever, and the only symptom is a wrong
                # number on a page that looks perfectly healthy.
                if stored.get("version") == PROFILE_VERSION:
                    return Success(profile_from_dict(stored))
            except (ValueError, KeyError, TypeError):
                # A profile written by an older *shape* of this code. Recomputing
                # is always correct here -- the snapshot it derives from cannot
                # have changed -- so a stale cache degrades to a slow request
                # rather than to a 500.
                pass

        running_key = (dataset.workspace_id, dataset.id)
        if running_key in _FAILED:
            _FAILED.discard(running_key)
            return Failure(
                ConflictError(
                    "The dataset profile could not be computed. Reload the page to try again."
                )
            )
        # No `await` from the cache check above to the registration below: a running
        # computation can only save its result and leave `_RUNNING` while this
        # coroutine is suspended, so it cannot slip between the two checks.
        running = _RUNNING.get(running_key)
        if running is None:
            try:
                raw = self._store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
            except FileNotFoundError:
                return Failure(NotFoundError("Stored dataset file", str(dataset.id)))
            started_at = datetime.now(UTC)
            task = asyncio.create_task(self._compute(running_key, dataset, raw))
            running = (started_at, task)
            _RUNNING[running_key] = running
        return Success(ProfileComputing(started_at=running[0], compounds=dataset.row_count))

    async def _compute(self, key: _Key, dataset: Dataset, raw: bytes) -> None:
        """Runs in the background, past the request that started it: a reader who
        leaves or reloads does not cancel it, and the saved result serves them."""
        try:
            # Off-thread for the same reason `GetScorecard` moves `build_scorecard`
            # off it: this is minutes of RDKit and BLAS on a large dataset, and on
            # the event loop it would stall every other request in the process.
            profile = await asyncio.to_thread(
                build_profile,
                frame=pl.read_parquet(io.BytesIO(raw)),
                structure_column=dataset.structure_column,
                target=dataset.target,
                normalizer=self._normalizer,
            )
            self._store.put_bytes(
                profile_key(dataset.workspace_id, dataset.id),
                json.dumps(profile_to_dict(profile)).encode(),
            )
        except Exception:
            logger.exception("Profiling dataset %s failed", dataset.id)
            _FAILED.add(key)
        finally:
            _RUNNING.pop(key, None)

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

The cost is that the first request for a large dataset is slow. That is the right
place for it: one viewer waits once, instead of every uploader waiting always.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import dataclass

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.data.build_profile import build_profile
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.data.profile import (
    PROFILE_VERSION,
    DatasetProfile,
    profile_from_dict,
    profile_to_dict,
)
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


def profile_key(workspace_id: uuid.UUID | str, dataset_id: uuid.UUID | str) -> str:
    return f"{workspace_id}/datasets/{dataset_id}/profile.json"


@dataclass(frozen=True, kw_only=True)
class GetDatasetProfileQuery:
    dataset_id: uuid.UUID


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
    ) -> Result[DatasetProfile, DomainError]:
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

        try:
            raw = self._store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
        except FileNotFoundError:
            return Failure(NotFoundError("Dataset snapshot", str(dataset.id)))

        # Off-thread for the same reason `GetScorecard` moves `build_scorecard`
        # off it: this is seconds of RDKit and BLAS, and running it on the event
        # loop would stall every other request in the process for the duration.
        profile = await asyncio.to_thread(
            build_profile,
            frame=pl.read_parquet(io.BytesIO(raw)),
            structure_column=dataset.structure_column,
            target=dataset.target,
            normalizer=self._normalizer,
        )
        self._store.put_bytes(key, json.dumps(profile_to_dict(profile)).encode())
        return Success(profile)

"""Page through a frozen Dataset's own rows.

Until this existed there was no way to look at the data a Dataset contains --
the snapshot was written at freeze time and read back only by training. A
scientist could see how many compounds survived validation and not one of them.

Deliberately narrow: structure, target and partition, sorted by target or by
partition. Not a general query surface over the uploader's other columns, and
not sortable by structure -- ordering compounds by their SMILES string is
alphabetical nonsense dressed up as chemistry, the same reason
`apply_result_view` refuses it for prediction results.

Offset paging rather than the keyset cursors the list endpoints use: those page
over a table whose rows arrive over time, where an offset silently skips or
repeats rows as the underlying set shifts. A snapshot is immutable, so page 3 of
a given sort is the same three rows today and next year, and an offset is the
simpler thing that is exactly as correct here.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Literal

import polars as pl
from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.data.snapshot import snapshot_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.domain.shared.errors import DomainError, NotFoundError

MAX_LIMIT = 200


@dataclass(frozen=True, kw_only=True)
class Compound:
    structure: str
    target: float | None
    split: str


@dataclass(frozen=True, kw_only=True)
class CompoundPage:
    items: list[Compound]
    total: int


@dataclass(frozen=True, kw_only=True)
class GetDatasetCompoundsQuery:
    dataset_id: uuid.UUID
    offset: int = 0
    limit: int = 50
    sort: Literal["target", "split"] | None = None
    descending: bool = False
    split: str | None = None


class GetDatasetCompounds:
    def __init__(self, repository: DatasetRepository, store: BlobStore) -> None:
        self._repository = repository
        self._store = store

    async def __call__(
        self, query: GetDatasetCompoundsQuery, auth: AuthContext | None = None
    ) -> Result[CompoundPage, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        dataset = await self._repository.get(auth.workspace_id, query.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(query.dataset_id)))

        try:
            raw = self._store.get_bytes(snapshot_key(dataset.workspace_id, dataset.id))
        except FileNotFoundError:
            return Failure(NotFoundError("Stored dataset file", str(dataset.id)))

        # ponytail: reads the whole Parquet per page, exactly as
        # `GetPredictionResults` does. `pl.scan_parquet` with a pushed-down
        # slice is the upgrade path for both, together, once either one is
        # measurably slow.
        frame = pl.read_parquet(io.BytesIO(raw)).select(
            pl.col(dataset.structure_column).alias("structure"),
            pl.col(dataset.target.column).cast(pl.Float64, strict=False).alias("target"),
            pl.col("split"),
        )

        if query.split is not None:
            frame = frame.filter(pl.col("split") == query.split)

        total = frame.height

        if query.sort is not None:
            # `structure` as the tie-break makes the order total, so two
            # requests for the same page of a column with ties (every row of a
            # binary target is 0.0 or 1.0) cannot return different rows -- the
            # same unstable-sort hazard `apply_result_view` documents.
            frame = frame.sort(
                [query.sort, "structure"],
                descending=[query.descending, False],
                nulls_last=[True, False],
            )

        limit = max(1, min(query.limit, MAX_LIMIT))
        page = frame.slice(max(0, query.offset), limit)
        return Success(
            CompoundPage(
                items=[
                    Compound(
                        structure=str(row["structure"]),
                        target=None if row["target"] is None else float(row["target"]),
                        split=str(row["split"]),
                    )
                    for row in page.iter_rows(named=True)
                ],
                total=total,
            )
        )

"""Read a Protocol's Scorecard -- the honesty artifact `build_scorecard` (Task
15) renders from the raw measurements `RunTraining` (Task 14) wrote.

Only `build_scorecard` decides what the Scorecard says; this use case's whole
job is finding the right `ScorecardInputs` blob and handing it over unchanged.

The blob is always written before its Protocol row (`train_protocol.py`'s
"blobs first, Protocol row last" ordering), so on every path that exists
today a fetched Protocol's scorecard blob is already there. That invariant
could stop holding -- a future retraining/versioning flow that creates a
Protocol row ahead of its scorecard, or an operator deleting a blob out from
under a live row -- so the read is defended anyway: a missing blob is a
`NotFoundError` (404), not a 500 raised mid-request.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, replace

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.data.compound_ids import read_compound_ids
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.build_scorecard import build_scorecard
from daikonstudio.application.execution.train_protocol import ScorecardInputs, scorecard_inputs_key
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.execution.scorecard import Scorecard
from daikonstudio.domain.shared.errors import DomainError, NotFoundError


@dataclass(frozen=True, kw_only=True)
class GetScorecardQuery:
    protocol_id: uuid.UUID


class GetScorecard:
    def __init__(
        self,
        protocols: ProtocolRepository,
        store: BlobStore,
        normalizer: StructureNormalizer,
        datasets: DatasetRepository,
    ) -> None:
        self._protocols = protocols
        self._store = store
        self._normalizer = normalizer
        self._datasets = datasets

    async def __call__(
        self, query: GetScorecardQuery, auth: AuthContext | None = None
    ) -> Result[Scorecard, DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        protocol = await self._protocols.get(auth.workspace_id, query.protocol_id)
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(query.protocol_id)))

        try:
            raw = self._store.get_bytes(scorecard_inputs_key(protocol.workspace_id, protocol.id))
        except FileNotFoundError:
            return Failure(NotFoundError("Scorecard", str(protocol.id)))
        inputs = ScorecardInputs.from_json(raw)

        # ponytail: build_scorecard computes Murcko scaffolds and an O(test x
        # train) Tanimoto matrix -- measured at 0.86s for 16k train/2k test
        # structures, which blocks every other request on the process's event
        # loop for the duration. `RunTraining` already offloads this same class
        # of work (asyncio.to_thread around engine.train/predict); do the same
        # here rather than let the most-viewed screen in the product serialize
        # behind it. Upgrade path if this still isn't enough: cache the
        # rendered card next to the blob (the inputs are immutable once
        # written, so there is nothing to invalidate).
        scorecard = await asyncio.to_thread(
            build_scorecard,
            task=TaskType(inputs.task),
            metrics=inputs.metrics,
            validation_metrics=inputs.validation_metrics,
            engine_id=inputs.engine_id,
            conditions=inputs.conditions,
            baseline_engine_id=inputs.baseline_engine_id,
            baseline_conditions=inputs.baseline_conditions,
            baseline_metrics=inputs.baseline_metrics,
            baseline_is_self=inputs.baseline_is_self,
            actual=inputs.actual,
            predicted=inputs.predicted,
            structures=inputs.structures,
            train_structures=inputs.train_structures,
            normalizer=self._normalizer,
            target_unit=inputs.target_unit,
            target_direction=inputs.target_direction,
            split_strategy=inputs.split_strategy,
            random_split_metrics=inputs.random_split_metrics,
            random_split_unavailable=inputs.random_split_unavailable,
            random_split_metrics_undefined=inputs.random_split_metrics_undefined,
            metrics_undefined=inputs.metrics_undefined,
            duplicate_spread=inputs.duplicate_spread,
        )
        # IDs are looked up now rather than stored with the inputs, so naming or
        # changing the dataset's identifier column shows here without retraining.
        ids = None
        if scorecard.worst_rows:
            dataset = await self._datasets.get(protocol.workspace_id, protocol.dataset_id)
            if dataset is not None:
                try:
                    ids = await asyncio.to_thread(
                        read_compound_ids,
                        self._store,
                        dataset,
                        [row.structure for row in scorecard.worst_rows],
                    )
                except FileNotFoundError:
                    ids = None
        if ids:
            scorecard = replace(
                scorecard,
                worst_rows=[
                    replace(row, compound_id=ids.get(row.structure))
                    for row in scorecard.worst_rows
                ],
            )
        return Success(scorecard)

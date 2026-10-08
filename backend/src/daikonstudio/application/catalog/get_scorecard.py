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
from daikonstudio.application.catalog.derive_readouts import target_columns_of
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.data.compound_ids import read_compound_ids
from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.execution.build_scorecard import (
    HeldOutChemistry,
    build_scorecard,
    held_out_chemistry,
)
from daikonstudio.application.execution.train_protocol import (
    ScorecardInputs,
    scorecard_chemistry_key,
    scorecard_inputs_key,
)
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.protocol_access import ProtocolAccess
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
        access: ProtocolAccess,
    ) -> None:
        self._access = access
        self._protocols = protocols
        self._store = store
        self._normalizer = normalizer
        self._datasets = datasets

    async def __call__(
        self, query: GetScorecardQuery, auth: AuthContext | None = None
    ) -> Result[list[Scorecard], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None

        protocol = await visible_protocol(
            self._protocols, self._access, auth, auth.workspace_id, query.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(query.protocol_id)))

        try:
            raw = self._store.get_bytes(scorecard_inputs_key(protocol.workspace_id, protocol.id))
        except FileNotFoundError:
            return Failure(NotFoundError("Scorecard", str(protocol.id)))
        inputs = ScorecardInputs.from_json(
            raw, legacy_column=target_columns_of(protocol.readouts)[0]
        )

        # The expensive half -- Murcko scaffolds and the O(test x train) Tanimoto
        # search -- is read, not computed: training stores it (`scorecard_chemistry_
        # key`). What is left is measured at 0.74 s for four targets at 40k test
        # compounds, off the event loop. The cards themselves are rendered on every
        # view, so a change to how a Scorecard reads reaches old Protocols too.
        chemistry = await self._chemistry(protocol.workspace_id, protocol.id, inputs)
        scorecards = await asyncio.to_thread(_build_all, inputs, chemistry)
        # IDs are looked up now rather than stored with the inputs, so naming or
        # changing the dataset's identifier column shows here without retraining.
        wanted = {row.structure for card in scorecards for row in card.worst_rows} | {
            row.structure for card in scorecards for row in [*card.ranked_high, *card.ranked_low]
        }
        ids = None
        if wanted:
            dataset = await self._datasets.get(protocol.workspace_id, protocol.dataset_id)
            if dataset is not None:
                try:
                    ids = await asyncio.to_thread(read_compound_ids, self._store, dataset, wanted)
                except FileNotFoundError:
                    ids = None
        if ids:
            scorecards = [
                replace(
                    card,
                    worst_rows=[
                        replace(row, compound_id=ids.get(row.structure)) for row in card.worst_rows
                    ],
                    ranked_high=[
                        replace(row, compound_id=ids.get(row.structure))
                        for row in card.ranked_high
                    ],
                    ranked_low=[
                        replace(row, compound_id=ids.get(row.structure)) for row in card.ranked_low
                    ],
                )
                for card in scorecards
            ]
        return Success(scorecards)

    async def _chemistry(
        self, workspace_id: uuid.UUID, protocol_id: uuid.UUID, inputs: ScorecardInputs
    ) -> HeldOutChemistry:
        """The stored chemistry, or, for a Protocol trained before training stored it,
        the chemistry computed once and stored for every later view.

        One computation per Protocol however many views arrive while it runs: a
        reload, or a second person opening the page, waits for the first instead of
        starting another. Shielded, so a closed tab does not abandon work every
        later view needs. The search itself runs in a separate process when it is
        large (see the RDKit normalizer), so the API keeps answering meanwhile.
        """
        key = scorecard_chemistry_key(workspace_id, protocol_id)
        try:
            return HeldOutChemistry.from_json(self._store.get_bytes(key))
        except FileNotFoundError:
            pass
        pending = _COMPUTING.get(protocol_id)
        if pending is None:
            pending = asyncio.ensure_future(self._compute_and_store(key, inputs))
            _COMPUTING[protocol_id] = pending
            pending.add_done_callback(lambda _: _COMPUTING.pop(protocol_id, None))
        return await asyncio.shield(pending)

    async def _compute_and_store(self, key: str, inputs: ScorecardInputs) -> HeldOutChemistry:
        # No structure kind passed, so this takes the molecule default -- which is correct
        # by construction rather than by assumption. This path only runs for a Protocol
        # whose chemistry was never stored at training time, meaning one trained before
        # 0.5.1, and sequence datasets could not be ingested at all until long after that.
        # A sequence Protocol always has its chemistry stored and never reaches here.
        chemistry = await asyncio.to_thread(
            held_out_chemistry,
            inputs.structures,
            inputs.train_structures,
            self._normalizer,
        )
        await asyncio.to_thread(self._store.put_bytes, key, chemistry.to_json())
        return chemistry


# Per API process, which is the unit that would otherwise repeat the work.
_COMPUTING: dict[uuid.UUID, asyncio.Future[HeldOutChemistry]] = {}


def _build_all(inputs: ScorecardInputs, chemistry: HeldOutChemistry) -> list[Scorecard]:
    """One Scorecard per target, sharing one `HeldOutChemistry`."""
    return [
        build_scorecard(
            target=target.column,
            task=TaskType(target.task),
            metrics=target.metrics,
            validation_metrics=target.validation_metrics,
            engine_id=inputs.engine_id,
            conditions=inputs.conditions,
            baseline_engine_id=inputs.baseline_engine_id,
            baseline_conditions=inputs.baseline_conditions,
            baseline_metrics=target.baseline_metrics,
            baseline_is_self=inputs.baseline_is_self,
            joint_model=inputs.joint_model,
            actual=target.actual,
            predicted=target.predicted,
            structures=inputs.structures,
            chemistry=chemistry,
            target_unit=target.target_unit,
            target_direction=target.target_direction,
            split_strategy=inputs.split_strategy,
            deduplicated=inputs.deduplicated,
            random_split_metrics=target.random_split_metrics,
            random_split_unavailable=inputs.random_split_unavailable,
            random_split_metrics_undefined=target.random_split_metrics_undefined,
            metrics_undefined=target.metrics_undefined,
            duplicate_spread=target.duplicate_spread,
            cutoff=target.cutoff,
            baseline_cutoff=target.baseline_cutoff,
            cutoff_note=target.cutoff_note,
        )
        for target in inputs.targets
    ]

"""Count errors within a user-chosen tolerance, without rebuilding the diagnostics."""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import AuthContext, require_authenticated
from daikonstudio.application.catalog.derive_readouts import target_columns_of
from daikonstudio.application.catalog.visibility import visible_protocol
from daikonstudio.application.execution.build_scorecard import HeldOutChemistry, similarity_groups
from daikonstudio.application.execution.train_protocol import (
    ScorecardInputs,
    scorecard_chemistry_key,
    scorecard_inputs_key,
)
from daikonstudio.application.ports.blob_store import BlobStore
from daikonstudio.application.ports.protocol_access import ProtocolAccess
from daikonstudio.application.ports.protocol_repository import ProtocolRepository
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError


@dataclass(frozen=True, kw_only=True)
class GetScorecardToleranceQuery:
    protocol_id: uuid.UUID
    target: str
    tolerance: float


@dataclass(frozen=True, kw_only=True)
class ToleranceBin:
    lower: float
    upper: float
    count: int
    within_count: int


@dataclass(frozen=True, kw_only=True)
class ToleranceResult:
    target: str
    tolerance: float
    within_count: int
    test_count: int
    by_similarity: list[ToleranceBin] = field(default_factory=list)


class GetScorecardTolerance:
    def __init__(
        self, protocols: ProtocolRepository, store: BlobStore, access: ProtocolAccess
    ) -> None:
        self._protocols = protocols
        self._store = store
        self._access = access

    async def __call__(
        self, query: GetScorecardToleranceQuery, auth: AuthContext | None = None
    ) -> Result[ToleranceResult, DomainError]:
        require_authenticated(auth)
        assert auth is not None
        if not math.isfinite(query.tolerance) or query.tolerance < 0:
            return Failure(ValidationError("Tolerance must be a finite, non-negative number."))
        protocol = await visible_protocol(
            self._protocols, self._access, auth, auth.workspace_id, query.protocol_id
        )
        if protocol is None:
            return Failure(NotFoundError("Protocol", str(query.protocol_id)))
        try:
            raw = await asyncio.to_thread(
                self._store.get_bytes, scorecard_inputs_key(protocol.workspace_id, protocol.id)
            )
        except FileNotFoundError:
            return Failure(NotFoundError("Scorecard", str(protocol.id)))
        # The main scorecard caches chemistry for legacy runs before exposing the
        # tolerance control. If unavailable, still report the exact overall count.
        try:
            chemistry_raw = await asyncio.to_thread(
                self._store.get_bytes, scorecard_chemistry_key(protocol.workspace_id, protocol.id)
            )
            similarities = HeldOutChemistry.from_json(chemistry_raw).similarities
        except FileNotFoundError:
            similarities = None
        return await asyncio.to_thread(
            _count, raw, target_columns_of(protocol.readouts)[0], query, similarities
        )


def _count(
    raw: bytes,
    legacy_column: str,
    query: GetScorecardToleranceQuery,
    similarities: list[float] | None = None,
) -> Result[ToleranceResult, DomainError]:
    inputs = ScorecardInputs.from_json(raw, legacy_column=legacy_column)
    target = next((t for t in inputs.targets if t.column == query.target), None)
    if target is None:
        return Failure(NotFoundError("Target", query.target))
    if target.prediction_kind != "value":
        return Failure(ValidationError("An error tolerance applies only to continuous values."))
    within = [
        abs(p - a) <= query.tolerance for a, p in zip(target.actual, target.predicted, strict=True)
    ]
    return Success(
        ToleranceResult(
            target=target.column,
            tolerance=query.tolerance,
            within_count=sum(within),
            test_count=len(target.actual),
            by_similarity=(
                [
                    ToleranceBin(
                        lower=similarities[group[0]],
                        upper=similarities[group[-1]],
                        count=len(group),
                        within_count=sum(within[i] for i in group),
                    )
                    for group in similarity_groups(similarities)
                ]
                if similarities is not None
                else []
            ),
        )
    )

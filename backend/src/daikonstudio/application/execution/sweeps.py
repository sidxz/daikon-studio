"""Fan-out: N training configs submitted as one named group.

A sweep is not a workflow. Nothing here waits for anything, nothing consumes
another run's output, and no step must survive a crash to be correct -- which
is precisely why this is a `sweep_id` column and a loop over the existing
`TrainProtocol` rather than an orchestration engine. The trigger that would
change that is named and unchanged: the first time one step consumes another
step's output, dependent state exists and a workflow engine starts earning
its keep.

Each child run keeps its own mandatory baseline. Twenty configs against one
shared baseline would mean nineteen runs waiting on a twentieth's output, and
that dependency is exactly what this feature was scoped to avoid. The honest
cost is a repeated baseline fit; the real fix is the deferred fit-result cache,
not a sweep-level special case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from returns.result import Failure, Result, Success

from daikonstudio.application.auth import (
    AuthContext,
    require_authenticated,
    require_editor,
)
from daikonstudio.application.engines.registry import EngineRegistry, UnknownEngineError
from daikonstudio.application.execution.train_protocol import (
    TrainProtocol,
    TrainProtocolCommand,
)
from daikonstudio.application.pagination import clamp_limit
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.run_repository import RunRepository, SweepSummary
from daikonstudio.domain.execution.run import Run
from daikonstudio.domain.shared.errors import (
    ConcurrencyConflictError,
    DomainError,
    NotFoundError,
    ValidationError,
)

# A hand-built comparison, not a search. The cap exists so one request cannot
# queue unbounded work; the per-workspace concurrency cap already governs how
# fast it drains.
MAX_CONFIGS = 50


@dataclass(frozen=True, kw_only=True)
class SweepConfig:
    """One point in the comparison. The engine varies as freely as its
    conditions do -- "chemprop versus ECFP4" and "depth 3 versus depth 5" are
    the same request shape, which is why there is no grid to expand."""

    engine_id: str
    conditions: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SubmitSweepCommand:
    """The dataset and the baseline are declared once, for the whole sweep.
    Configs measured on different data, or against different baselines, are not
    a comparison -- and a form that lets you build one silently produces a
    ranking that means nothing."""

    name: str
    dataset_id: uuid.UUID
    configs: list[SweepConfig]
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SweepResult:
    sweep_id: uuid.UUID
    runs: list[Run]


@dataclass(frozen=True, kw_only=True)
class ListSweepsQuery:
    limit: int = 50


@dataclass(frozen=True, kw_only=True)
class GetSweepQuery:
    sweep_id: uuid.UUID


@dataclass(frozen=True, kw_only=True)
class CancelSweepCommand:
    sweep_id: uuid.UUID


class SubmitSweep:
    """Creates N training runs sharing one `sweep_id`.

    Everything is validated before anything is created. A failure discovered
    halfway through the loop would leave a half-submitted sweep -- runs that
    will consume the fleet, appear in the ranking, and be indistinguishable
    from a sweep the user actually asked for. Conditions stay unvalidated, for
    the reason `TrainProtocol`'s own docstring gives: an invalid hyperparameter
    fails its own Run visibly, and only the engine's manifest can resolve a
    condition's default.
    """

    def __init__(
        self,
        datasets: DatasetRepository,
        engines: EngineRegistry,
        train: TrainProtocol,
    ) -> None:
        self._datasets = datasets
        self._engines = engines
        self._train = train

    async def __call__(
        self, command: SubmitSweepCommand, auth: AuthContext | None = None
    ) -> Result[SweepResult, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        if not command.configs:
            return Failure(ValidationError("A sweep needs at least one config"))
        if len(command.configs) > MAX_CONFIGS:
            return Failure(
                ValidationError(
                    f"A sweep is limited to {MAX_CONFIGS} configs; got {len(command.configs)}"
                )
            )

        dataset = await self._datasets.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))

        for engine_id in {config.engine_id for config in command.configs}:
            try:
                self._engines.get(engine_id)
            except UnknownEngineError:
                return Failure(NotFoundError("Engine", engine_id))
        if command.baseline_engine_id:
            try:
                self._engines.get(command.baseline_engine_id)
            except UnknownEngineError:
                return Failure(NotFoundError("Engine", command.baseline_engine_id))

        sweep_id = uuid.uuid4()
        runs: list[Run] = []
        for index, config in enumerate(command.configs, start=1):
            result = await self._train(
                TrainProtocolCommand(
                    # Indexed rather than named after the engine: two configs
                    # can share an engine and differ only in conditions, and an
                    # index never collides. The config itself is on the row.
                    name=f"{command.name} #{index}",
                    dataset_id=command.dataset_id,
                    engine_id=config.engine_id,
                    conditions=config.conditions,
                    baseline_engine_id=command.baseline_engine_id,
                    baseline_conditions=command.baseline_conditions,
                    sweep_name=command.name,
                ),
                auth=auth,
                sweep_id=sweep_id,
            )
            if isinstance(result, Failure):
                # Only reachable if the database itself fails: every domain
                # reason `TrainProtocol` can refuse for was checked above.
                return result
            runs.append(result.unwrap())
        return Success(SweepResult(sweep_id=sweep_id, runs=runs))


class ListSweeps:
    """The sweeps list page. One grouped query, no cursor -- see
    `sweep_summaries`."""

    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    async def __call__(
        self, query: ListSweepsQuery, auth: AuthContext | None = None
    ) -> Result[list[SweepSummary], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        limit = clamp_limit(query.limit)
        return Success(await self._runs.sweep_summaries(auth.workspace_id, limit=limit))


class GetSweep:
    """Every member of one sweep. Unknown or empty is `NotFoundError`, not an
    empty list: a sweep with no members does not exist, and returning `[]` for
    a mistyped id would render as a sweep that mysteriously lost its runs."""

    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    async def __call__(
        self, query: GetSweepQuery, auth: AuthContext | None = None
    ) -> Result[list[Run], DomainError]:
        require_authenticated(auth)
        assert auth is not None  # require_authenticated has already rejected None
        runs = await self._runs.list_by_sweep(auth.workspace_id, query.sweep_id)
        if not runs:
            return Failure(NotFoundError("Sweep", str(query.sweep_id)))
        return Success(runs)


class CancelSweep:
    """Cancel every member still doing work, and report how many that was.

    `Run.cancel()` owns the rule and the mechanism, unchanged: a pending run
    never starts, and a running one stops at its next checkpoint because the
    row is the channel. This use case only decides *which* rows.

    Members that already reached a terminal status are skipped rather than
    raising, which also makes a second cancel a no-op returning zero. A retry
    of a dropped request is not a conflict, and a run that succeeded a
    millisecond before the cancel arrived is not a failure of the cancel.

    A running member's own worker writes checkpoints on its own schedule --
    a progress update every few seconds, plus unthrottled writes at fit-leg
    boundaries -- so `self._runs.update(run)` below can lose the optimistic-
    concurrency race even though `run.cancel()` itself never raised. That is
    a `ConcurrencyConflictError`, not a terminal-status `DomainError`, and it
    must not abort the walk: a cascade that stops at the first member whose
    checkpoint won the race would cancel everything visited so far and leave
    every remaining member -- including still-pending ones -- running. On
    that race, re-read the row and retry the cancel exactly once; a member
    that turned terminal in the meantime is this cascade's success (the work
    is stopped), not its failure, so either outcome of the retry lets the
    loop move on to the next member.
    """

    def __init__(self, runs: RunRepository) -> None:
        self._runs = runs

    async def __call__(
        self, command: CancelSweepCommand, auth: AuthContext | None = None
    ) -> Result[int, DomainError]:
        require_authenticated(auth)
        require_editor(auth)
        assert auth is not None  # require_authenticated has already rejected None

        runs = await self._runs.list_by_sweep(auth.workspace_id, command.sweep_id)
        if not runs:
            return Failure(NotFoundError("Sweep", str(command.sweep_id)))

        cancelled = 0
        for run in runs:
            try:
                run.cancel()
            except DomainError:
                # Already terminal. Not this cancel's problem, and not an error:
                # the work this call exists to stop is already stopped.
                continue
            try:
                await self._runs.update(run)
            except ConcurrencyConflictError:
                current = await self._runs.get(auth.workspace_id, run.id)
                if current is None:
                    continue
                try:
                    current.cancel()
                except DomainError:
                    # Went terminal between our read and this retry -- the
                    # work is stopped either way, just not by us.
                    continue
                try:
                    await self._runs.update(current)
                except ConcurrencyConflictError:
                    # Lost the race twice. One retry is the contract; move on
                    # rather than let this member hold up the rest of the sweep.
                    continue
            cancelled += 1
        return Success(cancelled)

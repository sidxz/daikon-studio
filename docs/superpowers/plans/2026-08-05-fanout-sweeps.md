# Fan-out Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit N training configs as one named sweep, watch them together, and rank them on one page.

**Architecture:** Two nullable columns on `runs` (`sweep_id`, `metrics`), a `SubmitSweep` use case that loops the existing `TrainProtocol`, and a `features/sweeps/` frontend mirroring `features/runs/`. Nothing below the use-case layer learns about sweeps: the Postgres queue, the runner protocol, and the runner agents see ordinary pending rows with a lane.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy 2 async, Alembic, Postgres, pytest; Next.js App Router, TanStack Query, orval-generated client, vitest, biome.

**Spec:** `docs/superpowers/specs/2026-08-05-fanout-sweeps-design.md`

## Global Constraints

- Branch is `self-hosted-runners`. Do not merge, rebase, or push it — it is deliberately laptop-only pending the user's decision.
- Gates that must stay green: `make test-all` (497 passing at the start of this plan), `make lint` (ruff + mypy), `make lint-fe` (biome), `make test-fe` (51 passing). Import-linter's 3 contracts are part of `make test-all`.
- Layer rule (import-linter, "Clean Architecture layers"): `interface` → `infrastructure` → `application` → `domain`. Never the reverse. `domain` may not import polars, sqlalchemy, fastapi, or numpy.
- Bounded-context independence: `domain.catalog`, `domain.data`, `domain.execution` may not import each other. Cross-context references are plain UUIDs, never ForeignKeys.
- **The runner agents do not hot-reload.** After any change under `application/execution/` or `infrastructure/engines/`, run `make dev-worker` and `make dev-worker-gpu` or you will debug a fix that never loaded.
- Any test that inserts `runners` rows must clean up via `cleanup_registered_runners` in `backend/tests/helpers/runner_fixtures.py` — fixtures bound to the session-scoped NullPool engine have no per-test rollback, and a leaked row deterministically breaks `test_runner_repository.py::test_list_returns_all` under `make test-all`.
- All backend commands run from `backend/`. All frontend commands run from `frontend/`. `make` targets run from the repo root.
- Commit after every task. Use `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` in commit messages.

---

## File Structure

**Backend — created:**
- `backend/alembic/versions/010_sweeps_and_metrics.py` — the two columns and the index.
- `backend/src/daikonstudio/application/execution/sweeps.py` — `SubmitSweep`, `ListSweeps`, `GetSweep`, `CancelSweep`. All four are sweep-shaped reads and writes over the existing Run aggregate; they live together because they change together.
- `backend/src/daikonstudio/interface/routes/sweeps.py` — the four endpoints and their Pydantic bodies.
- `backend/tests/integration/test_sweeps.py` — use-case behaviour against real Postgres.
- `backend/tests/api/test_sweeps.py` — the HTTP contract.

**Backend — modified:**
- `domain/execution/run.py` — `sweep_id` and `metrics` constructor kwargs, `record_metrics()`.
- `infrastructure/persistence/sqlalchemy/execution/models.py` — two columns, one index.
- `infrastructure/persistence/sqlalchemy/execution/repository.py` — round-trip both columns, persist `metrics` on update, `list_by_sweep`, `sweep_summaries`.
- `application/ports/run_repository.py` — the two new methods and the `SweepSummary` read model.
- `application/execution/build_scorecard.py` — extract `primary_metric_for(task)`.
- `application/execution/train_protocol.py` — optional `sweep_id` kwarg; record metrics at the end of `RunTraining`.
- `infrastructure/runner/wire.py` — `metrics` and `sweep_id` on the envelopes.
- `interface/routes/runner_api.py` — apply `metrics` on update.
- `infrastructure/runner/ports.py` — send `metrics` on update.
- `infrastructure/di/container.py`, `interface/app.py` — register the four use cases and the router.

**Frontend — created:** `src/features/sweeps/{types,hooks,lib,components}/…`, `src/app/(dashboard)/sweeps/{page.tsx,new/page.tsx,[id]/page.tsx}`.

**Frontend — modified:** `src/shared/lib/navigation.ts` (one nav entry), `src/shared/lib/api/model` + `openapi.json` (regenerated, committed).

---

### Task 1: Schema — `sweep_id` and `metrics` columns

**Files:**
- Create: `backend/alembic/versions/010_sweeps_and_metrics.py`
- Modify: `backend/src/daikonstudio/domain/execution/run.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/models.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py`
- Test: `backend/tests/integration/test_sweeps.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Run(sweep_id: uuid.UUID | None = None, metrics: dict[str, Any] | None = None)`; `Run.record_metrics(*, primary_metric: str, value: float | None, baseline_value: float | None) -> None`; both columns round-tripping through `SqlAlchemyRunRepository`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/integration/test_sweeps.py`. Note the fixture style: `migrated_session` in `backend/tests/conftest.py` is the session-scoped, migrated engine — read the top of `backend/tests/integration/test_train_protocol.py` for how existing integration tests build a repository from it, and follow that exactly.

```python
"""Sweeps: N training runs submitted as one group, ranked together.

Against real Postgres and the real repository -- the columns exist to be
queried and grouped, and an ORM round-trip that never touches SQL would
not prove either.
"""

from __future__ import annotations

import uuid

import pytest

from daikonstudio.domain.execution.run import Run, RunKind
from daikonstudio.infrastructure.persistence.sqlalchemy.execution.repository import (
    SqlAlchemyRunRepository,
)


def _run(*, workspace_id: uuid.UUID, sweep_id: uuid.UUID | None = None, name: str = "s") -> Run:
    return Run(
        kind=RunKind.TRAINING,
        workspace_id=workspace_id,
        requested_by=uuid.uuid4(),
        cache_key="k" * 64,
        params={"name": name, "sweep_name": "BBBP comparison", "dataset_id": str(uuid.uuid4())},
        sweep_id=sweep_id,
    )


@pytest.mark.asyncio
async def test_sweep_id_and_metrics_round_trip(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    sweep_id = uuid.uuid4()
    run = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    await repository.add(run)

    run.start()
    run.record_metrics(primary_metric="mcc", value=0.603, baseline_value=0.632)
    await repository.update(run)

    stored = await repository.get(workspace_id, run.id)
    assert stored is not None
    assert stored.sweep_id == sweep_id
    assert stored.metrics == {
        "primary_metric": "mcc",
        "value": 0.603,
        "baseline_value": 0.632,
    }


@pytest.mark.asyncio
async def test_solo_run_has_no_sweep_id(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    run = _run(workspace_id=workspace_id)
    await repository.add(run)

    stored = await repository.get(workspace_id, run.id)
    assert stored is not None
    assert stored.sweep_id is None
    assert stored.metrics is None
```

Add the `sessions` fixture at the top of the file if `backend/tests/conftest.py` does not already provide one — check first, and reuse rather than redefine:

```python
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker


@pytest_asyncio.fixture
async def sessions(_migrated_engine: AsyncEngine):
    return async_sessionmaker(_migrated_engine, expire_on_commit=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -v`
Expected: FAIL with `TypeError: Run.__init__() got an unexpected keyword argument 'sweep_id'`.

- [ ] **Step 3: Add the domain fields**

In `domain/execution/run.py`, add two keyword parameters to `Run.__init__` (place `sweep_id` immediately after `protocol_id`, and `metrics` after `result_uri`, to keep the constructor ordered like the row):

```python
        sweep_id: uuid.UUID | None = None,
        metrics: dict[str, Any] | None = None,
```

And in the body, after the existing `self.protocol_id = protocol_id` block:

```python
        # Which sweep this Run belongs to, or None for an ordinary solo run --
        # which is nearly every row. Set once, at creation, by `SubmitSweep`;
        # `update()` never persists it, exactly like `params`, because "which
        # question was I part of" is an instruction and not an outcome.
        self.sweep_id = sweep_id
        # The headline number, denormalised out of the Scorecard blob so that
        # ranking N runs is a column read. An outcome, so `update()` does
        # persist it -- the same argument `protocol_id` makes above. None until
        # a training run reaches `ready`, and forever on a prediction run.
        self.metrics = metrics
```

Then add the method, next to `link_protocol`:

```python
    def record_metrics(
        self, *, primary_metric: str, value: float | None, baseline_value: float | None
    ) -> None:
        """The one number a sweep ranks on, plus what it was measured against.

        Deliberately not the full metric dict: everything else a scientist
        needs is in the Scorecard, and the reason this lives on the row at all
        is that building a Scorecard recomputes Tanimoto similarity over
        train x test. Ranking twenty runs must not pay that twenty times.

        `value` is nullable because a metric can be genuinely undefined -- a
        single-class test split makes every classification metric meaningless,
        and the Scorecard already says so. A ranked list shows such a run as
        unranked rather than as zero.
        """
        self.metrics = {
            "primary_metric": primary_metric,
            "value": value,
            "baseline_value": baseline_value,
        }
        self._touch()
```

- [ ] **Step 4: Add the columns and the index**

In `infrastructure/persistence/sqlalchemy/execution/models.py`, add to `RunModel` after `protocol_id`:

```python
    # NULL for an ordinary solo run. Set at creation and never updated -- see
    # `Run.sweep_id`. A bare indexed UUID with no `sweeps` table behind it:
    # the group's only state is its members, and a `GROUP BY` answers every
    # question the list page asks.
    sweep_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    # The headline metric, denormalised for ranking -- see `Run.record_metrics`.
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
```

And in `__table_args__`, after the protocol index:

```python
        # Backs both sweep queries: the member list, and the grouped summary
        # the sweeps list page reads.
        Index("ix_runs_workspace_sweep_id", "workspace_id", "sweep_id"),
```

- [ ] **Step 5: Write the migration**

Create `backend/alembic/versions/010_sweeps_and_metrics.py`:

```python
"""runs.sweep_id and runs.metrics

Fan-out: N training configs submitted as one named group. `sweep_id` is that
group, and there is no `sweeps` table behind it -- a sweep's only state is its
members, so a `GROUP BY` answers everything the list page asks and a second
aggregate inside the execution context would own no invariant the Run does not
already own.

`metrics` denormalises the headline number out of the Scorecard blob. Building
a Scorecard recomputes RDKit Tanimoto similarity over train x test on every
call; a page whose purpose is comparing twenty runs must not pay that twenty
times per visit.

No backfill for either. Every existing run predates sweeps, so `sweep_id` is
correctly NULL; and `metrics` for a historical run is recoverable only by
reading and parsing its Scorecard blob, which is exactly the cost this column
exists to avoid paying. Old runs rank as unmeasured, which is honest -- they
were never part of a sweep.

Revision ID: 010
Revises: 009
Create Date: 2026-08-05 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("sweep_id", sa.Uuid(), nullable=True))
    op.add_column("runs", sa.Column("metrics", postgresql.JSONB(), nullable=True))
    op.create_index("ix_runs_workspace_sweep_id", "runs", ["workspace_id", "sweep_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_workspace_sweep_id", table_name="runs")
    op.drop_column("runs", "metrics")
    op.drop_column("runs", "sweep_id")
```

- [ ] **Step 6: Round-trip both columns through the repository**

In `infrastructure/persistence/sqlalchemy/execution/repository.py`, add to `_to_domain`'s `Run(...)` call (after `protocol_id=model.protocol_id,`):

```python
        sweep_id=model.sweep_id,
        metrics=model.metrics,
```

Add the same two lines to `_to_model`'s `RunModel(...)` call, reading from `run`:

```python
        sweep_id=run.sweep_id,
        metrics=run.metrics,
```

In `update()`, add `metrics` to `.values(...)`, directly under the `protocol_id=model.protocol_id,` line:

```python
                    # An outcome, like protocol_id above -- see
                    # `Run.record_metrics`. `sweep_id` is deliberately absent:
                    # it is an instruction, and stays write-once like `params`.
                    metrics=model.metrics,
```

- [ ] **Step 7: Apply the migration and run the tests**

Run: `make migrate && cd backend && uv run pytest tests/integration/test_sweeps.py -v`
Expected: both tests PASS.

- [ ] **Step 8: Confirm nothing else regressed**

Run: `make test-all`
Expected: 499 passed (the 497 baseline plus this task's 2), import-linter 3/3.

- [ ] **Step 9: Commit**

```bash
git add backend/alembic/versions/010_sweeps_and_metrics.py \
        backend/src/daikonstudio/domain/execution/run.py \
        backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/models.py \
        backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py \
        backend/tests/integration/test_sweeps.py
git commit -m "feat(execution): runs.sweep_id and runs.metrics

A sweep is a group of runs, not a table: sweep_id plus a GROUP BY answers
every question the list page asks. metrics denormalises the headline
number so ranking N runs does not rebuild N Scorecards.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Training records its headline metric, all the way through the runner protocol

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/build_scorecard.py`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py`
- Modify: `backend/src/daikonstudio/infrastructure/runner/wire.py`
- Modify: `backend/src/daikonstudio/interface/routes/runner_api.py`
- Modify: `backend/src/daikonstudio/infrastructure/runner/ports.py`
- Test: `backend/tests/integration/test_train_protocol.py`, `backend/tests/api/test_runner_protocol.py`

**Interfaces:**
- Consumes: `Run.record_metrics(*, primary_metric, value, baseline_value)` from Task 1.
- Produces: `primary_metric_for(task: TaskType) -> str` in `application/execution/build_scorecard.py`; a `ready` training run whose `metrics` column is populated whether it ran inline or on a real runner agent.

**Why the wire path is in this task and not a later one:** `RunTraining` executes *on the runner agent*, whose `RunRepository` is `HttpRunRepository`. It POSTs an explicit field list. A `metrics` value set on the aggregate but absent from that list is silently dropped, and the column stays NULL for every run that did not go through `InlineEnqueuer` — meaning it works in tests and in `STUDIO_INLINE_JOBS=1` dev, and does nothing in production. The two halves are one deliverable.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/integration/test_train_protocol.py`, matching the fixtures the file already uses for its end-to-end training tests:

```python
@pytest.mark.asyncio
async def test_training_records_its_headline_metric(trained_run: Run) -> None:
    """The number a sweep ranks on lands on the row, not only in the blob.

    `trained_run` is a completed classification training run -- see the
    fixtures at the top of this file.
    """
    assert trained_run.metrics is not None
    assert trained_run.metrics["primary_metric"] == "mcc"
    assert isinstance(trained_run.metrics["value"], float)
    assert isinstance(trained_run.metrics["baseline_value"], float)
```

If no `trained_run` fixture exists, use whichever fixture or helper the neighbouring tests use to obtain a finished training `Run` and assert on that object instead — do not build a second training path.

Append to `backend/tests/api/test_runner_protocol.py`:

```python
@pytest.mark.asyncio
async def test_update_run_applies_metrics(client, claimed_run) -> None:
    """A runner's terminal update carries the headline metric.

    Without this the column is populated only under InlineEnqueuer -- green in
    tests, NULL in production.
    """
    response = await client.post(
        f"/api/v1/runner/runs/{claimed_run.id}",
        json={
            "status": "running",
            "expected_version": claimed_run.version,
            "metrics": {"primary_metric": "mcc", "value": 0.6, "baseline_value": 0.5},
        },
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_update_run_without_metrics_does_not_clear_them(client, claimed_run) -> None:
    """A bare progress update must not null out a metric a previous update set
    -- the same failure the security review caught for progress and phase."""
    await client.post(
        f"/api/v1/runner/runs/{claimed_run.id}",
        json={
            "status": "running",
            "expected_version": claimed_run.version,
            "metrics": {"primary_metric": "mcc", "value": 0.6, "baseline_value": 0.5},
        },
    )
    await client.post(
        f"/api/v1/runner/runs/{claimed_run.id}",
        json={"status": "running", "expected_version": claimed_run.version + 1, "progress": 0.5},
    )
    # Read it back through whichever helper this file already uses to load a run
    # by id, and assert `metrics` survived.
```

Match the existing fixture names in `test_runner_protocol.py` — read the top of that file first; `client` and `claimed_run` above are placeholders for whatever it actually provides, and the token/auth header setup it already does must be reused, not reinvented.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/integration/test_train_protocol.py -k headline tests/api/test_runner_protocol.py -k metrics -v`
Expected: FAIL — `assert None is not None` on the first, and a 422 `extra_forbidden` on the runner ones (`RunUpdateEnvelope` sets `model_config = _FORBID`).

- [ ] **Step 3: Extract the primary-metric choice**

In `application/execution/build_scorecard.py`, add above `build_scorecard`:

```python
def primary_metric_for(task: TaskType) -> str:
    """The one metric a Scorecard leads with, and the one a sweep ranks on.

    Shared so those two can never disagree about which number is the headline
    -- a ranked list ordered by a metric the Scorecard does not show is a
    silent lie about which model won.
    """
    return "mcc" if task is TaskType.BINARY_CLASSIFICATION else "rmse"
```

Then in the `return Scorecard(` block, replace:

```python
        primary_metric="mcc" if is_classification else "rmse",
```

with:

```python
        primary_metric=primary_metric_for(task),
```

- [ ] **Step 4: Record the metric at the end of training**

In `application/execution/train_protocol.py`, add to the imports:

```python
from daikonstudio.application.execution.build_scorecard import primary_metric_for
```

Then in `RunTraining.__call__`, immediately after the existing `run.link_protocol(protocol_id)` line and before `return result_uri`:

```python
        # Denormalised for ranking. Both dicts are already measured above, so
        # this is a lookup, not a computation -- and it rides out on the same
        # `succeed()` + `update()` write that persists `result_uri`, so a run
        # can never be READY with no metric on it.
        primary = primary_metric_for(task)
        run.record_metrics(
            primary_metric=primary,
            value=metrics.get(primary),
            baseline_value=baseline_metrics.get(primary),
        )
```

- [ ] **Step 5: Carry `metrics` and `sweep_id` over the wire**

In `infrastructure/runner/wire.py`, add to `RunEnvelope`'s fields (it mirrors every `Run.__init__` kwarg, so both belong):

```python
    sweep_id: uuid.UUID | None = None
    metrics: dict[str, Any] | None = None
```

Add the matching lines to `RunEnvelope.from_domain`'s `cls(...)` call:

```python
            sweep_id=run.sweep_id,
            metrics=run.metrics,
```

and to `RunEnvelope.to_domain`'s `Run(...)` call:

```python
            sweep_id=self.sweep_id,
            metrics=self.metrics,
```

Add one field to `RunUpdateEnvelope`, after `protocol_id`:

```python
    metrics: dict[str, Any] | None = None
```

`sweep_id` is deliberately *not* on `RunUpdateEnvelope`: a runner has no business changing which sweep a run belongs to, and the server-side repository does not persist it on update anyway.

- [ ] **Step 6: Apply it server-side**

In `interface/routes/runner_api.py`, inside `update_run`, add after the `error_message` block and before the `protocol_id` block:

```python
    if "metrics" in fields:
        run.metrics = body.metrics
```

Guarded by `model_fields_set` like every other optional field, so a bare progress update does not null out a metric a previous update set.

- [ ] **Step 7: Send it client-side**

In `infrastructure/runner/ports.py`, inside `HttpRunRepository.update`, add after the `if run.protocol_id is not None:` block:

```python
        # Conditional for the same reason `protocol_id` is: this is set once,
        # at the end of a training run, and every progress checkpoint before
        # that has it None. Sending that None explicitly would tell the server
        # to clear it on the very next heartbeat.
        if run.metrics is not None:
            fields["metrics"] = run.metrics
```

- [ ] **Step 8: Run the tests**

Run: `cd backend && uv run pytest tests/integration/test_train_protocol.py tests/api/test_runner_protocol.py tests/api/test_runner_ports.py -v`
Expected: PASS.

- [ ] **Step 9: Reload the runner agents and confirm the whole suite**

Run: `make dev-worker && make dev-worker-gpu && make test-all && make lint`
Expected: the suite passes, ruff and mypy clean. The agent restart matters — they hold the old `train_protocol.py` in memory otherwise.

- [ ] **Step 10: Commit**

```bash
git add backend/src/daikonstudio/application/execution/build_scorecard.py \
        backend/src/daikonstudio/application/execution/train_protocol.py \
        backend/src/daikonstudio/infrastructure/runner/wire.py \
        backend/src/daikonstudio/infrastructure/runner/ports.py \
        backend/src/daikonstudio/interface/routes/runner_api.py \
        backend/tests/integration/test_train_protocol.py \
        backend/tests/api/test_runner_protocol.py
git commit -m "feat(execution): training records its headline metric on the run

Extracts primary_metric_for so the Scorecard and a sweep ranking can never
disagree about which number is the headline. Carried over the runner
protocol as well as inline -- a field set on the aggregate but missing from
HttpRunRepository's POST is green in tests and NULL in production.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Repository reads for a sweep

**Files:**
- Modify: `backend/src/daikonstudio/application/ports/run_repository.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py`
- Test: `backend/tests/integration/test_sweeps.py`

**Interfaces:**
- Consumes: `Run.sweep_id` from Task 1.
- Produces: `SweepSummary` (frozen dataclass, in `application/ports/run_repository.py`) with fields `sweep_id: uuid.UUID`, `name: str | None`, `dataset_id: uuid.UUID | None`, `created_at: datetime`, `total: int`, `by_status: dict[str, int]`; and on `RunRepository`: `list_by_sweep(workspace_id: UUID, sweep_id: UUID) -> list[Run]` and `sweep_summaries(workspace_id: UUID, *, limit: int = 50) -> list[SweepSummary]`.

`SweepSummary` lives in the port module, not in `application/execution/sweeps.py`, because the use-case module imports the port — defining the return type in the use-case module would make the two import each other.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/integration/test_sweeps.py`:

```python
@pytest.mark.asyncio
async def test_list_by_sweep_returns_only_that_sweep(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    mine, theirs = uuid.uuid4(), uuid.uuid4()
    for _ in range(3):
        await repository.add(_run(workspace_id=workspace_id, sweep_id=mine))
    await repository.add(_run(workspace_id=workspace_id, sweep_id=theirs))
    await repository.add(_run(workspace_id=workspace_id))

    runs = await repository.list_by_sweep(workspace_id, mine)

    assert len(runs) == 3
    assert {run.sweep_id for run in runs} == {mine}


@pytest.mark.asyncio
async def test_list_by_sweep_is_workspace_scoped(sessions) -> None:
    """The filter is in the SQL, not applied after the fetch."""
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    await repository.add(_run(workspace_id=uuid.uuid4(), sweep_id=sweep_id))

    assert await repository.list_by_sweep(uuid.uuid4(), sweep_id) == []


@pytest.mark.asyncio
async def test_sweep_summaries_counts_by_status(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    sweep_id = uuid.uuid4()
    pending = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    finished = _run(workspace_id=workspace_id, sweep_id=sweep_id)
    await repository.add(pending)
    await repository.add(finished)
    finished.start()
    finished.succeed("file:///tmp/x.json")
    await repository.update(finished)

    summaries = await repository.sweep_summaries(workspace_id)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.sweep_id == sweep_id
    assert summary.name == "BBBP comparison"
    assert summary.total == 2
    assert summary.by_status == {"pending": 1, "ready": 1}


@pytest.mark.asyncio
async def test_sweep_summaries_ignores_solo_runs(sessions) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    workspace_id = uuid.uuid4()
    await repository.add(_run(workspace_id=workspace_id))

    assert await repository.sweep_summaries(workspace_id) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -v`
Expected: FAIL with `AttributeError: 'SqlAlchemyRunRepository' object has no attribute 'list_by_sweep'`.

- [ ] **Step 3: Declare the read model and the port methods**

In `application/ports/run_repository.py`, add one import (the module already imports `datetime` and `UUID`, and uses the bare `UUID` name — match it, do not introduce `import uuid` alongside it):

```python
from dataclasses import dataclass
```

Then above `class RunRepository`:

```python
@dataclass(frozen=True, kw_only=True)
class SweepSummary:
    """One row of the sweeps list, assembled by a GROUP BY rather than read
    from a `sweeps` table -- a sweep's only state is its members.

    `name` and `dataset_id` are read out of any member's `params` (every member
    of a sweep carries the same two, written once at submission). Both are
    nullable only to survive a malformed row; a sweep created by `SubmitSweep`
    always has them.

    `by_status` holds only the statuses actually present, so a caller reads
    `.get(status, 0)` rather than trusting a fixed key set that a new
    `RunStatus` member would silently invalidate.
    """

    sweep_id: UUID
    name: str | None
    dataset_id: UUID | None
    created_at: datetime
    total: int
    by_status: dict[str, int]
```

And on `RunRepository`, after `list`:

```python
    async def list_by_sweep(self, workspace_id: UUID, sweep_id: UUID) -> list[Run]:
        """Every member of one sweep, oldest first -- submission order, which
        is the order the configs were given in and therefore the order a user
        recognises. Unpaginated on purpose: a sweep is bounded by what a human
        typed into a form, and paging a comparison defeats the comparison."""
        ...

    async def sweep_summaries(
        self, workspace_id: UUID, *, limit: int = 50
    ) -> list[SweepSummary]: ...
```

- [ ] **Step 4: Implement them**

In `infrastructure/persistence/sqlalchemy/execution/repository.py`, extend the imports:

```python
from sqlalchemy import CursorResult, Select, func, select, tuple_
```

and

```python
from daikonstudio.application.ports.run_repository import SweepSummary
```

Then add the two methods to `SqlAlchemyRunRepository`, after `list`:

```python
    async def list_by_sweep(self, workspace_id: uuid.UUID, sweep_id: uuid.UUID) -> list[Run]:
        statement = (
            select(RunModel)
            .where(RunModel.workspace_id == workspace_id, RunModel.sweep_id == sweep_id)
            .order_by(RunModel.created_at, RunModel.id)
        )
        async with self._sessions() as session:
            result = await session.execute(statement)
            return [_to_domain(model) for model in result.scalars()]

    async def sweep_summaries(
        self, workspace_id: uuid.UUID, *, limit: int = 50
    ) -> list[SweepSummary]:
        """One row per sweep, via FILTER-ed aggregates so the LIMIT applies to
        sweeps rather than to (sweep, status) pairs.

        ponytail: no cursor. Sweeps are created by hand, a handful at a time --
        add keyset pagination here the day a workspace has more than `limit`,
        the same shape `list()` already uses.
        """
        counts = {
            status: func.count()
            .filter(RunModel.status == status)
            .label(f"n_{status}")
            for status in ("pending", "running", "ready", "failed", "cancelled")
        }
        statement = (
            select(
                RunModel.sweep_id,
                func.min(RunModel.created_at).label("created_at"),
                func.min(RunModel.params["sweep_name"].astext).label("name"),
                func.min(RunModel.params["dataset_id"].astext).label("dataset_id"),
                func.count().label("total"),
                *counts.values(),
            )
            .where(RunModel.workspace_id == workspace_id, RunModel.sweep_id.is_not(None))
            .group_by(RunModel.sweep_id)
            .order_by(func.min(RunModel.created_at).desc())
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return [
            SweepSummary(
                sweep_id=row.sweep_id,
                name=row.name,
                dataset_id=uuid.UUID(row.dataset_id) if row.dataset_id else None,
                created_at=row.created_at,
                total=row.total,
                by_status={
                    status: getattr(row, f"n_{status}")
                    for status in counts
                    if getattr(row, f"n_{status}")
                },
            )
            for row in rows
        ]
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/application/ports/run_repository.py \
        backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/repository.py \
        backend/tests/integration/test_sweeps.py
git commit -m "feat(execution): sweep member list and grouped summary

FILTER-ed aggregates so the LIMIT applies to sweeps, not to (sweep,
status) pairs. Name and dataset come from any member's params -- every
member carries the same two.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `SubmitSweep`

**Files:**
- Create: `backend/src/daikonstudio/application/execution/sweeps.py`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py`
- Test: `backend/tests/integration/test_sweeps.py`

**Interfaces:**
- Consumes: `SweepSummary`, `list_by_sweep`, `sweep_summaries` (Task 3); `Run.sweep_id` (Task 1).
- Produces: `TrainProtocol.__call__(command, auth=None, *, sweep_id: uuid.UUID | None = None)`; `SweepConfig(engine_id: str, conditions: dict[str, object])`; `SubmitSweepCommand(name, dataset_id, configs, baseline_engine_id, baseline_conditions)`; `SubmitSweep.__call__(command, auth) -> Result[SweepResult, DomainError]` where `SweepResult(sweep_id: uuid.UUID, runs: list[Run])`.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/integration/test_sweeps.py`. Build `SubmitSweep` from the same real `TrainProtocol` the neighbouring integration tests construct — a fake `TrainProtocol` here would prove nothing, since the entire design claim is that a sweep child and a solo run are the same object produced by the same code.

```python
@pytest.mark.asyncio
async def test_submit_creates_one_run_per_config_sharing_a_sweep_id(
    submit_sweep, dataset, auth
) -> None:
    result = await submit_sweep(
        SubmitSweepCommand(
            name="BBBP comparison",
            dataset_id=dataset.id,
            configs=[
                SweepConfig(engine_id="ecfp4-randomforest", conditions={"n_estimators": 100}),
                SweepConfig(engine_id="ecfp4-randomforest", conditions={"n_estimators": 300}),
                SweepConfig(engine_id="ecfp4-xgboost", conditions={}),
            ],
        ),
        auth=auth,
    )

    sweep = result.unwrap()
    assert len(sweep.runs) == 3
    assert {run.sweep_id for run in sweep.runs} == {sweep.sweep_id}
    assert [run.params["name"] for run in sweep.runs] == [
        "BBBP comparison #1",
        "BBBP comparison #2",
        "BBBP comparison #3",
    ]
    assert all(run.params["sweep_name"] == "BBBP comparison" for run in sweep.runs)


@pytest.mark.asyncio
async def test_an_unknown_engine_in_the_last_config_creates_no_runs(
    submit_sweep, dataset, auth, runs_repository
) -> None:
    """Pre-flight, not fail-halfway. A partially-submitted sweep is
    indistinguishable from a complete one, and nobody asked for it."""
    result = await submit_sweep(
        SubmitSweepCommand(
            name="doomed",
            dataset_id=dataset.id,
            configs=[
                SweepConfig(engine_id="ecfp4-randomforest", conditions={}),
                SweepConfig(engine_id="no-such-engine", conditions={}),
            ],
        ),
        auth=auth,
    )

    assert isinstance(result, Failure)
    assert await runs_repository.sweep_summaries(auth.workspace_id) == []


@pytest.mark.asyncio
async def test_an_empty_config_list_is_rejected(submit_sweep, dataset, auth) -> None:
    result = await submit_sweep(
        SubmitSweepCommand(name="empty", dataset_id=dataset.id, configs=[]),
        auth=auth,
    )
    assert isinstance(result, Failure)
```

Add to this file's imports:

```python
from returns.result import Failure

from daikonstudio.application.execution.sweeps import (
    SubmitSweep,
    SubmitSweepCommand,
    SweepConfig,
)
```

Reuse the `dataset`, `auth`, and repository fixtures the existing integration tests build; if `test_sweeps.py` has none yet, lift the setup from `tests/integration/test_train_protocol.py` rather than inventing a second way to create a Dataset. `STUDIO_INLINE_JOBS` must be **off** for these — the runs only need to be created and enqueued, not executed, and executing three real fits per test would make the file minutes long.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -k submit -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daikonstudio.application.execution.sweeps'`.

- [ ] **Step 3: Let `TrainProtocol` stamp a sweep id**

In `application/execution/train_protocol.py`, change `TrainProtocol.__call__`'s signature to:

```python
    async def __call__(
        self,
        command: TrainProtocolCommand,
        auth: AuthContext | None = None,
        *,
        sweep_id: uuid.UUID | None = None,
    ) -> Result[Run, DomainError]:
```

and add `sweep_id=sweep_id,` to the `Run(...)` construction, directly after `params=command.to_params(),`:

```python
            # Which sweep asked for this run, or None for a solo request. The
            # only difference between the two, deliberately: a sweep child is
            # the same object, with the same cache key, baseline resolution and
            # lane, produced by this same code path.
            sweep_id=sweep_id,
```

Add a paragraph to the class docstring:

```
    `sweep_id` is the one thing a caller may add to an otherwise identical
    request. It is stamped on the row rather than folded into `params` because
    `params` is write-once and unindexed: a mistyped grouping could never be
    corrected, and the cancel cascade and the sweeps list both filter on it.
```

- [ ] **Step 4: Write the use case**

Create `backend/src/daikonstudio/application/execution/sweeps.py`:

```python
"""Fan-out: N training configs submitted as one named group.

A sweep is not a workflow. Nothing here waits for anything, nothing consumes
another run's output, and no step must survive a crash to be correct -- which
is precisely why this is a `sweep_id` column and a loop over the existing
`TrainProtocol` rather than an orchestration engine. See
`docs/superpowers/specs/2026-08-05-fanout-sweeps-design.md` for the trigger
that would change that.

Each child run keeps its own mandatory baseline. Twenty configs against one
shared baseline would mean nineteen runs waiting on a twentieth's output, and
that dependency is exactly what this feature was scoped to avoid. The honest
cost is a repeated baseline fit; the real fix is the deferred fit-result cache,
not a sweep-level special case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace

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
from daikonstudio.application.ports.dataset_repository import DatasetRepository
from daikonstudio.application.ports.run_repository import RunRepository, SweepSummary
from daikonstudio.domain.execution.run import Run
from daikonstudio.domain.shared.errors import DomainError, NotFoundError, ValidationError

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
```

Note the `sweep_name` in `params`: `TrainProtocolCommand.to_params()` does not carry it yet. Add it in the next step rather than writing a second params path here.

- [ ] **Step 5: Carry the sweep's display name in params**

In `application/execution/train_protocol.py`, add a field to `TrainProtocolCommand`:

```python
    # The sweep's own name, repeated on every member. A label, not a grouping:
    # the id is a column precisely because `params` is write-once, and this
    # rides along so the sweeps list can title a group without a second table
    # or a second query. `None` on a solo run.
    sweep_name: str | None = None
```

Add it to `to_params`:

```python
            "sweep_name": self.sweep_name,
```

and to `from_params`:

```python
            sweep_name=params.get("sweep_name"),
```

Then in `sweeps.py`, add `sweep_name=command.name,` to the `TrainProtocolCommand(...)` construction.

`compute_cache_key` is deliberately untouched: two identical configs in different sweeps are the same work, and a display name must not change a content hash.

- [ ] **Step 6: Run the tests**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/src/daikonstudio/application/execution/sweeps.py \
        backend/src/daikonstudio/application/execution/train_protocol.py \
        backend/tests/integration/test_sweeps.py
git commit -m "feat(execution): SubmitSweep creates N runs under one sweep_id

Everything validated before anything is created -- a sweep that failed
halfway would be indistinguishable from one somebody asked for. TrainProtocol
gains one optional kwarg and is otherwise untouched, so a sweep child and a
solo run are the same object from the same code path.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `ListSweeps`, `GetSweep`, `CancelSweep`

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/sweeps.py`
- Test: `backend/tests/integration/test_sweeps.py`

**Interfaces:**
- Consumes: `list_by_sweep`, `sweep_summaries`, `SweepSummary` (Task 3); the query/command dataclasses declared in Task 4.
- Produces: `ListSweeps.__call__(query, auth) -> Result[list[SweepSummary], DomainError]`; `GetSweep.__call__(query, auth) -> Result[list[Run], DomainError]`; `CancelSweep.__call__(command, auth) -> Result[int, DomainError]` returning how many runs were actually cancelled.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/integration/test_sweeps.py`:

```python
@pytest.mark.asyncio
async def test_cancel_sweep_cancels_only_the_non_terminal_runs(
    sessions, cancel_sweep, auth
) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    queued = _run(workspace_id=auth.workspace_id, sweep_id=sweep_id)
    done = _run(workspace_id=auth.workspace_id, sweep_id=sweep_id)
    await repository.add(queued)
    await repository.add(done)
    done.start()
    done.succeed("file:///tmp/x.json")
    await repository.update(done)

    cancelled = (await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)).unwrap()

    assert cancelled == 1
    runs = {run.id: run.status for run in await repository.list_by_sweep(auth.workspace_id, sweep_id)}
    assert runs[queued.id] is RunStatus.CANCELLED
    assert runs[done.id] is RunStatus.READY


@pytest.mark.asyncio
async def test_cancel_sweep_is_idempotent(sessions, cancel_sweep, auth) -> None:
    """A second cancel is not an error. The first one already stopped the
    work, and a 409 here would make the retry of a dropped request look like
    a failure."""
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    await repository.add(_run(workspace_id=auth.workspace_id, sweep_id=sweep_id))

    first = (await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)).unwrap()
    second = (await cancel_sweep(CancelSweepCommand(sweep_id=sweep_id), auth=auth)).unwrap()

    assert (first, second) == (1, 0)


@pytest.mark.asyncio
async def test_cancel_unknown_sweep_is_not_found(cancel_sweep, auth) -> None:
    result = await cancel_sweep(CancelSweepCommand(sweep_id=uuid.uuid4()), auth=auth)
    assert isinstance(result, Failure)


@pytest.mark.asyncio
async def test_get_sweep_returns_members_in_submission_order(
    sessions, get_sweep, auth
) -> None:
    repository = SqlAlchemyRunRepository(sessions)
    sweep_id = uuid.uuid4()
    for index in range(3):
        await repository.add(
            _run(workspace_id=auth.workspace_id, sweep_id=sweep_id, name=f"s #{index + 1}")
        )

    runs = (await get_sweep(GetSweepQuery(sweep_id=sweep_id), auth=auth)).unwrap()

    assert [run.params["name"] for run in runs] == ["s #1", "s #2", "s #3"]
```

Extend the imports in this file with `RunStatus`, `CancelSweepCommand`, `GetSweepQuery`, and add `cancel_sweep`/`get_sweep` fixtures constructing `CancelSweep(repository)` and `GetSweep(repository)` from the same `sessions` fixture the other tests use.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -k "cancel or get_sweep" -v`
Expected: FAIL with `ImportError: cannot import name 'CancelSweep'`.

- [ ] **Step 3: Implement the three**

Append to `application/execution/sweeps.py`:

```python
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
        return Success(await self._runs.sweep_summaries(auth.workspace_id, limit=query.limit))


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
            await self._runs.update(run)
            cancelled += 1
        return Success(cancelled)
```

Add `RunStatus` to this module's domain import only if you end up needing it; the `try/except DomainError` above deliberately lets `Run.cancel()` remain the single place that knows which statuses are terminal.

- [ ] **Step 4: Run the tests**

Run: `cd backend && uv run pytest tests/integration/test_sweeps.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/application/execution/sweeps.py \
        backend/tests/integration/test_sweeps.py
git commit -m "feat(execution): list, read and cancel a sweep

Cancel skips already-terminal members rather than raising, which makes a
retry of a dropped request a no-op returning zero instead of a 409.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: HTTP endpoints

**Files:**
- Create: `backend/src/daikonstudio/interface/routes/sweeps.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Modify: `backend/src/daikonstudio/interface/app.py`
- Test: `backend/tests/api/test_sweeps.py`

**Interfaces:**
- Consumes: all four use cases from Tasks 4-5.
- Produces: `POST /api/v1/sweeps`, `GET /api/v1/sweeps`, `GET /api/v1/sweeps/{sweep_id}`, `POST /api/v1/sweeps/{sweep_id}/cancel`; response models `SweepResponse`, `SweepDetailResponse`, `SweepRunResponse`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/api/test_sweeps.py`, following the client and auth fixtures `backend/tests/api/test_protocols.py` already uses:

```python
"""The sweeps HTTP contract."""

from __future__ import annotations

import uuid

import pytest


@pytest.mark.asyncio
async def test_submit_returns_202_with_every_run(client, dataset_id) -> None:
    response = await client.post(
        "/api/v1/sweeps",
        json={
            "name": "BBBP comparison",
            "dataset_id": str(dataset_id),
            "configs": [
                {"engine_id": "ecfp4-randomforest", "conditions": {}},
                {"engine_id": "ecfp4-xgboost", "conditions": {}},
            ],
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert len(body["runs"]) == 2
    assert body["name"] == "BBBP comparison"
    assert all(run["sweep_id"] == body["sweep_id"] for run in body["runs"])
    assert body["runs"][0]["engine_id"] == "ecfp4-randomforest"


@pytest.mark.asyncio
async def test_unknown_engine_is_404_and_creates_nothing(client, dataset_id) -> None:
    response = await client.post(
        "/api/v1/sweeps",
        json={
            "name": "doomed",
            "dataset_id": str(dataset_id),
            "configs": [{"engine_id": "no-such-engine", "conditions": {}}],
        },
    )
    assert response.status_code == 404
    assert (await client.get("/api/v1/sweeps")).json()["items"] == []


@pytest.mark.asyncio
async def test_empty_configs_is_422(client, dataset_id) -> None:
    response = await client.post(
        "/api/v1/sweeps",
        json={"name": "empty", "dataset_id": str(dataset_id), "configs": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_then_read_then_cancel(client, dataset_id) -> None:
    submitted = (
        await client.post(
            "/api/v1/sweeps",
            json={
                "name": "round trip",
                "dataset_id": str(dataset_id),
                "configs": [{"engine_id": "ecfp4-randomforest", "conditions": {}}],
            },
        )
    ).json()
    sweep_id = submitted["sweep_id"]

    listed = (await client.get("/api/v1/sweeps")).json()["items"]
    assert [item["sweep_id"] for item in listed] == [sweep_id]
    assert listed[0]["total"] == 1

    detail = (await client.get(f"/api/v1/sweeps/{sweep_id}")).json()
    assert len(detail["runs"]) == 1

    assert (await client.post(f"/api/v1/sweeps/{sweep_id}/cancel")).status_code == 204
    after = (await client.get(f"/api/v1/sweeps/{sweep_id}")).json()
    assert after["runs"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_unknown_sweep_is_404(client) -> None:
    assert (await client.get(f"/api/v1/sweeps/{uuid.uuid4()}")).status_code == 404
```

These need `STUDIO_INLINE_JOBS` **off** so the runs stay pending; check how `tests/api/conftest.py` configures the container and follow it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/api/test_sweeps.py -v`
Expected: FAIL — every request 404s, because no router is mounted.

- [ ] **Step 3: Write the router**

Create `backend/src/daikonstudio/interface/routes/sweeps.py`:

```python
"""Sweep endpoints: submit N configs, list the groups, read one, cancel it.

`SweepRunResponse` is the one place `Run.params` is projected onto the wire.
That is deliberate and scoped: a ranked comparison is unreadable without the
engine and conditions each row was produced from, and every alternative --
re-fetching a Protocol per row, or reading N Scorecards -- costs a request per
run to recover something the row already holds.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from daikonstudio.application.execution.sweeps import (
    CancelSweep,
    CancelSweepCommand,
    GetSweep,
    GetSweepQuery,
    ListSweeps,
    ListSweepsQuery,
    MAX_CONFIGS,
    SubmitSweep,
    SubmitSweepCommand,
    SweepConfig,
)
from daikonstudio.application.ports.run_repository import SweepSummary
from daikonstudio.domain.execution.run import Run
from daikonstudio.interface.dependencies._container import use_case
from daikonstudio.interface.dependencies.auth import AuthDep
from daikonstudio.interface.error_handlers import result_to_response

router = APIRouter(prefix="/api/v1/sweeps", tags=["sweeps"])

SubmitSweepDep = Annotated[SubmitSweep, Depends(use_case(SubmitSweep))]
ListSweepsDep = Annotated[ListSweeps, Depends(use_case(ListSweeps))]
GetSweepDep = Annotated[GetSweep, Depends(use_case(GetSweep))]
CancelSweepDep = Annotated[CancelSweep, Depends(use_case(CancelSweep))]


class SweepConfigBody(BaseModel):
    engine_id: str
    conditions: dict[str, Any] = Field(default_factory=dict)


class SubmitSweepBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_id: uuid.UUID
    # min_length=1 here as well as in the use case: an empty sweep is a 422
    # about the request's shape, and saying so at the edge means the client
    # gets a field-level error instead of a domain message.
    configs: list[SweepConfigBody] = Field(min_length=1, max_length=MAX_CONFIGS)
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, Any] = Field(default_factory=dict)


class SweepRunResponse(BaseModel):
    id: uuid.UUID
    sweep_id: uuid.UUID | None
    name: str
    engine_id: str
    conditions: dict[str, Any]
    status: str
    progress: float
    phase: str | None
    error_message: str | None
    protocol_id: uuid.UUID | None
    # None until this run reaches `ready`. The ranked table shows such a row as
    # unranked rather than as zero -- see `Run.record_metrics`.
    metrics: dict[str, Any] | None
    created_at: datetime

    @classmethod
    def from_domain(cls, run: Run) -> SweepRunResponse:
        return cls(
            id=run.id,
            sweep_id=run.sweep_id,
            name=run.params.get("name", ""),
            engine_id=run.params.get("engine_id", ""),
            conditions=run.params.get("conditions", {}),
            status=run.status.value,
            progress=run.progress,
            phase=run.phase,
            error_message=run.error_message,
            protocol_id=run.protocol_id,
            metrics=run.metrics,
            created_at=run.created_at,
        )


class SweepResponse(BaseModel):
    sweep_id: uuid.UUID
    name: str | None
    dataset_id: uuid.UUID | None
    created_at: datetime
    total: int
    by_status: dict[str, int]

    @classmethod
    def from_domain(cls, summary: SweepSummary) -> SweepResponse:
        return cls(
            sweep_id=summary.sweep_id,
            name=summary.name,
            dataset_id=summary.dataset_id,
            created_at=summary.created_at,
            total=summary.total,
            by_status=summary.by_status,
        )


class SweepDetailResponse(BaseModel):
    sweep_id: uuid.UUID
    name: str | None
    dataset_id: uuid.UUID | None
    runs: list[SweepRunResponse]

    @classmethod
    def from_runs(cls, sweep_id: uuid.UUID, runs: list[Run]) -> SweepDetailResponse:
        # Every member carries the same two, written once at submission, so the
        # first one answers for the group without a second query.
        first = runs[0].params
        dataset_id = first.get("dataset_id")
        return cls(
            sweep_id=sweep_id,
            name=first.get("sweep_name"),
            dataset_id=uuid.UUID(dataset_id) if dataset_id else None,
            runs=[SweepRunResponse.from_domain(run) for run in runs],
        )


class SweepListResponse(BaseModel):
    items: list[SweepResponse]


@router.post("", response_model=SweepDetailResponse, status_code=202)
async def submit_sweep(
    body: SubmitSweepBody, auth: AuthDep, service: SubmitSweepDep
) -> SweepDetailResponse:
    command = SubmitSweepCommand(
        name=body.name,
        dataset_id=body.dataset_id,
        configs=[
            SweepConfig(engine_id=config.engine_id, conditions=config.conditions)
            for config in body.configs
        ],
        baseline_engine_id=body.baseline_engine_id,
        baseline_conditions=body.baseline_conditions,
    )
    sweep = result_to_response(await service(command, auth=auth))
    return SweepDetailResponse.from_runs(sweep.sweep_id, sweep.runs)


@router.get("", response_model=SweepListResponse)
async def list_sweeps(
    auth: AuthDep, service: ListSweepsDep, limit: int = 50
) -> SweepListResponse:
    summaries = result_to_response(await service(ListSweepsQuery(limit=limit), auth=auth))
    return SweepListResponse(items=[SweepResponse.from_domain(s) for s in summaries])


@router.get("/{sweep_id}", response_model=SweepDetailResponse)
async def get_sweep(
    sweep_id: uuid.UUID, auth: AuthDep, service: GetSweepDep
) -> SweepDetailResponse:
    runs = result_to_response(await service(GetSweepQuery(sweep_id=sweep_id), auth=auth))
    return SweepDetailResponse.from_runs(sweep_id, runs)


@router.post("/{sweep_id}/cancel", status_code=204)
async def cancel_sweep(
    sweep_id: uuid.UUID, auth: AuthDep, service: CancelSweepDep
) -> Response:
    result_to_response(await service(CancelSweepCommand(sweep_id=sweep_id), auth=auth))
    return Response(status_code=204)
```

Check the exact import paths for `AuthDep`, `use_case`, and `result_to_response` against `interface/routes/runs.py` before running — mirror that file rather than trusting the paths above.

- [ ] **Step 4: Register the use cases and the router**

In `infrastructure/di/container.py`, add the import:

```python
from daikonstudio.application.execution.sweeps import (
    CancelSweep,
    GetSweep,
    ListSweeps,
    SubmitSweep,
)
```

and the registrations, next to the existing `TrainProtocol` one:

```python
    container.define(
        SubmitSweep,
        lambda c: SubmitSweep(_datasets(c), c[EngineRegistry], c[TrainProtocol]),
    )
    container.define(ListSweeps, lambda c: ListSweeps(_runs(c)))
    container.define(GetSweep, lambda c: GetSweep(_runs(c)))
    container.define(CancelSweep, lambda c: CancelSweep(_runs(c)))
```

In `interface/app.py`, import `router as sweeps_router` alongside the others and add, keeping the existing alphabetical order:

```python
    app.include_router(sweeps_router)
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && uv run pytest tests/api/test_sweeps.py -v`
Expected: all PASS.

- [ ] **Step 6: Regenerate the TypeScript client**

Run: `make generate-api`
Expected: `frontend/src/shared/lib/api/model` gains `SweepResponse`, `SweepDetailResponse`, `SweepRunResponse`, `SubmitSweepBody`, `SweepConfigBody`, `SweepListResponse`, and `openapi.json` updates. Commit the regenerated files.

- [ ] **Step 7: Full backend gate**

Run: `make test-all && make lint`
Expected: green, import-linter 3/3.

- [ ] **Step 8: Commit**

```bash
git add backend/src/daikonstudio/interface/routes/sweeps.py \
        backend/src/daikonstudio/infrastructure/di/container.py \
        backend/src/daikonstudio/interface/app.py \
        backend/tests/api/test_sweeps.py \
        frontend/src/shared/lib/api openapi.json
git commit -m "feat(api): sweep endpoints

SweepRunResponse is the one place Run.params reaches the wire: a ranked
comparison is unreadable without the engine and conditions per row, and
every alternative costs a request per run to recover what the row holds.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Frontend data layer and ranking

**Files:**
- Create: `frontend/src/features/sweeps/types/index.ts`
- Create: `frontend/src/features/sweeps/hooks/query-keys.ts`
- Create: `frontend/src/features/sweeps/hooks/use-sweeps.ts`
- Create: `frontend/src/features/sweeps/lib/rank.ts`
- Create: `frontend/src/features/sweeps/lib/rank.test.ts`
- Create: `frontend/src/features/sweeps/index.ts`

**Interfaces:**
- Consumes: the generated `SweepResponse`, `SweepDetailResponse`, `SweepRunResponse` from Task 6.
- Produces: `rankRuns(runs: SweepRun[]): SweepRun[]`; `formatMetric(metrics: SweepRun["metrics"]): string`; `useSweeps()`, `useSweep(id)`, `useSubmitSweep()`, `useCancelSweep()`; `SWEEPS_KEY`, `SWEEP_KEY`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/features/sweeps/lib/rank.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { SweepRun } from "../types";
import { rankRuns } from "./rank";

function run(id: string, metrics: SweepRun["metrics"], status = "ready"): SweepRun {
  return { id, status, metrics } as SweepRun;
}

describe("rankRuns", () => {
  it("ranks a higher MCC first", () => {
    const ranked = rankRuns([
      run("a", { primary_metric: "mcc", value: 0.4, baseline_value: 0.3 }),
      run("b", { primary_metric: "mcc", value: 0.7, baseline_value: 0.3 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["b", "a"]);
  });

  it("ranks a lower RMSE first", () => {
    const ranked = rankRuns([
      run("a", { primary_metric: "rmse", value: 0.9, baseline_value: 1.1 }),
      run("b", { primary_metric: "rmse", value: 0.4, baseline_value: 1.1 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["b", "a"]);
  });

  it("puts unfinished runs last rather than treating them as zero", () => {
    const ranked = rankRuns([
      run("pending", null, "running"),
      run("scored", { primary_metric: "mcc", value: 0.1, baseline_value: 0.3 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["scored", "pending"]);
  });

  it("puts a run whose metric is undefined last, alongside the unfinished", () => {
    const ranked = rankRuns([
      run("undefined-metric", { primary_metric: "mcc", value: null, baseline_value: 0.3 }),
      run("scored", { primary_metric: "mcc", value: 0.1, baseline_value: 0.3 }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["scored", "undefined-metric"]);
  });

  it("keeps submission order among equally unrankable runs", () => {
    const ranked = rankRuns([
      run("first", null, "pending"),
      run("second", null, "pending"),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["first", "second"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm vitest run src/features/sweeps/lib/rank.test.ts`
Expected: FAIL — module `./rank` not found.

- [ ] **Step 3: Implement the ranking**

First create `frontend/src/features/sweeps/types/index.ts`, since `rank.ts` imports from it:

```ts
import type { SweepDetailResponse, SweepResponse, SweepRunResponse } from "@/shared/lib/api/model";

export type Sweep = SweepResponse;
export type SweepDetail = SweepDetailResponse;
export type SweepRun = SweepRunResponse;
```

Then create `frontend/src/features/sweeps/lib/rank.ts`:

```ts
import type { SweepRun } from "../types";

/**
 * Metrics where a smaller number is a better model. The server picks the
 * primary metric (`primary_metric_for` in build_scorecard.py) and sends its
 * name on the row, so this table only has to know the direction -- inferring
 * it from the value would have no way to tell 0.4 RMSE from 0.4 MCC.
 */
const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

/**
 * Best first; anything unrankable last, in submission order.
 *
 * Unrankable is not the same as bad: a run still training has no number yet,
 * and a run whose metric is genuinely undefined (a single-class test split
 * makes every classification metric meaningless) has none either. Sorting
 * those as zero would rank a pending run above a real one on an RMSE sweep
 * and below it on an MCC sweep, which is a ranking that says nothing true.
 */
export function rankRuns(runs: SweepRun[]): SweepRun[] {
  const scored = runs.filter((run) => typeof run.metrics?.value === "number");
  const unscored = runs.filter((run) => typeof run.metrics?.value !== "number");
  scored.sort((a, b) => {
    const lower = LOWER_IS_BETTER.has(String(a.metrics?.primary_metric ?? ""));
    const left = a.metrics?.value as number;
    const right = b.metrics?.value as number;
    return lower ? left - right : right - left;
  });
  return [...scored, ...unscored];
}

/** The headline number, or why there isn't one. */
export function formatMetric(metrics: SweepRun["metrics"]): string {
  if (typeof metrics?.value !== "number") return "—";
  return `${String(metrics.primary_metric).toUpperCase()} ${metrics.value.toFixed(3)}`;
}

/**
 * How far this run beat its own baseline, signed so that positive always means
 * better regardless of the metric's direction. Null when either side is
 * missing -- an unmeasured comparison must not render as a dead heat.
 */
export function baselineDelta(metrics: SweepRun["metrics"]): number | null {
  const value = metrics?.value;
  const baseline = metrics?.baseline_value;
  if (typeof value !== "number" || typeof baseline !== "number") return null;
  return LOWER_IS_BETTER.has(String(metrics?.primary_metric ?? ""))
    ? baseline - value
    : value - baseline;
}
```

- [ ] **Step 4: Run the test**

Run: `cd frontend && pnpm vitest run src/features/sweeps/lib/rank.test.ts`
Expected: 5 tests PASS.

- [ ] **Step 5: Write the hooks**

Create `frontend/src/features/sweeps/hooks/query-keys.ts`:

```ts
export const SWEEPS_KEY = ["sweeps"];
export const SWEEP_KEY = ["sweep"];
```

Create `frontend/src/features/sweeps/hooks/use-sweeps.ts`:

```ts
"use client";

import { isTerminal } from "@/features/runs";
import { API_V1, customInstance } from "@/shared/lib/api/custom-instance";
import type {
  SweepDetailResponse,
  SweepListResponse,
} from "@/shared/lib/api/model";
import { RUN_POLL_MS } from "@/shared/lib/query-defaults";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { SWEEPS_KEY, SWEEP_KEY } from "./query-keys";

export function useSweeps() {
  return useQuery({
    queryKey: SWEEPS_KEY,
    queryFn: () =>
      customInstance<SweepListResponse>({ url: `${API_V1}/sweeps`, method: "GET" }),
  });
}

/**
 * Polls until every member is terminal. A sweep is throttled by the
 * per-workspace concurrency cap, so the tail of a large one keeps arriving
 * long after the first few finish -- stopping at the first terminal run would
 * freeze the page mid-comparison.
 */
export function useSweep(id: string | undefined) {
  return useQuery({
    queryKey: [...SWEEP_KEY, id],
    queryFn: () =>
      customInstance<SweepDetailResponse>({ url: `${API_V1}/sweeps/${id}`, method: "GET" }),
    enabled: Boolean(id),
    refetchInterval: (query) => {
      const runs = query.state.data?.runs ?? [];
      return runs.length > 0 && runs.every((run) => isTerminal(run.status))
        ? false
        : RUN_POLL_MS;
    },
  });
}

export function useSubmitSweep() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      name: string;
      dataset_id: string;
      configs: { engine_id: string; conditions: Record<string, unknown> }[];
      baseline_engine_id?: string | null;
      baseline_conditions?: Record<string, unknown>;
    }) =>
      customInstance<SweepDetailResponse>({ url: `${API_V1}/sweeps`, method: "POST", data }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: SWEEPS_KEY }),
  });
}

export function useCancelSweep() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      customInstance<void>({ url: `${API_V1}/sweeps/${id}/cancel`, method: "POST" }),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: SWEEPS_KEY });
      queryClient.invalidateQueries({ queryKey: [...SWEEP_KEY, id] });
    },
  });
}
```

Create `frontend/src/features/sweeps/index.ts`:

```ts
export type { Sweep, SweepDetail, SweepRun } from "./types";
export { SWEEPS_KEY, SWEEP_KEY } from "./hooks/query-keys";
export { useCancelSweep, useSubmitSweep, useSweep, useSweeps } from "./hooks/use-sweeps";
export { baselineDelta, formatMetric, rankRuns } from "./lib/rank";
```

- [ ] **Step 6: Lint and test**

Run: `cd frontend && pnpm biome check --write src/features/sweeps && pnpm vitest run`
Expected: biome clean, all frontend tests pass (51 baseline + 5 new).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/features/sweeps
git commit -m "feat(sweeps-ui): data layer and direction-aware ranking

Unfinished runs and genuinely-undefined metrics sort last rather than as
zero -- zero would rank a pending run above a real one on an RMSE sweep
and below it on an MCC one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Sweeps list and detail pages

**Files:**
- Create: `frontend/src/features/sweeps/components/sweep-list.tsx`
- Create: `frontend/src/features/sweeps/components/sweep-detail.tsx`
- Create: `frontend/src/app/(dashboard)/sweeps/page.tsx`
- Create: `frontend/src/app/(dashboard)/sweeps/[id]/page.tsx`
- Modify: `frontend/src/features/sweeps/index.ts`
- Modify: `frontend/src/shared/lib/navigation.ts`

**Interfaces:**
- Consumes: `useSweeps`, `useSweep`, `useCancelSweep`, `rankRuns`, `formatMetric`, `baselineDelta` (Task 7).
- Produces: `SweepList`, `SweepDetail` components exported from `features/sweeps`.

- [ ] **Step 1: Add the nav entry**

In `frontend/src/shared/lib/navigation.ts`, add `Layers` to the lucide-react import and one item to the **Curate** group, after Protocols (training lives in Curate):

```ts
      { title: "Sweeps", href: "/sweeps", icon: Layers },
```

- [ ] **Step 2: Build the list**

Create `frontend/src/features/sweeps/components/sweep-list.tsx`. Read `frontend/src/features/runs/components/run-list.tsx` first and match its table, empty state, skeleton, and status-pill idioms exactly rather than inventing a second visual language. The list shows one row per sweep: name, when, total, and a compact per-status breakdown from `by_status`, linking to `/sweeps/{sweep_id}`. Provide an empty state pointing at `/sweeps/new`.

- [ ] **Step 3: Build the detail**

Create `frontend/src/features/sweeps/components/sweep-detail.tsx`. Take the card, table, skeleton, and status-pill chrome from `run-detail.tsx` and `run-list.tsx`; this is the logic it wraps:

```tsx
"use client";

import { RUN_STATUS_COPY, isTerminal } from "@/features/runs";
// Deep imports, not the feature barrel: `index.ts` re-exports this component,
// so importing from it here would be a cycle.
import { useCancelSweep, useSweep } from "../hooks/use-sweeps";
import { baselineDelta, formatMetric, rankRuns } from "../lib/rank";

export function SweepDetail({ id }: { id: string }) {
  const { data: sweep, isPending } = useSweep(id);
  const cancel = useCancelSweep();

  if (isPending || !sweep) return <SweepDetailSkeleton />;

  const ranked = rankRuns(sweep.runs);
  const live = sweep.runs.filter((run) => !isTerminal(run.status));

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{sweep.name ?? "Sweep"}</h1>
          {/*
            The per-workspace cap (STUDIO_WORKSPACE_MAX_ACTIVE_RUNS, default 10)
            means a large sweep drains in waves. Saying so is the difference
            between a fairness predicate doing its job and a product that looks
            broken. Only shown while it can actually bite.
          */}
          {live.length > 10 && (
            <p className="text-sm text-muted-foreground">
              A workspace runs 10 at a time, so this sweep finishes in waves.
            </p>
          )}
        </div>
        {live.length > 0 && (
          <Button
            variant="outline"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate(id)}
          >
            Cancel sweep
          </Button>
        )}
      </header>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>#</TableHead>
            <TableHead>Config</TableHead>
            <TableHead>Engine</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>vs baseline</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {ranked.map((run, index) => {
            const delta = baselineDelta(run.metrics);
            return (
              <TableRow key={run.id}>
                {/* Rank, not submission index -- the table is sorted, and
                    numbering it by position is the whole point. */}
                <TableCell>{run.metrics?.value == null ? "—" : index + 1}</TableCell>
                <TableCell>
                  {run.protocol_id ? (
                    <Link href={`/protocols/${run.protocol_id}`}>{run.name}</Link>
                  ) : (
                    run.name
                  )}
                </TableCell>
                <TableCell>{run.engine_id}</TableCell>
                <TableCell>{RUN_STATUS_COPY[run.status] ?? run.status}</TableCell>
                <TableCell>{formatMetric(run.metrics)}</TableCell>
                <TableCell>
                  {delta === null
                    ? "—"
                    : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
```

Import `Button`, `Table*`, and `Link` from wherever `run-list.tsx` imports them. Add a conditions cell if the existing table styling leaves room; the engine id alone is ambiguous when two configs share an engine, so at minimum render `run.conditions` as a compact key=value string in the Config cell's second line.

Reuse `RUN_STATUS_COPY` from `@/features/runs` rather than redeclaring the mapping.

- [ ] **Step 4: Wire the pages**

Create `frontend/src/app/(dashboard)/sweeps/page.tsx` and `frontend/src/app/(dashboard)/sweeps/[id]/page.tsx`, matching the structure of `src/app/(dashboard)/runs/page.tsx` and `src/app/(dashboard)/runs/[id]/page.tsx` (including how they read route params and set page headings). Export `SweepList` and `SweepDetail` from `features/sweeps/index.ts`.

- [ ] **Step 5: Verify in the browser**

Run: `make dev`, then open `http://localhost:3003/sweeps`.
Expected: the page renders with its empty state, the nav entry appears under Curate, and the ⌘K palette finds "Sweeps" (all three come from the single `navigation.ts` entry).

- [ ] **Step 6: Lint and test**

Run: `cd frontend && pnpm biome check --write src && pnpm vitest run`
Expected: clean, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/features/sweeps frontend/src/app/\(dashboard\)/sweeps \
        frontend/src/shared/lib/navigation.ts
git commit -m "feat(sweeps-ui): ranked list and detail pages

The detail page says a workspace runs 10 at a time -- without it a
throttled 20-run sweep reads as the product being stuck rather than the
fairness cap doing its job.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Submit form

**Files:**
- Create: `frontend/src/features/sweeps/components/sweep-form.tsx`
- Create: `frontend/src/features/sweeps/components/sweep-form.test.tsx`
- Create: `frontend/src/app/(dashboard)/sweeps/new/page.tsx`
- Modify: `frontend/src/features/sweeps/index.ts`

**Interfaces:**
- Consumes: `useSubmitSweep` (Task 7); `ConditionFields` and `resolveConditions` from `@/features/protocols`; `useEngines`, `enginesForTargetKind` from `@/features/engines`; `useDatasets`, `useDataset` from `@/features/datasets`.
- Produces: `SweepForm` exported from `features/sweeps`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/features/sweeps/components/sweep-form.test.tsx`, matching the render helpers and mocking style of `frontend/src/features/protocols/components/train-protocol-form.test.tsx` — read it first and reuse its query-client wrapper rather than building another.

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SweepForm } from "./sweep-form";

describe("SweepForm", () => {
  it("starts with one config row", () => {
    render(<SweepForm />, { wrapper: Wrapper });
    expect(screen.getAllByTestId("sweep-config-row")).toHaveLength(1);
  });

  it("adds and removes config rows", () => {
    render(<SweepForm />, { wrapper: Wrapper });
    fireEvent.click(screen.getByRole("button", { name: /add config/i }));
    expect(screen.getAllByTestId("sweep-config-row")).toHaveLength(2);
    fireEvent.click(screen.getAllByRole("button", { name: /remove config/i })[0]);
    expect(screen.getAllByTestId("sweep-config-row")).toHaveLength(1);
  });

  it("will not remove the last config row", () => {
    render(<SweepForm />, { wrapper: Wrapper });
    expect(screen.queryByRole("button", { name: /remove config/i })).toBeNull();
  });
});
```

Define `Wrapper` exactly as the protocols form test defines its own.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm vitest run src/features/sweeps/components/sweep-form.test.tsx`
Expected: FAIL — `./sweep-form` not found.

- [ ] **Step 3: Build the form**

Create `frontend/src/features/sweeps/components/sweep-form.tsx`, modelled on `train-protocol-form.tsx` — take its dataset select, engine select, `ConditionFields` usage, and card layout from there. This is the state and submit logic it wraps:

```tsx
"use client";

import { useDataset, useDatasets } from "@/features/datasets";
import { enginesForTargetKind, useEngines } from "@/features/engines";
import { resolveConditions } from "@/features/protocols";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useSubmitSweep } from "../hooks/use-sweeps";

interface ConfigRow {
  engineId: string;
  conditions: Record<string, unknown>;
}

export function SweepForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [baselineEngineId, setBaselineEngineId] = useState("");
  const [baselineConditions, setBaselineConditions] = useState<Record<string, unknown>>({});
  const [configs, setConfigs] = useState<ConfigRow[]>([{ engineId: "", conditions: {} }]);

  const datasets = useDatasets();
  const engines = useEngines();
  const { data: dataset } = useDataset(datasetId || undefined);
  const submit = useSubmitSweep();

  const available = dataset ? enginesForTargetKind(engines.data ?? [], dataset.target.kind) : [];
  const specsFor = (engineId: string) =>
    available.find((engine) => engine.id === engineId)?.conditions ?? [];

  function updateConfig(index: number, patch: Partial<ConfigRow>) {
    setConfigs((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  async function onSubmit() {
    const response = await submit.mutateAsync({
      name,
      dataset_id: datasetId,
      baseline_engine_id: baselineEngineId || null,
      baseline_conditions: resolveConditions(specsFor(baselineEngineId), baselineConditions),
      // Resolved against manifest defaults, exactly as the train form does.
      // Sending raw form state would make `{}` and `{n_estimators: 500}` two
      // different requests for work the server resolves identically.
      configs: configs.map((row) => ({
        engine_id: row.engineId,
        conditions: resolveConditions(specsFor(row.engineId), row.conditions),
      })),
    });
    router.push(`/sweeps/${response.sweep_id}`);
  }

  return (
    <form onSubmit={(event) => { event.preventDefault(); void onSubmit(); }}>
      {/* name / dataset / baseline fields -- copy from train-protocol-form.tsx */}

      {configs.map((row, index) => (
        <div key={index} data-testid="sweep-config-row">
          {/* engine select for `available`, then <ConditionFields> for
              specsFor(row.engineId), writing back through updateConfig */}
          {configs.length > 1 && (
            <Button
              type="button"
              variant="ghost"
              aria-label="Remove config"
              onClick={() => setConfigs((rows) => rows.filter((_, i) => i !== index))}
            >
              <XIcon aria-hidden />
            </Button>
          )}
        </div>
      ))}

      <Button
        type="button"
        variant="outline"
        onClick={() => setConfigs((rows) => [...rows, { engineId: "", conditions: {} }])}
      >
        Add config
      </Button>
      <Button type="submit" disabled={submit.isPending || !datasetId || !name}>
        Submit sweep
      </Button>
    </form>
  );
}
```

Check `enginesForTargetKind`'s real signature and `resolveConditions`'s export path before running — `resolveConditions` currently lives in `features/protocols/components/train-protocol-form.tsx` and may need adding to that feature's `index.ts` barrel. Do that rather than copying the function.

Give the remove buttons a real accessible name (`aria-label="Remove config"`), not an icon alone — and while you are in this area, the lane pills on the Runners page still lack `aria-pressed` (noted in the handoff); do not fix that here, it belongs to a Runners change.

- [ ] **Step 4: Run the test**

Run: `cd frontend && pnpm vitest run src/features/sweeps/components/sweep-form.test.tsx`
Expected: 3 tests PASS.

- [ ] **Step 5: Wire the page**

Create `frontend/src/app/(dashboard)/sweeps/new/page.tsx` rendering `SweepForm`, matching `src/app/(dashboard)/protocols/new/page.tsx`. Export `SweepForm` from `features/sweeps/index.ts`.

- [ ] **Step 6: Lint and test**

Run: `cd frontend && pnpm biome check --write src && pnpm vitest run && pnpm tsc --noEmit`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/features/sweeps frontend/src/app/\(dashboard\)/sweeps
git commit -m "feat(sweeps-ui): submit form

Reuses the engine picker and ConditionFields from the train form rather
than reimplementing condition rendering, and resolves conditions against
manifest defaults so {} and the default value are one request, not two.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Gates and live verification

**Files:** none created; this task proves the feature works against the real stack.

**Interfaces:**
- Consumes: everything above.
- Produces: a verified end-to-end sweep, and a handoff note recording what was and was not checked.

- [ ] **Step 1: Run every gate**

Run: `make test-all && make lint && make lint-fe && make test-fe`
Expected: backend suite green with import-linter 3/3, ruff and mypy clean, biome clean, frontend tests green. Record the actual numbers — do not claim a count you did not read.

- [ ] **Step 2: Restart everything, including the runner agents**

Run: `make dev`
The agents hold `application/execution/` in memory; without a restart a sweep will execute the pre-Task-2 `RunTraining` and every run's `metrics` will be NULL.

- [ ] **Step 3: Submit a real sweep through the browser**

At `http://localhost:3003/sweeps/new`, submit three configs on the BBBP dataset — `ecfp4-randomforest` at two different `n_estimators`, and `ecfp4-xgboost` at defaults — against the default baseline. Keep it to fast CPU engines; a chemprop sweep is bounded by the single server-wide `STUDIO_WORKER_JOB_TIMEOUT` with no per-lane override.

Verify, in order:
1. All three runs appear immediately as pending under one sweep.
2. The detail page polls and shows progress without a manual refresh.
3. Each finished run shows a metric and a signed baseline delta, best first.
4. Each finished run links to a Scorecard whose primary metric matches the ranked table's number for that row. A mismatch means `primary_metric_for` is not actually shared.
5. `SELECT sweep_id, metrics FROM runs WHERE sweep_id IS NOT NULL` shows populated `metrics` — this is the proof the runner wire path works, since the browser path uses real agents rather than `InlineEnqueuer`.

- [ ] **Step 4: Verify the cancel cascade against a live sweep**

Submit a second sweep of six configs, wait until some are running and others still pending, then cancel the sweep. Confirm the pending ones go straight to cancelled, the running ones stop at their next checkpoint rather than instantly, and any already-finished run keeps its `ready` status. Then press cancel again and confirm it returns 204 rather than an error.

- [ ] **Step 5: Record what happened**

Update `docs/superpowers/HANDOFF-fanout-sweeps.md` — or write its successor — with the state after this plan: what was verified live, what was not, and any new deferred items. Carry forward the still-open ones that this plan did not touch: `Dockerfile.gpu` has never been built, per-lane job timeouts do not exist, the reverse-proxy `merge_slashes` caveat before the first non-local runner, the semi-trusted runner boundary, and the Runners page's missing `aria-pressed` on the lane pills.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/HANDOFF-fanout-sweeps.md
git commit -m "docs: handoff after fan-out sweeps

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

**Do not** teach the queue, the runner protocol, or the runner agents about sweeps. `_CLAIM` in `infrastructure/persistence/sqlalchemy/execution/queue.py` is inside an adversarially-reviewed security boundary, and a sweep needs nothing from it. The only runner-facing change in this whole plan is one optional `metrics` field on an existing update envelope.

**Do not** put the sweep id in `Run.params`. `SqlAlchemyRunRepository.update` deliberately never persists `params`, so a grouping stored there could never be corrected, and it could not use an index.

**Do not** "fix" the per-workspace concurrency cap because a sweep felt slow. `STUDIO_WORKSPACE_MAX_ACTIVE_RUNS` defaulting to 10 is the fairness predicate; raising it lets one user's sweep starve the instance. Task 8 makes it visible instead.

**Expect** to re-run `make dev-worker` and `make dev-worker-gpu` after Tasks 2, 4, and 5. This has already cost one full debugging detour in this codebase.

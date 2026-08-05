# Self-Hosted Runners (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace arq/Valkey with a Postgres-backed job queue plus an HTTPS runner protocol, so any machine with a token can execute training/prediction runs — including external collaborators' GPU boxes — and delete Valkey from the stack.

**Architecture:** The queue is the `runs` table: a `pending` run with a `lane` set is claimable. Runner agents authenticate with per-runner bearer tokens and speak `/api/v1/runner/*`: claim (SKIP LOCKED + lease), read inputs, write progress, upload blobs, report terminal status. `RunTraining`/`RunPrediction` execute unmodified on the runner because the agent injects HTTP-backed implementations of the four ports they consume. Spec: `docs/superpowers/specs/2026-08-04-self-hosted-runners-design.md`.

**Tech Stack:** FastAPI, SQLAlchemy async + asyncpg, alembic, lagom, returns (Result), httpx, pydantic v2, Next.js + TanStack Query + orval-generated client.

## Global Constraints

- Layering (enforced by import-linter in `make test`): `domain` imports nothing internal; `application` imports domain; `infrastructure` imports application+domain; `interface` may import all. New wire models live in `infrastructure/runner/wire.py` so both `interface/routes/runner_api.py` and the HTTP port adapters can import them.
- Runner protocol routes are prefixed `/api/v1/runner` and MUST be added to `sentinel.protect(app, exclude_paths=[...])` in `interface/app.py` — the SDK does prefix matching (`path == p or path.startswith(p + "/")`, verified in sentinel_auth/middleware.py:111).
- Run statuses are `pending|running|ready|failed|cancelled` (no "queued" status exists). Claimable = `status='pending' AND lane IS NOT NULL AND claimed_by IS NULL AND attempts < max_attempts`.
- Tokens: `f"drt_{secrets.token_urlsafe(32)}"`, stored as `hashlib.sha256(token.encode()).hexdigest()` (64 hex chars). Plaintext returned exactly once, at creation.
- A runner NEVER writes status `cancelled` (cancel is user-initiated); the server rejects it.
- `OMP_NUM_THREADS=1` on every process that trains (Makefile already documents why; keep it on the new runner targets).
- New settings (in `Settings`): `workspace_max_active_runs: int = 10`, `runner_lease_seconds: int = 600`, `runner_max_attempts: int = 3`, `runner_upload_max_bytes: int = 1_073_741_824`, `runner_online_threshold_seconds: int = 15`. Deleted settings: `redis_url`, `worker_lane`, `worker_max_jobs`. Kept: `worker_job_timeout` (now served to runners in the claim response), `inline_jobs`.
- Every use case returns `returns.result.Result` and routes unwrap with `result_to_response` — copy the pattern from `interface/routes/runs.py`.
- Commit after every task. `make dev` workers are broken from Task 4 until Task 10 — use `STUDIO_INLINE_JOBS=1` in `backend/.env` meanwhile.
- Backend tests: `cd backend && set -a && . ./.env && set +a && env OMP_NUM_THREADS=1 uv run pytest <path> -v` (integration tests need Docker running for testcontainers).

---

### Task 1: Migration 009 — runners table + queue columns on runs

**Files:**
- Create: `backend/alembic/versions/009_runners_and_queue.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/runners/__init__.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/runners/models.py`
- Modify: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/models.py` (add 4 columns to `RunModel`)
- Test: `backend/tests/integration/test_migrations.py` (existing file — add assertions if it verifies schema; otherwise the alembic upgrade in the root conftest is the test)

**Interfaces:**
- Produces: `runners` table (`id` Uuid PK, `name` String(128) unique not null, `lanes` JSONB not null, `token_hash` String(64) unique not null, `created_at`/`updated_at` tz timestamps server_default now(), `last_seen_at` tz nullable, `revoked_at` tz nullable, `version` Integer not null). `runs` gains `lane` String(32) nullable, `claimed_by` Uuid nullable, `lease_expires_at` tz nullable, `attempts` Integer not null server_default '0'; partial index `ix_runs_claimable` on `(lane, created_at)` WHERE `status = 'pending'`.
- Produces: `RunnerModel` (SQLAlchemy) in `runners/models.py`; extended `RunModel`.

- [ ] **Step 1: Write the migration**

```python
"""runners + queue columns

The runs table becomes the job queue: a pending run with a lane set is
claimable by a registered runner. Runners are instance-level (no
workspace_id) -- see the 2026-08-04 self-hosted-runners spec.

Revision ID: 009
Revises: 008
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "runners",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("lanes", JSONB(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_runners_name"),
        sa.UniqueConstraint("token_hash", name="uq_runners_token_hash"),
    )

    op.add_column("runs", sa.Column("lane", sa.String(length=32), nullable=True))
    op.add_column("runs", sa.Column("claimed_by", sa.Uuid(), nullable=True))
    op.add_column("runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "runs", sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False)
    )
    op.create_index(
        "ix_runs_claimable",
        "runs",
        ["lane", "created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    # Cutover backfill. arq's in-flight queue disappears with Valkey, so:
    # pending runs become claimable on the default lane (locally the gpu lane
    # ran on CPU anyway; a wrong-lane pending run at cutover re-runs slower,
    # not wrongly), and running runs get an already-expired lease so the first
    # sweep requeues them instead of leaving them RUNNING forever.
    op.execute("UPDATE runs SET lane = 'default' WHERE status IN ('pending', 'running')")
    op.execute("UPDATE runs SET lease_expires_at = now() WHERE status = 'running'")


def downgrade() -> None:
    op.drop_index("ix_runs_claimable", table_name="runs")
    op.drop_column("runs", "attempts")
    op.drop_column("runs", "lease_expires_at")
    op.drop_column("runs", "claimed_by")
    op.drop_column("runs", "lane")
    op.drop_table("runners")
```

- [ ] **Step 2: Add `RunnerModel` and extend `RunModel`**

`runners/models.py` (mirror the mixin usage in `execution/models.py` — note: NO `WorkspaceIdMixin`, runners are instance-level):

```python
"""Runner rows -- registered machines allowed to claim runs. Instance-level:
no workspace_id, a runner serves lanes, and lanes cross workspaces."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from daikonstudio.infrastructure.persistence.sqlalchemy.base import (
    Base,
    EntityModelMixin,
    VersionMixin,
)


class RunnerModel(Base, EntityModelMixin, VersionMixin):
    __tablename__ = "runners"

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    lanes: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("name", name="uq_runners_name"),
        UniqueConstraint("token_hash", name="uq_runners_token_hash"),
    )
```

Check `base.py` first: if `EntityModelMixin` bundles `workspace_id`, use the individual mixins so runners get `id`/`created_at`/`updated_at` WITHOUT `workspace_id` (the agent-verified layout has `WorkspaceIdMixin` separate, so this should be direct).

In `execution/models.py`, add to `RunModel` after `error_message`:

```python
    # Queue columns -- see the 2026-08-04 self-hosted-runners spec. `lane` is
    # NULL until the enqueuer sets it; only laned pending runs are claimable.
    lane: Mapped[str | None] = mapped_column(String(32), nullable=True)
    claimed_by: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
```

(Add the missing `DateTime`/`Integer`/`datetime` imports. Do NOT add these fields to the domain `Run` — queue mechanics stay out of the aggregate; `SqlAlchemyRunRepository._to_model` must therefore not set them and `update()` must not overwrite them, which is already true since both enumerate columns explicitly.)

- [ ] **Step 3: Run migrations against the test DB to verify**

Run: `cd backend && set -a && . ./.env && set +a && env OMP_NUM_THREADS=1 uv run pytest tests/integration/test_migrations.py -v`
Expected: PASS (root conftest upgrades to head; a failing migration fails collection/setup)

- [ ] **Step 4: Commit**

```bash
git add backend/alembic/versions/009_runners_and_queue.py backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/
git commit -m "feat: runners table + queue columns on runs (migration 009)"
```

---

### Task 2: Runner domain aggregate + repository

**Files:**
- Create: `backend/src/daikonstudio/domain/runners/__init__.py`
- Create: `backend/src/daikonstudio/domain/runners/runner.py`
- Create: `backend/src/daikonstudio/application/ports/runner_repository.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/runners/repository.py`
- Test: `backend/tests/unit/runners/test_runner.py`, `backend/tests/integration/test_runner_repository.py`

**Interfaces:**
- Produces: `Runner(AggregateRoot)` with kwargs `name: str, lanes: tuple[str, ...], token_hash: str, last_seen_at: datetime | None = None, revoked_at: datetime | None = None, id/created_at/updated_at/version` (same entity-kwargs pattern as `Run.__init__`). Property `is_revoked: bool`. Method `revoke() -> None` (idempotent — keeps the first `revoked_at`).
- Produces: port `RunnerRepository(Protocol)`:

```python
class RunnerRepository(Protocol):
    async def add(self, runner: Runner) -> None: ...
    async def get(self, runner_id: uuid.UUID) -> Runner | None: ...
    async def get_by_token_hash(self, token_hash: str) -> Runner | None: ...
    async def list(self) -> list[Runner]: ...
    async def touch_last_seen(self, runner_id: uuid.UUID) -> None: ...
    async def revoke(self, runner_id: uuid.UUID) -> None: ...
```

- Produces: `SqlAlchemyRunnerRepository(sessions: async_sessionmaker)` implementing it. `touch_last_seen`/`revoke` are targeted `UPDATE ... SET last_seen_at|revoked_at = now()` statements (no version bump — a heartbeat is not an edit; add a one-line comment saying so). `revoke` only sets `revoked_at` when it is NULL (idempotent). `add` maps `lanes` tuple→list for JSONB; domain reconstruction maps back to tuple.

- [ ] **Step 1: Write failing unit tests for the aggregate**

```python
import uuid
from datetime import UTC, datetime

from daikonstudio.domain.runners.runner import Runner


def _runner(**overrides):
    defaults = dict(name="gpu-01", lanes=("gpu",), token_hash="a" * 64)
    defaults.update(overrides)
    return Runner(**defaults)


def test_new_runner_is_not_revoked() -> None:
    assert _runner().is_revoked is False


def test_revoke_sets_timestamp_once() -> None:
    runner = _runner()
    runner.revoke()
    first = runner.revoked_at
    assert runner.is_revoked and first is not None
    runner.revoke()
    assert runner.revoked_at == first  # idempotent, keeps the original instant


def test_lanes_are_a_tuple() -> None:
    assert _runner(lanes=("default", "gpu")).lanes == ("default", "gpu")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/unit/runners/ -v` → FAIL (module not found)

- [ ] **Step 3: Implement `Runner`** (follow `Run.__init__`'s shape; `revoke()` sets `self.revoked_at = datetime.now(UTC)` only if currently None, then `_touch()` like Run does)

- [ ] **Step 4: Unit tests pass** — `uv run pytest tests/unit/runners/ -v`

- [ ] **Step 5: Write failing repository integration test**

In `tests/integration/test_runner_repository.py`, mirror the fixture style of the existing `tests/integration/test_run_repository.py` (read it first; reuse its sessionmaker fixture approach). Cover: `add` + `get` roundtrip preserves all fields incl. lanes tuple; `get_by_token_hash` finds it and returns `None` for unknown hash; `list` returns all; `touch_last_seen` sets a timestamp visible on re-`get`; `revoke` sets `revoked_at`, is idempotent, and duplicate `name` on `add` raises (assert `IntegrityError` or the mapped `ConflictError`, whichever the repo does — map `IntegrityError` to `daikonstudio.domain.shared.errors.ConflictError` inside `add`, matching how dataset duplicates are surfaced).

- [ ] **Step 6: Run to verify failure**, **Step 7: Implement `SqlAlchemyRunnerRepository`** (mirror `execution/repository.py`'s `_to_domain`/`_to_model` symmetry), **Step 8: Tests pass**

Run: `uv run pytest tests/integration/test_runner_repository.py -v`

- [ ] **Step 9: Commit** — `git commit -m "feat: Runner aggregate + repository"`

---

### Task 3: RunQueue port + SKIP LOCKED claim implementation

**Files:**
- Create: `backend/src/daikonstudio/application/ports/run_queue.py`
- Create: `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/execution/queue.py`
- Test: `backend/tests/integration/test_run_queue.py`

**Interfaces:**
- Produces: port `RunQueue(Protocol)`:

```python
class RunQueue(Protocol):
    async def sweep(self, *, max_attempts: int) -> None:
        """Requeue expired leases; fail runs that exhausted their attempts."""
        ...

    async def claim_next(
        self,
        *,
        runner_id: uuid.UUID,
        lanes: Sequence[str],
        lease_seconds: int,
        max_active_per_workspace: int,
        max_attempts: int,
    ) -> uuid.UUID | None: ...

    async def verify_claim(
        self,
        run_id: uuid.UUID,
        *,
        runner_id: uuid.UUID,
        lease_seconds: int,
        require_active: bool,
    ) -> bool:
        """True iff `runner_id` holds the claim on `run_id`; extends the lease
        as a side effect (every authenticated run-scoped call is a heartbeat).
        With require_active=True, also demands status pending/running -- used
        by write endpoints so nothing mutates a terminal run's satellites."""
        ...

    async def set_lane(self, run_id: uuid.UUID, lane: str) -> None: ...

    async def active_run_by_runner(self) -> dict[uuid.UUID, uuid.UUID]:
        """runner_id -> the run it currently holds (claimed pending or running)."""
        ...
```

- Produces: `SqlAlchemyRunQueue(sessions: async_sessionmaker)` implementing it with `sqlalchemy.text()` statements.

- [ ] **Step 1: Write failing integration tests** (this is the highest-risk logic in the plan — test it hard; use the same sessionmaker fixture pattern as `tests/integration/test_run_repository.py`, inserting runs directly via `RunModel`)

Test cases, each a separate function:

```python
# helpers: _pending_run(lane="default", workspace_id=None, attempts=0) inserts a
# RunModel with status "pending", minimal valid fields, returns its id.

async def test_claim_returns_oldest_matching_lane(queue, session_factory): ...
    # two pending default-lane runs + one gpu -> claim(lanes=["default"]) returns the older default

async def test_claim_skips_other_lanes(queue, ...): ...
    # only gpu pending -> claim(lanes=["default"]) returns None

async def test_claim_sets_claimant_lease_and_attempts(queue, ...): ...
    # after claim: claimed_by == runner_id, lease_expires_at ~ now+lease, attempts == 1

async def test_claimed_run_is_not_claimable_again(queue, ...): ...

async def test_two_concurrent_claims_get_different_runs(queue, ...): ...
    # two pending runs; run claim_next twice concurrently via asyncio.gather on
    # two separate sessions -> the two returned ids differ (SKIP LOCKED)

async def test_sweep_requeues_expired_lease(queue, ...): ...
    # claimed + running with lease_expires_at in the past -> after sweep:
    # status pending, claimed_by NULL, lease NULL, progress 0, phase NULL

async def test_sweep_fails_run_after_max_attempts(queue, ...): ...
    # pending, claimed_by NULL, attempts == 3 -> after sweep(max_attempts=3):
    # status failed, error_message mentions the lease

async def test_workspace_cap_blocks_claim(queue, ...): ...
    # cap=1, one RUNNING run in workspace W, one pending in W, one pending in W2
    # -> claim returns the W2 run, not the W one

async def test_lane_null_is_never_claimable(queue, ...): ...

async def test_verify_claim_true_for_claimant_and_extends_lease(queue, ...): ...
async def test_verify_claim_false_for_other_runner(queue, ...): ...
async def test_verify_claim_require_active_false_on_terminal(queue, ...): ...
async def test_set_lane_makes_run_claimable(queue, ...): ...
async def test_cancelled_pending_run_is_not_claimed(queue, ...): ...
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/integration/test_run_queue.py -v`

- [ ] **Step 3: Implement `SqlAlchemyRunQueue`**

```python
"""The runs table as a job queue. Raw SQL by design: SKIP LOCKED, a partial
index, and make_interval are Postgres idioms, and hiding them behind the ORM
would only obscure the one query whose exact shape is the point."""

_SWEEP_REQUEUE = text("""
    UPDATE runs
    SET status = 'pending', claimed_by = NULL, lease_expires_at = NULL,
        progress = 0.0, phase = NULL, updated_at = now()
    WHERE lease_expires_at < now() AND status IN ('pending', 'running')
""")

_SWEEP_FAIL = text("""
    UPDATE runs
    SET status = 'failed', updated_at = now(),
        error_message = 'runner lease expired after ' || attempts || ' attempts'
    WHERE status = 'pending' AND claimed_by IS NULL AND lane IS NOT NULL
      AND attempts >= :max_attempts
""")

# ponytail: the fairness cap is a correlated count per candidate row -- fine
# while the pending queue is small; switch to a windowed CTE if it ever isn't.
_CLAIM = text("""
    WITH candidate AS (
        SELECT r.id FROM runs r
        WHERE r.status = 'pending' AND r.claimed_by IS NULL
          AND r.lane = ANY(:lanes) AND r.attempts < :max_attempts
          AND (SELECT count(*) FROM runs a
               WHERE a.workspace_id = r.workspace_id AND a.status = 'running')
              < :cap
        ORDER BY r.created_at
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    UPDATE runs
    SET claimed_by = :runner_id, attempts = attempts + 1, updated_at = now(),
        lease_expires_at = now() + make_interval(secs => :lease_seconds)
    FROM candidate WHERE runs.id = candidate.id
    RETURNING runs.id
""")

_VERIFY = text("""
    UPDATE runs
    SET lease_expires_at = now() + make_interval(secs => :lease_seconds)
    WHERE id = :run_id AND claimed_by = :runner_id
      AND (NOT :require_active OR status IN ('pending', 'running'))
    RETURNING id
""")
```

`claim_next` runs sweep's two statements first, then `_CLAIM`, in one session/transaction; returns the id or None. `set_lane` is `UPDATE runs SET lane = :lane, updated_at = now() WHERE id = :run_id`. `active_run_by_runner` is `SELECT DISTINCT ON (claimed_by) claimed_by, id FROM runs WHERE claimed_by IS NOT NULL AND status IN ('pending','running') ORDER BY claimed_by, created_at DESC`. Bind `:lanes` as a list (asyncpg handles `= ANY($1::varchar[])`; if the text() binding fights you, use `sqlalchemy.bindparam("lanes", expanding=True)` with `r.lane IN :lanes` instead — equivalent semantics).

- [ ] **Step 4: Tests pass** — `uv run pytest tests/integration/test_run_queue.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat: RunQueue port + SKIP LOCKED claim/lease/sweep"`

---

### Task 4: Kill arq — jobs module with port-based ctx, DbEnqueuer

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/jobs.py`
- Delete: `backend/src/daikonstudio/infrastructure/worker.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`
- Modify: `backend/tests/unit/execution/test_worker.py` → rename `test_jobs.py`
- Delete: `backend/tests/unit/execution/test_lanes.py` (its subject — arq queue names — no longer exists; lane routing is now covered by Task 3's `test_claim_skips_other_lanes`/`test_set_lane_makes_run_claimable`)

**Interfaces:**
- Produces: `infrastructure/jobs.py` with:
  - `run_job(ctx: dict[str, Any], run_id: uuid.UUID) -> None` — same body/semantics as today's `worker.py::run_job` (ConflictError-on-start drop, RunInterrupted handling, FAILED-before-reraise, re-read-before-succeed), but `_load`/`_save`/`_train`/`_predict` use `ctx["runs"]`, `ctx["datasets"]`, `ctx["protocols"]`, `ctx["store"]` (port objects) instead of building SqlAlchemy repositories from `ctx["sessions"]`.
  - `build_sqlalchemy_ctx(sessions, store, *, job_deadline_seconds=None) -> dict` — the one place the SqlAlchemy repo trio is assembled into a ctx (used by `InlineEnqueuer` now, and by nothing else once arq is gone).
  - `InlineEnqueuer` — moved from worker.py, now calling `run_job(build_sqlalchemy_ctx(...), run_id)`; docstring and swallow-semantics unchanged.
  - `DbEnqueuer(queue: RunQueue)` implementing `JobEnqueuer`: `enqueue(run_id, lane)` → `await self._queue.set_lane(run_id, lane)`.
- Consumes: `RunQueue.set_lane` (Task 3).

- [ ] **Step 1: Move + adapt the tests.** `git mv backend/tests/unit/execution/test_worker.py backend/tests/unit/execution/test_jobs.py`; update imports to `daikonstudio.infrastructure.jobs`; wherever the tests build a ctx with `{"sessions": ...}` switch to the fake-repo ports the tests likely already stub (read the file first — if it fakes repositories via `ctx["sessions"]`, introduce plain fake port objects; the assertions stay identical). Add one new test:

```python
async def test_db_enqueuer_sets_lane() -> None:
    calls: list[tuple[uuid.UUID, str]] = []

    class FakeQueue:
        async def set_lane(self, run_id, lane):
            calls.append((run_id, lane))

    run_id = uuid.uuid4()
    await DbEnqueuer(FakeQueue()).enqueue(run_id, lane="gpu")
    assert calls == [(run_id, "gpu")]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/unit/execution/test_jobs.py -v` → FAIL (no module `jobs`)

- [ ] **Step 3: Create `jobs.py`, delete `worker.py`.** Port `run_job`/`_HANDLERS`/`_train`/`_predict`/`_load`/`_save`/`InlineEnqueuer` over (docstrings included — they carry the at-least-once/SystemExit rationale); drop `WorkerSettings`, `ArqEnqueuer`, `queue_for`, `_HARD_TIMEOUT_MARGIN_SECONDS`, the module-level `Settings()` read, and every `arq` import. `_train`'s deadline comes from `ctx.get("job_deadline_seconds")` exactly as before.

- [ ] **Step 4: Rewire the container.** In `di/container.py`: remove the `ArqEnqueuer` import/binding; import `DbEnqueuer, InlineEnqueuer` from `daikonstudio.infrastructure.jobs`; add `from daikonstudio.application.ports.run_queue import RunQueue` and `from daikonstudio.infrastructure.persistence.sqlalchemy.execution.queue import SqlAlchemyRunQueue`; then:

```python
    container.define(
        RunQueue,  # type: ignore[type-abstract]
        lambda c: SqlAlchemyRunQueue(c[async_sessionmaker]),
    )
    # Plain factory, NOT Singleton: both branches depend on c[async_sessionmaker],
    # which tests override per test -- see the InlineEnqueuer caching incident
    # documented in the JobEnqueuer comment this replaces.
    container.define(
        JobEnqueuer,  # type: ignore[type-abstract]
        lambda c: (
            InlineEnqueuer(c[async_sessionmaker], c[BlobStore])
            if resolved.inline_jobs
            else DbEnqueuer(c[RunQueue])
        ),
    )
```

Also bind `RunnerRepository` here (Task 5 needs it): `container.define(RunnerRepository, lambda c: SqlAlchemyRunnerRepository(c[async_sessionmaker]))`, and expose the resolved settings for interface dependencies that need config values (Task 7's lease extension): `container.define(Settings, Singleton(lambda: resolved))`.

- [ ] **Step 5: Chase the imports.** `grep -rn "infrastructure.worker\|ArqEnqueuer\|WorkerSettings" backend/src backend/tests` — fix every hit (the Makefile/pyproject/docker hits wait for Tasks 10–11).

- [ ] **Step 6: Full unit + api suites pass** — `uv run pytest tests/unit tests/api -v` (api tests exercise `InlineEnqueuer` end-to-end via `inline_jobs=True`)

- [ ] **Step 7: Commit** — `git commit -m "feat: DbEnqueuer + port-based job ctx; delete arq worker"`

---

### Task 5: Runner management — use cases + `/api/v1/runners` routes

**Files:**
- Create: `backend/src/daikonstudio/application/runners/__init__.py`
- Create: `backend/src/daikonstudio/application/runners/manage.py`
- Create: `backend/src/daikonstudio/interface/routes/runners.py`
- Modify: `backend/src/daikonstudio/infrastructure/di/container.py`, `backend/src/daikonstudio/interface/app.py`, `backend/src/daikonstudio/settings.py`
- Test: `backend/tests/unit/runners/test_manage.py`, `backend/tests/api/test_runners.py`

**Interfaces:**
- Produces (in `manage.py`, all following the command/Result/auth conventions of `application/execution/*`):

```python
@dataclass(frozen=True)
class CreateRunnerCommand:
    name: str
    lanes: tuple[str, ...]

@dataclass(frozen=True)
class CreatedRunner:
    runner: Runner
    token: str  # plaintext, surfaced exactly once

class CreateRunner:
    def __init__(self, runners: RunnerRepository) -> None: ...
    async def __call__(self, command, *, auth: AuthContext | None) -> Result[CreatedRunner, DomainError]:
        # require_editor(auth); ValidationError on empty name / empty lanes /
        # any blank lane string; token = f"drt_{secrets.token_urlsafe(32)}";
        # token_hash = sha256 hexdigest; repo.add (ConflictError on dup name)

@dataclass(frozen=True)
class RunnerStatusView:
    runner: Runner
    online: bool          # last_seen_at within online_threshold_seconds of now
    current_run_id: uuid.UUID | None

class ListRunners:
    def __init__(self, runners: RunnerRepository, queue: RunQueue, *, online_threshold_seconds: int) -> None: ...
    async def __call__(self, *, auth) -> Result[list[RunnerStatusView], DomainError]: ...

@dataclass(frozen=True)
class RevokeRunnerCommand:
    runner_id: uuid.UUID

class RevokeRunner:
    def __init__(self, runners: RunnerRepository) -> None: ...
    async def __call__(self, command, *, auth) -> Result[None, DomainError]:
        # require_editor; NotFoundError("Runner", id) if repo.get is None; repo.revoke
```

- Produces (routes, Sentinel-authed via `AuthDep`, prefix `/api/v1/runners`): `POST ""` → 201 `{id, name, lanes, token, created_at}`; `GET ""` → `[{id, name, lanes, online, last_seen_at, revoked, current_run_id, created_at}]` (plain list — single-digit runners, no pagination); `POST "/{runner_id}/revoke"` → 204. Request body `extra="forbid"` like every other route module.
- Settings added here: `runner_online_threshold_seconds: int = 15` (+ the other four new settings from Global Constraints — add them all now in one edit).

- [ ] **Step 1: Failing unit tests for the use cases** (fake `RunnerRepository`/`RunQueue` as plain classes recording calls): create returns token with `drt_` prefix whose sha256 matches the stored hash; create rejects empty lanes with `ValidationError`; viewer role is refused (copy how existing use-case tests assert `AuthorizationError` — check `tests/unit/` for a `require_editor` example first); list computes `online` from a `last_seen_at` 5 s ago (True) vs 60 s ago (False) vs None (False) and joins `current_run_id` from the queue's `active_run_by_runner`; revoke on unknown id → `NotFoundError`.
- [ ] **Step 2: Verify failure**, **Step 3: Implement `manage.py`**, **Step 4: Unit tests pass** — `uv run pytest tests/unit/runners/ -v`
- [ ] **Step 5: Failing api tests** (`tests/api/test_runners.py`, using the existing `client`/`viewer_client` fixtures): POST creates and returns a token once; GET lists it (no token field in list responses — assert `"token" not in item`); POST revoke → subsequent GET shows `revoked: true`; viewer POST → 403; duplicate name → 409.
- [ ] **Step 6: Verify failure**, **Step 7: Wire routes + container bindings + `app.include_router(runners_router)`**, **Step 8: api tests pass** — `uv run pytest tests/api/test_runners.py -v`
- [ ] **Step 9: Commit** — `git commit -m "feat: runner registration, listing, revocation API"`

---

### Task 6: Wire envelopes (shared serialization for protocol + client)

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/runner/__init__.py`
- Create: `backend/src/daikonstudio/infrastructure/runner/wire.py`
- Test: `backend/tests/unit/runners/test_wire.py`

**Interfaces:**
- Produces pydantic models, each with `from_domain(obj) -> Self` and `to_domain() -> <domain type>`:
  - `RunEnvelope`: every `Run.__init__` kwarg verbatim (`id, kind, workspace_id, requested_by, cache_key, params, protocol_id, status, progress, phase, result_uri, error_message, created_at, updated_at, version`). Enums as their `str` values on the wire; `to_domain` casts back via `RunKind(...)`/`RunStatus(...)`.
  - `RunUpdateEnvelope`: only the mutable fields a handler writes — `status: str, progress: float, phase: str | None, result_uri: str | None, error_message: str | None, protocol_id: uuid.UUID | None, expected_version: int`.
  - `DatasetEnvelope`: every `Dataset.__init__` kwarg (`domain/data/dataset.py:33-64`), with nested `TargetSpecWire`/`SplitSpecWire`/`ValidationReportWire` mirroring `TargetSpec` (`domain/data/target.py:42-52`), `SplitSpec` (`domain/data/split.py:17-22`), `ValidationReport` (`domain/data/validation.py:31-38`) field-for-field — open those three files and copy the exact field names/types; the roundtrip test below is the enforcement that nothing drifted.
  - `ProtocolEnvelope`: every `InSilicoProtocol.__init__` kwarg (`domain/catalog/protocol.py:33-52`) with `ReadoutWire` mirroring `Readout` (`name, type, unit, direction, description`; `type` via `ReadoutType(...)`), `conditions: dict[str, Any]` (the domain exposes a `MappingProxyType` — `from_domain` does `dict(protocol.conditions)`), `status` via `ProtocolStatus(...)`.
  - `ClaimResponse`: `{run: RunEnvelope, deadline_seconds: int}`.
  - `BlobPutResponse`: `{uri: str}`. `RunUpdateResponse`: `{version: int}`.
- Consumed by: Task 7 (routes) and Task 8 (HTTP ports). Living in infrastructure keeps the import direction legal for both.

- [ ] **Step 1: Failing roundtrip tests** — for each envelope, build a fully-populated domain object (every optional field set, ≥2 readouts, non-trivial `params`/`conditions`), then assert `Envelope.from_domain(obj).to_domain()` equals the original field-by-field (compare `__dict__` where the domain types lack `__eq__`; `Readout` is a frozen dataclass so `==` works). Add one JSON trip: `Envelope.model_validate_json(envelope.model_dump_json()).to_domain()` — this is what actually crosses the wire.
- [ ] **Step 2: Verify failure**, **Step 3: Implement**, **Step 4: Pass** — `uv run pytest tests/unit/runners/test_wire.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat: runner protocol wire envelopes"`

---

### Task 7: Runner protocol — auth dependencies + `/api/v1/runner/*` routes

**Files:**
- Create: `backend/src/daikonstudio/application/execution/claim_run.py`
- Create: `backend/src/daikonstudio/interface/dependencies/runner_auth.py`
- Create: `backend/src/daikonstudio/interface/routes/runner_api.py`
- Modify: `backend/src/daikonstudio/interface/app.py`, `backend/src/daikonstudio/infrastructure/di/container.py`
- Test: `backend/tests/api/test_runner_protocol.py`

**Interfaces:**
- Produces `ClaimRun` use case: `__init__(queue: RunQueue, runs: RunRepository, *, lease_seconds: int, max_active_per_workspace: int, max_attempts: int, deadline_seconds: int)`; `__call__(*, runner: Runner) -> Result[tuple[Run, int] | None, DomainError]` — sweep, claim against `runner.lanes` (refuse with `AuthorizationError` if `runner.is_revoked` — defense in depth, the dependency already 401s), load via `runs.get_by_id`, return `(run, deadline_seconds)` or `Success(None)` when the queue is empty. Container-bound with values from `Settings` (`runner_lease_seconds`, `workspace_max_active_runs`, `runner_max_attempts`, `worker_job_timeout`).
- Produces dependencies in `runner_auth.py`:

```python
async def get_runner(request: Request, container=Depends(get_container)) -> Runner:
    # "Authorization: Bearer drt_..." -> sha256 -> RunnerRepository.get_by_token_hash
    # 401 (WWW-Authenticate: Bearer) when header missing/malformed/unknown/revoked.
    # On success: await repo.touch_last_seen(runner.id)  -- the liveness signal.

RunnerDep = Annotated[Runner, Depends(get_runner)]

def claimed_run(*, require_active: bool):  # dependency factory
    async def _dep(run_id: uuid.UUID, runner: RunnerDep, container=...) -> Run:
        # queue.verify_claim(run_id, runner_id=runner.id,
        #                    lease_seconds=settings.runner_lease_seconds,
        #                    require_active=require_active)  -> False => 403
        # RunRepository.get_by_id(run_id) -> None => 404 (check BEFORE verify so
        # an unknown id is 404, not 403)
        # returns the loaded Run
    return _dep

ClaimedRunRead = Annotated[Run, Depends(claimed_run(require_active=False))]
ClaimedRunWrite = Annotated[Run, Depends(claimed_run(require_active=True))]
```

  Note `verify_claim` extends the lease on every call — reads included. That is deliberate: the training reporter's periodic `get_by_id` (its cancellation check) is thereby the heartbeat, with no separate heartbeat endpoint.
- Produces routes (`router = APIRouter(prefix="/api/v1/runner", tags=["runner-protocol"])`):

| route | contract |
|---|---|
| `POST /claim` | `RunnerDep`. 200 `ClaimResponse` or 204 empty. |
| `GET /runs/{run_id}` | `ClaimedRunRead` → `RunEnvelope`. |
| `POST /runs/{run_id}` | `ClaimedRunWrite` + body `RunUpdateEnvelope` → `RunUpdateResponse`. 422 if `body.status == "cancelled"` (runners never cancel) or status not in `{"running","ready","failed"}`; 409 if `body.expected_version != current.version` or repo raises the concurrency error. Server applies ONLY the mutable fields onto the freshly loaded run (`status/progress/phase/result_uri/error_message/protocol_id` — never `params`, `kind`, `workspace_id`), sets `run.version = body.expected_version`, calls the container-resolved `RunRepository.update` (resolve repositories through the container like every other route — never construct them in the route). |
| `GET /runs/{run_id}/dataset` | `ClaimedRunRead`. `dataset_id = run.params.get("dataset_id")` → 404 if absent; `DatasetRepository.get(run.workspace_id, dataset_id)` → 404 if missing; `DatasetEnvelope`. |
| `GET /runs/{run_id}/protocol` | `ClaimedRunRead`. 404 unless `run.protocol_id`; `ProtocolRepository.get(run.workspace_id, run.protocol_id)` → `ProtocolEnvelope`. |
| `POST /runs/{run_id}/protocol` | `ClaimedRunWrite` + body `ProtocolEnvelope`. 422 unless `body.workspace_id == run.workspace_id`; `ProtocolRepository.add(body.to_domain())` → 201. |
| `GET /runs/{run_id}/blobs/{key:path}` | `ClaimedRunRead`. 403 unless `key.startswith(f"{run.workspace_id}/")`; `store.get_bytes(key)` → `Response(content, media_type="application/octet-stream")`, 404 on backend miss (catch `FileNotFoundError` and `OSError`). |
| `PUT /runs/{run_id}/blobs/{key:path}` | `ClaimedRunWrite`. Same prefix guard. 413 if `Content-Length` or actual body length exceeds `settings.runner_upload_max_bytes` (`body = await request.body()`; the bytes-based `BlobStore` port means in-memory is inherent — do both checks). `store.put_bytes(key, body)` → `BlobPutResponse(uri=...)`. |

- Modify `app.py`: `sentinel.protect(app, exclude_paths=["/health", "/version", "/docs", "/openapi.json", "/api/v1/runner"])` + `app.include_router(runner_api_router)`.

- [ ] **Step 1: Failing api tests** (`tests/api/test_runner_protocol.py`; note the api `app` fixture uses `inline_jobs=True`, which is irrelevant here — these tests never enqueue, they insert runs/runners through the container's repositories directly). Helper at top: `async def _register_runner(app, lanes) -> tuple[uuid.UUID, dict]` creating a runner via `CreateRunner` resolved from `app.state.container` and returning `(id, {"Authorization": f"Bearer {token}"})`. Cover at minimum:
  - no/garbage/revoked token on `POST /claim` → 401 each; Sentinel headers NOT required (this is the exclude_paths proof — the request carries only the runner token).
  - empty queue claim → 204; pending default-lane run + default runner → 200 envelope with matching id, `deadline_seconds == settings.worker_job_timeout`.
  - gpu-lane pending run + default-lane runner → 204 (registered lanes, not client-supplied).
  - `GET /runs/{id}` by non-claimant runner → 403; by claimant → full envelope including `params`.
  - update: claimant posts `status=running` then `ready` with correct `expected_version` → 200s and versions increment; `cancelled` → 422; stale `expected_version` → 409; update on a run another runner claimed → 403.
  - blob GET with key outside `f"{workspace_id}/"` → 403; PUT roundtrip inside the prefix → uri returned and `GET` returns the same bytes; PUT larger than a test-sized cap (override `runner_upload_max_bytes=64` on the test Settings) → 413; PUT after the run went `ready` → 403 (write dep `require_active`).
  - dataset endpoint: training-style run with `params={"dataset_id": ...}` and a dataset created via the normal api fixtures → envelope matches; prediction-style run without `dataset_id` → 404.
- [ ] **Step 2: Verify failure**, **Step 3: Implement** (claim_run.py, runner_auth.py, runner_api.py, app.py, container bindings for `ClaimRun`), **Step 4: Pass** — `uv run pytest tests/api/test_runner_protocol.py -v`
- [ ] **Step 5: Full api suite still green** (Sentinel exclusion must not leak: `uv run pytest tests/api -v`)
- [ ] **Step 6: Commit** — `git commit -m "feat: runner protocol endpoints with token auth + lease fencing"`

---

### Task 8: HTTP port implementations (the runner-side adapters)

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/runner/ports.py`
- Test: `backend/tests/api/test_runner_ports.py`

**Interfaces:**
- Produces:

```python
class RunnerApiClient:
    """One claimed run's view of the studio. Bound to a run_id because every
    protocol URL is run-scoped -- constructing it per job is the design."""
    def __init__(
        self,
        base_url: str,          # e.g. "https://studio.example.org"
        token: str,
        run_id: uuid.UUID,
        *,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,   # test seams
    ) -> None: ...
    # owns: self._api = httpx.AsyncClient(base_url=f"{base_url}/api/v1/runner", headers=auth, timeout=60.0, transport=async_transport)
    #        self._blobs = httpx.Client(same, transport=sync_transport)
    async def aclose(self) -> None: ...

class HttpRunRepository:     # implements the RunRepository methods the job path calls
    def __init__(self, client: RunnerApiClient) -> None: ...
    async def get_by_id(self, run_id) -> Run | None      # GET /runs/{id}; 404 -> None
    async def update(self, run) -> None                  # POST /runs/{id} from run's mutable fields
                                                         # + expected_version=run.version;
                                                         # on 200: run.version = resp["version"]
                                                         # on 409: raise the concurrency error type that
                                                         #   execution/repository.py raises on rowcount 0
                                                         #   (grep its import there and reuse it verbatim)
    # add/get/find_by_cache_key/list: raise NotImplementedError("api-side only")

class HttpDatasetRepository:
    async def get(self, workspace_id, dataset_id) -> Dataset | None   # GET /runs/{run_id}/dataset
    # add/list/find_by_content_hash: NotImplementedError

class HttpProtocolRepository:
    async def get(self, workspace_id, protocol_id) -> InSilicoProtocol | None  # GET /runs/{run_id}/protocol
    async def add(self, protocol) -> None                                      # POST /runs/{run_id}/protocol
    # update/list: NotImplementedError

class HttpBlobStore:         # sync, like the BlobStore port
    def get_bytes(self, key: str) -> bytes               # GET /runs/{run_id}/blobs/{key}
    def put_bytes(self, key: str, data: bytes) -> str    # PUT ...; returns resp["uri"]
    # exists/delete: NotImplementedError (job path verified to use only get/put)

def build_http_ctx(base_url, token, run_id, *, deadline_seconds, **transports) -> dict[str, Any]:
    # {"runs": ..., "datasets": ..., "protocols": ..., "store": ...,
    #  "job_deadline_seconds": deadline_seconds, "_client": client}  -- ctx for jobs.run_job
```

- Consumes: wire envelopes (Task 6), routes (Task 7), `run_job` ctx contract (Task 4).
- Test helper produced for reuse in Task 10 — put it in `backend/tests/helpers/sync_asgi.py` (create `tests/helpers/__init__.py`):

```python
class SyncAsgiTransport(httpx.BaseTransport):
    """Bridges a SYNC httpx.Client onto an ASGI app for tests. Each request is
    run to completion (body fully read) on a private single-thread executor's
    own event loop, so it is safe to call from inside another running loop --
    which is exactly what a handler's synchronous BlobStore call does."""
    def __init__(self, app) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _round_trip() -> httpx.Response:
            response = await self._inner.handle_async_request(request)
            content = await response.aread()
            return httpx.Response(response.status_code, headers=response.headers, content=content)
        return self._pool.submit(asyncio.run, _round_trip()).result()
```

- [ ] **Step 1: Failing tests** in `tests/api/test_runner_ports.py` — these run against the real app fixture over transports (no mocking of the server): register a runner + insert a claimable run (reuse Task 7's `_register_runner` helper — move it into `tests/helpers/runner_fixtures.py` now that two files need it), claim it via a raw client, then build the ports with `async_transport=httpx.ASGITransport(app=app)` and `sync_transport=SyncAsgiTransport(app)` and assert:
  - `HttpRunRepository.get_by_id` returns a `Run` equal to the envelope fields; unknown id → `None`.
  - `update` after local `run.start()` persists `running` and bumps `run.version` by 1; a second stale copy raises the concurrency error on `update`.
  - `HttpBlobStore.put_bytes`/`get_bytes` roundtrip bytes and return a uri string.
  - `HttpProtocolRepository.add` then `.get` roundtrips a protocol (link `run.protocol_id` via an update in between, since GET reads `run.protocol_id`).
  - `HttpDatasetRepository.get` returns the dataset for a run whose params carry `dataset_id`.
- [ ] **Step 2: Verify failure**, **Step 3: Implement `ports.py` (+ helpers move)**, **Step 4: Pass** — `uv run pytest tests/api/test_runner_ports.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat: HTTP implementations of the four job ports"`

---

### Task 9: The runner agent + dev seed CLI

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/runner/agent.py`
- Create: `backend/src/daikonstudio/infrastructure/runner/__main__.py`
- Create: `backend/src/daikonstudio/infrastructure/runner/seed.py`
- Test: `backend/tests/unit/runners/test_agent.py`

**Interfaces:**
- Produces `agent.py`:

```python
class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDIO_", env_file=".env", extra="ignore")
    url: str                      # STUDIO_URL, e.g. https://studio.example.org
    runner_token: str             # STUDIO_RUNNER_TOKEN, the drt_... secret
    poll_seconds: float = 3.0

async def poll_once(api: httpx.AsyncClient, settings: AgentSettings) -> bool:
    """One iteration: claim; if a run was granted, execute it via jobs.run_job
    with HTTP ports. Returns True iff a job was executed (drives test + the
    no-sleep-while-busy loop). run_job re-raises handler failures after
    persisting FAILED -- catch Exception here, log via structlog, return True:
    the agent must survive a failed job. SystemExit/KeyboardInterrupt/CancelledError
    propagate -- same shutdown contract run_job's docstring describes."""

async def main() -> None:
    # AgentSettings(); one long-lived AsyncClient for /claim;
    # loop: executed = await poll_once(...); if not executed: sleep(poll_seconds)
```

  `poll_once` builds the ctx with `build_http_ctx(settings.url, settings.runner_token, run_id, deadline_seconds=payload["deadline_seconds"])` and closes it (`await ctx["_client"].aclose()`) in a finally. `__main__.py` is two lines: `asyncio.run(agent.main())` under `if __name__ == "__main__":` — entrypoint `python -m daikonstudio.infrastructure.runner`.
- Produces `seed.py` — local-dev bootstrap, NOT for production (module docstring says so): upserts two runners straight into the DB via `Settings().database_url`:

| name | lanes | fixed token |
|---|---|---|
| `dev-local-default` | `["default"]` | `drt_dev_default` |
| `dev-local-gpu` | `["gpu"]` | `drt_dev_gpu` |

  Upsert = `INSERT ... ON CONFLICT (name) DO UPDATE SET token_hash = excluded.token_hash, lanes = excluded.lanes, revoked_at = NULL`. Runs as `python -m daikonstudio.infrastructure.runner.seed`, prints the two names it ensured. (`# ponytail: fixed plaintext dev tokens; fine while the dev DB only listens on 127.0.0.1 -- production runners come from the UI.`)

- [ ] **Step 1: Failing unit tests for `poll_once`** using `httpx.MockTransport`: a 204 claim → returns False and no job ran; a 200 claim whose subsequent `run_job` is monkeypatched to record the ctx/run_id → returns True, ctx carried the right `job_deadline_seconds`, and the recorded run_id matches; `run_job` raising `RuntimeError` → `poll_once` returns True without propagating.
- [ ] **Step 2: Verify failure**, **Step 3: Implement all three modules**, **Step 4: Pass** — `uv run pytest tests/unit/runners/test_agent.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat: runner agent loop + dev runner seeding"`

---

### Task 10: End-to-end integration test + Makefile/compose cutover

**Files:**
- Create: `backend/tests/integration/test_runner_full_loop.py`
- Modify: `Makefile`, `docker-compose.yml`
- Modify: `backend/src/daikonstudio/settings.py` (delete `redis_url`, `worker_lane`, `worker_max_jobs`; the four comments referencing arq go with them)

**Interfaces:**
- Consumes: everything. This is the proof of the spec's central claim — handlers run unmodified over HTTP ports.

- [ ] **Step 1: Write the integration test.** Read `backend/tests/integration/conftest.py` first and reuse its app/auth fixtures the way `test_full_loop.py` does, with one difference: this test's container must use `inline_jobs=False` (build a local `app` fixture: `create_app()` then override the container exactly as `tests/api/conftest.py` lines 150-166 do, but `Settings(inline_jobs=False, blob_base_url=f"file://{tmp_path}", database_url=<the root conftest's testcontainer URL>)`, and override `async_sessionmaker` with one built on `create_async_engine(url, poolclass=NullPool)` — NullPool is required because the `SyncAsgiTransport` thread makes DB calls from a second event loop). Flow, asserting at every stage:
  1. Upload the existing fixture CSV + create a dataset (same calls as `test_full_loop.py`).
  2. Submit training via `POST /api/v1/protocols` (or wherever `test_full_loop.py` submits it — mirror it). Poll `GET /api/v1/runs/{id}`: status stays `pending` — nothing executes inline anymore.
  3. Register a runner via `POST /api/v1/runners` (lanes `["default"]`), grab the token.
  4. Drive the agent's real code path in-process: `api = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s/api/v1/runner", headers=runner_auth)`, then monkeypatch-free call of `poll_once` is not possible (it builds network clients), so instead do what `poll_once` does with the test transports: POST /claim → 200; `ctx = build_http_ctx("http://s", token, run_id, deadline_seconds=..., async_transport=ASGITransport(app), sync_transport=SyncAsgiTransport(app))`; `await run_job(ctx, run_id)`.
  5. Assert via the USER api: run `ready`, `protocol_id` set, `GET /api/v1/protocols/{id}/scorecard` (or the shape `test_full_loop.py` checks) succeeds — the artifact a remote runner uploaded is readable server-side.
  6. Second claim returns 204 (queue drained).
- [ ] **Step 2: Run it** — `uv run pytest tests/integration/test_runner_full_loop.py -v` (slow — a real ECFP4 fit; that is the point). Expected: PASS.
- [ ] **Step 3: Makefile cutover.** Replace the `ARQ`/`WORKER`/`WORKER_GPU` definitions:

```make
# Runner agents replace the arq workers: same jobs, but claimed over the HTTP
# runner protocol (see docs/superpowers/specs/2026-08-04-self-hosted-runners-design.md).
# Lanes live on the server-side runner rows that `make seed-runners` ensures.
# OMP_NUM_THREADS=1 is load-bearing -- see the original explanation below (kept).
RUNNER     := env OMP_NUM_THREADS=1 STUDIO_URL=http://localhost:8002 uv run python -m daikonstudio.infrastructure.runner
WORKER     := env STUDIO_RUNNER_TOKEN=drt_dev_default $(RUNNER)
WORKER_GPU := env STUDIO_RUNNER_TOKEN=drt_dev_gpu $(RUNNER)
```

  Add `seed-runners: ## Ensure the two local dev runners exist` running `$(BACKEND) && $(BE_ENV) && uv run python -m daikonstudio.infrastructure.runner.seed`, call it from `up` (after `migrate`), keep `dev`/`dev-worker`/`dev-worker-gpu`/`stop` targets working against the new commands (the pid-file logic is unchanged; the pkill comment about identical arq command lines still applies verbatim to the agents — update its wording from "arq" to "runner agent"). Update `up`'s echo and the header comment ("Two workers, because engines declare which lane...") to describe agents + lanes-on-the-server. Delete `valkey` from `docker-compose.yml` and from `up`'s compose invocation/echo.
- [ ] **Step 4: Settings prune** — delete `redis_url`, `worker_lane`, `worker_max_jobs` and their comment blocks; update the `worker_job_timeout` comment (soft deadline, served to runners via the claim response; the old "arq's hard job_timeout is derived" sentence goes). `grep -rn "redis_url\|worker_lane\|worker_max_jobs" backend/` must come back empty.
- [ ] **Step 5: Manual smoke** — `make up && make dev`, wait, then in the UI (or curl with dev Sentinel auth) submit a training run with `STUDIO_INLINE_JOBS=0`; `tail .logs/worker.log` shows a claim + fit; run reaches `ready`. Record the result honestly in the commit message.
- [ ] **Step 6: Commit** — `git commit -m "feat: end-to-end runner loop; make dev runs agents; Valkey removed from compose"`

---

### Task 11: Dependency + Docker cleanup

**Files:**
- Modify: `backend/pyproject.toml` (remove `arq>=0.26` and `redis>=5.2.0`), then `cd backend && uv sync`
- Modify: `backend/Dockerfile` (line-34 comment: worker override becomes `python -m daikonstudio.infrastructure.runner` with `STUDIO_URL`/`STUDIO_RUNNER_TOKEN` env)
- Modify: `backend/Dockerfile.gpu` (drop `STUDIO_WORKER_LANE=gpu STUDIO_WORKER_MAX_JOBS=1` from ENV — lanes are server-side now; `CMD ["python", "-m", "daikonstudio.infrastructure.runner"]`)

- [ ] **Step 1: Remove deps + sync.** `grep -rn "import arq\|from arq\|import redis\|from redis" backend/src backend/tests` must be empty first.
- [ ] **Step 2: Full backend suite + import-linter** — `make test` → PASS. Also `make test-all` if time allows (runs the slow integration suite including both full-loop tests).
- [ ] **Step 3: Build the CPU image to prove the Dockerfile still assembles** — `cd backend && docker build -t daikon-runner:cpu .` → succeeds. (GPU image build is environment-dependent; do NOT block on it — note in the commit if skipped.)
- [ ] **Step 4: Commit** — `git commit -m "chore: drop arq/redis; runner entrypoints in Docker images"`

---

### Task 12: Runners UI

**Files:**
- Create: `frontend/src/features/runners/types/index.ts`, `hooks/query-keys.ts`, `hooks/use-runners.ts`, `components/runner-list.tsx`, `components/new-runner-dialog.tsx`, `index.ts`
- Create: `frontend/src/app/(dashboard)/runners/page.tsx`
- Modify: `frontend/src/shared/lib/navigation.ts`

**Interfaces:**
- Consumes: `/api/v1/runners` endpoints (Task 5) through the orval-generated client (`make generate-api` regenerates `frontend/src/shared/lib/api/model` types + `openapi.json`).

- [ ] **Step 1: `make generate-api`** — commit the refreshed `openapi.json` + generated model files together with this task.
- [ ] **Step 2: Feature module**, copying the runs feature's structure exactly (`features/runs` is the template — same barrel/index style, same `customInstance` + `API_V1` usage, same query-key pattern):
  - `hooks/use-runners.ts`: `useRunners()` (`GET ${API_V1}/runners`, `refetchInterval: 10_000` — the online dot should move without a reload), `useCreateRunner()` (mutation POST, invalidates `RUNNERS_KEY`), `useRevokeRunner()` (mutation POST `/{id}/revoke`, invalidates).
  - `components/runner-list.tsx` (`"use client"`): table of name / lanes (badge per lane) / status dot (green "online" when `online`, gray "offline", red "revoked") / last seen (relative) / current run (link to `/runs/{id}` when present) / a Revoke button with a confirm step; loading-skeleton, error, and empty states copied from `run-list.tsx`'s branch structure. Empty state copy: "No runners yet. Add one to run training on your own hardware."
  - `components/new-runner-dialog.tsx`: name input + lane multi-select (`default`, `gpu` — read the lane list from the engines the catalog exposes if the generated client offers it cheaply; otherwise a constant `const KNOWN_LANES = ["default", "gpu"] as const` with a comment pointing at `EngineManifest.lane`). On success, a **token-reveal step**: monospace one-time token inside a ready-to-copy command block, with copy button and the warning "This token is shown once. Treat it like a password."

```text
docker run -d --restart unless-stopped \
  -e STUDIO_URL={apiOrigin} \
  -e STUDIO_RUNNER_TOKEN={token} \
  daikon-runner:gpu
```

  (`apiOrigin` from `getApiBaseUrl()` stripped of its path; image tag `daikon-runner:gpu` when the gpu lane is selected, else `:cpu`.) Dialog must not be closable-by-accident during the reveal step (require an explicit "I've copied it" button).
- [ ] **Step 3: Page + nav** — `app/(dashboard)/runners/page.tsx` re-exporting the list like `runs/page.tsx` does; add `{ title: "Runners", href: "/runners", icon: Server }` (lucide `Server`) to the **Catalog** group in `navigation.ts`.
- [ ] **Step 4: Checks** — `make lint-fe && make test-fe` → PASS. Manual: `make dev`, open `/runners`, see the two seeded dev runners online, create + revoke a third.
- [ ] **Step 5: Commit** — `git commit -m "feat: Runners page — register, monitor, revoke"`

---

### Task 13: Final verification + spec sync

- [ ] **Step 1:** `make test-all && make lint && make lint-fe && make test-fe` — all green.
- [ ] **Step 2:** `grep -rni "arq\|valkey\|redis" backend/src Makefile docker-compose.yml backend/pyproject.toml` — remaining hits must be deliberate (historical docstrings are fine only where they explain a surviving design decision; update any that describe deleted machinery).
- [ ] **Step 3:** Re-read the spec (`docs/superpowers/specs/2026-08-04-self-hosted-runners-design.md`) against what was built; fix any drift in the spec, not the code, if the deviation was one of the four already-approved ones (pending vs QUEUED wording, workspace-prefix blob scoping, default-lane backfill, `/api/v1/runner` prefix). Anything else: stop and surface it.
- [ ] **Step 4: Commit** — `git commit -m "docs: sync runner spec with as-built phase 1"`

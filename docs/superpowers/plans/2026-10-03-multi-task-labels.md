# Multi-Task Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Dataset carries an ordered tuple of target columns. Chemprop (and later MoLFormer-XL) learn all targets jointly. Every other engine trains one model per target behind one fan-out adapter. Every surface downstream (prediction results, Scorecards, the sweep table, the dataset pages) handles N targets.

**Architecture:**
- `Dataset.targets` is the source of truth, stored as a JSONB array (migration 013).
- `EngineRegistry.get()` wraps every engine that does not declare `supports_multitask` in `FanOut` (`application/engines/fan_out.py`). Everything downstream of the registry sees one shape:
  - metrics nested by target column;
  - long-format predictions with a `target` column;
  - a bare artifact at N = 1 and a zip container at N > 1.
- Training writes a per-target `ScorecardInputs`. The Scorecard read path renders one card per target, sharing the expensive test-set chemistry.

**Tech Stack:**
- Backend: Python 3.13, FastAPI, SQLAlchemy 2 + asyncpg + Alembic, polars, scikit-learn / XGBoost / LightGBM, chemprop v2 + Lightning, transformers.
- Frontend: Next.js, React Query, AG Grid, orval, Vitest, biome.

**Spec:** `docs/superpowers/specs/2026-10-03-multi-task-labels-design.md`. Read it in full. Also read the traps in `docs/superpowers/plans/2026-10-03-multi-task-labels-handoff.md`. This plan argues from both. Where the survey below found the spec wrong or silent, **this plan wins**, and the section "Corrections to the spec" says why.

---

## Decisions taken while planning (user, 2026-10-03)

1. **Every row carries a value for every target (dense labels).**
   - The upload gate applies today's per-cell rules to each target column. A row that is missing any target is rejected, and the reason names the column.
   - All targets therefore share one set of train and test rows. That is what lets `ScorecardInputs` store the test structures once.
   - Sparse label matrices (the case where joint learning actually helps) are a follow-up plan. Because of this, the spec's "a NaN label does not contribute to chemprop's loss" test is **dropped**, and MoLFormer needs no masked loss.
2. **Uncertainty is per target.**
   - With one target, prediction results keep the column `uncertainty`, so every existing results file stays readable.
   - With N > 1 targets, each target gets a `{column}_uncertainty` column.
   - The results API returns `uncertainty` as an object keyed by target column. The triage grid shows one Uncertainty column per target.

## Corrections to the spec (from the code survey)

| # | Spec says / omits | Reality | Resolution (task) |
|---|---|---|---|
| 1 | `TrainContext.task` stays a single field | The adapter receives one context covering every target of a mixed-kind dataset, so a single `task` cannot describe it | `TrainContext.targets: dict[str, TaskType]` (column → task, ordered). `task` and `target_column` become properties that **raise** when they would be ambiguous. Engines keep reading `ctx.task` / `ctx.target_column` unchanged. (T5) |
| 2 | silent | `PredictContext` has no target names. The adapter needs them at N = 1 to tag rows, and chemprop needs them to label its output. `RunPrediction` deliberately loads no Dataset | `PredictContext.target_columns`, recovered from the Protocol's readouts by `target_columns_of(readouts)` (T4, T5) |
| 3 | silent | `prepare_frame` validates and deduplicates exactly one target. `ValidationReport.duplicate_spread` is one float. `ConflictRow` does not say which column conflicts | Per-target gate and dedup. `duplicate_spread: dict[str, float]`. `ConflictRow.column`. Stored reports are migrated in 013 (T1, T2) |
| 4 | `runs.metrics` keyed by target column, legacy read through a wrapper | JSONB does not preserve object key order, and the sweep table must show targets in the scientist's order. A read-time wrapper needs the dataset's target name on every runs list | `{"targets": [{column, primary_metric, value, baseline_value}, …]}`. Existing rows are rewritten in migration 013, which can join `datasets`, so no read-time wrapper is needed (T7) |
| 5 | silent | The runner wire protocol pins `DatasetEnvelope.target` and `RunMetricsWire`'s three keys | Wire models change. **The API and the runners must be deployed together.** A stale runner gets a 422 on its metrics write (T1, T7) |
| 6 | "The run page shows N scorecards" | Scorecards render on the **protocol** page. The run page links there | N scorecards as tabs on the protocol page (F5) |
| 7 | "Target step: one kind radio for all targets" | Contradicts the approved mixed-kind decision | One kind choice **per target**, with unit and direction under each numeric one (F2) |
| 8 | Sweep table: sortable columns | Today the table has no clickable headers. It is a fixed ranking on one metric (`rank.ts`) | Per-target columns, sorted by clicking a header. Default sort only when N = 1, which keeps today's look (F6) |
| 9 | silent | `SubmitSweep` validates every config *before* creating any run, so a sweep is never half-submitted | The joint-engine mixed-kind refusal goes into that pre-flight too, not only into `TrainProtocol` (T9) |
| 10 | silent | `build_scorecard` runs a train × test Tanimoto search, about a minute at 324k compounds. N scorecards would pay it N times | Computed once per Protocol (`held_out_chemistry`) and shared across targets (T8) |
| 11 | silent | The dataset profile (target distribution, cliffs, descriptor correlations) and the compounds tab are single-target | Profile per target: `?target=<index>`, cached per index. Compounds carry every target, and you can sort by any one of them (T3, F3) |
| 12 | silent | A Scorecard does not name its target and does not say whether it came from joint or per-target training | `Scorecard.target` and `Scorecard.joint_model`. `joint_model` is recorded at training time, not inferred later from the manifest (T8) |
| 13 | `ScorecardInputs` "becomes per-target" | `structures` / `train_structures` are 15+ MB at 324k compounds, and duplicating them N times is waste | Run-level fields stay once. Per-target fields move to `targets: list[TargetInputs]`. Legacy blobs are upgraded on read (T8) |

## Global Constraints

- **Layers** (`lint-imports`, three contracts in `backend/pyproject.toml`):
  - `interface → infrastructure → application → domain`.
  - The domain imports no numpy, polars, sklearn, sqlalchemy or fastapi.
  - `domain.catalog`, `domain.data` and `domain.execution` may not import each other.
- **Gates before every commit that touches backend code:**
  ```
  cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
  ```
  plus the task's tests run with `OMP_NUM_THREADS=1`. Skipping mypy has turned `main` red before.
- **Gates for frontend commits:**
  ```
  cd frontend && pnpm lint && pnpm exec tsc --noEmit && pnpm test
  ```
- **Migration number is `013`.** The dev DB is at `012`.
- **The runner agent does not hot-reload.** Run `make dev-worker` (and `make dev-worker-gpu` for chemprop) after any engine, training or wire change before checking anything live.
- **UI and user-facing copy:** American spelling, academic and plain. No em-dash clause chains, no rhetorical-question headings. Conventions are in `docs/copy-audit.md`.
- **Commits:**
  - Conventional subject (`feat(datasets): …`, `fix(…)`, `test(…)`, `docs(…)`).
  - End with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - **Never** add a `Claude-Session:` trailer or any claude.ai URL.
- **Comment density:** this codebase writes "why" comments on non-obvious decisions. Match that. Do not narrate the obvious.
- **`content_hash` must not change.** A one-target upload must freeze to the same bytes as before this branch. Column order and dtypes of `prepare_frame`'s output are load-bearing (see T2).
- **Never catch `RunInterrupted`** in new code (handoff trap 4).

## Review Focus

These five failure modes are not covered by the spec's tests. They are the most likely to bite a real user, most likely first. Each one is pinned by a test in the task named.

1. **Anything trained before this branch must behave exactly as before.** That covers a bare artifact, a legacy `ScorecardInputs` blob, a results file with a plain `uncertainty` column, migrated `runs.metrics`, and a cached `profile.json`. Pinned in T6 (bare artifact), T8 (legacy blob renders as one card named after its readout), T11 (byte-for-byte identical one-target results frame) and T7 (migration round trip).
2. **Awkward column names.** These include `IC50 {nM}` (breaks `str.format`), `a/b`, a leading slash, and pairs like `x` + `x_probability`. Zip entries and profile blob keys use indexes, never column names. Pinned in T2 (braces in the reason text, the collision check) and T6 (a column named `/odd {name}` round-trips through the container).
3. **Cancel or deadline in the middle of a fan-out.** Cancelling during target 2 of 4 must stop at once and train nothing more. The deadline must scale with the target count. Pinned in T6 (`RunInterrupted` from fit 2 of 4 propagates, and fits 3 and 4 never start) and T10 (claim deadline × scale).
4. **A mixed-kind dataset sent to a joint engine without going through the UI.** That means direct API calls, a sweep whose last config is chemprop, or a chosen baseline that is a joint engine. Pinned in T9: an API-path refusal before any run exists, and a sweep pre-flight that creates no runs.
5. **The 324k × 4 real dataset.** Risks: the fan-out deadline, Scorecard read time, per-target profile time, and artifact size. Pinned in T21's real-data run, which records measured timings.

---

## File map

**Create**
- `backend/alembic/versions/013_dataset_targets.py`: targets list, per-target report shapes, run headline metrics.
- `backend/src/daikonstudio/application/engines/fan_out.py`: `FanOut`, one model per target behind the engine contract.
- `backend/tests/unit/engines/test_fan_out.py`
- `backend/tests/unit/data/test_targets.py`
- `backend/tests/unit/data/test_dataset.py`
- `backend/tests/unit/catalog/test_derive_readouts.py`
- `frontend/src/shared/lib/targets.ts` and `targets.test.ts`
- `frontend/src/features/engines/components/targets-hint.tsx`

**Modify (backend)**
- domain:
  - `domain/data/{target,dataset,validation}.py`
  - `domain/execution/{run,scorecard}.py`
- application:
  - `application/data/{prepare_frame,create_dataset,compound_ids,set_dataset_id_column,get_dataset_compounds,get_dataset_profile}.py`
  - `application/engines/{context,manifest,protocol,registry}.py`
  - `application/catalog/{derive_readouts,get_scorecard}.py`
  - `application/execution/{train_protocol,predict_with_protocol,build_scorecard,sweeps,claim_run}.py`
- infrastructure:
  - `infrastructure/engines/{_scoring,chemprop_dmpnn,molformer_xl}.py`
  - `infrastructure/persistence/sqlalchemy/data/{models,repository}.py`
  - `infrastructure/runner/wire.py`
- interface:
  - `interface/routes/{datasets,engines,protocols,runs}.py`

**Modify (frontend)**
- `frontend/openapi.json` and `src/shared/lib/api/**` (regenerated)
- datasets:
  - `features/datasets/{types/index.ts,lib/draft-from-upload.ts,hooks/use-datasets.ts}`
  - `features/datasets/components/{dataset-wizard,dataset-list,dataset-detail,compound-browser,dataset-profile-view,validation-report-view}.tsx`
- engines: `features/engines/{types/index.ts,index.ts}`
- protocols:
  - `features/protocols/components/{train-protocol-form,protocol-detail}.tsx`
  - `features/protocols/hooks/use-protocols.ts`
- sweeps:
  - `features/sweeps/{lib/rank.ts,components/sweep-detail.tsx,components/sweep-form.tsx,index.ts}`
- runs: `features/runs/components/{triage-grid,predict-wizard}.tsx`
- e2e: `tests/e2e/api-mock.ts`

---

### Task 0: Rebase onto `main` and record a green baseline

The branch was cut at `6ec809d`. `main` has since gained `62deb95` (the `read_compound_ids` fix the handoff listed as queued). This task folds it in and records the baseline.

- [ ] **Step 1: Rebase**

```bash
cd /Users/sidx/workspace/daikon-studio
git checkout multi-task-labels
git rebase main
git log --oneline -5
```

Expected:
- The three `docs(plan)` commits plus this plan's commit sit on top of `62deb95`.
- No conflicts (the branch is docs only).

- [ ] **Step 2: Baseline gates**

```bash
cd backend && uv run ruff check src tests && uv run mypy src && uv run lint-imports && OMP_NUM_THREADS=1 uv run pytest -q
```

Expected:
- All pass. Docker must be running for the testcontainers Postgres.
- Note the test count. Every later task should only add to it.

---

### Task 1: Store a dataset's targets as an ordered list

This task covers the domain, persistence, migration 013 (the dataset half), the wire format, and every reader that only needs a mechanical change. Callers that genuinely need multi-target logic keep working through a **transitional** `Dataset.single_target()`, which raises on a multi-target dataset. Task 9 deletes it.

**Files:**
- Modify:
  - `backend/src/daikonstudio/domain/data/target.py:31-43`
  - `backend/src/daikonstudio/domain/data/dataset.py`
  - `backend/src/daikonstudio/domain/data/validation.py`
- Modify:
  - `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/models.py:25`
  - `backend/src/daikonstudio/infrastructure/persistence/sqlalchemy/data/repository.py:26-60`
- Create: `backend/alembic/versions/013_dataset_targets.py`
- Modify: `backend/src/daikonstudio/infrastructure/runner/wire.py` (the `TargetSpecWire` user `DatasetEnvelope`, and `ValidationReportWire`)
- Modify (application data):
  - `backend/src/daikonstudio/application/data/prepare_frame.py` (report shapes only)
  - `backend/src/daikonstudio/application/data/create_dataset.py`
  - `backend/src/daikonstudio/application/data/compound_ids.py:34`
  - `backend/src/daikonstudio/application/data/set_dataset_id_column.py:56`
  - `backend/src/daikonstudio/application/data/get_dataset_profile.py:164`
  - `backend/src/daikonstudio/application/data/get_dataset_compounds.py:97`
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (every `dataset.target`)
- Modify: `backend/src/daikonstudio/interface/routes/datasets.py` (`ConflictRowResponse`, `ValidationReportResponse`, `DatasetResponse.from_domain`)
- Test:
  - `backend/tests/integration/test_migrations.py`
  - `backend/tests/unit/data/test_dataset.py`
  - `backend/tests/unit/data/test_validation.py`

**Interfaces:**
- Produces:
  - `Dataset(…, targets: tuple[TargetSpec, ...], …)`
  - `Dataset.target_columns -> tuple[str, ...]`
  - `Dataset.single_target() -> TargetSpec` (transitional)
  - `check_id_column(columns, *, id_column, structure_column, target_columns: Sequence[str])`
  - `ConflictRow(structure, column, values, row_numbers)`
  - `ValidationReport.duplicate_spread: dict[str, float]` (keyed by target column; numeric targets that had replicates only)
  - Migration 013 module constants `TARGETS_FROM_TARGET`, `KEY_SPREAD_BY_TARGET`, `TAG_CONFLICT_COLUMNS`, `COUNT_MULTI_TARGET`, `TARGET_FROM_TARGETS`, `UNKEY_SPREAD`, `UNTAG_CONFLICT_COLUMNS`
- Consumes: nothing new.

- [ ] **Step 1: Write the failing migration test**

Append to `backend/tests/integration/test_migrations.py` (add `import json` at the top):

```python
def _load_migration(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(f"alembic/versions/{name}.py"))
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


async def _json(session, sql: str, **params) -> object:
    # `::text` then json.loads: independent of whichever jsonb codec the driver uses.
    return json.loads(await session.scalar(text(sql), params))


_SINGLE_TARGET = {"column": "y", "kind": "numeric", "unit": "nM", "direction": "low"}
_REPORT_012 = {
    "total_rows": 3,
    "valid_rows": 3,
    "invalid": [],
    "conflicting": [{"structure": "CCO", "values": [0, 1], "row_numbers": [1, 2]}],
    "duplicates_collapsed": 1,
    "salts_flagged": 0,
    "duplicate_spread": 0.25,
}


@pytest.mark.asyncio
async def test_013_moves_a_single_target_into_a_list_and_back(migrated_session):
    migration = _load_migration("013_dataset_targets")
    # Re-create the 012 column beside the 013 one, inside this test's rolled-back
    # transaction, so both directions run against real rows.
    await migrated_session.execute(text("ALTER TABLE datasets ADD COLUMN target JSONB"))
    dataset_id = uuid.uuid4()
    await migrated_session.execute(
        text(
            "INSERT INTO datasets (id, workspace_id, name, structure_column, target, targets,"
            " split, content_hash, snapshot_uri, row_count, validation_report, version,"
            " created_at, updated_at) VALUES (:id, :ws, 'd', 'smiles',"
            " CAST(CAST(:target AS text) AS jsonb), '[]', '{}', 'h', 'x', 3,"
            " CAST(CAST(:report AS text) AS jsonb), 1, now(), now())"
        ),
        {
            "id": dataset_id,
            "ws": uuid.uuid4(),
            "target": json.dumps(_SINGLE_TARGET),
            "report": json.dumps(_REPORT_012),
        },
    )

    for statement in (
        migration.TARGETS_FROM_TARGET,
        migration.KEY_SPREAD_BY_TARGET,
        migration.TAG_CONFLICT_COLUMNS,
    ):
        await migrated_session.execute(text(statement))

    assert await _json(
        migrated_session, "SELECT targets::text FROM datasets WHERE id = :id", id=dataset_id
    ) == [_SINGLE_TARGET]
    report = await _json(
        migrated_session,
        "SELECT validation_report::text FROM datasets WHERE id = :id",
        id=dataset_id,
    )
    assert report["duplicate_spread"] == {"y": 0.25}
    assert report["conflicting"][0]["column"] == "y"

    await migrated_session.execute(text("UPDATE datasets SET target = NULL WHERE id = :id"), {"id": dataset_id})
    for statement in (
        migration.TARGET_FROM_TARGETS,
        migration.UNKEY_SPREAD,
        migration.UNTAG_CONFLICT_COLUMNS,
    ):
        await migrated_session.execute(text(statement))

    assert await _json(
        migrated_session, "SELECT target::text FROM datasets WHERE id = :id", id=dataset_id
    ) == _SINGLE_TARGET
    assert await _json(
        migrated_session,
        "SELECT validation_report::text FROM datasets WHERE id = :id",
        id=dataset_id,
    ) == _REPORT_012


@pytest.mark.asyncio
async def test_013_downgrade_counts_the_datasets_it_would_truncate(migrated_session):
    migration = _load_migration("013_dataset_targets")
    await migrated_session.execute(
        text(
            "INSERT INTO datasets (id, workspace_id, name, structure_column, targets, split,"
            " content_hash, snapshot_uri, row_count, validation_report, version, created_at,"
            " updated_at) VALUES (:id, :ws, 'd', 'smiles',"
            ' \'[{"column": "a", "kind": "binary"}, {"column": "b", "kind": "binary"}]\','
            " '{}', 'h2', 'x', 3, '{}', 1, now(), now())"
        ),
        {"id": uuid.uuid4(), "ws": uuid.uuid4()},
    )
    assert await migrated_session.scalar(text(migration.COUNT_MULTI_TARGET)) == 1
```

Also refactor `test_011_backfills_each_protocols_creator_from_its_training_run` to use `_load_migration("011_created_by")`.

- [ ] **Step 2: Write the failing domain tests**

Create `backend/tests/unit/data/test_dataset.py`:

```python
import uuid

import pytest

from daikonstudio.domain.data.dataset import Dataset, check_id_column
from daikonstudio.domain.data.split import SplitSpec, SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import ValidationReport
from daikonstudio.domain.shared.errors import ValidationError


def _dataset(*columns: str) -> Dataset:
    return Dataset(
        workspace_id=uuid.uuid4(),
        name="d",
        structure_column="smiles",
        targets=tuple(TargetSpec(column=c, kind=TargetKind.BINARY) for c in columns),
        split=SplitSpec(strategy=SplitStrategy.RANDOM, seed=1),
        content_hash="h",
        snapshot_uri="x",
        row_count=1,
        validation_report=ValidationReport(total_rows=1, valid_rows=1),
    )


def test_target_columns_keep_the_order_chosen():
    assert _dataset("b", "a").target_columns == ("b", "a")


def test_single_target_refuses_a_dataset_with_several():
    assert _dataset("y").single_target().column == "y"
    with pytest.raises(ValidationError):
        _dataset("a", "b").single_target()


def test_an_identifier_may_not_be_any_of_the_targets():
    with pytest.raises(ValidationError):
        check_id_column(
            ["smiles", "a", "b", "id"],
            id_column="b",
            structure_column="smiles",
            target_columns=("a", "b"),
        )
```

Add to `backend/tests/unit/data/test_validation.py`:

```python
def test_a_conflict_names_the_target_column_its_labels_disagree_in():
    frame = pl.DataFrame({"smiles": ["CCO", "OCC"], "y": [0, 1]})
    _, report = prepare_frame(frame, "smiles", BINARY, NORMALIZER)
    assert report.conflicting[0].column == "y"
```

Then update the existing spread assertion in `test_numeric_duplicates_are_averaged_and_spread_retained` to `assert report.duplicate_spread == {"y": pytest.approx(2.0)}`.

- [ ] **Step 3: Run the new tests to verify they fail**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/data/test_dataset.py tests/unit/data/test_validation.py -k "order or several or identifier or names_the_target or averaged" tests/integration/test_migrations.py -q
```

Expected:
- `test_dataset.py` fails at import (`targets` is an unexpected keyword).
- The migration test fails with `FileNotFoundError` for `013_dataset_targets.py`.
- The validation tests fail on `.column` and on the dict comparison.

- [ ] **Step 4: Domain changes**

In `domain/data/target.py`, add one entry to `RESERVED_TARGET_COLUMNS` (inside the set, after `"compound_id"`):

```python
        # The long-format column every engine's `predict()` output carries once a
        # Dataset can hold several targets (`application/engines/fan_out.py`). Never
        # persisted today; reserved for the same defensive reason as `row_id`.
        "target",
```

In `domain/data/dataset.py`:
- Replace the `target: TargetSpec` parameter with `targets: tuple[TargetSpec, ...]`.
- Replace `self.target = target` with the block below.
- Add the property and the method.
- Replace `check_id_column`.

```python
        # What the scientist is predicting, in the order they chose the columns, and
        # never empty (`check_targets`, at creation). There is deliberately no `target`
        # shortcut to the first one: a caller that took it would train on one target
        # and silently drop the rest.
        self.targets = targets
```

```python
    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(target.column for target in self.targets)

    def single_target(self) -> TargetSpec:
        """Transitional, deleted by Task 9 of the multi-task plan: the one target of a
        one-target Dataset. Raises rather than picking the first, for the reason
        `targets` gives."""
        if len(self.targets) != 1:
            raise ValidationError("This step does not handle datasets with several targets yet.")
        return self.targets[0]


def check_id_column(
    columns: Sequence[str],
    *,
    id_column: str,
    structure_column: str,
    target_columns: Sequence[str],
) -> None:
    """An identifier is any stored column but the ones that already mean something."""
    if id_column in (structure_column, "split") or id_column in target_columns:
        raise ValidationError(
            "Choose an identifier column other than the structure, target or split column."
        )
    if id_column not in columns:
        raise ValidationError(f"Column '{id_column}' is not in the uploaded file.")
```

In `domain/data/validation.py`:
- Add `column: str` to `ConflictRow`, after `structure`, with a short comment: a compound can conflict in one binary target and agree in another, and the file is fixed in that column.
- Change `ValidationReport.duplicate_spread` to:

```python
    # Keyed by target column. Only numeric targets with at least one replicate group
    # appear: a binary target has no spread, and a column with no replicates has
    # nothing to measure, which is different from a spread of zero.
    duplicate_spread: dict[str, float] = field(default_factory=dict)
```

and in `report_from_dict`: `duplicate_spread=dict(data.get("duplicate_spread") or {}),`.

- [ ] **Step 5: Migration 013 (dataset half)**

Create `backend/alembic/versions/013_dataset_targets.py`:

```python
"""datasets.targets, and the per-target shapes that come with it

A Dataset carries an ordered list of target specs instead of exactly one; every
existing row becomes a one-element list, in place. Two values inside
`validation_report` change shape in the same step, because each only means
something per target once there can be several:

- `duplicate_spread`, a number or null, becomes an object keyed by target column
  (empty when there was nothing to measure);
- every `conflicting` entry gains the `column` its labels disagree in.

Downgrade restores the single-target shapes, and refuses while any dataset has
more than one target: dropping the others would silently destroy data that
protocols and runs still cite.

Revision ID: 013
Revises: 012
Create Date: 2026-10-03 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | None = None
depends_on: str | None = None

TARGETS_FROM_TARGET = "UPDATE datasets SET targets = jsonb_build_array(target)"

KEY_SPREAD_BY_TARGET = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{duplicate_spread}',
    CASE
        WHEN jsonb_typeof(validation_report -> 'duplicate_spread') = 'number'
        THEN jsonb_build_object(target ->> 'column', validation_report -> 'duplicate_spread')
        ELSE '{}'::jsonb
    END
)
"""

TAG_CONFLICT_COLUMNS = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{conflicting}',
    (
        SELECT COALESCE(
            jsonb_agg(entry || jsonb_build_object('column', target ->> 'column')), '[]'::jsonb
        )
        FROM jsonb_array_elements(validation_report -> 'conflicting') AS entry
    )
)
WHERE jsonb_typeof(validation_report -> 'conflicting') = 'array'
"""

COUNT_MULTI_TARGET = "SELECT count(*) FROM datasets WHERE jsonb_array_length(targets) > 1"

TARGET_FROM_TARGETS = "UPDATE datasets SET target = targets -> 0"

UNKEY_SPREAD = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{duplicate_spread}',
    COALESCE(validation_report -> 'duplicate_spread' -> (target ->> 'column'), 'null'::jsonb)
)
"""

UNTAG_CONFLICT_COLUMNS = """
UPDATE datasets
SET validation_report = jsonb_set(
    validation_report,
    '{conflicting}',
    (
        SELECT COALESCE(jsonb_agg(entry - 'column'), '[]'::jsonb)
        FROM jsonb_array_elements(validation_report -> 'conflicting') AS entry
    )
)
WHERE jsonb_typeof(validation_report -> 'conflicting') = 'array'
"""


def upgrade() -> None:
    op.add_column("datasets", sa.Column("targets", postgresql.JSONB(), nullable=True))
    op.execute(TARGETS_FROM_TARGET)
    op.execute(KEY_SPREAD_BY_TARGET)
    op.execute(TAG_CONFLICT_COLUMNS)
    op.alter_column("datasets", "targets", nullable=False)
    op.drop_column("datasets", "target")


def downgrade() -> None:
    if op.get_bind().execute(sa.text(COUNT_MULTI_TARGET)).scalar_one():
        raise RuntimeError(
            "Cannot downgrade below 013: some datasets have more than one target, and the "
            "single-target schema would silently drop the others. Delete those datasets "
            "and their protocols first."
        )
    op.add_column("datasets", sa.Column("target", postgresql.JSONB(), nullable=True))
    op.execute(TARGET_FROM_TARGETS)
    op.execute(UNKEY_SPREAD)
    op.execute(UNTAG_CONFLICT_COLUMNS)
    op.alter_column("datasets", "target", nullable=False)
    op.drop_column("datasets", "targets")
```

- [ ] **Step 6: Persistence and wire**

**`data/models.py`:** replace the `target` column with:
```python
    # Ordered as the scientist chose them; see `Dataset.targets`.
    targets: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
```

**`data/repository.py`:**
- In `_to_domain`: `targets=tuple(target_from_dict(t) for t in model.targets),`
- In `_to_model`: `targets=[target_to_dict(t) for t in dataset.targets],`

**`runner/wire.py`:**
- In `DatasetEnvelope`:
  - Field: `targets: list[TargetSpecWire]`
  - `from_domain`: `targets=[TargetSpecWire.from_domain(t) for t in dataset.targets],`
  - `to_domain`: `targets=tuple(t.to_domain() for t in self.targets),`
- In `ValidationReportWire`: `duplicate_spread: dict[str, float] = {}`. Pydantic copies mutable defaults, so this is safe.

- [ ] **Step 7: Mechanical reader updates**

**`prepare_frame.py`:**
- Numeric branch: replace the `duplicate_spread = …` line with
  ```python
  duplicate_spread = {target.column: sum(spreads) / len(spreads)} if spreads else {}
  ```
- Binary branch: add `column=target.column,` to the `ConflictRow(...)` call.

**`create_dataset.py`:**
- Build the Dataset with `targets=(command.target,)`.
- Call `check_id_column(..., target_columns=(command.target.column,))`.
- The command keeps `target` until Task 2.

**`compound_ids.py:34`:** `reserved = {dataset.structure_column, *dataset.target_columns, "split"}`

**`set_dataset_id_column.py:56`:** `target_columns=dataset.target_columns,`

**`train_protocol.py`, `get_dataset_profile.py`, `get_dataset_compounds.py`:**
- Replace every `dataset.target` with `dataset.single_target()`:
  ```
  grep -n "dataset\.target\b" src/daikonstudio/application
  ```
  lists them all.
- In `train_protocol.py`, also change
  `duplicate_spread=dataset.validation_report.duplicate_spread`
  to
  `duplicate_spread=dataset.validation_report.duplicate_spread.get(dataset.single_target().column)`.

**`interface/routes/datasets.py`:**
- `ConflictRowResponse` gains `column: str`, with a comment matching the domain one.
- `ValidationReportResponse.duplicate_spread: dict[str, float]` (comment: keyed by target column).
- `DatasetResponse.from_domain` builds `target=` from `dataset.single_target()`. Task 2 makes it a list.

- [ ] **Step 8: Update fixtures that construct the old shapes**

Run:
```
cd backend && grep -rn "Dataset(\|ConflictRow(\|duplicate_spread" tests src | grep -v "^src/daikonstudio/domain"
```

Update each hit:
- `Dataset(... target=X ...)` → `targets=(X,)`. Leave `CreateDatasetCommand(target=...)` alone until Task 2.
- `ConflictRow(...)` → add `column="y"`, or whatever that fixture's target column is.
- `duplicate_spread=0.3` → `duplicate_spread={"y": 0.3}`.
- `duplicate_spread is None` / `== None` assertions on a report → `== {}`.
- Wire tests (`tests/unit/runners/test_wire.py`) and runner fixtures (`tests/helpers/runner_fixtures.py`) that build a `DatasetEnvelope` or a `Dataset` → `targets=`.

- [ ] **Step 9: Run tests and gates**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit tests/integration/test_migrations.py tests/api/test_datasets.py tests/api/test_runner_protocol.py -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add -A backend
git commit -m "feat(datasets): store a dataset's targets as an ordered list

Migration 013 turns datasets.target into a one-element targets array and
keys the stored assay spread and conflicts by target column.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Gate and deduplicate every target at upload

**Files:**
- Modify:
  - `backend/src/daikonstudio/domain/data/target.py`
  - `backend/src/daikonstudio/application/data/prepare_frame.py`
  - `backend/src/daikonstudio/application/data/create_dataset.py`
  - `backend/src/daikonstudio/interface/routes/datasets.py`
- Test:
  - `backend/tests/unit/data/test_targets.py` (create)
  - `backend/tests/unit/data/test_validation.py`
  - `backend/tests/api/test_datasets.py`

**Interfaces:**
- Consumes: Task 1's `Dataset(targets=…)`, `ConflictRow.column`, `ValidationReport.duplicate_spread: dict`.
- Produces:
  - `probability_column(column: str) -> str`
  - `uncertainty_column(column: str, *, target_count: int) -> str`
  - `prediction_columns(targets: Sequence[TargetSpec]) -> list[str]`
  - `check_targets(targets: Sequence[TargetSpec]) -> None`, which raises `ValidationError`
  - `prepare_frame(frame, structure_column, targets: Sequence[TargetSpec], normalizer)`
  - `CreateDatasetCommand.targets: tuple[TargetSpec, ...]`
  - HTTP `CreateDatasetBody.targets: list[TargetBody]` (min 1)
  - `DatasetResponse.targets: list[TargetBody]`

- [ ] **Step 1: Failing tests for the target rules**

Create `backend/tests/unit/data/test_targets.py`:

```python
import pytest

from daikonstudio.domain.data.target import (
    TargetKind,
    TargetSpec,
    check_targets,
    prediction_columns,
)
from daikonstudio.domain.shared.errors import ValidationError


def numeric(column: str) -> TargetSpec:
    return TargetSpec(column=column, kind=TargetKind.NUMERIC)


def binary(column: str) -> TargetSpec:
    return TargetSpec(column=column, kind=TargetKind.BINARY)


def test_several_distinct_targets_of_mixed_kinds_are_accepted():
    check_targets([numeric("solubility"), binary("reactive")])


def test_no_target_is_refused():
    with pytest.raises(ValidationError, match="at least one"):
        check_targets([])


def test_a_target_chosen_twice_is_refused():
    with pytest.raises(ValidationError, match="'y' is chosen more than once"):
        check_targets([numeric("y"), numeric("y")])


@pytest.mark.parametrize("column", ["target", "uncertainty", "row_id", "structure"])
def test_a_reserved_column_name_is_refused(column):
    with pytest.raises(ValidationError, match="cannot be used as a target column"):
        check_targets([numeric(column)])


def test_a_binary_target_beside_its_own_probability_column_is_refused():
    with pytest.raises(ValidationError, match="foo_probability"):
        check_targets([binary("foo"), numeric("foo_probability")])


def test_with_several_targets_an_uncertainty_column_can_clash_too():
    with pytest.raises(ValidationError, match="foo_uncertainty"):
        check_targets([numeric("foo"), numeric("foo_uncertainty")])


def test_one_target_writes_no_per_target_uncertainty_column():
    assert prediction_columns([binary("y")]) == ["y_probability", "y"]
    assert prediction_columns([binary("a"), numeric("b")]) == [
        "a_probability",
        "a",
        "a_uncertainty",
        "b",
        "b_uncertainty",
    ]
```

- [ ] **Step 2: Failing tests for the gate**

In `backend/tests/unit/data/test_validation.py`:

Change every existing `prepare_frame(frame, "smiles", NUMERIC, NORMALIZER)` / `BINARY` call to pass a tuple: `(NUMERIC,)` / `(BINARY,)`.

Update the three reason strings the gate now words per column:
- `"Missing target value"` → `"Missing value for target 'y'"`
- `"Target value is not numeric: 'NA'"` → `"Target 'y' is not numeric: 'NA'"`
- `"Binary target must be 0 or 1 (found 'active')"` → `"Target 'y' must be 0 or 1 (found 'active')"` (and likewise for `'2'`)

Append:

```python
SOLUBILITY = TargetSpec(column="solubility", kind=TargetKind.NUMERIC)
REACTIVE = TargetSpec(column="reactive", kind=TargetKind.BINARY)


def test_every_target_is_gated_and_the_reason_names_its_column():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "solubility": ["1.0", "2.0", "3.0"],
            "reactive": ["0", "", "1"],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (SOLUBILITY, REACTIVE), NORMALIZER)
    assert prepared.height == 2
    assert [row.reason for row in report.invalid] == ["Missing value for target 'reactive'"]


def test_duplicates_collapse_per_target_kind_and_keep_column_order():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "c1ccccc1"],
            "solubility": [1.0, 3.0, 9.0],
            "reactive": [1, 1, 0],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (SOLUBILITY, REACTIVE), NORMALIZER)
    assert prepared.columns == ["smiles", "solubility", "reactive"]
    row = prepared.filter(pl.col("smiles") == "CCO")
    assert row["solubility"].item() == 2.0
    assert row["reactive"].item() == 1
    assert report.duplicates_collapsed == 1
    assert report.duplicate_spread == {"solubility": pytest.approx(2.0)}


def test_a_conflict_in_one_binary_target_drops_the_compound_and_names_the_column():
    frame = pl.DataFrame(
        {
            "smiles": ["CCO", "OCC", "CCN"],
            "solubility": [1.0, 1.0, 2.0],
            "reactive": [0, 1, 0],
        }
    )
    prepared, report = prepare_frame(frame, "smiles", (SOLUBILITY, REACTIVE), NORMALIZER)
    assert prepared["smiles"].to_list() == ["CCN"]
    assert [(c.column, c.values) for c in report.conflicting] == [("reactive", [0, 1])]


def test_a_column_name_with_braces_is_reported_verbatim():
    target = TargetSpec(column="IC50 {nM}", kind=TargetKind.NUMERIC)
    frame = pl.DataFrame({"smiles": ["CCO", "CCN"], "IC50 {nM}": ["1.0", "NA"]})
    _, report = prepare_frame(frame, "smiles", (target,), NORMALIZER)
    assert report.invalid[0].reason == "Target 'IC50 {nM}' is not numeric: 'NA'"
```

- [ ] **Step 3: Failing API tests**

In `backend/tests/api/test_datasets.py`:
- Change `create_body` to emit `"targets": [NUMERIC_TARGET]`.
- Change every `target={...}` override in this file to `targets=[{...}]`.

Then append:

```python
TWO_TARGET_CSV = (
    b"smiles,solubility,reactive\n"
    b"CCO,1.0,0\nc1ccccc1,5.0,1\nCCN,2.0,0\nc1ccncc1,6.0,1\nCCCO,1.5,1\n"
    b"Cc1ccccc1,5.5,0\nCCCN,2.5,1\nc1ccsc1,6.5,0\nCCCCO,1.2,0\nC1CCCCC1,4.0,1\n"
)


async def test_a_dataset_keeps_several_targets_in_the_order_chosen(client, csv_upload):
    upload_ref = await csv_upload(TWO_TARGET_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "reactive", "kind": "binary"},
                {"column": "solubility", "kind": "numeric", "unit": "logS", "direction": "high"},
            ],
        ),
    )
    assert response.status_code == 201, response.text
    assert [t["column"] for t in response.json()["targets"]] == ["reactive", "solubility"]


async def test_a_dataset_is_refused_when_any_one_target_is_degenerate(client, csv_upload):
    header, *rows = TWO_TARGET_CSV.decode().strip().split("\n")
    csv = "\n".join([f"{header},flag", *(f"{row},0" for row in rows)]) + "\n"
    upload_ref = await csv_upload(csv.encode())
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "solubility", "kind": "numeric"},
                {"column": "flag", "kind": "binary"},
            ],
        ),
    )
    assert response.status_code == 422, response.text
    assert "'flag'" in response.text


async def test_the_structure_column_cannot_also_be_a_target(client, csv_upload):
    upload_ref = await csv_upload(TWO_TARGET_CSV)
    response = await client.post(
        "/api/v1/datasets",
        json=create_body(upload_ref, targets=[{"column": "smiles", "kind": "numeric"}]),
    )
    assert response.status_code == 422, response.text
```

- [ ] **Step 4: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/data/test_targets.py tests/unit/data/test_validation.py tests/api/test_datasets.py -q
```

Expected: FAIL. There are import errors for `check_targets` and `prediction_columns`, the `prepare_frame` signature does not match, and the API returns 422 for `targets`.

- [ ] **Step 5: Target rules in the domain**

Append to `domain/data/target.py` (add `from collections.abc import Sequence` and `from daikonstudio.domain.shared.errors import ValidationError`):

```python
def probability_column(column: str) -> str:
    """The readout a binary target's P(class=1) is written under. The one definition,
    shared by `derive_readouts` and the collision check below."""
    return f"{column}_probability"


def uncertainty_column(column: str, *, target_count: int) -> str:
    """Where prediction results store one target's per-compound uncertainty.

    Plain `uncertainty` for a one-target Protocol: it is the name every results file
    written before several targets existed already uses, so those stay readable.
    `{column}_uncertainty` beside each target otherwise, because one number per row
    cannot describe four models.
    """
    return "uncertainty" if target_count == 1 else f"{column}_uncertainty"


def prediction_columns(targets: Sequence[TargetSpec]) -> list[str]:
    """Every column a prediction for these targets writes, besides the fixed ones."""
    names: list[str] = []
    for target in targets:
        if target.kind is TargetKind.BINARY:
            names.append(probability_column(target.column))
        names.append(target.column)
        if len(targets) > 1:
            names.append(uncertainty_column(target.column, target_count=len(targets)))
    return names


def check_targets(targets: Sequence[TargetSpec]) -> None:
    """The invariants a Dataset's targets hold, checked once at creation.

    Every one of these is a silent overwrite downstream if it slips through: a
    prediction results frame is a dict of columns, so two derived names that
    collide keep whichever was written last.
    """
    if not targets:
        raise ValidationError("Choose at least one column to predict.")
    columns = [target.column for target in targets]
    for column in columns:
        if column in RESERVED_TARGET_COLUMNS:
            raise ValidationError(
                f"'{column}' cannot be used as a target column",
                detail=(
                    "The application writes a column with this name to prediction "
                    "results and exports. Rename the column in your file. "
                    f"Reserved names: {', '.join(sorted(RESERVED_TARGET_COLUMNS))}."
                ),
            )
    repeated = sorted({column for column in columns if columns.count(column) > 1})
    if repeated:
        raise ValidationError(f"'{repeated[0]}' is chosen more than once as a target.")
    names = prediction_columns(targets)
    clashing = sorted({name for name in names if names.count(name) > 1})
    if clashing:
        raise ValidationError(
            f"Two targets would write the same prediction column, '{clashing[0]}'.",
            detail=(
                "A binary target named x is predicted as x and x_probability, and with "
                "several targets each one also gets x_uncertainty. Rename one of the "
                "columns in your file."
            ),
        )
```

- [ ] **Step 6: Per-target gate and dedup**

Replace `_validate_target` and `prepare_frame` in `application/data/prepare_frame.py`. Keep the module docstring and `read_csv_upload`. Add `from collections.abc import Sequence`.

```python
def _validate_target(
    frame: pl.DataFrame, target: TargetSpec, row_numbers: list[int]
) -> tuple[pl.DataFrame, list[int], list[InvalidRow]]:
    """Rows whose target cannot be trained on, rejected here with their row
    numbers rather than as `Input y contains NaN` minutes later in a worker.

    Returns the frame with the target cast (Float64 for NUMERIC, Int64 for
    BINARY), the surviving row numbers, and one InvalidRow per rejected row.
    Three reasons, in the words a scientist needs: an empty cell, text where a
    number belongs (`NA`, `<10`, `12,5`), or a binary label that is not 0 or 1.
    Each names its column, since with several targets "missing value" alone does
    not say which cell to fill. Built with f-strings, never `str.format`: a column
    named `IC50 {nM}` would otherwise be read as a format field.
    """
    column = target.column
    raw = frame[column]
    text = raw.cast(pl.String, strict=False).fill_null("").str.strip_chars()
    numeric = (
        raw.str.strip_chars().cast(pl.Float64, strict=False)
        if raw.dtype == pl.String
        else raw.cast(pl.Float64, strict=False)
    )
    empty = text == ""
    if target.kind is TargetKind.BINARY:
        ok = numeric.is_in([0.0, 1.0]).fill_null(False) & ~empty

        def reason(value: str) -> str:
            return f"Target '{column}' must be 0 or 1 (found '{value}')"

        cast_to: pl.DataType = pl.Int64()
    else:
        # is_finite, not is_not_null: polars parses "nan" and "inf" to floats that
        # are not null, and either one is the `Input y contains NaN` failure this
        # gate exists to stop.
        ok = numeric.is_finite().fill_null(False) & ~empty

        def reason(value: str) -> str:
            return f"Target '{column}' is not numeric: '{value}'"

        cast_to = pl.Float64()
    invalid = [
        InvalidRow(
            row_number=row_numbers[index],
            value=text[index],
            reason=f"Missing value for target '{column}'" if empty[index] else reason(text[index]),
        )
        for index in range(frame.height)
        if not ok[index]
    ]
    kept = frame.filter(ok).with_columns(numeric.filter(ok).cast(cast_to).alias(column))
    kept_rows = [number for number, keep in zip(row_numbers, ok.to_list(), strict=True) if keep]
    return kept, kept_rows, invalid


def prepare_frame(
    frame: pl.DataFrame,
    structure_column: str,
    targets: Sequence[TargetSpec],
    normalizer: StructureNormalizer,
) -> tuple[pl.DataFrame, ValidationReport]:
    # ... unchanged from the current body down to and including the `valid_frame = ...`
    # line that writes the canonical structures back ...

    # The target gate runs after the structure gate so a row that fails both is
    # reported once, for its structure -- the thing the scientist fixes first. The
    # targets are gated in the order chosen, and a row is reported for the first one
    # it fails.
    for target in targets:
        valid_frame, row_numbers, bad_targets = _validate_target(valid_frame, target, row_numbers)
        invalid.extend(bad_targets)
    invalid.sort(key=lambda row: row.row_number)
    valid_rows = valid_frame.height

    # ... `salts_flagged` and the `valid_rows == 0` early return, unchanged ...

    target_columns = {target.column for target in targets}
    other_columns = [
        c for c in frame.columns if c != structure_column and c not in target_columns
    ]
    # (keep the existing ponytail comment about extra columns here)
    keep_others = [pl.col(c).first() for c in other_columns]

    # (keep the existing comment above `row_numbers_by_structure`)
    row_numbers_by_structure: dict[str, list[int]] = {}
    for structure, row_number in zip(
        valid_frame[structure_column].to_list(), row_numbers, strict=True
    ):
        row_numbers_by_structure.setdefault(structure, []).append(row_number)

    # One pass over the duplicate groups covers every target. A measured value is
    # averaged and its replicate spread kept; a binary label must agree across the
    # replicates, and a structure whose labels disagree in any binary target is a
    # data problem for the scientist to resolve, not one a majority vote papers over.
    # Output column order -- structure, the targets in order, then the rest -- is
    # what it was with one target, so a one-target upload freezes to the same bytes
    # and the same `content_hash` as before targets could be several.
    aggregations: list[pl.Expr] = [pl.len().alias("_n")]
    helpers = ["_n"]
    for index, target in enumerate(targets):
        values = pl.col(target.column)
        if target.kind is TargetKind.NUMERIC:
            aggregations += [
                (values.max() - values.min()).alias(f"_spread_{index}"),
                values.mean().alias(target.column),
            ]
            helpers.append(f"_spread_{index}")
        else:
            aggregations += [
                values.n_unique().alias(f"_n_unique_{index}"),
                values.alias(f"_values_{index}"),
                values.first().alias(target.column),
            ]
            helpers += [f"_n_unique_{index}", f"_values_{index}"]
    grouped = valid_frame.group_by(structure_column, maintain_order=True).agg(
        *aggregations, *keep_others
    )

    conflicting: list[ConflictRow] = []
    is_conflict = pl.Series([False] * grouped.height)
    for index, target in enumerate(targets):
        if target.kind is not TargetKind.BINARY:
            continue
        clash = grouped[f"_n_unique_{index}"] > 1
        conflicting += [
            ConflictRow(
                structure=str(row[structure_column]),
                column=target.column,
                values=list(row[f"_values_{index}"]),
                row_numbers=sorted(row_numbers_by_structure[str(row[structure_column])]),
            )
            for row in grouped.filter(clash).iter_rows(named=True)
        ]
        is_conflict = is_conflict | clash
    agreeing = grouped.filter(~is_conflict)

    group_sizes = agreeing["_n"].to_list()
    # Groups of size one carry no spread information and are excluded from the
    # mean entirely -- they don't count as "zero spread", they count as nothing.
    duplicate_spread: dict[str, float] = {}
    for index, target in enumerate(targets):
        if target.kind is not TargetKind.NUMERIC:
            continue
        spreads = [
            float(spread)
            for n, spread in zip(group_sizes, agreeing[f"_spread_{index}"].to_list(), strict=True)
            if n > 1
        ]
        if spreads:
            duplicate_spread[target.column] = sum(spreads) / len(spreads)

    return agreeing.drop(helpers), ValidationReport(
        total_rows=total_rows,
        valid_rows=valid_rows,
        invalid=invalid,
        conflicting=conflicting,
        duplicates_collapsed=sum(n - 1 for n in group_sizes),
        salts_flagged=salts_flagged,
        duplicate_spread=duplicate_spread,
    )
```

The two old kind-specific branches (`if target.kind is TargetKind.NUMERIC: …` and the BINARY block after it) are deleted. The loop above replaces both.

- [ ] **Step 7: `CreateDataset` over targets**

In `application/data/create_dataset.py`:
- `CreateDatasetCommand.target: TargetSpec` → `targets: tuple[TargetSpec, ...]`.
- In `__call__`, replace the reserved-name block (keep its C1 comment, re-pointed at `check_targets`) with:

```python
        try:
            check_targets(command.targets)
        except ValidationError as error:
            return Failure(error)
        target_columns = [target.column for target in command.targets]
        if command.structure_column in target_columns:
            return Failure(
                ValidationError("The structure column cannot also be a column to predict.")
            )
```

Then:
- `missing` iterates over `(command.structure_column, *target_columns)`.
- `check_id_column(..., target_columns=target_columns)`.
- `prepare_frame(frame, command.structure_column, command.targets, self._normalizer)`.
- The degenerate guard runs per target:

```python
        for target in command.targets:
            degenerate = _degenerate_partition(split_frame, target)
            if degenerate is not None:
                return Failure(degenerate)
```

- The Dataset is built with `targets=command.targets`.
- In `_degenerate_partition`, name the column in the message, not only in the detail:
  `f"After splitting, every compound in the {partition} set has the same '{target.column}' {kind}. A model trained or evaluated on it would not be meaningful."`
- Remove the now-unused `RESERVED_TARGET_COLUMNS` import, and import `check_targets`.

- [ ] **Step 8: HTTP body and response**

In `interface/routes/datasets.py`:
- `CreateDatasetBody.target: TargetBody` → `targets: list[TargetBody] = Field(min_length=1)`.
- The route builds `targets=tuple(TargetSpec(column=t.column, kind=t.kind, unit=t.unit, direction=t.direction) for t in body.targets)`.
- `DatasetResponse.target` → `targets: list[TargetBody]`, and in `from_domain`: `targets=[TargetBody.model_validate(target_to_dict(t)) for t in dataset.targets],`.

- [ ] **Step 9: Update fixtures that build the old command/body**

Run:
```
cd backend && grep -rn "CreateDatasetCommand(\|\"target\": {\|\"target\":{" tests
```

For each hit:
- `target=TargetSpec(...)` in a `CreateDatasetCommand` → `targets=(TargetSpec(...),)`.
- `"target": {...}` in an HTTP JSON body → `"targets": [{...}]`.

Files known to need it:
- `tests/api/test_protocols.py:73`
- `tests/api/test_runs.py:65`
- `tests/integration/test_train_protocol.py` (`Studio.dataset`)
- `tests/integration/test_sweeps.py` (the `dataset` fixture)
- `tests/integration/test_full_loop.py`
- `tests/api/test_dataset_id_column.py`
- `tests/api/test_triage_round_trip.py`

- [ ] **Step 10: Run tests and gates**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit tests/api tests/integration/test_train_protocol.py -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected:
- All pass.
- Any test that pins a snapshot `content_hash` passes **unchanged**. A failure there means the column order or dtype moved, and the fix is in `prepare_frame`, not in the test.

- [ ] **Step 11: Commit**

```bash
git add -A backend
git commit -m "feat(datasets): gate, deduplicate and check every target at upload

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Dataset compounds and profile, per target

**Files:**
- Modify:
  - `backend/src/daikonstudio/application/data/get_dataset_compounds.py`
  - `backend/src/daikonstudio/application/data/get_dataset_profile.py`
  - `backend/src/daikonstudio/interface/routes/datasets.py` (`CompoundResponse`, the compounds and profile routes)
- Test:
  - `backend/tests/api/test_datasets.py`
  - `backend/tests/api/test_dataset_profile.py`

**Interfaces:**
- Consumes: `Dataset.targets` and `Dataset.target_columns` (Task 1).
- Produces:
  - `Compound.targets: dict[str, float | None]`
  - `GetDatasetCompoundsQuery.target: int = 0` (the index that `sort="target"` orders by)
  - HTTP `GET /datasets/{id}/compounds?sort=target&target=<index>`; `CompoundResponse.targets`
  - `profile_key(workspace_id, dataset_id, *, target: int) -> str`
  - `GetDatasetProfileQuery.target: int = 0`
  - HTTP `GET /datasets/{id}/profile?target=<index>`

- [ ] **Step 1: Failing tests**

Append to `tests/api/test_datasets.py`:

```python
async def test_compounds_carry_every_target_and_sort_by_any_one(client, csv_upload):
    upload_ref = await csv_upload(TWO_TARGET_CSV)
    created = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "solubility", "kind": "numeric"},
                {"column": "reactive", "kind": "binary"},
            ],
        ),
    )
    dataset_id = created.json()["id"]
    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/compounds",
        params={"sort": "target", "target": 1, "sort_dir": "desc"},
    )
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert set(items[0]["targets"]) == {"solubility", "reactive"}
    assert items[0]["targets"]["reactive"] == 1.0
    assert items[-1]["targets"]["reactive"] == 0.0

    out_of_range = await client.get(
        f"/api/v1/datasets/{dataset_id}/compounds", params={"sort": "target", "target": 2}
    )
    assert out_of_range.status_code == 422
```

In `tests/api/test_dataset_profile.py`:
- Give `_until_ready` a `params: dict | None = None` argument, passed to `client.get`.
- Append:

```python
async def test_each_target_has_its_own_profile(client, csv_upload):
    from tests.api.test_datasets import TWO_TARGET_CSV, create_body

    upload_ref = await csv_upload(TWO_TARGET_CSV)
    created = await client.post(
        "/api/v1/datasets",
        json=create_body(
            upload_ref,
            targets=[
                {"column": "solubility", "kind": "numeric"},
                {"column": "reactive", "kind": "binary"},
            ],
        ),
    )
    dataset_id = created.json()["id"]

    first = await _until_ready(client, dataset_id)
    second = await _until_ready(client, dataset_id, params={"target": 1})

    assert first.json()["target_kind"] == "numeric"
    assert second.json()["target_kind"] == "binary"
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/api/test_datasets.py -k compounds_carry tests/api/test_dataset_profile.py -k own_profile -q
```

Expected:
- `single_target()` raises, so the request returns 422/500.
- Or the unknown `target` param is ignored and the assertions fail.

- [ ] **Step 3: Compounds**

In `get_dataset_compounds.py`:
- Update the module docstring's "structure, target, partition" to "structure, every target, partition".
- `Compound.target: float | None` → `targets: dict[str, float | None]`, with a comment: keyed by target column, in the Dataset's order.
- `GetDatasetCompoundsQuery` gains `target: int = 0`, with a comment: which target `sort="target"` orders by, as an index into `Dataset.targets`. An index rather than a name, so no client-supplied string ever reaches the frame.
- In `__call__`, after the dataset lookup:

```python
        if not 0 <= query.target < len(dataset.targets):
            return Failure(
                ValidationError(f"Target index {query.target} is out of range for this dataset.")
            )
        columns = dataset.target_columns
```

In the `select`, replace the single target expression with `*(pl.col(column).cast(pl.Float64, strict=False) for column in columns),`. The reserved-name check guarantees no target is named `structure`, `split` or `compound_id`.

The sort becomes:

```python
        if query.sort is not None:
            key = columns[query.target] if query.sort == "target" else "split"
            frame = frame.sort(
                [key, "structure"],
                descending=[query.descending, False],
                nulls_last=[True, False],
            )
```

and each item becomes:

```python
                    Compound(
                        structure=str(row["structure"]),
                        targets={
                            column: None if row[column] is None else float(row[column])
                            for column in columns
                        },
                        split=str(row["split"]),
                        compound_id=row["compound_id"],
                    )
```

In the route:
- `CompoundResponse.target` → `targets: dict[str, float | None]`, with `from_domain` passing `targets=compound.targets`.
- Add the query param `target: Annotated[int, Query(ge=0)] = 0` and pass `target=target`.

- [ ] **Step 4: Profile**

In `get_dataset_profile.py`:

```python
def profile_key(workspace_id: uuid.UUID | str, dataset_id: uuid.UUID | str, *, target: int) -> str:
    """One profile per target. Target 0's keeps the name every profile had before a
    Dataset could hold several targets, so those stay cached. An index, never the
    column name, so nothing a CSV header contains reaches a blob key."""
    name = "profile.json" if target == 0 else f"profile-{target}.json"
    return f"{workspace_id}/datasets/{dataset_id}/{name}"
```

Then:
- `GetDatasetProfileQuery` gains `target: int = 0`.
- `_Key = tuple[uuid.UUID, uuid.UUID, int]`.
- After the dataset lookup, refuse out-of-range indexes with the same `ValidationError` as the compounds route.
- `key = profile_key(dataset.workspace_id, dataset.id, target=query.target)`.
- `running_key = (dataset.workspace_id, dataset.id, query.target)`.
- `_compute` passes `target=dataset.targets[key[2]]` to `build_profile` and saves under `profile_key(..., target=key[2])`.
- Delete the `single_target()` uses in this file.
- Add one `ponytail:` line to the module docstring: the structure-only sections (similarity, scaffolds, descriptors) are recomputed per target; split the profile into structure and per-target halves if multi-target datasets are profiled often.

In the route, add `target: Annotated[int, Query(ge=0)] = 0` and pass it to the query.

- [ ] **Step 5: Run tests and gates**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/api/test_datasets.py tests/api/test_dataset_profile.py tests/unit/data -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS. Existing compounds tests that read `item["target"]` must be updated to `item["targets"]["y"]`.

- [ ] **Step 6: Commit**

```bash
git add -A backend
git commit -m "feat(datasets): show and sort compounds by any target, profile each one

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Readouts from each target's own kind

**Files:**
- Modify: `backend/src/daikonstudio/application/catalog/derive_readouts.py`
- Modify:
  - `backend/src/daikonstudio/application/execution/train_protocol.py` (the `derive_readouts` call)
  - `backend/tests/unit/catalog/test_protocol.py` (call sites)
- Test: `backend/tests/unit/catalog/test_derive_readouts.py` (create)

**Interfaces:**
- Consumes: `probability_column` (Task 2).
- Produces:
  - `derive_readouts(targets: Sequence[TargetSpec]) -> tuple[Readout, ...]` (the `task` parameter is gone)
  - `target_columns_of(readouts: Sequence[Readout]) -> tuple[str, ...]`

- [ ] **Step 1: Failing test**

Create `backend/tests/unit/catalog/test_derive_readouts.py`:

```python
from daikonstudio.application.catalog.derive_readouts import derive_readouts, target_columns_of
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec

SOLUBILITY = TargetSpec(column="solubility", kind=TargetKind.NUMERIC, unit="logS", direction=Direction.HIGH)
REACTIVE = TargetSpec(column="reactive", kind=TargetKind.BINARY)


def test_a_mixed_dataset_gets_a_value_beside_a_probability_and_class_pair():
    readouts = derive_readouts((SOLUBILITY, REACTIVE))
    assert [(r.name, r.type.value, r.unit) for r in readouts] == [
        ("solubility", "numeric", "logS"),
        ("reactive_probability", "probability", None),
        ("reactive", "class", None),
    ]


def test_four_binary_targets_give_eight_readouts():
    targets = tuple(TargetSpec(column=c, kind=TargetKind.BINARY) for c in "abcd")
    assert len(derive_readouts(targets)) == 8


def test_the_targets_are_recovered_from_the_readouts_in_order():
    assert target_columns_of(derive_readouts((REACTIVE, SOLUBILITY))) == ("reactive", "solubility")
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && uv run pytest tests/unit/catalog/test_derive_readouts.py -q
```

Expected: FAIL. `target_columns_of` does not exist, and `derive_readouts` requires `task`.

- [ ] **Step 3: Implement**

Replace the function in `derive_readouts.py`. Keep the module docstring, and update its last paragraph: it now reads each target's own kind, so `TaskType` is no longer imported.

```python
from collections.abc import Sequence

from daikonstudio.domain.catalog.readout import Readout, ReadoutType
from daikonstudio.domain.data.target import Direction, TargetKind, TargetSpec, probability_column


def derive_readouts(targets: Sequence[TargetSpec]) -> tuple[Readout, ...]:
    """Each target's readouts, from that target's own kind, in the Dataset's order.

    From the kind and not from a training task, because one Dataset may mix kinds:
    a solubility value beside a reactivity flag is one numeric readout beside a
    probability/class pair.
    """
    return tuple(readout for target in targets for readout in _readouts_for(target))


def _readouts_for(target: TargetSpec) -> tuple[Readout, ...]:
    direction = target.direction.value if target.direction is not None else None
    if target.kind is TargetKind.NUMERIC:
        return (
            Readout(
                name=target.column,
                type=ReadoutType.NUMERIC,
                unit=target.unit,
                direction=direction,
                description=f"Predicted {target.column}",
            ),
        )
    return (
        Readout(
            name=probability_column(target.column),
            type=ReadoutType.PROBABILITY,
            unit=None,
            direction=Direction.HIGH.value,
            description=f"Probability that {target.column} is positive",
        ),
        Readout(
            name=target.column,
            type=ReadoutType.CLASS,
            unit=None,
            direction=direction,
            description=f"Predicted {target.column} class",
        ),
    )


def target_columns_of(readouts: Sequence[Readout]) -> tuple[str, ...]:
    """The target columns a Protocol predicts, recovered from its readouts.

    Every target contributes exactly one readout named after its own column --
    NUMERIC for a measured value, CLASS for a binary one -- and only a binary target
    adds a second, PROBABILITY, readout. So the non-probability readouts, in order,
    are the targets, in order. This is what lets prediction and the Scorecard name a
    Protocol's targets without loading its training Dataset, which `RunPrediction`
    deliberately never does.
    """
    return tuple(r.name for r in readouts if r.type is not ReadoutType.PROBABILITY)
```

Update the call in `train_protocol.py` to `derive_readouts(dataset.targets)`. Fix the three calls in `tests/unit/catalog/test_protocol.py`: drop the `TaskType` argument and pass a tuple of targets.

- [ ] **Step 4: Run tests and gates, then commit**

Run:
```
cd backend && uv run pytest tests/unit/catalog -q && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add -A backend
git commit -m "feat(catalog): derive readouts per target, from each target's own kind

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The engine contract carries several targets

**Files:**
- Modify (engine contract):
  - `backend/src/daikonstudio/application/engines/context.py`
  - `backend/src/daikonstudio/application/engines/manifest.py`
  - `backend/src/daikonstudio/application/engines/protocol.py`
- Modify (engines):
  - `backend/src/daikonstudio/infrastructure/engines/_scoring.py` (`_score`)
  - `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py:370-384`
  - `backend/src/daikonstudio/infrastructure/engines/molformer_xl.py:460-486`
- Modify (callers):
  - `backend/src/daikonstudio/application/execution/train_protocol.py`
  - `backend/src/daikonstudio/application/execution/predict_with_protocol.py:331-339`
  - `backend/src/daikonstudio/interface/routes/engines.py`
- Test:
  - `backend/tests/unit/engines/test_engine_contract.py`
  - every `TrainContext(` / `PredictContext(` construction in `backend/tests`

**Interfaces:**
- Consumes: `target_columns_of` (Task 4).
- Produces:
  - `TrainContext(frame, targets: dict[str, TaskType], structure_column, conditions, seed, report)`, where:
    - `targets` maps column to task and is ordered;
    - `.target_columns -> tuple[str, ...]`;
    - `.target_column -> str` raises `ValueError` if there are several targets;
    - `.task -> TaskType` raises `ValueError` if the kinds are mixed.
  - `TrainResult.metrics: dict[str, dict[str, float]]` and `validation_metrics: dict[str, dict[str, float]] | None`, both keyed by target column.
  - `PredictContext.target_columns: tuple[str, ...]`
  - `EngineManifest.supports_multitask: bool = False`
  - HTTP `EngineManifestResponse.supports_multitask: bool`

- [ ] **Step 1: Failing tests**

Append to `backend/tests/unit/engines/test_engine_contract.py`:

```python
import polars as pl

from daikonstudio.application.engines.context import TrainContext


def _ctx(targets):
    return TrainContext(
        frame=pl.DataFrame(), targets=targets, structure_column="smiles", conditions={}, seed=1
    )


def test_a_single_target_context_names_its_target_and_task():
    ctx = _ctx({"y": TaskType.REGRESSION})
    assert ctx.target_columns == ("y",)
    assert ctx.target_column == "y"
    assert ctx.task is TaskType.REGRESSION


def test_target_column_raises_on_a_context_with_several_targets():
    ctx = _ctx({"a": TaskType.REGRESSION, "b": TaskType.REGRESSION})
    assert ctx.task is TaskType.REGRESSION
    with pytest.raises(ValueError, match="2 targets"):
        _ = ctx.target_column


def test_task_raises_when_the_targets_mix_kinds():
    with pytest.raises(ValueError, match="different kinds"):
        _ = _ctx({"a": TaskType.REGRESSION, "b": TaskType.BINARY_CLASSIFICATION}).task


def test_a_manifest_does_not_learn_several_targets_jointly_unless_it_says_so():
    assert MANIFEST.supports_multitask is False
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && uv run pytest tests/unit/engines/test_engine_contract.py -q
```

Expected: FAIL (`targets` is an unexpected keyword).

- [ ] **Step 3: Contract**

In `context.py`, replace the `TrainContext` fields `task` and `target_column` with `targets`, and add the properties. Keep the class docstring and extend it with the paragraph shown.

```python
@dataclass(frozen=True, kw_only=True)
class TrainContext:
    """`frame` carries the dataset columns plus a `split` column of train/validation/test.

    `targets` maps each target column to its task, in the Dataset's order, and is
    authoritative. An engine must NEVER infer regression-vs-classification from the
    target values: a regression target whose values happen to all be 0.0 or 1.0
    would silently train a classifier. The Dataset's TargetSpecs are the only source
    of truth for what is being predicted.

    `target_column` and `task` are the single-target reading every engine without
    `supports_multitask` uses, and they raise rather than guess when the context
    holds more than they can describe. Such an engine is never handed one: the
    registry wraps it in `FanOut`, which splits the context per target first. The
    raise is the loud form of that guarantee -- a first-element answer here would
    train on one target and silently drop the rest.
    """

    frame: pl.DataFrame
    targets: dict[str, TaskType]
    structure_column: str
    conditions: dict[str, object]
    seed: int
    report: ProgressReporter = _no_op
    """(keep the existing report docstring)"""

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(self.targets)

    @property
    def target_column(self) -> str:
        if len(self.targets) != 1:
            raise ValueError(
                f"This training context carries {len(self.targets)} targets "
                f"({', '.join(self.targets)}); `target_column` describes exactly one. An "
                "engine without `supports_multitask` must be wrapped in FanOut."
            )
        return next(iter(self.targets))

    @property
    def task(self) -> TaskType:
        tasks = set(self.targets.values())
        if len(tasks) != 1:
            raise ValueError(
                "The targets in this training context are of different kinds; `task` "
                "describes one. Only a uniform-kind dataset reaches a joint engine."
            )
        return tasks.pop()
```

`TrainResult`:

```python
    artifact: bytes
    # Keyed by target column: {"solubility": {"rmse": 0.61, ...}}. A one-target fit
    # has exactly one key, so the shape is uniform.
    metrics: dict[str, dict[str, float]]
    validation_metrics: dict[str, dict[str, float]] | None = None
```

`PredictContext` gains, last:

```python
    # The Protocol's targets, in order (`target_columns_of(protocol.readouts)`). A
    # joint engine labels its output rows with them; `FanOut` uses them to pick the
    # artifact for each target and to tag every row.
    target_columns: tuple[str, ...]
```

In `manifest.py`, `EngineManifest` gains, after `is_baseline`:

```python
    # Learns every target of a Dataset in one model. It selects the training path
    # and labels the result ("one joint model" vs "one model per target"); it does
    # not decide which engines a Dataset may use -- every engine accepts every
    # Dataset, the rest through `FanOut`.
    supports_multitask: bool = False
```

In `protocol.py`, change the `predict` docstring to:
`"""Returns columns: row_id (int), value (float), uncertainty (float | null) -- plus target (str), one row per (compound, target), when the manifest declares supports_multitask. FanOut adds it for every other engine."""`

- [ ] **Step 4: Engines nest their metrics**

In `_scoring.py`, rename the current `_score` to `_metrics_on` (same body) and add:

```python
def _score(
    model: Any,
    test_rows: pl.DataFrame,
    ctx: TrainContext,
    is_classification: bool,
    featurizer: Featurizer = ecfp4,
) -> dict[str, dict[str, float]]:
    """`_metrics_on`, keyed by the one target this fit trained on -- the shape
    `TrainResult.metrics` takes now that a Dataset can hold several. These engines
    always fit one target; `FanOut` merges the keys."""
    return {ctx.target_column: _metrics_on(model, test_rows, ctx, is_classification, featurizer)}
```

`_score_validation` calls `_score` and so returns the nested shape unchanged. Fix its return annotation to `dict[str, dict[str, float]] | None`. The five sklearn-API engines then need no edit.

In `chemprop_dmpnn.py` and `molformer_xl.py`:
```python
metrics = {ctx.target_column: score(test_rows)}
validation_metrics = {ctx.target_column: score(validation_rows)} if validation_rows.height > 0 else None
```

- [ ] **Step 5: Callers**

In `train_protocol.py`, still single-target through `dataset.single_target()` (Task 9 makes it a loop):
- `_train_off_thread` builds `TrainContext(frame=frame, targets={target.column: task}, structure_column=dataset.structure_column, conditions=conditions, seed=dataset.split.seed, report=self._reporter(run, span))`, with `target = dataset.single_target()`.
- The test-set replay passes `target_columns=dataset.target_columns` to `PredictContext`.
- Read `chosen.metrics[target.column]`, `baseline_result.metrics[target.column]`, `chosen.validation_metrics[target.column]` and, in `_optimism_gap`, `result.metrics[target.column]`.

In `predict_with_protocol.py`, pass `target_columns=target_columns_of(protocol.readouts)` to the `PredictContext`.

In `routes/engines.py`, `EngineManifestResponse` gains `supports_multitask: bool`, set in `from_manifest`.

- [ ] **Step 6: Update test constructions**

Run:
```
cd backend && grep -rn "TrainContext(\|PredictContext(\|\.metrics\[\|result.metrics\b\|\"rmse\" in\|\"mcc\" in" tests/unit/engines tests/unit/execution
```

For each `TrainContext(`:
- replace `task=X, ... target_column="y"` with `targets={"y": X}`.

For each `PredictContext(`:
- add `target_columns=("y",)` (the frame's target).

For each metrics assertion on a `TrainResult`:
- read `result.metrics["y"]`.

Fake/stub engines in `tests/unit/execution` and `test_engine_contract.py` that return `TrainResult(metrics={...})`:
- nest it under their target column.

- [ ] **Step 7: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit tests/integration/test_train_protocol.py tests/api/test_runs.py tests/api/test_engines.py -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add -A backend
git commit -m "feat(engines): carry targets, not one target, through the engine contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The fan-out adapter

**Files:**
- Create: `backend/src/daikonstudio/application/engines/fan_out.py`
- Modify: `backend/src/daikonstudio/application/engines/registry.py` (`get`, `baseline`)
- Test: `backend/tests/unit/engines/test_fan_out.py` (create)

**Interfaces:**
- Consumes: `TrainContext.targets`, `PredictContext.target_columns`, `EngineManifest.supports_multitask` (Task 5).
- Produces:
  - `FanOut(inner: Engine)`, which satisfies `Engine`.
  - `EngineRegistry.get()` and `.baseline()` return a `FanOut` around every engine whose manifest does not declare `supports_multitask`.
  - Every engine's `predict` output, as seen through the registry, has a `target` column.

- [ ] **Step 1: Failing tests**

Create `backend/tests/unit/engines/test_fan_out.py`:

```python
import pickle
import zipfile
from io import BytesIO

import polars as pl
import pytest

from daikonstudio.application.engines.context import (
    PredictContext,
    RunInterrupted,
    TrainContext,
    TrainResult,
)
from daikonstudio.application.engines.fan_out import FanOut
from daikonstudio.application.engines.manifest import EngineManifest, TaskType
from daikonstudio.application.engines.registry import EngineRegistry

_SINGLE = EngineManifest(
    id="single", version="1", name="Single", description="d", tasks=tuple(TaskType)
)
_JOINT = EngineManifest(
    id="joint", version="1", name="Joint", description="d", tasks=tuple(TaskType),
    supports_multitask=True,
)


class _Recorder:
    """Trains by remembering what it was asked; its artifact names the target."""

    def __init__(self, interrupt_on: int | None = None) -> None:
        self.contexts: list[TrainContext] = []
        self._interrupt_on = interrupt_on

    @staticmethod
    def manifest() -> EngineManifest:
        return _SINGLE

    def train(self, ctx: TrainContext) -> TrainResult:
        self.contexts.append(ctx)
        if self._interrupt_on == len(self.contexts):
            raise RunInterrupted("cancelled", cancelled=True)
        for fraction in (0.0, 0.5, 1.0):
            ctx.report(fraction, "fitting")
        column = ctx.target_column  # raises if FanOut handed it several
        return TrainResult(
            artifact=pickle.dumps({"column": column, "task": ctx.task.value}),
            metrics={column: {"rmse": float(len(self.contexts))}},
            validation_metrics={column: {"rmse": 0.0}},
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        trained_on = pickle.loads(ctx.artifact)["column"]
        return pl.DataFrame(
            {
                "row_id": pl.Series(range(ctx.frame.height), dtype=pl.Int64),
                "value": pl.Series([float(len(trained_on))] * ctx.frame.height),
                "uncertainty": pl.Series([None] * ctx.frame.height, dtype=pl.Float64),
            }
        )


def _ctx(targets: dict[str, TaskType], report=lambda fraction, phase: None) -> TrainContext:
    return TrainContext(
        frame=pl.DataFrame({"smiles": ["C"]}),
        targets=targets,
        structure_column="smiles",
        conditions={},
        seed=1,
        report=report,
    )


def _predict(engine: FanOut, artifact: bytes, columns: tuple[str, ...]) -> pl.DataFrame:
    return engine.predict(
        PredictContext(
            frame=pl.DataFrame({"smiles": ["C", "CC"]}),
            structure_column="smiles",
            artifact=artifact,
            conditions={},
            target_columns=columns,
        )
    )


def test_one_target_stores_the_inner_artifact_verbatim_and_still_tags_rows():
    inner = _Recorder()
    result = FanOut(inner).train(_ctx({"y": TaskType.REGRESSION}))
    assert pickle.loads(result.artifact)["column"] == "y"  # bare bytes, no container
    predictions = _predict(FanOut(inner), result.artifact, ("y",))
    assert predictions["target"].to_list() == ["y", "y"]


def test_several_targets_round_trip_through_one_container():
    columns = ("aggregator", "/odd {name}", "reactive")
    inner = _Recorder()
    result = FanOut(inner).train(_ctx(dict.fromkeys(columns, TaskType.BINARY_CLASSIFICATION)))
    assert zipfile.is_zipfile(BytesIO(result.artifact))
    predictions = _predict(FanOut(inner), result.artifact, columns)
    for column in columns:
        rows = predictions.filter(pl.col("target") == column)
        # each target's rows came from its own sub-model
        assert rows["value"].to_list() == [float(len(column))] * 2


def test_metrics_nest_per_target_in_order():
    result = FanOut(_Recorder()).train(
        _ctx({"a": TaskType.REGRESSION, "b": TaskType.BINARY_CLASSIFICATION})
    )
    assert list(result.metrics) == ["a", "b"]
    assert result.validation_metrics == {"a": {"rmse": 0.0}, "b": {"rmse": 0.0}}


def test_each_sub_fit_sees_one_target_and_that_targets_own_task():
    inner = _Recorder()
    FanOut(inner).train(_ctx({"a": TaskType.REGRESSION, "b": TaskType.BINARY_CLASSIFICATION}))
    assert [ctx.targets for ctx in inner.contexts] == [
        {"a": TaskType.REGRESSION},
        {"b": TaskType.BINARY_CLASSIFICATION},
    ]


def test_progress_spans_zero_to_one_once_across_all_sub_fits():
    seen: list[float] = []
    FanOut(_Recorder()).train(
        _ctx(dict.fromkeys("abcd", TaskType.REGRESSION), report=lambda f, _: seen.append(f))
    )
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert seen == sorted(seen)


def test_an_interruption_in_the_second_of_four_fits_propagates_and_stops_the_rest():
    inner = _Recorder(interrupt_on=2)
    with pytest.raises(RunInterrupted):
        FanOut(inner).train(_ctx(dict.fromkeys("abcd", TaskType.REGRESSION)))
    assert len(inner.contexts) == 2


def test_predicting_with_targets_the_container_was_not_built_for_fails_loudly():
    result = FanOut(_Recorder()).train(_ctx(dict.fromkeys("ab", TaskType.REGRESSION)))
    with pytest.raises(ValueError, match="trained on"):
        _predict(FanOut(_Recorder()), result.artifact, ("a", "c"))


def test_the_registry_wraps_only_engines_that_cannot_learn_targets_jointly():
    class _Joint(_Recorder):
        @staticmethod
        def manifest() -> EngineManifest:
            return _JOINT

    joint = _Joint()
    registry = EngineRegistry({"single": _Recorder(), "joint": joint})
    assert isinstance(registry.get("single"), FanOut)
    assert registry.get("joint") is joint
    assert registry.get("single").manifest() is _SINGLE
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && uv run pytest tests/unit/engines/test_fan_out.py -q
```

Expected: FAIL with `ModuleNotFoundError: daikonstudio.application.engines.fan_out`.

- [ ] **Step 3: Implement `FanOut`**

Create `backend/src/daikonstudio/application/engines/fan_out.py`:

```python
"""One model per target, behind the single-engine contract.

Five engines in the roster fit exactly one target: a random forest, a boosted tree
or a Gaussian process has one output. This adapter is what lets every one of them
accept a Dataset with several targets anyway -- it trains the inner engine once per
target, on the same rows and the same split, so a per-target baseline and a joint
model are compared on identical ground. The registry wraps every engine that does
not declare `supports_multitask`, at every target count including one, so the
long-format output below is produced in exactly one place.

Three rules this file exists to keep:

- **`RunInterrupted` propagates.** Nothing here catches anything. Swallowing a
  cancellation raised in fit two of four would keep burning a runner for fits
  three and four (see `context.py`).
- **Progress is mapped, not forwarded.** `ctx.report` is progress within one fit,
  0.0 to 1.0; target i of N reports into its own slice, (i + f) / N, or the bar
  would reset to zero N times.
- **The container appears only at N > 1.** At one target the artifact is the inner
  engine's bytes verbatim: every Protocol trained before targets could be several
  has a bare artifact in blob storage, and a container there would break prediction
  for all of them. At N > 1 the artifacts are packed into one zip -- boring,
  inspectable, no pickle -- under index names, never column names, since a CSV
  header can hold a slash or a leading `/` that a zip entry name would rewrite.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import polars as pl

from daikonstudio.application.engines.context import (
    PredictContext,
    ProgressReporter,
    TrainContext,
    TrainResult,
)
from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine

_TARGETS_ENTRY = "targets.json"


class FanOut:
    def __init__(self, inner: Engine) -> None:
        self._inner = inner

    def manifest(self) -> EngineManifest:
        return self._inner.manifest()

    def train(self, ctx: TrainContext) -> TrainResult:
        columns = ctx.target_columns
        count = len(columns)
        results: list[TrainResult] = []
        for index, (column, task) in enumerate(ctx.targets.items()):
            if count > 1:
                # Before each sub-fit, not only inside it: tree and GP engines never
                # report on their own, and this checkpoint is what lets a cancel or a
                # deadline stop the run between targets.
                ctx.report(index / count, f"Training on {column} ({index + 1} of {count})")
            results.append(
                self._inner.train(
                    replace(ctx, targets={column: task}, report=_slice(ctx.report, index, count))
                )
            )
        validation = {
            key: value
            for result in results
            if result.validation_metrics is not None
            for key, value in result.validation_metrics.items()
        }
        return TrainResult(
            artifact=results[0].artifact if count == 1 else _pack(columns, results),
            metrics={key: value for result in results for key, value in result.metrics.items()},
            validation_metrics=validation or None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        columns = ctx.target_columns
        artifacts = [ctx.artifact] if len(columns) == 1 else _unpack(ctx.artifact, columns)
        return pl.concat(
            [
                self._inner.predict(
                    replace(ctx, artifact=artifact, target_columns=(column,))
                ).with_columns(pl.lit(column, dtype=pl.String).alias("target"))
                for column, artifact in zip(columns, artifacts, strict=True)
            ]
        )


def _slice(report: ProgressReporter, index: int, count: int) -> ProgressReporter:
    if count == 1:
        return report

    def sliced(fraction: float, phase: str) -> None:
        report((index + min(max(fraction, 0.0), 1.0)) / count, phase)

    return sliced


def _pack(columns: tuple[str, ...], results: list[TrainResult]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_TARGETS_ENTRY, json.dumps(list(columns)))
        for index, result in enumerate(results):
            archive.writestr(f"{index}.bin", result.artifact)
    return buffer.getvalue()


def _unpack(artifact: bytes, columns: tuple[str, ...]) -> list[bytes]:
    with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
        trained = tuple(json.loads(archive.read(_TARGETS_ENTRY)))
        if trained != columns:
            raise ValueError(
                f"This artifact was trained on {list(trained)}, not on the requested "
                f"targets {list(columns)}."
            )
        return [archive.read(f"{index}.bin") for index in range(len(columns))]
```

In `registry.py`, import `FanOut` and add the wrapping:

```python
def _uniform(engine: Engine) -> Engine:
    """Every engine as one that accepts several targets: as itself when it learns
    them jointly, inside `FanOut` otherwise. Applied at lookup so no caller can
    reach an unwrapped single-target engine by accident."""
    return engine if engine.manifest().supports_multitask else FanOut(engine)
```

- `get()` returns `_uniform(self._engines[engine_id])` (inside the existing `try`).
- `baseline()` returns `_uniform(baselines[0])`.
- `manifests()` is unchanged.

- [ ] **Step 4: Run tests and gates**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit tests/integration/test_train_protocol.py tests/api/test_runs.py -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected:
- PASS.
- Existing one-target training and prediction still work: the long format at N = 1 is the old frame plus a `target` column.
- If a test asserts `registry.get(x) is some_engine` for a non-multitask engine, change it to compare `.manifest()`.
- If mypy rejects `FanOut` as an `Engine` (its `manifest` is an instance method, while the protocol declares a staticmethod), change `Engine.manifest` in `protocol.py` to a plain `def manifest(self) -> EngineManifest: ...`. Every concrete engine's staticmethod still satisfies it.

- [ ] **Step 5: Commit**

```bash
git add -A backend
git commit -m "feat(engines): fan out one model per target for engines that fit one

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: A training run's headline metrics, per target

**Files:**
- Modify: `backend/src/daikonstudio/domain/execution/run.py:157-177`
- Modify: `backend/src/daikonstudio/infrastructure/runner/wire.py` (`RunMetricsWire`)
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (the `record_metrics` call)
- Modify: `backend/alembic/versions/013_dataset_targets.py` (add the run-metrics statements)
- Test:
  - `backend/tests/unit/execution/test_run.py`
  - `backend/tests/unit/runners/test_wire.py`
  - `backend/tests/integration/test_migrations.py`
  - `backend/tests/integration/test_train_protocol.py`
  - `backend/tests/integration/test_sweeps.py`
  - `backend/tests/api/test_runner_protocol.py`

**Interfaces:**
- Consumes: migration 013 from Task 1.
- Produces:
  - `TargetHeadline(column, primary_metric, value, baseline_value)` in `domain/execution/run.py`
  - `Run.record_metrics(headlines: Sequence[TargetHeadline])`, which writes `{"targets": [ {column, primary_metric, value, baseline_value}, … ]}`
  - `RunMetricsWire.targets: list[TargetHeadlineWire]`
  - Migration constants `NEST_RUN_METRICS`, `FLATTEN_RUN_METRICS`

- [ ] **Step 1: Failing tests**

Append to `tests/unit/execution/test_run.py`:

```python
from daikonstudio.domain.execution.run import TargetHeadline


def test_headline_metrics_are_recorded_per_target_in_order():
    run = Run(kind=RunKind.TRAINING, workspace_id=uuid.uuid4(), requested_by=uuid.uuid4(), cache_key="k")
    run.record_metrics(
        [
            TargetHeadline(column="b", primary_metric="mcc", value=0.4, baseline_value=0.3),
            TargetHeadline(column="a", primary_metric="rmse", value=None, baseline_value=0.9),
        ]
    )
    assert run.metrics == {
        "targets": [
            {"column": "b", "primary_metric": "mcc", "value": 0.4, "baseline_value": 0.3},
            {"column": "a", "primary_metric": "rmse", "value": None, "baseline_value": 0.9},
        ]
    }
```

(Match the imports already at the top of that file.)

Append to `tests/integration/test_migrations.py`:

```python
@pytest.mark.asyncio
async def test_013_nests_a_training_runs_headline_under_its_datasets_target(migrated_session):
    migration = _load_migration("013_dataset_targets")
    workspace, dataset_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await migrated_session.execute(
        text(
            "INSERT INTO datasets (id, workspace_id, name, structure_column, targets, split,"
            " content_hash, snapshot_uri, row_count, validation_report, version, created_at,"
            ' updated_at) VALUES (:id, :ws, \'d\', \'smiles\', \'[{"column": "y", "kind":'
            " \"numeric\"}]', '{}', 'h3', 'x', 3, '{}', 1, now(), now())"
        ),
        {"id": dataset_id, "ws": workspace},
    )
    flat = {"primary_metric": "rmse", "value": 0.5, "baseline_value": 0.7}
    await migrated_session.execute(
        text(
            "INSERT INTO runs (id, workspace_id, kind, requested_by, cache_key, params, status,"
            " progress, attempts, version, created_at, updated_at, metrics) VALUES (:id, :ws,"
            " 'training', :by, 'k', CAST(CAST(:params AS text) AS jsonb), 'ready', 1, 0, 1,"
            " now(), now(), CAST(CAST(:metrics AS text) AS jsonb))"
        ),
        {
            "id": run_id,
            "ws": workspace,
            "by": uuid.uuid4(),
            "params": json.dumps({"dataset_id": str(dataset_id)}),
            "metrics": json.dumps(flat),
        },
    )

    await migrated_session.execute(text(migration.NEST_RUN_METRICS))
    assert await _json(
        migrated_session, "SELECT metrics::text FROM runs WHERE id = :id", id=run_id
    ) == {"targets": [{"column": "y", **flat}]}

    await migrated_session.execute(text(migration.FLATTEN_RUN_METRICS))
    assert await _json(
        migrated_session, "SELECT metrics::text FROM runs WHERE id = :id", id=run_id
    ) == flat
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/execution/test_run.py tests/integration/test_migrations.py -q
```

Expected: FAIL. `TargetHeadline` and `NEST_RUN_METRICS` do not exist.

- [ ] **Step 3: Domain**

In `domain/execution/run.py`, add `from dataclasses import asdict, dataclass` and `from collections.abc import Sequence` as needed.

```python
@dataclass(frozen=True, kw_only=True)
class TargetHeadline:
    """One target's headline number on a training Run, and its baseline's."""

    column: str
    primary_metric: str
    value: float | None
    baseline_value: float | None
```

Replace `record_metrics`. Keep its docstring's reasoning, and add the paragraph shown:

```python
    def record_metrics(self, headlines: Sequence[TargetHeadline]) -> None:
        """(existing docstring) ...

        One headline per target, in the Dataset's order, as a list rather than an
        object keyed by column: JSONB does not keep key order, and the sweep table
        shows targets in the order the scientist chose them.
        """
        self.metrics = {"targets": [asdict(headline) for headline in headlines]}
        self._touch()
```

- [ ] **Step 4: Wire**

In `wire.py`, replace `RunMetricsWire`'s fields:

```python
class TargetHeadlineWire(BaseModel):
    model_config = _FORBID

    column: str
    primary_metric: str
    value: float | None
    baseline_value: float | None


class RunMetricsWire(BaseModel):
    """Mirrors exactly what `Run.record_metrics` writes (`domain/execution/run.py`):
    one headline per target -- the only shape a runner-reported training `metrics`
    payload may take. (keep the existing security paragraph)"""

    model_config = _FORBID

    targets: list[TargetHeadlineWire]
```

`runner_api.py` already stores `body.metrics.model_dump()`, which is now `{"targets": [...]}`.

- [ ] **Step 5: Migration statements**

In `013_dataset_targets.py`, add the run half to the module docstring: a training run's flat `runs.metrics` headline is nested under its dataset's single target, so every reader sees one shape. Then add:

```python
# `-> 'primary_metric' IS NOT NULL` rather than the `?` operator: a bare `?` reads
# as a bind marker to some drivers.
NEST_RUN_METRICS = """
UPDATE runs AS r
SET metrics = jsonb_build_object(
    'targets',
    jsonb_build_array(r.metrics || jsonb_build_object('column', d.targets -> 0 ->> 'column'))
)
FROM datasets AS d
WHERE r.kind = 'training'
  AND r.metrics -> 'primary_metric' IS NOT NULL
  AND d.id::text = r.params ->> 'dataset_id'
"""

FLATTEN_RUN_METRICS = """
UPDATE runs
SET metrics = (metrics -> 'targets' -> 0) - 'column'
WHERE kind = 'training' AND metrics -> 'targets' IS NOT NULL
"""
```

- In `upgrade()`, execute `NEST_RUN_METRICS` right after `TARGETS_FROM_TARGET`.
- In `downgrade()`, execute `FLATTEN_RUN_METRICS` right after the multi-target guard.

- [ ] **Step 6: Training writes it**

In `train_protocol.py`, replace the `run.record_metrics(primary_metric=..., value=..., baseline_value=...)` call with:

```python
        run.record_metrics(
            [
                TargetHeadline(
                    column=target.column,
                    primary_metric=primary,
                    value=metrics.get(primary),
                    baseline_value=baseline_metrics.get(primary),
                )
            ]
        )
```

with `target = dataset.single_target()`. Task 9 turns this into the loop.

- [ ] **Step 7: Update fixtures**

Run:
```
cd backend && grep -rn "record_metrics(\|primary_metric\b.*value\|RunMetricsWire(" tests
```

Convert each to the list shape:
- `test_sweeps.py` (`test_sweep_id_and_metrics_round_trip`)
- `test_train_protocol.py` (`test_training_records_its_headline_metric`, which now asserts `run.metrics["targets"][0]["primary_metric"] == "rmse"`)
- `test_wire.py`
- `tests/api/test_runner_protocol.py` (the PATCH body's `metrics`)

- [ ] **Step 8: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit tests/integration/test_migrations.py tests/integration/test_sweeps.py tests/integration/test_train_protocol.py tests/api/test_runner_protocol.py tests/api/test_sweeps.py -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add -A backend
git commit -m "feat(runs): record a training run's headline metric per target

Migration 013 nests every existing headline under its dataset's target.
Runners and the API must be deployed together: the wire shape changed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Scorecards, one per target

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (`ScorecardInputs`, new `TargetInputs`, how `RunTraining` builds them)
- Modify: `backend/src/daikonstudio/application/execution/build_scorecard.py`
- Modify: `backend/src/daikonstudio/domain/execution/scorecard.py`
- Modify: `backend/src/daikonstudio/application/catalog/get_scorecard.py`
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py:258-330, 395-410`
- Test:
  - `backend/tests/unit/execution/test_scorecard.py`
  - `backend/tests/unit/catalog/test_get_scorecard.py`
  - `backend/tests/integration/test_train_protocol.py`
  - `backend/tests/api/test_protocols.py`

**Interfaces:**
- Consumes: `target_columns_of` (Task 4).
- Produces:
  - `TargetInputs`: the per-target fields, listed below.
  - `ScorecardInputs`: the run-level fields plus `joint_model: bool = False` and `targets: list[TargetInputs]`.
  - `ScorecardInputs.from_json(data: bytes, *, legacy_column: str = "") -> ScorecardInputs`
  - `HeldOutChemistry(similarities: list[float] | None, scaffolds: list[str])`
  - `held_out_chemistry(structures, train_structures, normalizer) -> HeldOutChemistry`
  - `build_scorecard(*, target: str, joint_model: bool = False, chemistry: HeldOutChemistry, ...)`. The `normalizer` and `train_structures` parameters are removed.
  - `Scorecard.target: str` and `Scorecard.joint_model: bool`
  - `GetScorecard` returns `Result[list[Scorecard], DomainError]`
  - HTTP `GET /protocols/{id}/scorecard` → `list[ScorecardResponse]`, each with `target` and `joint_model`

`TargetInputs` per-target fields:
- `column`
- `task`
- `metrics`
- `validation_metrics`
- `actual`
- `predicted`
- `prediction_kind`
- `baseline_metrics`
- `random_split_metrics`
- `random_split_metrics_undefined`
- `metrics_undefined`
- `duplicate_spread`
- `target_unit`
- `target_direction`

`ScorecardInputs` run-level fields:
- `protocol_id`
- `run_id`
- `dataset_id`
- `engine_id`
- `conditions`
- `structures`
- `train_structures`
- `baseline_engine_id`
- `baseline_conditions`
- `baseline_is_self`
- `random_split_unavailable`
- `split_strategy`

- [ ] **Step 1: Failing tests**

In `tests/unit/catalog/test_get_scorecard.py`:
- Add the imports: `json`, `from dataclasses import replace`, `TargetInputs` (from `train_protocol`), `Readout` and `ReadoutType` (from `domain.catalog.readout`), `RdkitStructureNormalizer`.
- Give `_FakeProtocol` a `readouts` attribute: one `Readout(name="y", type=ReadoutType.NUMERIC, unit=None, direction=None, description="d")`.
- Rewrite `_inputs` to build the new shape: run-level fields plus `targets=[TargetInputs(column="y", ...)]`.

Then add:

```python
def _target(column: str, **overrides) -> TargetInputs:
    fields = dict(
        column=column,
        task="regression",
        metrics={"rmse": 0.5, "mae": 0.4, "r2": 0.1},
        actual=[1.0, 2.0],
        predicted=[1.1, 1.9],
        prediction_kind="value",
        baseline_metrics={"rmse": 0.7, "mae": 0.6, "r2": 0.0},
        random_split_metrics=None,
        random_split_metrics_undefined=None,
        metrics_undefined=None,
        duplicate_spread=None,
        target_unit=None,
        target_direction=None,
    )
    fields.update(overrides)
    return TargetInputs(**fields)


async def test_one_scorecard_per_target_and_a_degenerate_one_does_not_sink_its_siblings(
    monkeypatch,
) -> None:
    workspace_id, protocol_id = uuid.uuid4(), uuid.uuid4()
    inputs = replace(
        _inputs(protocol_id),
        targets=[
            _target(
                "active",
                task="binary_classification",
                prediction_kind="probability",
                metrics={"mcc": None, "balanced_accuracy": None, "auroc": None, "auprc": None},
                baseline_metrics={"mcc": None, "balanced_accuracy": None, "auroc": None, "auprc": None},
                metrics_undefined={"mcc": "Undefined: all test-set compounds have the same 'active' value."},
                actual=[0.0, 0.0],
                predicted=[0.2, 0.3],
            ),
            _target("solubility"),
        ],
    )
    cards = await _scorecards(workspace_id, protocol_id, inputs)
    assert [card.target for card in cards] == ["active", "solubility"]
    assert cards[0].metrics_undefined is not None
    assert cards[1].metrics["rmse"] == 0.5 and cards[1].metrics_undefined is None


async def test_the_test_set_chemistry_is_computed_once_for_every_target(monkeypatch) -> None:
    calls = 0
    real = module.held_out_chemistry

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "held_out_chemistry", counting)
    workspace_id, protocol_id = uuid.uuid4(), uuid.uuid4()
    inputs = replace(_inputs(protocol_id), targets=[_target("a"), _target("b"), _target("c")])
    assert len(await _scorecards(workspace_id, protocol_id, inputs)) == 3
    assert calls == 1


async def test_a_blob_written_before_targets_could_be_several_is_one_scorecard() -> None:
    workspace_id, protocol_id = uuid.uuid4(), uuid.uuid4()
    legacy = json.dumps(
        {
            "protocol_id": str(protocol_id), "run_id": "r", "dataset_id": "d",
            "engine_id": "ecfp4-xgboost", "task": "regression", "conditions": {},
            "metrics": {"rmse": 0.5}, "actual": [1.0, 2.0], "predicted": [1.1, 1.9],
            "prediction_kind": "value", "structures": ["CCO", "CCN"],
            "train_structures": ["CCC"], "baseline_engine_id": "ecfp4-randomforest",
            "baseline_metrics": {"rmse": 0.7}, "baseline_is_self": False,
            "random_split_metrics": None, "random_split_unavailable": None,
            "random_split_metrics_undefined": None, "metrics_undefined": None,
            "duplicate_spread": None, "target_unit": None, "target_direction": None,
            "split_strategy": "random",
        }
    ).encode()
    cards = await _scorecards_from_blob(workspace_id, protocol_id, legacy)
    assert [card.target for card in cards] == ["y"]  # named after the protocol's readout
```

Write the helpers `_scorecards(workspace_id, protocol_id, inputs)` and `_scorecards_from_blob(workspace_id, protocol_id, blob)` once. They call `GetScorecard(_FakeProtocols(...), _FakeStore(blob), RdkitStructureNormalizer(), _NoDatasets())(GetScorecardQuery(protocol_id=protocol_id), auth=FakeAuth(workspace_id=workspace_id))` and `.unwrap()`. Use the fakes already in this file, and match how the existing off-thread test constructs `FakeAuth`.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/catalog/test_get_scorecard.py -q
```

Expected: FAIL (`TargetInputs` cannot be imported).

- [ ] **Step 3: `ScorecardInputs` and `TargetInputs`**

In `train_protocol.py`:
- Split `ScorecardInputs`.
- Move the docstring paragraphs about `prediction_kind`, `metrics_undefined` / `random_split_metrics_undefined`, `validation_metrics` and `target_unit` / `target_direction` onto `TargetInputs`.
- Keep the `baseline_is_self`, `random_split_unavailable` and `split_strategy` paragraphs on `ScorecardInputs`.

```python
@dataclass(frozen=True, kw_only=True)
class TargetInputs:
    """One target's share of `ScorecardInputs`: everything measured about it.

    (moved docstring paragraphs)
    """

    column: str
    task: str
    metrics: dict[str, float | None]
    validation_metrics: dict[str, float | None] | None = None
    actual: list[float]
    predicted: list[float]
    prediction_kind: str
    baseline_metrics: dict[str, float | None]
    random_split_metrics: dict[str, float | None] | None
    random_split_metrics_undefined: dict[str, str] | None
    metrics_undefined: dict[str, str] | None
    duplicate_spread: float | None
    target_unit: str | None
    target_direction: str | None


_PER_TARGET = tuple(f.name for f in fields(TargetInputs) if f.name != "column")


@dataclass(frozen=True, kw_only=True)
class ScorecardInputs:
    """Everything measured during a training run, before anyone interprets it.

    Run-level facts appear once; everything measured per target is in `targets`,
    in the Dataset's order. `structures` and `train_structures` are run-level on
    purpose: every target shares one split and one set of test rows, and at 324k
    compounds those two lists are most of the blob.

    (remaining run-level paragraphs)
    """

    protocol_id: str
    run_id: str
    dataset_id: str
    engine_id: str
    conditions: dict[str, Any]
    structures: list[str]
    train_structures: list[str]
    baseline_engine_id: str
    baseline_conditions: dict[str, Any] = field(default_factory=dict)
    baseline_is_self: bool
    random_split_unavailable: str | None
    split_strategy: str
    # True when one model learned every target; False for one model per target --
    # and for every Protocol trained before targets could be several, which had one
    # target and one model either way. Recorded here rather than read off the
    # engine's manifest later, so the Scorecard says what happened, not what the
    # engine would do today.
    joint_model: bool = False
    targets: list[TargetInputs]

    def to_json(self) -> bytes:
        # (keep the allow_nan comment)
        return json.dumps(asdict(self), allow_nan=False).encode()

    @classmethod
    def from_json(cls, data: bytes, *, legacy_column: str = "") -> ScorecardInputs:
        """`legacy_column` names the one target of a blob written before targets
        could be several, which stored its per-target fields at the top level and
        never recorded the column. Only `GetScorecard` renders that name, so only it
        needs to pass one."""
        raw = json.loads(data)
        if "targets" not in raw:
            raw["targets"] = [
                {"column": legacy_column, **{k: raw.pop(k) for k in _PER_TARGET if k in raw}}
            ]
        raw["targets"] = [TargetInputs(**target) for target in raw["targets"]]
        return cls(**raw)
```

Add `fields` to the `dataclasses` import.

In `RunTraining.__call__` (still single-target via `target = dataset.single_target()`), build:

```python
        inputs = ScorecardInputs(
            protocol_id=str(protocol_id),
            run_id=str(run.id),
            dataset_id=str(dataset.id),
            engine_id=manifest.id,
            conditions=conditions,
            structures=[str(s) for s in test_rows[dataset.structure_column].to_list()],
            train_structures=[str(s) for s in train_rows[dataset.structure_column].to_list()],
            baseline_engine_id=baseline_manifest.id,
            baseline_conditions=baseline_conditions,
            baseline_is_self=baseline_is_self,
            random_split_unavailable=random_split_unavailable,
            split_strategy=dataset.split.strategy.value,
            joint_model=manifest.supports_multitask,
            targets=[TargetInputs(column=target.column, task=task.value, ...)],
        )
```

Move the existing per-target values into that one `TargetInputs`.

- [ ] **Step 4: Shared chemistry and the per-target card**

In `domain/execution/scorecard.py`, add two `Scorecard` fields, documented in the class docstring:

```python
    #: The target column this card scores. A Protocol has one card per target.
    target: str
    #: True when one model learned every target, False for one model per target.
    joint_model: bool
```

In `build_scorecard.py`:

```python
@dataclass(frozen=True, kw_only=True)
class HeldOutChemistry:
    """The test set's chemistry: nearest-neighbour similarity to the training set and
    each compound's Murcko scaffold. Identical for every target of a Protocol, since
    every target shares one split -- so it is computed once and handed to each
    target's `build_scorecard`. At 324k compounds the similarity search is the
    expensive half of the page; four targets must not pay it four times."""

    similarities: list[float] | None
    scaffolds: list[str]


def held_out_chemistry(
    structures: list[str], train_structures: list[str], normalizer: StructureNormalizer
) -> HeldOutChemistry:
    # (move the existing comment about `nearest_neighbour_tanimoto([], [])` here)
    similarities = (
        normalizer.nearest_neighbour_tanimoto(structures, train_structures)
        if structures and train_structures
        else None
    )
    return HeldOutChemistry(
        similarities=similarities,
        scaffolds=[normalizer.murcko_scaffold(structure) for structure in structures],
    )
```

Change `build_scorecard`:
- remove the `train_structures` and `normalizer` parameters;
- add `target: str`, `joint_model: bool = False` and `chemistry: HeldOutChemistry`;
- read `similarities = chemistry.similarities` and `scaffolds = chemistry.scaffolds` in place of computing them;
- pass `target=target, joint_model=joint_model` to `Scorecard(...)`.

Remove the `StructureNormalizer` import if it is now unused there; `held_out_chemistry` still needs it.

- [ ] **Step 5: `GetScorecard` returns one card per target**

In `get_scorecard.py`:

```python
    async def __call__(
        self, query: GetScorecardQuery, auth: AuthContext | None = None
    ) -> Result[list[Scorecard], DomainError]:
        ...
        inputs = ScorecardInputs.from_json(
            raw, legacy_column=target_columns_of(protocol.readouts)[0]
        )
        # (keep the ponytail comment about to_thread, now about the whole list)
        scorecards = await asyncio.to_thread(_build_all, inputs, self._normalizer)
        # IDs are looked up now rather than stored with the inputs, so naming or
        # changing the dataset's identifier column shows here without retraining.
        wanted = {row.structure for card in scorecards for row in card.worst_rows}
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
                        replace(row, compound_id=ids.get(row.structure))
                        for row in card.worst_rows
                    ],
                )
                for card in scorecards
            ]
        return Success(scorecards)


def _build_all(inputs: ScorecardInputs, normalizer: StructureNormalizer) -> list[Scorecard]:
    """One Scorecard per target, sharing one `HeldOutChemistry`."""
    chemistry = held_out_chemistry(inputs.structures, inputs.train_structures, normalizer)
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
            random_split_metrics=target.random_split_metrics,
            random_split_unavailable=inputs.random_split_unavailable,
            random_split_metrics_undefined=target.random_split_metrics_undefined,
            metrics_undefined=target.metrics_undefined,
            duplicate_spread=target.duplicate_spread,
        )
        for target in inputs.targets
    ]
```

Import `held_out_chemistry` from `build_scorecard` at module level, so the monkeypatch in Step 1 sees it as `module.held_out_chemistry`. Import `target_columns_of` from `derive_readouts`.

- [ ] **Step 6: Route**

In `routes/protocols.py`:
- `ScorecardResponse` gains `target: str` and `joint_model: bool`, with a docstring line: one card per target; `joint_model` says whether one model learned them all.
- `from_domain` sets both.
- The route becomes `response_model=list[ScorecardResponse]` and returns `[ScorecardResponse.from_domain(card) for card in result_to_response(...)]`.

- [ ] **Step 7: Update fixtures**

Run:
```
cd backend && grep -rn "build_scorecard(\|scorecard_for\|ScorecardInputs(\|/scorecard\"" tests
```

- `build_scorecard(...)` calls in `tests/unit/execution/test_scorecard.py`:
  - pass `target="y"`;
  - pass `chemistry=held_out_chemistry(structures, train_structures, NORMALIZER)`;
  - drop `normalizer=` and `train_structures=`.
- Integration reads of `inputs.metrics` / `inputs.baseline_metrics` / `inputs.actual` / `inputs.predicted` / `inputs.task` / `inputs.metrics_undefined` / `inputs.random_split_metrics*` / `inputs.target_unit` → `inputs.targets[0].<field>`.
- `test_scorecard_inputs_reads_a_blob_written_before_baseline_conditions_existed` → also assert `inputs.targets[0].column == ""` with no `legacy_column`.
- API scorecard reads → `response.json()[0]`.

- [ ] **Step 8: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit tests/integration/test_train_protocol.py tests/integration/test_full_loop.py tests/api/test_protocols.py -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add -A backend
git commit -m "feat(scorecard): one scorecard per target, sharing the test-set chemistry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Train on every target, and refuse a joint engine mixed kinds

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py`
- Modify: `backend/src/daikonstudio/application/execution/sweeps.py:136-150`
- Modify: `backend/src/daikonstudio/domain/data/dataset.py` (delete `single_target`)
- Test:
  - `backend/tests/integration/test_train_protocol.py`
  - `backend/tests/integration/test_sweeps.py`
  - `backend/tests/unit/data/test_dataset.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `joint_kind_error(manifest: EngineManifest, dataset: Dataset) -> ValidationError | None` in `train_protocol.py`.
  - `RunTraining` trains every target. Its `ScorecardInputs.targets`, `run.metrics["targets"]` and the readouts all cover every target, in order.

- [ ] **Step 1: Failing integration tests**

In `tests/integration/test_train_protocol.py` (add `import io, zipfile`; import `ValidationError` from `daikonstudio.domain.shared.errors`):

```python
def _two_target_csv() -> bytes:
    numbers = tuple(1.0 + 0.37 * index for index in range(len(_STRUCTURES)))
    labels = _alternating_values()
    rows = "\n".join(
        f"{smiles},{number},{int(label)}"
        for smiles, number, label in zip(_STRUCTURES, numbers, labels, strict=True)
    )
    return f"smiles,y,active\n{rows}\n".encode()


MIXED = (
    TargetSpec(column="y", kind=TargetKind.NUMERIC, unit="logS", direction=Direction.HIGH),
    TargetSpec(column="active", kind=TargetKind.BINARY),
)
```

Extend `Studio.dataset` with two keyword arguments:
- `targets: tuple[TargetSpec, ...] | None = None`, defaulting to the current single `y` spec;
- `csv: bytes | None = None`, defaulting to `_csv(values)`.

Thread both into the command and the upload.

```python
async def test_a_mixed_kind_dataset_trains_one_model_per_target(studio: Studio) -> None:
    dataset = await studio.dataset(targets=MIXED, csv=_two_target_csv())
    run = await studio.wait(
        await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    )
    assert run.status is RunStatus.READY, run.error_message

    protocol = await studio.protocol_for(run)
    assert [(r.name, r.type.value) for r in protocol.readouts] == [
        ("y", "numeric"),
        ("active_probability", "probability"),
        ("active", "class"),
    ]
    inputs = await studio.scorecard_for(run)
    assert [(t.column, t.task) for t in inputs.targets] == [
        ("y", "regression"),
        ("active", "binary_classification"),
    ]
    assert inputs.joint_model is False
    assert "rmse" in inputs.targets[0].metrics
    assert "mcc" in inputs.targets[1].baseline_metrics
    assert len(inputs.targets[1].predicted) == len(inputs.structures)
    assert [h["column"] for h in run.metrics["targets"]] == ["y", "active"]
    artifact = studio.store.get_bytes(artifact_key(studio.auth.workspace_id, protocol.id))
    assert zipfile.is_zipfile(io.BytesIO(artifact))


async def test_a_single_target_protocol_still_stores_a_bare_artifact(studio: Studio) -> None:
    dataset = await studio.dataset()
    run = await studio.wait(
        await studio.train(dataset_id=dataset.id, engine_id="ecfp4-xgboost", conditions={})
    )
    protocol = await studio.protocol_for(run)
    artifact = studio.store.get_bytes(artifact_key(studio.auth.workspace_id, protocol.id))
    assert not zipfile.is_zipfile(io.BytesIO(artifact))


async def test_a_joint_engine_is_refused_a_mixed_kind_dataset_before_a_run_exists(
    studio: Studio,
) -> None:
    dataset = await studio.dataset(targets=MIXED, csv=_two_target_csv())
    result = await studio._train(
        TrainProtocolCommand(
            name="joint", dataset_id=dataset.id, engine_id="chemprop-dmpnn", conditions={}
        ),
        studio.auth,
    )
    error = result.failure()
    assert isinstance(error, ValidationError)
    assert "same kind" in str(error)


async def test_a_joint_baseline_is_refused_a_mixed_kind_dataset_too(studio: Studio) -> None:
    dataset = await studio.dataset(targets=MIXED, csv=_two_target_csv())
    result = await studio._train(
        TrainProtocolCommand(
            name="joint baseline",
            dataset_id=dataset.id,
            engine_id="ecfp4-xgboost",
            conditions={},
            baseline_engine_id="chemprop-dmpnn",
        ),
        studio.auth,
    )
    assert isinstance(result.failure(), ValidationError)
```

In `tests/integration/test_sweeps.py`, add a `mixed_dataset` fixture. Copy the `dataset` fixture, but:
- upload `_mixed_csv()`, which adds an `active` column of `(index // 2) % 2` values to `_csv()`'s rows;
- use `targets=(TargetSpec(column="y", kind=TargetKind.NUMERIC), TargetSpec(column="active", kind=TargetKind.BINARY))`.

Then:

```python
@pytest.mark.asyncio
async def test_a_joint_engine_in_the_last_config_on_a_mixed_dataset_creates_no_runs(
    submit_sweep, mixed_dataset, auth, runs_repository
) -> None:
    result = await submit_sweep(
        SubmitSweepCommand(
            name="doomed",
            dataset_id=mixed_dataset.id,
            configs=[
                SweepConfig(engine_id="ecfp4-randomforest", conditions={}),
                SweepConfig(engine_id="chemprop-dmpnn", conditions={}),
            ],
        ),
        auth=auth,
    )
    assert isinstance(result, Failure)
    assert await runs_repository.sweep_summaries(auth.workspace_id) == []
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/integration/test_train_protocol.py -k "mixed or bare or joint" tests/integration/test_sweeps.py -k mixed -q
```

Expected: FAIL. `single_target()` raises in the worker, the run is FAILED, and no refusal happens at enqueue.

- [ ] **Step 3: `joint_kind_error` and enqueue checks**

In `train_protocol.py`:
- import `EngineManifest` from `manifest`;
- import `TargetHeadline` from `domain/execution/run`;
- import `TargetSpec` from `domain/data/target`.

```python
def joint_kind_error(manifest: EngineManifest, dataset: Dataset) -> ValidationError | None:
    """Why a joint engine cannot train on this Dataset, or None when it can.

    An engine that declares `supports_multitask` learns every target in one model
    with one loss, and adding a squared error to a cross-entropy needs a relative
    weighting nobody can set honestly. So a Dataset that mixes measured values and
    active/inactive labels trains only on engines that fit one model per target.
    Checked at enqueue (`TrainProtocol`, `SubmitSweep`), which refuse before a Run
    exists, and again by the worker for a Run enqueued before the rule existed.
    """
    kinds = {target.kind for target in dataset.targets}
    if not manifest.supports_multitask or len(kinds) < 2:
        return None
    numeric = [t.column for t in dataset.targets if t.kind is TargetKind.NUMERIC]
    binary = [t.column for t in dataset.targets if t.kind is TargetKind.BINARY]
    return ValidationError(
        f"{manifest.name} trains one joint model, so every target must be the same kind. "
        f"This dataset has measured values ({', '.join(numeric)}) and active/inactive "
        f"labels ({', '.join(binary)}).",
        detail=(
            "Choose an engine that trains one model per target, or a dataset whose "
            "targets are all one kind."
        ),
    )
```

In `TrainProtocol.__call__`:
- Keep the resolved engine: `engine = self._engines.get(command.engine_id)` inside the existing `try`.
- After the dataset is loaded and workspace-checked:

```python
        for candidate in (engine, baseline):
            refused = joint_kind_error(candidate.manifest(), dataset)
            if refused is not None:
                return Failure(refused)
```

In `SubmitSweep.__call__`, after the existing engine-existence checks (import `joint_kind_error`):

```python
        # Same pre-flight reason as the existence checks above: a refusal from
        # `TrainProtocol` halfway through the loop below would leave a sweep that
        # looks complete and is not.
        baseline = (
            self._engines.get(command.baseline_engine_id)
            if command.baseline_engine_id
            else self._engines.baseline()
        )
        engine_ids = sorted({config.engine_id for config in command.configs})
        for engine in [*(self._engines.get(engine_id) for engine_id in engine_ids), baseline]:
            refused = joint_kind_error(engine.manifest(), dataset)
            if refused is not None:
                return Failure(refused)
```

- [ ] **Step 4: `RunTraining` loops over targets**

Replace `_task_for`:

```python
def _task_for(target: TargetSpec) -> TaskType:
    """The one place a task is decided, and it reads the target's spec -- never its
    values. `TrainContext.targets` is authoritative precisely so an engine cannot
    look at a column of 0.0s and 1.0s and decide for itself."""
    return (
        TaskType.BINARY_CLASSIFICATION
        if target.kind is TargetKind.BINARY
        else TaskType.REGRESSION
    )


def _check_capable(manifest: EngineManifest, dataset: Dataset, *, role: str = "") -> None:
    """Before any compute: can this engine train every target here? Milliseconds,
    against a fit measured in minutes."""
    for task in dict.fromkeys(_task_for(target) for target in dataset.targets):
        if task not in manifest.tasks:
            raise ValidationError(
                f"{role}{manifest.name} does not support {_task_label(task)}. "
                f"Supported tasks: {', '.join(map(_task_label, manifest.tasks))}."
            )
    refused = joint_kind_error(manifest, dataset)
    if refused is not None:
        raise refused
```

In `__call__`:
- `targets = {target.column: _task_for(target) for target in dataset.targets}`.
- `_check_capable(manifest, dataset)` replaces the inline task check.
- `_check_capable(baseline_manifest, dataset, role="The baseline engine: ")` replaces the baseline one. Keep the existing baseline message wording if a test pins it: `grep -n "The baseline engine" tests`.

Thread `targets: dict[str, TaskType]` through `_fit`, `_train_off_thread` and `_optimism_gap` in place of `task: TaskType`. `_train_off_thread` builds `TrainContext(..., targets=targets, ...)`.

Change `_undefined_reasons` to take the column instead of the dataset:

```python
def _undefined_reasons(
    undefined: set[str], column: str, train_rows: pl.DataFrame, test_rows: pl.DataFrame
) -> dict[str, str] | None:
    (same body, with `column` already a parameter)
```

`_optimism_gap` returns per-target results:

```python
    ) -> tuple[dict[str, tuple[dict[str, float | None], dict[str, str] | None]] | None, str | None]:
        """(existing docstring; "Returns (per target: (metrics, metrics_undefined)),
        unavailable_reason)")"""
        if dataset.split.strategy is not SplitStrategy.SCAFFOLD:
            return None, None
        await self._progress(...)  # unchanged
        try:
            random_frame = assign_split(...)  # unchanged
            result = await self._train_off_thread(
                run, engine, dataset, targets, conditions, random_frame, _RANDOM_SPLIT_SPAN
            )
            train_rows = random_frame.filter(pl.col("split") == "train")
            test_rows = random_frame.filter(pl.col("split") == "test")
            gap: dict[str, tuple[dict[str, float | None], dict[str, str] | None]] = {}
            for column in targets:
                metrics, undefined = _measured(result.metrics[column])
                gap[column] = (metrics, _undefined_reasons(undefined, column, train_rows, test_rows))
            return gap, None
        except RunInterrupted:
            raise  # (keep the comment)
        except Exception as exc:
            return None, user_facing_error(exc)  # (keep the comment)
```

After the fits, build every target's inputs and headline:

```python
        train_rows = frame.filter(pl.col("split") == "train")
        test_rows = frame.filter(pl.col("split") == "test")
        predictions = await asyncio.to_thread(
            engine.predict,
            PredictContext(
                frame=test_rows,
                structure_column=dataset.structure_column,
                artifact=chosen.artifact,
                conditions=conditions,
                target_columns=dataset.target_columns,
            ),
        )

        protocol_id = uuid.uuid4()
        per_target: list[TargetInputs] = []
        headlines: list[TargetHeadline] = []
        for target in dataset.targets:
            task = targets[target.column]
            metrics, undefined = _measured(chosen.metrics[target.column])
            baseline_metrics, baseline_undefined = _measured(baseline_result.metrics[target.column])
            gap = random_split.get(target.column) if random_split is not None else None
            # `predict` returns one row per (compound, target); this target's rows, in
            # test-set order, line up with `actual` below.
            predicted = predictions.filter(pl.col("target") == target.column).sort("row_id")
            per_target.append(
                TargetInputs(
                    column=target.column,
                    task=task.value,
                    metrics=metrics,
                    validation_metrics=(
                        _measured(chosen.validation_metrics[target.column])[0]
                        if chosen.validation_metrics is not None
                        else None
                    ),
                    actual=[float(v) for v in test_rows[target.column].to_list()],
                    predicted=[float(v) for v in predicted["value"].to_list()],
                    prediction_kind=(
                        "probability" if task is TaskType.BINARY_CLASSIFICATION else "value"
                    ),
                    baseline_metrics=baseline_metrics,
                    random_split_metrics=gap[0] if gap is not None else None,
                    random_split_metrics_undefined=gap[1] if gap is not None else None,
                    metrics_undefined=_undefined_reasons(
                        undefined | baseline_undefined, target.column, train_rows, test_rows
                    ),
                    duplicate_spread=dataset.validation_report.duplicate_spread.get(target.column),
                    target_unit=target.unit,
                    target_direction=(
                        target.direction.value if target.direction is not None else None
                    ),
                )
            )
            primary = primary_metric_for(task)
            headlines.append(
                TargetHeadline(
                    column=target.column,
                    primary_metric=primary,
                    value=metrics.get(primary),
                    baseline_value=baseline_metrics.get(primary),
                )
            )
```

Then:
- `ScorecardInputs(..., random_split_unavailable=random_split_unavailable, targets=per_target)`. Rename the first element of `_optimism_gap`'s return to `random_split`.
- `readouts=derive_readouts(dataset.targets)`.
- `run.record_metrics(headlines)`.

- [ ] **Step 5: Delete the transitional accessor**

```bash
cd backend && grep -rn "single_target" src tests
```

- Delete `Dataset.single_target` and `test_single_target_refuses_a_dataset_with_several`.
- Any remaining hit in `src` is a bug. Fix it with the real multi-target logic, not a new shim.

Expected: no hits.

- [ ] **Step 6: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports
```

Expected: the full suite passes.

```bash
git add -A backend
git commit -m "feat(training): train every target, and refuse a joint engine mixed kinds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Scale a fan-out run's deadline by its target count

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py` (`deadline_scale`, and `TrainProtocol` stamping it into params)
- Modify: `backend/src/daikonstudio/application/execution/claim_run.py:75`
- Test:
  - `backend/tests/unit/runners/test_claim_deadline.py`
  - `backend/tests/integration/test_train_protocol.py`

**Interfaces:**
- Consumes: `TrainProtocol` (Task 9).
- Produces:
  - `deadline_scale(manifest: EngineManifest, dataset: Dataset) -> int`
  - Training `run.params["deadline_scale"]`
  - `ClaimRun` multiplies the lane deadline by it, so the agent's hard kill (deadline + grace) scales too.

- [ ] **Step 1: Failing tests**

Append to `tests/unit/runners/test_claim_deadline.py`:

```python
async def test_a_fan_out_run_gets_the_lane_deadline_once_per_target():
    run = _run("gpu")
    run.params = {"deadline_scale": 4}
    runner = Runner(name="gpu-box", lanes=("gpu",), token_hash="h")
    assert (await _claim(run)(runner=runner)).unwrap() == (run, 4 * 7200, 600)
```

In `test_a_mixed_kind_dataset_trains_one_model_per_target` (Task 9), add `assert run.params["deadline_scale"] == 2`.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/runners/test_claim_deadline.py tests/integration/test_train_protocol.py -k "fan_out or mixed_kind" -q
```

Expected: FAIL.

- [ ] **Step 3: Implement**

In `train_protocol.py`:

```python
def deadline_scale(manifest: EngineManifest, dataset: Dataset) -> int:
    """How many times over its lane's deadline a training Run may take.

    A fan-out engine fits once per target in each of its legs -- the model and the
    random-split comparison -- so four targets take about four times as long as one
    against a budget sized for one. A joint engine fits once regardless.

    ponytail: ignores the baseline, which fans out too -- a joint chemprop run on
    four targets still fits four random forests. Cheap next to chemprop's own fit
    today; scale by the baseline as well if one ever dominates.
    """
    return 1 if manifest.supports_multitask else len(dataset.targets)
```

In `TrainProtocol.__call__`:
`params={**command.to_params(), "deadline_scale": deadline_scale(engine.manifest(), dataset)},`

`from_params` ignores keys it does not read, and `RetryRun` re-enqueues the same row, so a retry keeps the scale.

In `claim_run.py`, after the lane lookup:

```python
        # A fan-out training run fits once per target; `TrainProtocol` stamped how
        # many lane budgets that is worth. Absent on prediction runs and on runs
        # enqueued before targets could be several, where it is 1.
        deadline *= int(run.params.get("deadline_scale", 1))
```

- [ ] **Step 4: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/runners tests/integration/test_train_protocol.py -q && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add -A backend
git commit -m "feat(runs): scale a fan-out run's deadline by its target count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Prediction results for every target

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/predict_with_protocol.py`:
  - the `RunPrediction` readouts block, lines 359-390;
  - `PredictionRow`;
  - `GetPredictionResults`.
- Modify: `backend/src/daikonstudio/interface/routes/runs.py` (`PredictionResponse.uncertainty`)
- Test:
  - `backend/tests/api/test_runs.py`
  - `backend/tests/api/test_triage_round_trip.py`

**Interfaces:**
- Consumes:
  - `target_columns_of` (Task 4)
  - `probability_column`, `uncertainty_column` (Task 2)
  - Long-format predictions (Task 6)
- Produces:
  - Results parquet with each target's readout columns.
  - `uncertainty` (N = 1) or `{column}_uncertainty` (N > 1).
  - `PredictionRow.uncertainty: dict[str, float | None]` and HTTP `PredictionResponse.uncertainty: dict[str, float | None]`, keyed by target column.
  - Sort and filter accept every uncertainty column.

- [ ] **Step 1: Failing tests**

In `tests/api/test_runs.py`:
- In `test_results_carry_structure_readouts_uncertainty_and_applicability`, change the uncertainty assertions to `assert row["uncertainty"] == {"y": None}`.
- Append:

```python
async def test_a_one_target_results_file_keeps_the_plain_uncertainty_column(
    client, published_protocol_id, prediction_upload_ref
):
    """Every results file written before several targets existed has `uncertainty`,
    not `y_uncertainty`; a one-target protocol must keep writing and sorting by it."""
    run_id = (await _predict(client, published_protocol_id, prediction_upload_ref)).json()["id"]
    response = await client.get(
        f"/api/v1/runs/{run_id}/results", params={"sort_by": "uncertainty"}
    )
    assert response.status_code == 200, response.text


def _two_target_training_csv() -> bytes:
    rows = "\n".join(
        f"{smiles},{1.0 + 0.37 * index},{(index // 2) % 2}"
        for index, smiles in enumerate(_STRUCTURES)
    )
    return f"smiles,y,active\n{rows}\n".encode()


async def test_a_two_target_protocol_writes_every_readout_and_its_own_uncertainty(
    client, csv_upload, prediction_upload_ref
):
    upload_ref = await csv_upload(_two_target_training_csv())
    dataset = await client.post(
        "/api/v1/datasets",
        json={
            "name": "panel",
            "upload_ref": upload_ref,
            "structure_column": "smiles",
            "targets": [
                {"column": "y", "kind": "numeric", "unit": "logS", "direction": "high"},
                {"column": "active", "kind": "binary"},
            ],
            "split": {"strategy": "random", "seed": 1},
        },
    )
    assert dataset.status_code == 201, dataset.text
    trained = await _train(client, dataset.json()["id"], engine_id="ecfp4-randomforest")
    assert trained.status_code == 202, trained.text
    protocol_id = (await client.get("/api/v1/protocols")).json()["items"][0]["id"]
    assert (await client.post(f"/api/v1/protocols/{protocol_id}/publish")).status_code == 204

    run_id = (await _predict(client, protocol_id, prediction_upload_ref)).json()["id"]
    response = await client.get(
        f"/api/v1/runs/{run_id}/results",
        params={"sort_by": "active_uncertainty", "sort_dir": "desc"},
    )
    assert response.status_code == 200, response.text
    row = response.json()["items"][0]
    assert set(row["readouts"]) == {"y", "active_probability", "active"}
    assert set(row["uncertainty"]) == {"y", "active"}
    # the random forest reports a spread for each target separately
    assert isinstance(row["uncertainty"]["y"], float)
    assert isinstance(row["uncertainty"]["active"], float)
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/api/test_runs.py -q
```

Expected: FAIL. `uncertainty` is a float, and two readouts destructure badly at `predict_with_protocol.py:384` (handoff trap 2).

- [ ] **Step 3: `RunPrediction`**

Replace the block from `values = predictions["value"].to_list()` to `columns["uncertainty"] = predictions["uncertainty"]` with the code below. Keep the reserved-names comment and extend it to mention `{target}_uncertainty`.

```python
        target_columns = target_columns_of(protocol.readouts)
        readouts = {readout.name: readout for readout in protocol.readouts}
        columns: dict[str, pl.Series] = {"structure": pl.Series(structures)}
        columns["input_row"] = pl.Series(input_rows, dtype=pl.Int64)
        if compound_ids is not None:
            columns["compound_id"] = pl.Series(compound_ids, dtype=pl.String)
        for column in target_columns:
            # One row per (compound, target) from `predict`; this target's, in input order.
            part = predictions.filter(pl.col("target") == column).sort("row_id")
            values = part["value"].to_list()
            if readouts[column].type is ReadoutType.CLASS:
                # `value` is P(class=1); the hard label is the standard 0.5 decision
                # threshold over it -- the engine's own `predict()` only ever returns
                # the probability (see `_scoring.py`), so this is the one place a
                # class label exists.
                #
                # ponytail: 0.5 is fixed, not configurable -- there is nowhere for a
                # scientist to ask for a different operating point (e.g. to trade
                # recall for precision on an imbalanced assay). Upgrade path: accept
                # it as a prediction condition once someone needs one.
                columns[probability_column(column)] = pl.Series(values, dtype=pl.Float64)
                columns[column] = pl.Series(
                    [1.0 if v >= 0.5 else 0.0 for v in values], dtype=pl.Float64
                )
            else:
                columns[column] = pl.Series(values, dtype=pl.Float64)
            columns[uncertainty_column(column, target_count=len(target_columns))] = part[
                "uncertainty"
            ]
        columns["applicability"] = pl.Series(similarities, dtype=pl.Float64)
```

At N = 1 this writes exactly the old frame: the same columns in the same order (`structure`, `input_row`, `compound_id`, readouts, `uncertainty`, `applicability`).

Imports:
- `target_columns_of` from `application.catalog.derive_readouts`
- `ReadoutType` from `domain.catalog.readout`
- `probability_column` and `uncertainty_column` from `domain.data.target`

- [ ] **Step 4: `GetPredictionResults` and the response**

`PredictionRow.uncertainty: dict[str, float | None]`. Add a comment: keyed by target column, each target's own model's spread; `None` per target where the engine has none to report.

In `GetPredictionResults.__call__`:

```python
        target_columns = target_columns_of(protocol.readouts)
        uncertainty_columns = {
            column: uncertainty_column(column, target_count=len(target_columns))
            for column in target_columns
        }
        viewed = apply_result_view(
            frame,
            columns={readout.name for readout in protocol.readouts}
            | set(uncertainty_columns.values())
            | {"applicability"},
            sort=query.sort,
            filters=query.filters,
        )
```

and in each `PredictionRow`:
`uncertainty={column: row.get(name) for column, name in uncertainty_columns.items()},`

In `routes/runs.py`:
- `PredictionResponse.uncertainty: dict[str, float | None]`.
- Update the docstring: one entry per target, keyed by its column.

- [ ] **Step 5: Update fixtures**

```bash
cd backend && grep -rn "\"uncertainty\"\]\|\[\"uncertainty\"\]\|uncertainty is None\|uncertainty=" tests/api tests/integration
```

Convert scalar expectations to the dict keyed by the protocol's target column (`"y"` in these fixtures).

- [ ] **Step 6: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/api tests/integration -q && uv run ruff check src tests && uv run mypy src && uv run lint-imports
```

Expected: PASS.

```bash
git add -A backend
git commit -m "feat(predictions): write every target's readouts and its own uncertainty

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Chemprop learns its targets jointly

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py`
- Test: `backend/tests/unit/engines/test_chemprop_dmpnn.py`

**Interfaces:**
- Consumes: `TrainContext.target_columns` / `.task`, `PredictContext.target_columns` (Task 5).
- Produces:
  - `_MANIFEST.supports_multitask = True`
  - chemprop `predict` returns long format with a `target` column
  - old one-target checkpoints still predict

- [ ] **Step 1: Failing test**

Append to `tests/unit/engines/test_chemprop_dmpnn.py`. Reuse that file's frame fixture and its chemprop `importorskip` guard. The frame below assumes ≥ 12 rows with a train/validation/test split; adapt `_frame()` to the file's helper name.

```python
def test_two_targets_train_jointly_and_predict_in_long_format():
    frame = _frame().with_columns((pl.col("y") * 2.0 + 1.0).alias("z"))
    engine = ChempropDMPNN()
    result = engine.train(
        TrainContext(
            frame=frame,
            targets={"y": TaskType.REGRESSION, "z": TaskType.REGRESSION},
            structure_column="smiles",
            conditions={"epochs": 2},
            seed=1,
        )
    )
    assert list(result.metrics) == ["y", "z"]
    assert "rmse" in result.metrics["z"]

    predictions = engine.predict(
        PredictContext(
            frame=frame,
            structure_column="smiles",
            artifact=result.artifact,
            conditions={},
            target_columns=("y", "z"),
        )
    )
    assert predictions.height == 2 * frame.height
    assert predictions["target"].unique().sort().to_list() == ["y", "z"]


def test_chemprop_declares_that_it_learns_targets_jointly():
    assert ChempropDMPNN.manifest().supports_multitask is True
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -q
```

Expected:
- FAIL: `target_column` raises on two targets, and `supports_multitask` is False.
- If chemprop is not installed, these are SKIPPED. Run them on the GPU-lane environment (`make dev-worker-gpu`'s venv) before calling the task done.

- [ ] **Step 3: Implement**

In `chemprop_dmpnn.py`:
- `_MANIFEST` gains `supports_multitask=True`.
- Update the description: "Learns every target of a dataset in one model."
- `_datapoints(structures, targets: list[Sequence[float]] | None = None)`. Each datapoint gets `y=np.array([float(v) for v in row])`. Keep the docstring's point that `y` is 1-D of length n_tasks.
- `_forward` returns a 2-D array, one column per task:

```python
    return torch.cat(batches).cpu().numpy().reshape(len(dataset), -1)
```

and its docstring says "(molecules, tasks)".
- `_build_model(..., n_tasks: int)` passes `n_tasks=n_tasks` to `BinaryClassificationFFN(...)` and `RegressionFFN(...)`.

In `train`:

```python
        columns = ctx.target_columns
        # One task for the whole fit: a joint engine is only ever handed a dataset
        # whose targets share a kind (`joint_kind_error`, at enqueue).
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        ...
        train_set = MoleculeDataset(
            _datapoints(train_rows[ctx.structure_column].to_list(), train_rows.select(columns).rows())
        )
        validation_set = MoleculeDataset(
            _datapoints(
                validation_rows[ctx.structure_column].to_list(), validation_rows.select(columns).rows()
            )
        )
        ...
        model = _build_model(..., n_tasks=len(columns))
        ...
        def score(rows: pl.DataFrame) -> dict[str, dict[str, float]]:
            dataset = MoleculeDataset(_datapoints(rows[ctx.structure_column].to_list()))
            values = _forward(trainer, model, dataset)
            scored: dict[str, dict[str, float]] = {}
            for index, column in enumerate(columns):
                truth = rows[column].to_numpy()
                if is_classification:
                    scored[column] = classification_metrics(
                        truth,
                        (values[:, index] >= 0.5).astype(float),
                        values[:, index],
                        train_has_both_classes=train_rows[column].n_unique() >= 2,
                    )
                else:
                    scored[column] = regression_metrics(truth, values[:, index])
            return scored

        metrics = score(test_rows)
        validation_metrics = score(validation_rows) if validation_rows.height > 0 else None
```

Delete the old `train_has_both_classes` line, since it is now per column inside `score`. The scaler comments stay: `normalize_targets` fits one scaler per column.

In `predict`:

```python
        values = _forward(trainer, model, dataset)
        row_ids = pl.Series(range(values.shape[0]), dtype=pl.Int64)
        # (keep the explicit-dtypes and ponytail-uncertainty comments)
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "row_id": row_ids,
                        "value": pl.Series([float(v) for v in values[:, index]], dtype=pl.Float64),
                        "uncertainty": pl.Series([None] * values.shape[0], dtype=pl.Float64),
                        "target": pl.Series([column] * values.shape[0], dtype=pl.String),
                    }
                )
                for index, column in enumerate(ctx.target_columns)
            ]
        )
```

An old one-target checkpoint yields `values` of shape (n, 1), and `ctx.target_columns` has length 1, so it still predicts.

- [ ] **Step 4: Run tests (with chemprop installed) and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_chemprop_dmpnn.py tests/unit/engines -q && uv run mypy src && uv run lint-imports
```

Expected: PASS, not SKIPPED, in an environment with chemprop.

```bash
git add -A backend
git commit -m "feat(engines): chemprop learns every target of a dataset jointly

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Regenerate the API client and add the shared frontend helpers

**Files:**
- Modify: `frontend/openapi.json` and `frontend/src/shared/lib/api/**`, both regenerated.
- Create:
  - `frontend/src/shared/lib/targets.ts`
  - `frontend/src/shared/lib/targets.test.ts`
- Modify:
  - `frontend/src/features/engines/types/index.ts`
  - `frontend/src/features/engines/types/index.test.ts`
  - `frontend/src/features/engines/index.ts`

**Interfaces:**
- Consumes: the backend contract after Tasks 1–12.
- Produces:
  - `targetsOf(readouts: ReadoutResponse[]): string[]`
  - `uncertaintyColumn(target: string, targetCount: number): string`
  - `enginesForTargets(engines: Engine[], targets: Pick<TargetBody, "kind">[]): Engine[]`, which replaces `enginesForTargetKind`
  - `jointEnginesRefused(engines: Engine[], targets: Pick<TargetBody, "kind">[]): Engine[]`
  - `trainingKind(engine: Engine, targetCount: number): string | null`

- [ ] **Step 1: Regenerate**

Run:
```
make generate-api
```

Then:
```
cd frontend && git diff --stat openapi.json src/shared/lib/api
```

Expected: the diff shows these changes, and **nothing else**. An unrelated change means `main` and the snapshot had drifted; stop and report.
- `targets: TargetBody[]` on dataset create/response
- `CompoundResponse.targets`
- `duplicate_spread` as a record
- `ConflictRowResponse.column`
- `EngineManifestResponse.supports_multitask`
- `ScorecardResponse.target` / `joint_model`
- `PredictionResponse.uncertainty` as a record
- `TargetHeadlineWire`

- [ ] **Step 2: Failing tests**

Create `frontend/src/shared/lib/targets.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { ReadoutResponse } from "@/shared/lib/api/model";
import { targetsOf, uncertaintyColumn } from "./targets";

const readout = (name: string, type: ReadoutResponse["type"]): ReadoutResponse => ({
  name,
  type,
  unit: null,
  direction: null,
  description: "",
});

describe("targetsOf", () => {
  it("recovers each target once, in order, from its readouts", () => {
    expect(
      targetsOf([
        readout("solubility", "numeric"),
        readout("reactive_probability", "probability"),
        readout("reactive", "class"),
      ]),
    ).toEqual(["solubility", "reactive"]);
  });
});

describe("uncertaintyColumn", () => {
  it("keeps the plain name for one target and suffixes it for several", () => {
    expect(uncertaintyColumn("y", 1)).toBe("uncertainty");
    expect(uncertaintyColumn("y", 2)).toBe("y_uncertainty");
  });
});
```

In `features/engines/types/index.test.ts`, replace the `enginesForTargetKind` tests:

```ts
const rf = { id: "rf", tasks: ["regression", "binary_classification"], supports_multitask: false } as Engine;
const chemprop = { id: "chemprop", tasks: ["regression", "binary_classification"], supports_multitask: true } as Engine;
const regressor = { id: "gp-reg", tasks: ["regression"], supports_multitask: false } as Engine;

it("offers an engine only if it supports every target's task", () => {
  expect(enginesForTargets([rf, regressor], [{ kind: "numeric" }, { kind: "binary" }])).toEqual([rf]);
});

it("withholds a joint engine from a dataset that mixes kinds, and names it", () => {
  const mixed = [{ kind: "numeric" as const }, { kind: "binary" as const }];
  expect(enginesForTargets([rf, chemprop], mixed)).toEqual([rf]);
  expect(jointEnginesRefused([rf, chemprop], mixed)).toEqual([chemprop]);
  expect(enginesForTargets([rf, chemprop], [{ kind: "binary" }, { kind: "binary" }])).toEqual([rf, chemprop]);
});

it("says how an engine trains several targets, and nothing for one", () => {
  expect(trainingKind(chemprop, 4)).toBe("Trains one joint model on all 4 targets.");
  expect(trainingKind(rf, 4)).toBe("Trains 4 separate models, one per target.");
  expect(trainingKind(rf, 1)).toBeNull();
});
```

- [ ] **Step 3: Run to verify failure**

Run:
```
cd frontend && pnpm vitest run src/shared/lib/targets.test.ts src/features/engines/types/index.test.ts
```

Expected: FAIL (missing exports).

- [ ] **Step 4: Implement**

Create `frontend/src/shared/lib/targets.ts`:

```ts
import type { ReadoutResponse } from "@/shared/lib/api/model";

/**
 * The target columns a Protocol predicts, recovered from its readouts. Mirrors
 * the backend's `target_columns_of`: every target has exactly one readout named
 * after its own column, and only a binary one adds a second, probability, one.
 */
export function targetsOf(readouts: ReadoutResponse[]): string[] {
  return readouts.filter((readout) => readout.type !== "probability").map((readout) => readout.name);
}

/**
 * The results column holding one target's uncertainty, which is also the name
 * the results endpoint sorts and filters by. Mirrors the backend's
 * `uncertainty_column`: plain `uncertainty` for a one-target Protocol.
 */
export function uncertaintyColumn(target: string, targetCount: number): string {
  return targetCount === 1 ? "uncertainty" : `${target}_uncertainty`;
}
```

In `features/engines/types/index.ts`, replace `enginesForTargetKind` with the code below. Keep `TASK_FOR_TARGET_KIND`, and import `TargetBody` from `@/shared/lib/api/model`.

```ts
type HasKind = Pick<TargetBody, "kind">;

/**
 * Engines that can train on every one of these targets. An engine must support
 * each target's task; a joint engine (`supports_multitask`) learns them all in one
 * model, so the server refuses it a dataset that mixes measured values and
 * active/inactive labels, and so does this list.
 */
export function enginesForTargets(engines: Engine[], targets: HasKind[]): Engine[] {
  const kinds = new Set(targets.map((target) => target.kind));
  const tasks = [...kinds].map((kind) => TASK_FOR_TARGET_KIND[kind]);
  return engines.filter(
    (engine) =>
      tasks.every((task) => engine.tasks.includes(task)) &&
      (!engine.supports_multitask || kinds.size <= 1),
  );
}

/** Joint engines left out of `enginesForTargets` because the targets mix kinds. */
export function jointEnginesRefused(engines: Engine[], targets: HasKind[]): Engine[] {
  const mixed = new Set(targets.map((target) => target.kind)).size > 1;
  return mixed ? engines.filter((engine) => engine.supports_multitask) : [];
}

/** How an engine trains on several targets; nothing to say about one. */
export function trainingKind(engine: Engine, targetCount: number): string | null {
  if (targetCount < 2) return null;
  return engine.supports_multitask
    ? `Trains one joint model on all ${targetCount} targets.`
    : `Trains ${targetCount} separate models, one per target.`;
}
```

In `features/engines/index.ts`, export the three new helpers and remove `enginesForTargetKind`.

- [ ] **Step 5: Run tests and typecheck, then commit**

Run:
```
cd frontend && pnpm vitest run src/shared/lib/targets.test.ts src/features/engines && pnpm lint
```

Expected: PASS.

`pnpm exec tsc --noEmit` will now list every frontend consumer of the old shapes. That list is the worklist for F2–F7. Do not fix them here.

```bash
git add -A frontend
git commit -m "feat(frontend): regenerate the API client for several targets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14 (F2): Choose several targets when uploading

**Files:**
- Modify:
  - `frontend/src/features/datasets/types/index.ts:55-83`
  - `frontend/src/features/datasets/lib/draft-from-upload.ts`
  - `frontend/src/features/datasets/lib/draft-from-upload.test.ts`
  - `frontend/src/features/datasets/components/dataset-wizard.tsx:102-388`
  - `frontend/src/features/datasets/hooks/use-datasets.ts:89-96` (`CreateDatasetInput`)

**Interfaces:**
- Consumes: `TargetBody`, regenerated in Task 13.
- Produces:
  - `DraftTarget { column; kind; unit; direction }`
  - `DatasetDraft.targets: DraftTarget[]`
  - `draftTarget(column: string, rows: Record<string, string>[]): DraftTarget`
  - `toggleTarget(draft: DatasetDraft, column: string, chosen: boolean, rows: Record<string, string>[]): DatasetDraft`
  - `withColumns(draft, changes: Partial<Pick<DatasetDraft, "structureColumn" | "targets">>)`

- [ ] **Step 1: Failing tests**

Replace the target-related expectations in `draft-from-upload.test.ts` with:

```ts
import { describe, expect, it } from "vitest";
import { draftFromUpload, toggleTarget, withColumns } from "./draft-from-upload";

const rows = [
  { smiles: "CCO", solubility: "1.2", reactive: "0", id: "A1" },
  { smiles: "CCN", solubility: "3.4", reactive: "1", id: "A2" },
];
const columns = ["smiles", "solubility", "reactive", "id"];

describe("draftFromUpload", () => {
  it("starts with the first non-structure column as the only target", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    expect(draft.targets).toEqual([
      { column: "solubility", kind: "numeric", unit: "", direction: "high" },
    ]);
  });
});

describe("toggleTarget", () => {
  it("adds targets in the order chosen, guessing each one's kind, and removes them", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    const both = toggleTarget(draft, "reactive", true, rows);
    expect(both.targets.map((t) => [t.column, t.kind])).toEqual([
      ["solubility", "numeric"],
      ["reactive", "binary"],
    ]);
    expect(toggleTarget(both, "solubility", false, rows).targets.map((t) => t.column)).toEqual([
      "reactive",
    ]);
  });

  it("clears an identifier that becomes a target", () => {
    const draft = { ...draftFromUpload(columns, rows, "panel.csv"), idColumn: "id" };
    expect(toggleTarget(draft, "id", true, rows).idColumn).toBeNull();
  });
});

describe("withColumns", () => {
  it("drops a target that becomes the structure column", () => {
    const draft = draftFromUpload(columns, rows, "panel.csv");
    expect(withColumns(draft, { structureColumn: "solubility" }).targets).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd frontend && pnpm vitest run src/features/datasets/lib/draft-from-upload.test.ts
```

Expected: FAIL.

- [ ] **Step 3: Types and draft helpers**

In `types/index.ts`:

```ts
/** One column the wizard will predict, with how it is measured. */
export interface DraftTarget {
  column: string;
  kind: TargetKind;
  unit: string;
  direction: Direction | "";
}

/** What the wizard accumulates. The file is held client-side until step 4. */
export interface DatasetDraft {
  file: File | null;
  name: string;
  structureColumn: string;
  /** In the order chosen; that order is kept on the dataset and everywhere after. */
  targets: DraftTarget[];
  strategy: SplitStrategy;
  seed: number;
  /** The column holding compound IDs, or null for none. */
  idColumn: string | null;
}
```

`EMPTY_DRAFT` changes: `targets: []` replaces `targetColumn`, `kind`, `unit` and `direction`.

`draft-from-upload.ts`:

```ts
import { guessIdColumn } from "@/shared/lib/guess-id-column";
import { type DatasetDraft, type DraftTarget, EMPTY_DRAFT } from "../types";
import { guessStructureColumn, looksBinary } from "./parse-csv";

/** A newly chosen target, its kind guessed from the preview rows. */
export function draftTarget(column: string, rows: Record<string, string>[]): DraftTarget {
  return { column, kind: looksBinary(rows, column) ? "binary" : "numeric", unit: "", direction: "high" };
}

/** The wizard's starting draft for an upload: every column guessed from its headers. */
export function draftFromUpload(
  columns: string[],
  rows: Record<string, string>[],
  fileName: string,
): DatasetDraft {
  const structureColumn = guessStructureColumn(columns);
  const first = columns.find((column) => column !== structureColumn);
  return {
    ...EMPTY_DRAFT,
    name: fileName.replace(/\.csv$/i, ""),
    structureColumn,
    targets: first ? [draftTarget(first, rows)] : [],
    idColumn: guessIdColumn(
      columns.filter((column) => column !== first),
      structureColumn,
    ),
  };
}

/**
 * A column change that keeps the structure, the targets and the identifier
 * distinct: a target that becomes the structure column is dropped, and an
 * identifier that becomes either is cleared.
 */
export function withColumns(
  draft: DatasetDraft,
  changes: Partial<Pick<DatasetDraft, "structureColumn" | "targets">>,
): DatasetDraft {
  const next = { ...draft, ...changes };
  const targets = next.targets.filter((target) => target.column !== next.structureColumn);
  const clash =
    next.idColumn === next.structureColumn ||
    targets.some((target) => target.column === next.idColumn);
  return { ...next, targets, idColumn: clash ? null : next.idColumn };
}

/** Choose or unchoose one column to predict. New targets go last. */
export function toggleTarget(
  draft: DatasetDraft,
  column: string,
  chosen: boolean,
  rows: Record<string, string>[],
): DatasetDraft {
  const others = draft.targets.filter((target) => target.column !== column);
  return withColumns(draft, {
    targets: chosen ? [...others, draftTarget(column, rows)] : others,
  });
}
```

- [ ] **Step 4: Wizard steps and submit**

In `dataset-wizard.tsx`, import `Checkbox` from `@/shared/components/ui/checkbox`, and `toggleTarget` plus `type DraftTarget`.

Add, inside the component:

```tsx
  function patchTarget(column: string, changes: Partial<DraftTarget>) {
    setDraft((prev) => ({
      ...prev,
      targets: prev.targets.map((target) =>
        target.column === column ? { ...target, ...changes } : target,
      ),
    }));
  }
```

`submit()`'s body: `targets` replaces `target` (and update `CreateDatasetInput.target` to `targets: {...}[]` in `use-datasets.ts`):

```ts
        targets: draft.targets.map((target) => ({
          column: target.column,
          kind: target.kind,
          unit: target.kind === "numeric" && target.unit.trim() ? target.unit.trim() : null,
          direction: target.kind === "numeric" && target.direction ? target.direction : null,
        })),
```

`canContinue`:

```ts
  const canContinue = [
    Boolean(draft.file),
    Boolean(draft.name.trim() && draft.structureColumn && draft.targets.length > 0),
    draft.targets.length > 0,
    Boolean(draft.strategy),
  ][step];
```

Columns step: replace the "Value to predict" `Select` block with:

```tsx
                <div className="space-y-1.5">
                  <Label>Values to predict</Label>
                  <div className="max-h-48 space-y-1.5 overflow-y-auto rounded-md border p-2">
                    {preview.columns
                      .filter((column) => column !== draft.structureColumn)
                      .map((column) => (
                        <label key={column} className="flex items-center gap-2 text-sm">
                          <Checkbox
                            checked={draft.targets.some((target) => target.column === column)}
                            onCheckedChange={(checked) =>
                              setDraft((prev) =>
                                toggleTarget(prev, column, checked === true, preview.rows),
                              )
                            }
                          />
                          <span className="font-mono">{column}</span>
                        </label>
                      ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Choose one or more. Every compound needs a value in each.
                  </p>
                </div>
```

The identifier `Select` excludes every target:
`column !== draft.structureColumn && !draft.targets.some((target) => target.column === column)`.

Target step: replace the whole `{step === 2 && (...)}` block with:

```tsx
          {step === 2 && (
            <div className="space-y-6">
              {draft.targets.map((target) => (
                <div key={target.column} className="space-y-4">
                  <div className="space-y-2">
                    <Label>
                      What kind of value is <span className="font-mono">{target.column}</span>?
                    </Label>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {(["numeric", "binary"] as const).map((kind) => (
                        <button
                          key={kind}
                          type="button"
                          onClick={() => patchTarget(target.column, { kind })}
                          className={`h-full rounded-lg border p-3 text-left transition-colors ${
                            target.kind === kind
                              ? "border-primary bg-primary/5"
                              : "border-border hover:bg-muted/40"
                          }`}
                        >
                          <span className="text-sm font-medium">{TARGET_KIND_COPY[kind].title}</span>
                          <span className="mt-1 block text-xs text-muted-foreground">
                            {TARGET_KIND_COPY[kind].detail}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                  {target.kind === "numeric" && (
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <Label htmlFor={`unit-${target.column}`}>Unit</Label>
                        <Input
                          id={`unit-${target.column}`}
                          value={target.unit}
                          onChange={(event) => patchTarget(target.column, { unit: event.target.value })}
                          placeholder="µM, log mol/L, kcal/mol…"
                        />
                        <p className="text-xs text-muted-foreground">Shown with every predicted value.</p>
                      </div>
                      <div className="space-y-1.5">
                        <Label>Preferred direction</Label>
                        <Select
                          value={target.direction || "high"}
                          onValueChange={(value) =>
                            patchTarget(target.column, { direction: value as "high" | "low" })
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="high">Higher is better</SelectItem>
                            <SelectItem value="low">Lower is better</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                          Used when ranking triage results and when judging a model against its
                          baseline.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
```

Remove the now-unused `looksBinary` and `withColumns` imports if biome flags them. The structures `Select`'s `withColumns(prev, { structureColumn: value })` call stays.

- [ ] **Step 5: Run tests, lint and typecheck the touched files, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/datasets && pnpm lint
```

Expected: PASS.

```bash
git add -A frontend
git commit -m "feat(datasets): choose several values to predict at upload, each with its kind

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15 (F3): Dataset pages show every target

**Files:**
- Modify:
  - `frontend/src/features/datasets/components/dataset-list.tsx:26-27`
  - `frontend/src/features/datasets/components/dataset-detail.tsx:33, 98-120, 186`
  - `frontend/src/features/datasets/components/compound-browser.tsx`
  - `frontend/src/features/datasets/components/compound-browser.test.tsx`
  - `frontend/src/features/datasets/components/dataset-profile-view.tsx`
  - `frontend/src/features/datasets/components/validation-report-view.tsx:67-80, 125-145`
  - `frontend/src/features/datasets/hooks/use-datasets.ts` (`useDatasetProfile`, `CompoundQuery`)

**Interfaces:**
- Consumes: `DatasetResponse.targets`, `CompoundResponse.targets`, `duplicate_spread` record, `ConflictRowResponse.column`, and the `target` index query parameter.
- Produces:
  - `useDatasetProfile(id, target = 0)`
  - `DatasetProfileView({ dataset, target, profile })`

- [ ] **Step 1: Failing test**

In `compound-browser.test.tsx`, change the fixtures:
- dataset: `targets: [{ column: "y", kind: "numeric", unit: null }, { column: "active", kind: "binary", unit: null }]`
- item: `targets: { y: 1, active: 0 }`

Add:

```tsx
it("shows a column per target and sorts by the one clicked", async () => {
  render(<CompoundBrowser dataset={dataset} />);
  expect(await screen.findByRole("columnheader", { name: "active" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /active/ }));
  expect(useDatasetCompoundsMock).toHaveBeenLastCalledWith(
    dataset.id,
    expect.objectContaining({ sort: "target", target: 1 }),
  );
});
```

The mock name must match whatever this file already mocks `useDatasetCompounds` as. Use `@testing-library/user-event` if the file already does; otherwise use `fireEvent.click`.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd frontend && pnpm vitest run src/features/datasets/components/compound-browser.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Implement**

**`use-datasets.ts`:**
- `CompoundQuery` gains `target?: number`.
- `useDatasetProfile(id: string | undefined, target = 0)`: `queryKey: [...DATASET_PROFILE_KEY, id, target]` and `params: { target }` on the request.

**`compound-browser.tsx`:**
- Add `const [sortTarget, setSortTarget] = useState(0);`.
- The query adds `target: sortTarget`.
- Replace the single sort button with:

```tsx
        <div className="flex flex-wrap gap-1">
          {dataset.targets.map((target, index) => (
            <Button
              key={target.column}
              variant={index === sortTarget ? "secondary" : "ghost"}
              size="sm"
              onClick={() =>
                reset(() => {
                  if (index === sortTarget) setDescending((d) => !d);
                  else {
                    setSortTarget(index);
                    setDescending(false);
                  }
                })
              }
            >
              {target.column}
              {index === sortTarget &&
                (descending ? <ArrowDown className="size-3.5" /> : <ArrowUp className="size-3.5" />)}
            </Button>
          ))}
        </div>
```

- The single target header becomes:
  ```tsx
  {dataset.targets.map((target) => (
    <TableHead key={target.column} className="text-right">{target.column}</TableHead>
  ))}
  ```
- The single target cell becomes:

```tsx
                  {dataset.targets.map((target) => (
                    <TableCell key={target.column} className="text-right">
                      <ReadoutValue
                        value={compound.targets[target.column] ?? null}
                        unit={target.unit}
                        precision={3}
                      />
                    </TableCell>
                  ))}
```

- The skeleton `colSpan` becomes `3 + dataset.targets.length + (dataset.id_column ? 1 : 0)`.

**`dataset-list.tsx`:**

```tsx
        <span className="font-mono">{dataset.targets.map((target) => target.column).join(", ")}</span>
        {dataset.targets.length === 1 && dataset.targets[0].unit && (
          <span className="font-mono">{dataset.targets[0].unit}</span>
        )}
```

**`dataset-detail.tsx`:**
- Add `const [profileTarget, setProfileTarget] = useState(0);`.
- `const profile = useDatasetProfile(datasetId, profileTarget);`.
- In the "What this predicts" card:
  - Replace the three fields `Target`, `Kind` and `Unit and direction` with the field below.
  - Change the grid to `sm:grid-cols-3`.

```tsx
                <Field
                  label={dataset.targets.length === 1 ? "Target" : "Targets"}
                  value={
                    <ul className="space-y-1">
                      {dataset.targets.map((target) => (
                        <li key={target.column}>
                          <span className="font-mono">{target.column}</span>
                          <span className="ml-2 text-muted-foreground">
                            {target.kind === "numeric" ? "Measured value" : "Active / inactive"}
                            {target.unit ? ` · ${target.unit}` : ""}
                            {target.direction
                              ? ` · ${target.direction === "high" ? "higher" : "lower"} is better`
                              : ""}
                          </span>
                        </li>
                      ))}
                    </ul>
                  }
                />
```

- Above the profile, when `dataset.targets.length > 1`, render a `Select` with
  `value={String(profileTarget)}` and `onValueChange={(v) => setProfileTarget(Number(v))}`. It has one `SelectItem` per target (`value={String(index)}`, label the column), and its `Label` reads "Profile for".
- Pass `target={dataset.targets[profileTarget]}` to `DatasetProfileView`.

**`dataset-profile-view.tsx`:**
- `DatasetProfileView`, `TargetSection` and `CliffSection` take `target: TargetBody` (import it from `@/shared/lib/api/model`).
- Replace `dataset.target.unit` with `target.unit` and `dataset.target.column` with `target.column`.
- The noise floor becomes `const noiseFloor = dataset.validation_report.duplicate_spread[target.column] ?? null;`.

**`validation-report-view.tsx`:**
- Replace the noise-floor card's condition and body:

```tsx
      {Object.keys(report.duplicate_spread).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Assay noise floor</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1 text-sm">
              {Object.entries(report.duplicate_spread).map(([column, spread]) => (
                <li key={column}>
                  Repeat measurements of <span className="font-mono">{column}</span> disagreed by{" "}
                  <span className="font-mono font-medium">{spread.toFixed(3)}</span> on average.
                </li>
              ))}
            </ul>
            <p className="mt-1 text-sm text-muted-foreground">
              Model error below this level is within experimental error. The scorecard reports it as
              the noise floor.
            </p>
          </CardContent>
        </Card>
      )}
```

- In the conflicts table, add a `<th className="pb-2 pr-4 font-medium">Target</th>` before "Labels". Add the matching `<td className="py-1.5 pr-4 font-mono text-xs">{row.column}</td>`. Change the row key to `` `${row.structure}-${row.column}` ``.

- [ ] **Step 4: Run tests, lint and typecheck, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/datasets && pnpm lint
```

Expected: PASS.

```bash
git add -A frontend
git commit -m "feat(datasets): show, sort and profile every target on the dataset pages

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16 (F4): Train and sweep forms offer engines for every target

**Files:**
- Create: `frontend/src/features/engines/components/targets-hint.tsx` (export it from `features/engines/index.ts`)
- Modify:
  - `frontend/src/features/protocols/components/train-protocol-form.tsx:92-93, 278-320`
  - `frontend/src/features/protocols/components/train-protocol-form.test.tsx`
- Modify:
  - `frontend/src/features/sweeps/components/sweep-form.tsx:48, 116-123`
  - `frontend/src/features/sweeps/components/sweep-form.test.tsx`

**Interfaces:**
- Consumes: `enginesForTargets`, `jointEnginesRefused`, `trainingKind` (Task 13).
- Produces: `TargetsHint({ dataset, engines })`.

- [ ] **Step 1: Failing test**

In `train-protocol-form.test.tsx`:
- Make the mocked dataset `targets: [{ kind: "numeric", column: "logS" }, { kind: "binary", column: "reactive" }]`.
- Give the mocked engines `supports_multitask` (one joint engine with both tasks, one fan-out engine with both tasks).

Add:

```tsx
it("offers only engines that can train every target, and says why a joint one is missing", async () => {
  renderForm();
  expect(await screen.findByText(/Chemprop D-MPNN train one joint model/)).toBeInTheDocument();
  // the joint engine is not offered for a mixed-kind dataset
  await openEngineSelect();
  expect(screen.queryByRole("option", { name: /Chemprop/ })).not.toBeInTheDocument();
});
```

`renderForm` / `openEngineSelect` are this file's existing helpers. If it opens the select inline, follow that pattern instead.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd frontend && pnpm vitest run src/features/protocols/components/train-protocol-form.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: `TargetsHint`**

Create `frontend/src/features/engines/components/targets-hint.tsx`:

```tsx
import type { DatasetResponse } from "@/shared/lib/api/model";
import { type Engine, jointEnginesRefused } from "../types";

function separator(index: number, count: number): string {
  if (index === 0) return "";
  if (index < count - 1) return ", ";
  return count > 2 ? ", and " : " and ";
}

/**
 * What a training request on this dataset will predict, and -- when its targets
 * mix kinds -- which joint engines are not offered and why, so a missing option
 * reads as a rule rather than a bug.
 */
export function TargetsHint({ dataset, engines }: { dataset: DatasetResponse; engines: Engine[] }) {
  const { targets } = dataset;
  const refused = jointEnginesRefused(engines, targets);
  return (
    <div className="space-y-1 text-xs text-muted-foreground">
      <p>
        Predicting{" "}
        {targets.map((target, index) => (
          <span key={target.column}>
            {separator(index, targets.length)}
            <span className="font-mono">{target.column}</span>
          </span>
        ))}
        {targets.length === 1 && targets[0].unit && (
          <>
            {" "}
            in <span className="font-mono">{targets[0].unit}</span>
          </>
        )}
        , held out by {dataset.split.strategy} split.
      </p>
      {refused.length > 0 && (
        <p>
          {refused.map((engine) => engine.name).join(" and ")} train one joint model and need
          targets of a single kind, so they are not offered for this dataset.
        </p>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Use it in both forms**

**`train-protocol-form.tsx`:**
- `eligible = engines.data && dataset ? enginesForTargets(engines.data, dataset.targets) : []`.
- Replace the "Predicting …" paragraph (lines 278-289) with `{dataset && <TargetsHint dataset={dataset} engines={engines.data ?? []} />}`.
- Under `{engine && <EngineExplainer engineId={engine.id} />}`, add:

```tsx
            {engine && dataset && trainingKind(engine, dataset.targets.length) && (
              <p className="text-xs text-muted-foreground">
                {trainingKind(engine, dataset.targets.length)}
              </p>
            )}
```

**`sweep-form.tsx`:**
- `const available = dataset ? enginesForTargets(engines.data ?? [], dataset.targets) : [];`.
- Replace the "Predicting …" lines 116-123 with `<TargetsHint dataset={dataset} engines={engines.data ?? []} />`, inside the existing `dataset &&` guard.

Update `sweep-form.test.tsx`'s dataset fixture to `targets: [...]`.

- [ ] **Step 5: Run tests and lint, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/protocols src/features/sweeps src/features/engines && pnpm lint
```

Expected: PASS.

```bash
git add -A frontend
git commit -m "feat(training): offer engines that can train every target, and say how

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17 (F5): One scorecard per target on the protocol page

**Files:**
- Modify:
  - `frontend/src/features/protocols/hooks/use-protocols.ts:75-88`
  - `frontend/src/features/protocols/components/protocol-detail.tsx:119-121`
  - `frontend/src/features/runs/components/predict-wizard.tsx:42-49`

**Interfaces:**
- Consumes: `ScorecardResponse[]` (with `target` and `joint_model`).
- Produces: `useScorecard(id)` returns `ScorecardResponse[]`.

- [ ] **Step 1: Implement**

In `use-protocols.ts`, change the request to `customInstance<ScorecardResponse[]>(...)`.

In `protocol-detail.tsx`, import the `Tabs`, `TabsContent`, `TabsList` and `TabsTrigger` components from `@/shared/components/ui/tabs`, and add:

```tsx
/**
 * One scorecard per target. One target renders exactly as before; several share a
 * header that says whether one model learned them all or each has its own, so a
 * row of per-target numbers is never mistaken for joint learning.
 */
function Scorecards({ scorecards }: { scorecards: ScorecardResponse[] }) {
  const [first] = scorecards;
  if (scorecards.length === 1) return <ScorecardView scorecard={first} />;
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        {first.joint_model
          ? `One model learned all ${scorecards.length} targets jointly. Each tab scores it on one target.`
          : `${scorecards.length} separate models, one per target, trained on the same compounds and split.`}
      </p>
      <Tabs defaultValue={first.target}>
        <TabsList>
          {scorecards.map((card) => (
            <TabsTrigger key={card.target} value={card.target} className="font-mono">
              {card.target}
            </TabsTrigger>
          ))}
        </TabsList>
        {scorecards.map((card) => (
          <TabsContent key={card.target} value={card.target}>
            <ScorecardView scorecard={card} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
```

and render `{scorecard.data && <Scorecards scorecards={scorecard.data} />}`.

In `predict-wizard.tsx`, replace `.join(" and ")` with an American list:

```tsx
          {new Intl.ListFormat("en-US", { type: "conjunction" }).format(
            protocol.readouts.map((readout) =>
              readout.unit ? `${readout.name} (${readout.unit})` : readout.name,
            ),
          )}
```

Update the comment above it: a Protocol declares one readout per numeric target and two per binary one, so the list can run to many.

- [ ] **Step 2: Tests, lint and typecheck, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/protocols src/features/runs && pnpm lint
```

Expected:
- PASS.
- Fixtures typed `as ScorecardResponse` in `largest-errors.test.tsx` etc. are unaffected; they render one card.

```bash
git add -A frontend
git commit -m "feat(protocols): show one scorecard per target, and how they were trained

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 18 (F6): The sweep table shows every target, sorted on demand

**Files:**
- Modify:
  - `frontend/src/features/sweeps/lib/rank.ts` (rewrite)
  - `frontend/src/features/sweeps/lib/rank.test.ts`
  - `frontend/src/features/sweeps/components/sweep-detail.tsx`
  - `frontend/src/features/sweeps/index.ts:4`

**Interfaces:**
- Consumes: `run.metrics = { targets: Headline[] }` (Task 7).
- Produces:
  - `Headline`
  - `headlines(metrics)`
  - `sweepTargets(runs)`
  - `headlineFor(run, column)`
  - `isRankable(headline)`
  - `sortRuns(runs, column: string | null)`
  - `formatMetric(headline)`
  - `baselineDelta(headline)`

- [ ] **Step 1: Failing tests**

Rewrite `rank.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { SweepRun } from "../types";
import { baselineDelta, formatMetric, headlineFor, sortRuns, sweepTargets } from "./rank";

const run = (id: string, targets: [string, string, number | null, number | null][]): SweepRun =>
  ({
    id,
    metrics: {
      targets: targets.map(([column, primary_metric, value, baseline_value]) => ({
        column,
        primary_metric,
        value,
        baseline_value,
      })),
    },
  }) as unknown as SweepRun;

const a = run("a", [["reactive", "mcc", 0.2, 0.1], ["solubility", "rmse", 0.5, 0.7]]);
const b = run("b", [["reactive", "mcc", 0.4, 0.1], ["solubility", "rmse", 0.9, 0.7]]);
const pending = { id: "p", metrics: null } as unknown as SweepRun;

describe("sweepTargets", () => {
  it("lists every target once, in the order the runs report them", () => {
    expect(sweepTargets([pending, a, b])).toEqual(["reactive", "solubility"]);
  });
});

describe("sortRuns", () => {
  it("keeps submission order until a target is chosen", () => {
    expect(sortRuns([a, b, pending], null).map((r) => r.id)).toEqual(["a", "b", "p"]);
  });

  it("ranks by the chosen target in that metric's direction, unmeasured last", () => {
    expect(sortRuns([pending, a, b], "reactive").map((r) => r.id)).toEqual(["b", "a", "p"]);
    expect(sortRuns([pending, a, b], "solubility").map((r) => r.id)).toEqual(["a", "b", "p"]);
  });
});

describe("formatMetric and baselineDelta", () => {
  it("read one target's headline, signed so positive is always better", () => {
    expect(formatMetric(headlineFor(a, "solubility"))).toBe("RMSE 0.500");
    expect(baselineDelta(headlineFor(a, "solubility"))).toBeCloseTo(0.2);
    expect(baselineDelta(headlineFor(b, "reactive"))).toBeCloseTo(0.3);
    expect(formatMetric(headlineFor(pending, "reactive"))).toBe("—");
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run:
```
cd frontend && pnpm vitest run src/features/sweeps/lib/rank.test.ts
```

Expected: FAIL.

- [ ] **Step 3: Rewrite `rank.ts`**

```ts
import type { SweepRun } from "../types";

/** Metrics where a smaller number is a better model (see `primary_metric_for`). */
const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

/** One target's headline on a training run (`Run.record_metrics`). */
export interface Headline {
  column: string;
  primary_metric: string;
  value: number | null;
  baseline_value: number | null;
}

/** A run's per-target headlines; empty while it has none. `metrics` is untyped in the contract. */
export function headlines(metrics: SweepRun["metrics"]): Headline[] {
  const targets = (metrics as { targets?: unknown } | null)?.targets;
  return Array.isArray(targets) ? (targets as Headline[]) : [];
}

/**
 * Every target any member reports, in the order reported. A sweep has one
 * dataset, so every finished member lists the same targets in the same order;
 * a pending one lists none.
 */
export function sweepTargets(runs: SweepRun[]): string[] {
  const seen: string[] = [];
  for (const run of runs) {
    for (const headline of headlines(run.metrics)) {
      if (!seen.includes(headline.column)) seen.push(headline.column);
    }
  }
  return seen;
}

export function headlineFor(run: SweepRun, column: string): Headline | undefined {
  return headlines(run.metrics).find((headline) => headline.column === column);
}

/** Rankable exactly when the value is a real number (see `sortRuns`). */
export function isRankable(headline: Headline | undefined): boolean {
  return typeof headline?.value === "number";
}

/**
 * Runs ordered by one target's headline, best first; anything unrankable last,
 * in submission order. `null` keeps submission order: with several targets no
 * single number says which run won -- an average would invent one, the worst
 * target would bury a model that is excellent at the others -- so the table
 * picks no winner until someone picks a column.
 *
 * Unrankable is not the same as bad: a run still training has no number yet, and
 * a run whose metric is genuinely undefined has none either.
 */
export function sortRuns(runs: SweepRun[], column: string | null): SweepRun[] {
  if (column === null) return runs;
  const scored = runs.filter((run) => isRankable(headlineFor(run, column)));
  const unscored = runs.filter((run) => !isRankable(headlineFor(run, column)));
  scored.sort((left, right) => {
    const a = headlineFor(left, column) as Headline;
    const b = headlineFor(right, column) as Headline;
    const difference = (a.value as number) - (b.value as number);
    return LOWER_IS_BETTER.has(a.primary_metric) ? difference : -difference;
  });
  return [...scored, ...unscored];
}

/** The headline number, or why there isn't one. */
export function formatMetric(headline: Headline | undefined): string {
  if (!headline || !isRankable(headline)) return "—";
  return `${headline.primary_metric.toUpperCase()} ${(headline.value as number).toFixed(3)}`;
}

/**
 * How far this run beat its own baseline on one target, signed so that positive
 * always means better. Null when either side is missing -- an unmeasured
 * comparison must not render as a dead heat.
 */
export function baselineDelta(headline: Headline | undefined): number | null {
  if (typeof headline?.value !== "number" || typeof headline.baseline_value !== "number") {
    return null;
  }
  return LOWER_IS_BETTER.has(headline.primary_metric)
    ? headline.baseline_value - headline.value
    : headline.value - headline.baseline_value;
}
```

In `sweeps/index.ts`, export `sortRuns, sweepTargets, headlineFor, isRankable, formatMetric, baselineDelta` and drop `rankRuns`.

- [ ] **Step 4: The table**

In `sweep-detail.tsx`:
- Import `useState` from `react`, `ArrowDown` from `lucide-react`, and the new helpers.
- Add `const [sortBy, setSortBy] = useState<string | null>(null);` as the first line of the component, before any early return (rules of hooks).
- After `if (!sweep) return …`:

```tsx
  const targets = sweepTargets(sweep.runs);
  // One target sorts by default, so a single-target sweep reads exactly as it
  // always has; several wait for a click (see `sortRuns`).
  const active = sortBy ?? (targets.length === 1 ? targets[0] : null);
  const ordered = sortRuns(sweep.runs, active);
```

Header: replace the `Score` and `Improvement over baseline` heads with:

```tsx
              {targets.map((column) => (
                <TableHead key={column}>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 font-mono hover:underline"
                    onClick={() => setSortBy(column)}
                    aria-sort={active === column ? "descending" : undefined}
                  >
                    {column}
                    {active === column && <ArrowDown className="size-3.5" />}
                  </button>
                </TableHead>
              ))}
```

Body:
- Iterate `ordered`.
- The rank cell is `{active && isRankable(headlineFor(run, active)) ? index + 1 : "—"}`.
- Replace the two metric cells with:

```tsx
                  {targets.map((column) => {
                    const headline = headlineFor(run, column);
                    const delta = baselineDelta(headline);
                    return (
                      <TableCell key={column}>
                        <div>{formatMetric(headline)}</div>
                        {delta !== null && (
                          <div className="text-xs text-muted-foreground">
                            {`${delta >= 0 ? "+" : ""}${delta.toFixed(3)} vs baseline`}
                          </div>
                        )}
                      </TableCell>
                    );
                  })}
```

Delete the old `const delta = baselineDelta(run.metrics);` line.

- [ ] **Step 5: Run tests, lint and typecheck, then commit**

Run:
```
cd frontend && pnpm vitest run src/features/sweeps && pnpm lint
```

Expected: PASS.

```bash
git add -A frontend
git commit -m "feat(sweeps): one sortable column per target, and no default winner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 19 (F7): One uncertainty column per target in the triage grid

**Files:**
- Modify:
  - `frontend/src/features/runs/components/triage-grid.tsx:136-147`
  - `frontend/tests/e2e/api-mock.ts:165`

**Interfaces:**
- Consumes: `PredictionResponse.uncertainty` (a record), `targetsOf`, `uncertaintyColumn` (Task 13).

- [ ] **Step 1: Implement**

In `triage-grid.tsx`, import `targetsOf` and `uncertaintyColumn` from `@/shared/lib/targets`. Replace the single `Uncertainty` column in `base.push(...)` with a loop before the `Applicability` push:

```tsx
    // One per target: each target's own model reports its own spread. The colId is
    // the results column the API sorts and filters by -- plain `uncertainty` for a
    // one-target Protocol, as every results file before several targets used.
    const targets = targetsOf(readouts);
    for (const target of targets) {
      base.push({
        headerName: targets.length === 1 ? "Uncertainty" : `Uncertainty (${target})`,
        colId: uncertaintyColumn(target, targets.length),
        width: 150,
        ...NUMBER_FILTER,
        valueGetter: (params) => params.data?.uncertainty?.[target] ?? null,
        // Null for XGBoost, which has no ensemble spread to report. Rendered as
        // absence rather than as a fabricated zero.
        cellRenderer: (params: { value: number | null }) => (
          <ReadoutValue value={params.value} precision={3} />
        ),
      });
    }
```

In `tests/e2e/api-mock.ts`, change `uncertainty: null` to `uncertainty: {}`. Also run `grep -n "target" tests/e2e/api-mock.ts`. If the mock dataset there has `target: {...}`, make it `targets: [{...}]`.

- [ ] **Step 2: Full frontend gates, then commit**

Run:
```
cd frontend && pnpm lint && pnpm exec tsc --noEmit && pnpm test
```

Expected:
- PASS.
- `tsc` is clean. Any remaining error is a consumer the tasks above missed: fix it in place, the same way.

```bash
git add -A frontend
git commit -m "feat(triage): one uncertainty column per target

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 20 (Phase 2): MoLFormer-XL learns its targets jointly

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/molformer_xl.py`
- Test: `backend/tests/unit/engines/test_molformer_xl.py`

**Interfaces:**
- Consumes: the same contract as Task 12.
- Produces:
  - `supports_multitask=True`
  - The artifact bundle gains `n_tasks: int`, and `target_mean` / `target_std` become lists.
  - Old artifacts (scalar mean and std, no `n_tasks`) still predict.

Labels are dense (Decision 1), so the loss needs no mask. `BCEWithLogitsLoss` / `MSELoss` over a `(batch, n_tasks)` tensor averages every label, which is the right objective.

- [ ] **Step 1: Failing test**

Append to `test_molformer_xl.py`, mirroring Task 12's chemprop test with the same skip guard as the file's existing tests:
- a two-target regression frame;
- metrics keyed `["y", "z"]`;
- predict returns `2 × rows` with a `target` column;
- `MolformerXL.manifest().supports_multitask is True`.

- [ ] **Step 2: Run to verify failure**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines/test_molformer_xl.py -q
```

Expected: FAIL. The run must have the weights available (see `_require_transformers`); a SKIP does not count.

- [ ] **Step 3: Implement**

- `_MANIFEST` gains `supports_multitask=True`.
- `_load_backbone(*, freeze_encoder: bool, num_labels: int)` passes `num_labels=num_labels`. Update the docstring: one logit per target.
- `_collate`: each example's target is a tuple of floats, so `targets = torch.tensor([list(item[1]) for item in batch], dtype=torch.float32)` has shape `(batch, n_tasks)`.
- `_build_module.forward` returns `output.logits` unchanged, shape `(batch, n_tasks)`. Delete the `reshape(-1)` and its comment. Training and validation losses compare `(batch, n_tasks)` to `(batch, n_tasks)`.
- `_logits` returns `torch.cat(batches).cpu().numpy()`, shape `(molecules, n_tasks)`. Update the docstring.
- `_unlabelled(structures, n_tasks)` returns `[(smiles, (0.0,) * n_tasks) for smiles in structures]`.
- `_standardize` takes the train rows as `list[Sequence[float]]` and returns arrays:

```python
def _standardize(targets: list[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Per-target mean and standard deviation of the training values, for regression.
    (keep the existing reasoning; a zero deviation becomes 1.0 per column)"""
    import numpy as np

    array = np.asarray(targets, dtype=float)
    deviation = array.std(axis=0)
    return array.mean(axis=0).tolist(), np.where(deviation > 0.0, deviation, 1.0).tolist()
```

- `_to_values` broadcasts: `np.asarray(logits) * np.asarray(target_std) + np.asarray(target_mean)` for regression. That works for a scalar from an old bundle, or a per-column list.

In `train`:
- `columns = ctx.target_columns`;
- `is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION`, with the comment from Task 12;
- the mean and std come from `_standardize(train_rows.select(columns).rows())`, or `([0.0] * n, [1.0] * n)` for classification;
- `examples(rows)` pairs each SMILES with `tuple((float(v) - m) / s for v, m, s in zip(row, means, stds))`;
- `score` loops over columns exactly as in Task 12;
- the bundle stores `n_tasks=len(columns)`.

In `predict`:
- `n_tasks = int(bundle.get("n_tasks", 1))`, passed to `_load_backbone` and `_unlabelled`;
- build the long frame per target column exactly as in Task 12.

- [ ] **Step 4: Run tests and gates, then commit**

Run:
```
cd backend && OMP_NUM_THREADS=1 uv run pytest tests/unit/engines -q && uv run mypy src && uv run lint-imports
```

Expected: PASS, not SKIPPED, for the MoLFormer tests.

```bash
git add -A backend
git commit -m "feat(engines): MoLFormer-XL learns every target of a dataset jointly

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 21: Verify on the real four-label dataset

The suite is synthetic. Twice on this project a green suite has shipped a bug that one real run caught. This task is not optional.

**Files:**
- Create, **do not commit**: `backend/verify_multitask.py`.

- [ ] **Step 1: Full gates**

Run:
```
cd backend && uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run lint-imports && OMP_NUM_THREADS=1 uv run pytest -q
cd ../frontend && pnpm lint && pnpm exec tsc --noEmit && pnpm test
```

Expected: all green.

- [ ] **Step 2: Migrate the dev database and check it**

Run:
```
make up && make migrate && docker compose exec -T postgres psql -U studio -c "SELECT version_num FROM alembic_version; SELECT targets FROM datasets LIMIT 3; SELECT metrics FROM runs WHERE kind = 'training' AND metrics IS NOT NULL LIMIT 3;"
```

Expected:
- the version is `013`;
- every dataset has a one-element `targets` list;
- training metrics look like `{"targets": [{"column": …}]}`.

Note: once this has run, the dev DB cannot downgrade past 013 while any multi-target dataset exists. That is by design.

- [ ] **Step 3: Real-data script through the real code path, rolled back**

Check the header row: `head -1 ~/Documents/Cage-Fusion-Paper/Dataset/train.csv`. Set `STRUCTURE` below to its SMILES column.

```python
"""Throwaway: the 324,803-row, four-label nuisance set through the real training and
scorecard path, against the dev Postgres, inside a transaction that is rolled back."""

import asyncio
import io
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import polars as pl
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from daikonstudio.application.catalog.get_scorecard import GetScorecard, GetScorecardQuery
from daikonstudio.application.execution.train_protocol import artifact_key
from daikonstudio.domain.data.split import SplitStrategy
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.settings import Settings
from tests.integration.test_train_protocol import Studio

STRUCTURE = "smiles"  # set from the header row
LABELS = ("aggregator", "luciferase_inhibitor", "reactive", "promiscuous")


async def main(rows: int, engine_id: str) -> None:
    frame = pl.read_csv(
        Path.home() / "Documents/Cage-Fusion-Paper/Dataset/train.csv", infer_schema=False
    ).rename({STRUCTURE: "smiles"})
    if rows:
        frame = frame.sample(n=rows, seed=7)
    engine = create_async_engine(Settings().database_url)
    async with engine.connect() as connection:
        await connection.begin()
        with tempfile.TemporaryDirectory() as blobs:
            studio = Studio(
                async_sessionmaker(
                    bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
                ),
                Path(blobs),
            )
            started = time.monotonic()
            dataset = await studio.dataset(
                strategy=SplitStrategy.SCAFFOLD,
                targets=tuple(TargetSpec(column=c, kind=TargetKind.BINARY) for c in LABELS),
                csv=frame.select("smiles", *LABELS).write_csv().encode(),
            )
            print(f"dataset: {dataset.row_count} rows in {time.monotonic() - started:.0f}s")

            started = time.monotonic()
            run = await studio.wait(
                await studio.train(dataset_id=dataset.id, engine_id=engine_id, conditions={})
            )
            print(f"train {engine_id}: {run.status.value} in {time.monotonic() - started:.0f}s")
            print("  error:", run.error_message)
            print("  headlines:", run.metrics)
            print("  deadline_scale:", run.params.get("deadline_scale"))

            protocol = await studio.protocol_for(run)
            artifact = studio.store.get_bytes(artifact_key(studio.auth.workspace_id, protocol.id))
            print(f"  artifact: {len(artifact) / 1e6:.1f} MB, zip={zipfile.is_zipfile(io.BytesIO(artifact))}")

            started = time.monotonic()
            cards = (
                await GetScorecard(studio.protocols, studio.store, studio.normalizer, studio.datasets)(
                    GetScorecardQuery(protocol_id=protocol.id), studio.auth
                )
            ).unwrap()
            print(f"scorecards: {len(cards)} in {time.monotonic() - started:.0f}s")
            for card in cards:
                print(f"  {card.target}: {card.metrics} vs baseline {card.baseline_metrics}")
        await connection.rollback()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), sys.argv[2]))
```

Run a smoke test first, then the full set:
```
cd backend && OMP_NUM_THREADS=1 uv run python verify_multitask.py 20000 ecfp4-xgboost
cd backend && OMP_NUM_THREADS=1 uv run python verify_multitask.py 0 ecfp4-xgboost
```

If chemprop is installed:
```
cd backend && OMP_NUM_THREADS=1 uv run python verify_multitask.py 20000 chemprop-dmpnn
```

Expected:
- `ready`, with 4 headlines in `aggregator, luciferase_inhibitor, reactive, promiscuous` order.
- `deadline_scale` is 4 for XGBoost and 1 for chemprop.
- The artifact is a zip for XGBoost and bare for chemprop.
- 4 scorecards.

Record:
- the wall-clock time for the full XGBoost run, against `4 × worker_job_timeout` (1800 s);
- the scorecard read time;
- the artifact size.

**Expect no lift from joint learning on this data and treat that as a pass.** The four labels come from four near-disjoint source lists (see the spec's "Verification on real data").

Delete `verify_multitask.py` when done.

- [ ] **Step 4: Live UI pass (needs the human for OAuth)**

1. Restart everything so the agents load the new code: `make dev`. It restarts the API, frontend and both runner agents; the agents do not hot-reload.
2. Ask the human to sign in. There is no test bypass for auth, by design.
3. In the browser:
   - upload the CSV with all four labels chosen;
   - check that the Target step shows four kind choices;
   - check the dataset page lists four targets, and that the compounds tab sorts by `reactive`;
   - check the profile selector switches targets.
4. Train `ecfp4-xgboost`:
   - the progress phase names each target in turn;
   - the protocol page shows four tabs and the "4 separate models" line.
5. Check the training form for a mixed-kind dataset (any two-column numeric + binary CSV): chemprop is not offered, and the note says why.
6. Run a two-config sweep: four target columns, no rank until a header is clicked.
7. Publish and predict on 50 compounds: the triage grid shows eight readout columns and four uncertainty columns (for `ecfp4-randomforest` they are filled; for XGBoost, "—").

Trust the DOM over the accessibility tree when checking.

- [ ] **Step 5: Report**

Write the measured numbers and anything that could not be checked into the hand-back message. Then use `superpowers:finishing-a-development-branch`.

The memory note `multi-task-labels.md` must be updated with:
- what landed;
- the timings;
- the deploy caveat: the API and runners ship together, because the wire shapes changed.

---

## Self-review notes

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| Domain targets and invariants | T1, T2 |
| Persistence | T1, T7 |
| Engine contract | T5 |
| Fan-out adapter | T6 |
| Native multi-task | T12, T20 |
| Readouts and prediction | T4, T11 |
| Scorecard | T8 |
| API and frontend | T2, T3, T13–T19 |
| Job timeout | T10 |
| Testing list | T2, T6, T8, T9, T11, T12 |
| Real-data verification | T21 |

The spec's chemprop NaN-mask test is dropped by Decision 1.

**Known ceilings, recorded in code as `ponytail:` comments:**
- `deadline_scale` ignores the baseline's fan-out.
- The profile recomputes the structure sections per target.
- There is no cap on the target count. A 600-target ToxCast upload would fan out 600 fits per leg, and the deadline scales to match. Add a cap at upload if that ever happens.

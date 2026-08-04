# Choosable Baseline and Pretrained Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a training request name the engine *and* conditions it is measured against, and let chemprop start from CheMeleon's pretrained weights — so "pretrained vs not" becomes an ordinary baseline choice.

**Architecture:** The baseline stops being a registry lookup (`EngineRegistry.baseline()`) and becomes a resolved *(engine, conditions)* pair carried in `run.params`. Because `baseline_is_self` already compares engine id *and* conditions, "same engine, different pretraining" is distinguished with no new concept. Pretrained weights ship as one `enum` condition on `chemprop-dmpnn`, loaded through chemprop 2.3.0's documented `BondMessagePassing(**hyper_parameters)` path.

**Tech Stack:** Python 3.14, FastAPI, pydantic v2, SQLAlchemy (async), polars, arq, chemprop 2.3.0 + torch + lightning (gpu extra only), pytest. Frontend: Next.js, React, TypeScript, TanStack Query, shadcn/radix, orval codegen.

**Spec:** `docs/superpowers/specs/2026-08-04-choosable-baseline-and-pretrained-weights-design.md`

## Global Constraints

- **A baseline comparison is mandatory and never skippable.** Choosable ≠ optional. No code path may produce a Scorecard with no comparison.
- **Every chemprop / torch / lightning import stays inside a function**, never at module scope. `default_registry()` instantiates `ChempropDMPNN` at import time, so the constructor must stay import-free. Guarded by `tests/unit/engines/test_engine_contract.py::test_the_registry_loads_with_no_gpu_extra_installed`.
- **`EngineManifest` and `ConditionSpec` must stay plain JSON-serializable data.** Guarded by `test_every_registered_manifest_round_trips_through_json`.
- **`ScorecardInputs` new fields must carry a default.** `from_json` is `cls(**json.loads(data))`; a field without a default breaks reads of every blob written before today.
- **`run.params` is write-once.** `SqlAlchemyRunRepository.update()` never persists it. Store resolved values at creation, never expect to rewrite them.
- **`OMP_NUM_THREADS=1` is load-bearing**, not tuning. Do not remove it from the Makefile or either Dockerfile (handoff §3).
- **`torch.load` must pass `weights_only=True`.** The CheMeleon file is a bare two-key tensor dict; the default unpickles arbitrary objects.
- CheMeleon exact values, copied verbatim — URL `https://zenodo.org/records/15460715/files/chemeleon_mp.pt`, size `34859448` bytes, MD5 `6a80b54fdb7de37ef0374d302f01e8ce`, licence MIT, pins `depth=6` / `d_h=2048` / V2 atom featurizer / mean aggregation.
- Run `make test`, `ruff check`, `ruff format --check` and `mypy` before each commit. The suite is 343 passing at the start of this plan; import-linter keeps 3 contracts.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backend/src/daikonstudio/application/engines/manifest.py` | Manifest data + condition validation | Add `lane_for()` |
| `backend/src/daikonstudio/application/execution/train_protocol.py` | Command, enqueue use case, worker handler, `ScorecardInputs` | Baseline pair throughout |
| `backend/src/daikonstudio/interface/routes/protocols.py` | HTTP DTOs | Two request fields, one response field |
| `backend/src/daikonstudio/domain/execution/scorecard.py` | Rendered Scorecard | One field |
| `backend/src/daikonstudio/application/execution/build_scorecard.py` | Inputs → Scorecard | Pass one field through |
| `backend/src/daikonstudio/settings.py` | Config | `pretrained_weights_dir` |
| `backend/src/daikonstudio/infrastructure/engines/_pretrained.py` | **New.** Fetch, verify and cache foundation-model weights | Create |
| `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py` | The engine | `pretrained` condition + load path |
| `backend/Dockerfile.gpu` | GPU worker image | Bake weights in |
| `frontend/src/features/engines/types/index.ts` | Engine vocabulary translation | CheMeleon's pinned values |
| `frontend/src/features/protocols/components/train-protocol-form.tsx` | Training form | Baseline picker, pinning |
| `frontend/src/features/protocols/components/condition-fields.tsx` | Manifest-driven fields | `pinned` prop |
| `frontend/src/features/protocols/components/scorecard-view.tsx` | Verdict + metrics | Dynamic baseline copy |

`_pretrained.py` is its own file rather than more lines in `chemprop_dmpnn.py`: downloading, checksumming and atomically caching a file has nothing to do with message passing, and a second engine wanting a foundation model should not import a D-MPNN module to get it.

---

### Task 1: `lane_for` — routing a run that needs two engines

**Files:**
- Modify: `backend/src/daikonstudio/application/engines/manifest.py` (append after `validate_conditions`)
- Test: `backend/tests/unit/execution/test_lanes.py`

**Interfaces:**
- Produces: `lane_for(*manifests: EngineManifest) -> str`, used by Task 3 and by the sibling retry plan.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/execution/test_lanes.py`:

```python
from daikonstudio.application.engines.manifest import EngineManifest, TaskType, lane_for


def _manifest(engine_id: str, lane: str) -> EngineManifest:
    return EngineManifest(
        id=engine_id,
        version="1.0.0",
        name=engine_id,
        description="",
        tasks=(TaskType.REGRESSION,),
        lane=lane,
    )


def test_lane_for_returns_default_when_every_engine_is_default_lane():
    assert lane_for(_manifest("a", "default"), _manifest("b", "default")) == "default"


def test_lane_for_returns_the_non_default_lane_whichever_side_declares_it():
    """A run fits the chosen engine *and* the baseline in one process, so it must
    land on a worker that can serve both. Choosing ecfp4-randomforest (default)
    with a chemprop-dmpnn baseline (gpu) previously routed to a default-lane
    worker, which fit the chosen engine and then died in _require_chemprop()."""
    assert lane_for(_manifest("chosen", "default"), _manifest("base", "gpu")) == "gpu"
    assert lane_for(_manifest("chosen", "gpu"), _manifest("base", "default")) == "gpu"


def test_lane_for_is_stable_when_both_declare_the_same_non_default_lane():
    assert lane_for(_manifest("a", "gpu"), _manifest("b", "gpu")) == "gpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/execution/test_lanes.py -v`
Expected: FAIL with `ImportError: cannot import name 'lane_for'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend/src/daikonstudio/application/engines/manifest.py`:

```python
def lane_for(*manifests: EngineManifest) -> str:
    """The lane a Run needs when more than one engine must fit inside it.

    A training Run fits the chosen engine and the baseline in one process, so it
    has to land on a worker that can serve both. Any non-default lane wins over
    the default one, because the default lane is the "no special hardware"
    lane -- a gpu-lane worker can run an ECFP4 fit, but not the reverse.

    ponytail: two lanes, so "the non-default one" is unambiguous. A Run wanting
    two *different* non-default lanes has no home; if a third lane ever exists,
    reject that pair at enqueue rather than silently picking the first.
    """
    return next((m.lane for m in manifests if m.lane != DEFAULT_LANE), DEFAULT_LANE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/execution/test_lanes.py -v`
Expected: PASS, all existing tests in the file still green

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/application/engines/manifest.py backend/tests/unit/execution/test_lanes.py
git commit -m "feat: lane_for picks a queue that can serve every engine in a run"
```

---

### Task 2: The command carries a baseline pair

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py:209-231` (`TrainProtocolCommand`)
- Test: `backend/tests/unit/execution/test_train_protocol_command.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `TrainProtocolCommand` with `baseline_engine_id: str | None = None` and `baseline_conditions: dict[str, object]` (default empty). `to_params()` emits keys `baseline_engine_id`, `baseline_conditions`; `from_params()` reads both with `.get()` defaults.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/execution/test_train_protocol_command.py`:

```python
"""The params round trip, including runs written before the baseline was choosable.

`run.params` is write-once and never migrated, so `from_params` must keep
reading rows created by an older deploy. That is the regression these defaults
exist to prevent, not a hypothetical.
"""

from __future__ import annotations

import uuid

from daikonstudio.application.execution.train_protocol import TrainProtocolCommand


def test_params_round_trip_carries_the_baseline_pair():
    command = TrainProtocolCommand(
        name="BBBP — D-MPNN",
        dataset_id=uuid.uuid4(),
        engine_id="chemprop-dmpnn",
        conditions={"pretrained": "CheMeleon"},
        baseline_engine_id="chemprop-dmpnn",
        baseline_conditions={"pretrained": "none"},
    )
    restored = TrainProtocolCommand.from_params(command.to_params())
    assert restored == command


def test_from_params_reads_a_run_written_before_the_baseline_was_choosable():
    legacy = {
        "name": "ESOL — RF",
        "dataset_id": str(uuid.uuid4()),
        "engine_id": "ecfp4-randomforest",
        "conditions": {"n_estimators": 500},
    }
    restored = TrainProtocolCommand.from_params(legacy)
    assert restored.baseline_engine_id is None
    assert restored.baseline_conditions == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/execution/test_train_protocol_command.py -v`
Expected: FAIL with `TypeError: TrainProtocolCommand.__init__() got an unexpected keyword argument 'baseline_engine_id'`

- [ ] **Step 3: Write minimal implementation**

Replace `train_protocol.py:209-231` entirely with:

```python
@dataclass(frozen=True, kw_only=True)
class TrainProtocolCommand:
    name: str
    dataset_id: uuid.UUID
    engine_id: str
    conditions: dict[str, object]
    # What this Run is measured against. `None` means "whatever the registry
    # flags as the default baseline", and survives only until `TrainProtocol`
    # resolves it -- what lands in `run.params` is always a concrete id, so a
    # queued Run cannot be silently retargeted by a registry change before a
    # worker picks it up. It stays `None` when read back off a row written
    # before the baseline was choosable.
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, object] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_id": str(self.dataset_id),
            "engine_id": self.engine_id,
            "conditions": self.conditions,
            "baseline_engine_id": self.baseline_engine_id,
            "baseline_conditions": self.baseline_conditions,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> TrainProtocolCommand:
        return cls(
            name=params["name"],
            dataset_id=uuid.UUID(params["dataset_id"]),
            engine_id=params["engine_id"],
            conditions=params["conditions"],
            baseline_engine_id=params.get("baseline_engine_id"),
            baseline_conditions=params.get("baseline_conditions") or {},
        )
```

Add `field` to the existing `dataclasses` import at the top of the module (it currently imports `dataclass` and `asdict`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/execution/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/application/execution/train_protocol.py backend/tests/unit/execution/test_train_protocol_command.py
git commit -m "feat: TrainProtocolCommand carries the baseline engine and conditions"
```

---

### Task 3: Resolve the baseline at enqueue

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py:264-305` (`TrainProtocol.__call__`)
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py:62-76` (`TrainProtocolBody`), and the POST handler at `:232`
- Modify: `backend/tests/unit/execution/test_lanes.py:34-46` (`_StubEngine`) and `:167` (the `training_setup` registry) — **the existing fixture breaks without this**
- Test: `backend/tests/unit/execution/test_lanes.py` (extend)

**Interfaces:**
- Consumes: `lane_for` (Task 1), `TrainProtocolCommand` (Task 2).
- Produces: `run.params["baseline_engine_id"]` always a concrete engine id; the Run enqueued on `lane_for(manifest, baseline_manifest)`; cache key includes both baseline parts.

- [ ] **Step 1: Fix the existing fixture first — this change breaks it**

`tests/unit/execution/test_lanes.py:167` builds `EngineRegistry({"heavy": _StubEngine("heavy", "gpu")})`. That registry has **no baseline engine**, so the moment `TrainProtocol` starts resolving one, `EngineRegistry.baseline()` raises `UnknownEngineError("no baseline engine registered")` and the existing, currently-passing `test_training_is_enqueued_to_its_engines_lane` fails. Fix the stub before writing anything new.

Give `_StubEngine` an `is_baseline` flag (`test_lanes.py:34-46`):

```python
class _StubEngine:
    def __init__(self, engine_id: str, lane: str, *, is_baseline: bool = False) -> None:
        self._manifest = EngineManifest(
            id=engine_id,
            version="1.0.0",
            name=engine_id,
            description="",
            tasks=(TaskType.REGRESSION,),
            lane=lane,
            is_baseline=is_baseline,
        )
```

and register a default baseline in `training_setup` (`:167`):

```python
    registry = EngineRegistry(
        {
            "heavy": _StubEngine("heavy", "gpu"),
            "plain": _StubEngine("plain", DEFAULT_LANE, is_baseline=True),
        }
    )
```

- [ ] **Step 2: Write the failing tests**

Append to `backend/tests/unit/execution/test_lanes.py`, beside the existing lane tests. `training_setup` yields the 4-tuple `(enqueuer, command, auth, use_case)`, and `_RecordingEnqueuer` records into `.lanes`.

```python
async def test_an_absent_baseline_resolves_to_the_registry_default(training_setup: Any) -> None:
    enqueuer, command, auth, use_case = training_setup

    run = (await use_case(command, auth)).unwrap()

    assert run.params["baseline_engine_id"] == "plain"


async def test_a_named_baseline_is_stored_resolved_never_as_none(training_setup: Any) -> None:
    """The concrete id, never None: params are write-once, so a Run storing None
    would be measured against whatever the registry flags at the moment a worker
    dequeues it -- which may not be what the user was shown."""
    _enqueuer, command, auth, use_case = training_setup

    run = (
        await use_case(
            replace(command, baseline_engine_id="plain", baseline_conditions={"k": 1}), auth
        )
    ).unwrap()

    assert run.params["baseline_engine_id"] == "plain"
    assert run.params["baseline_conditions"] == {"k": 1}


async def test_an_unknown_baseline_is_a_404_before_anything_is_enqueued(
    training_setup: Any,
) -> None:
    enqueuer, command, auth, use_case = training_setup

    result = await use_case(replace(command, baseline_engine_id="not-an-engine"), auth)

    assert isinstance(result.failure(), NotFoundError)
    assert enqueuer.lanes == []


async def test_a_gpu_baseline_pulls_a_default_lane_run_onto_the_gpu_queue(
    training_setup: Any,
) -> None:
    """The failure this prevents: a default-lane worker fits the chosen engine
    fine and then dies in the baseline's _require_chemprop()."""
    enqueuer, command, auth, use_case = training_setup

    result = await use_case(
        replace(command, engine_id="plain", baseline_engine_id="heavy"), auth
    )

    result.unwrap()
    assert enqueuer.lanes == ["gpu"]


async def test_the_baseline_pair_changes_the_cache_key(training_setup: Any) -> None:
    _enqueuer, command, auth, use_case = training_setup

    first = (await use_case(replace(command, baseline_engine_id="plain"), auth)).unwrap()
    second = (
        await use_case(
            replace(command, baseline_engine_id="plain", baseline_conditions={"k": 1}), auth
        )
    ).unwrap()

    assert first.cache_key != second.cache_key
```

Add `from dataclasses import replace` and `from daikonstudio.domain.shared.errors import NotFoundError` to the file's imports (`NotFoundError` is already imported at `:29`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/unit/execution/test_lanes.py -v`
Expected: the five new tests FAIL — `run.params` has no `baseline_engine_id`, and the unknown-baseline case returns Success. The pre-existing lane tests must still PASS after the Step 1 fixture fix.

- [ ] **Step 4: Write minimal implementation**

In `train_protocol.py`, add `from dataclasses import replace` to the imports, and `from daikonstudio.application.engines.manifest import lane_for` alongside the existing manifest imports.

Replace the body of `TrainProtocol.__call__` from `try: engine = ...` (`:271`) through `return Success(run)` (`:305`) with:

```python
        try:
            engine = self._engines.get(command.engine_id)
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", command.engine_id))

        # Same argument as `engine_id` above: a registry membership check with
        # exactly one possible answer, so an unknown baseline is a synchronous
        # 404 rather than a 202 for a Run that cannot succeed. The *conditions*
        # stay unvalidated here, for the reason in this class's docstring.
        try:
            baseline = (
                self._engines.get(command.baseline_engine_id)
                if command.baseline_engine_id
                else self._engines.baseline()
            )
        except UnknownEngineError:
            return Failure(NotFoundError("Engine", command.baseline_engine_id or "baseline"))

        dataset = await self._datasets.get(auth.workspace_id, command.dataset_id)
        if dataset is None:
            return Failure(NotFoundError("Dataset", str(command.dataset_id)))
        # Belt and braces: the read above is already workspace-scoped in SQL, so
        # this cannot fire today. It stays because it is the guard that has to
        # hold if a future caller ever hands us a Dataset it fetched elsewhere.
        require_same_workspace(auth, dataset.workspace_id, entity_type="Dataset")

        # Pin the resolved id into what gets persisted. `params` is write-once,
        # so a Run storing `None` would be measured against whatever the registry
        # flags at the moment a worker dequeues it -- which may not be what the
        # user was shown when they submitted.
        command = replace(command, baseline_engine_id=baseline.manifest().id)

        run = Run(
            kind=RunKind.TRAINING,
            workspace_id=auth.workspace_id,
            requested_by=auth.user_id,
            # The Dataset's content hash pins the data *and* the split, so this key
            # identifies "this data, split this way, through this engine, with these
            # conditions, measured against this baseline". Nothing reuses a training
            # Run today -- only predictions are cached -- but fit-result caching is a
            # live deferred item, and a key omitting the baseline would let it serve
            # a run whose comparison was against a different model.
            cache_key=compute_cache_key(
                kind="training",
                content_hash=dataset.content_hash,
                engine_id=command.engine_id,
                conditions=sorted(command.conditions.items()),
                baseline_engine_id=command.baseline_engine_id,
                baseline_conditions=sorted(command.baseline_conditions.items()),
            ),
            params=command.to_params(),
        )
        await self._runs.add(run)
        # Both engines fit inside this one Run, so the queue has to serve both.
        await self._enqueuer.enqueue(
            run.id, lane=lane_for(engine.manifest(), baseline.manifest())
        )
        return Success(run)
```

Then in `interface/routes/protocols.py`, add to `TrainProtocolBody` after `conditions` (`:76`):

```python
    # Optional: absent means the registry's flagged default baseline. A
    # comparison always happens -- this chooses which one, it does not skip it.
    baseline_engine_id: str | None = None
    baseline_conditions: dict[str, Any] = Field(default_factory=dict)
```

And pass both through wherever the handler at `:232` constructs `TrainProtocolCommand`, adding `baseline_engine_id=body.baseline_engine_id, baseline_conditions=body.baseline_conditions`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/unit/ -v && uv run ruff check && uv run mypy src`
Expected: PASS, including the pre-existing lane tests

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/application/execution/train_protocol.py backend/src/daikonstudio/interface/routes/protocols.py backend/tests/unit/execution/test_lanes.py
git commit -m "feat: choose the baseline engine and conditions on a training request"
```

---

### Task 4: The worker fits the chosen baseline

**Files:**
- Modify: `backend/src/daikonstudio/application/execution/train_protocol.py:359-361` and `:420-448` and `ScorecardInputs` at `:170-192`
- Test: `backend/tests/integration/test_train_protocol.py` (extend)

**Interfaces:**
- Consumes: `TrainProtocolCommand.baseline_engine_id` / `.baseline_conditions` (Task 2).
- Produces: `ScorecardInputs.baseline_conditions: dict[str, Any]` (defaulted).

- [ ] **Step 1: Widen the `Studio` test helper**

`tests/integration/test_train_protocol.py:171-180` defines `Studio.train(*, dataset_id, engine_id, conditions)`, which builds the `TrainProtocolCommand`. Add the two new keyword arguments so the tests below can reach them, defaulting both so all ~25 existing call sites keep working unchanged:

```python
    async def train(
        self,
        *,
        dataset_id: uuid.UUID,
        engine_id: str,
        conditions: dict[str, object],
        baseline_engine_id: str | None = None,
        baseline_conditions: dict[str, object] | None = None,
    ) -> Run:
        command = TrainProtocolCommand(
            name="a trained model",
            dataset_id=dataset_id,
            engine_id=engine_id,
            conditions=conditions,
            baseline_engine_id=baseline_engine_id,
            baseline_conditions=baseline_conditions or {},
        )
        return (await self._train(command, self.auth)).unwrap()
```

Note `Studio` wires the **real** `default_registry()` (`:150`), so these tests exercise the actual shipped engines. `studio.scorecard_for(run)` returns the written `ScorecardInputs`.

- [ ] **Step 2: Write the failing tests**

Add to `backend/tests/integration/test_train_protocol.py`. The file already has `test_choosing_the_baseline_engine_says_so_instead_of_faking_a_comparison` (`:377`) and `test_non_default_conditions_still_earn_a_real_baseline` (`:396`) — read both first; these extend that story rather than duplicating it.

```python
async def test_a_chosen_baseline_is_the_one_that_gets_fit(studio: Studio) -> None:
    """The Scorecard names the baseline that actually ran, not the flagged default."""
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id,
        engine_id="ecfp4-randomforest",
        conditions={"n_estimators": 20},
        baseline_engine_id="ecfp4-xgboost",
        baseline_conditions={"n_estimators": 20},
    )
    await studio.wait(run)

    inputs = await studio.scorecard_for(run)
    assert inputs.baseline_engine_id == "ecfp4-xgboost"
    # validate_conditions fills the rest of the manifest's defaults.
    assert inputs.baseline_conditions["n_estimators"] == 20
    assert inputs.baseline_is_self is False


async def test_same_engine_different_conditions_is_not_a_self_comparison(
    studio: Studio,
) -> None:
    """The case that makes 'pretrained vs not' work: one engine id, two settings.
    If this collapsed into baseline_is_self the second fit would never run and
    the Scorecard would present one result twice as though it were a comparison."""
    dataset = await studio.dataset()
    run = await studio.train(
        dataset_id=dataset.id,
        engine_id="ecfp4-randomforest",
        conditions={"n_estimators": 20},
        baseline_engine_id="ecfp4-randomforest",
        baseline_conditions={"n_estimators": 200},
    )
    await studio.wait(run)

    inputs = await studio.scorecard_for(run)
    assert inputs.baseline_is_self is False
    assert inputs.baseline_conditions["n_estimators"] == 200


async def test_a_baseline_that_cannot_serve_the_task_fails_the_run_readably(
    studio: Studio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three shipped engines declare both tasks, so this guard is unreachable
    with the real registry -- and that is exactly why it needs a test. Register a
    regression-only engine and make it the baseline for a binary dataset.

    The failure must name the *baseline* engine. Reusing the chosen engine's
    message would send a scientist to inspect the model they picked, which is
    fine, over a baseline they never chose.
    """
    regression_only = _StubSingleTaskEngine("regression-only")
    monkeypatch.setitem(studio.registry_engines, "regression-only", regression_only)

    dataset = await studio.dataset(kind=TargetKind.BINARY)
    run = await studio.train(
        dataset_id=dataset.id,
        engine_id="ecfp4-randomforest",
        conditions={"n_estimators": 20},
        baseline_engine_id="regression-only",
    )
    reloaded = await studio.wait(run)

    assert reloaded.status is RunStatus.FAILED
    assert "regression-only" in (reloaded.error_message or "")
    assert "binary_classification" in (reloaded.error_message or "")


def test_scorecard_inputs_reads_a_blob_written_before_baseline_conditions_existed() -> None:
    """`from_json` is `cls(**json.loads(data))`. Without a default, every
    Scorecard blob written before this change becomes unreadable."""
    dataset = _scorecard_inputs_fixture()  # build one valid instance at module scope
    legacy = json.loads(dataset.to_json())
    del legacy["baseline_conditions"]

    restored = ScorecardInputs.from_json(json.dumps(legacy).encode())

    assert restored.baseline_conditions == {}
```

Two supporting pieces that test needs, both in the same file:

- `_StubSingleTaskEngine` — a minimal engine declaring `tasks=(TaskType.REGRESSION,)` only. Copy the shape of `_StubEngine` in `tests/unit/execution/test_lanes.py:34-52`; `train`/`predict` can raise `NotImplementedError`, because the guard fires before either is called.
- `Studio` currently constructs `default_registry()` inline at `:150`. Hold it on the instance instead (`self.registry_engines = {...}` fed into `EngineRegistry`, exposed so a test can add to it), so a test can register an engine without monkeypatching a module-level function that the worker also imports.

The run must land `FAILED` rather than raising, because `InlineEnqueuer` runs the job through `run_job`, which catches and records the error on the row — that is the same path a real worker takes.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_train_protocol.py -v`
Expected: FAIL — `train()` rejects the new kwargs, and `ScorecardInputs` has no `baseline_conditions`

- [ ] **Step 4: Write minimal implementation**

Add to `ScorecardInputs` immediately after `baseline_engine_id` (`:182`):

```python
    baseline_engine_id: str
    # Defaulted because `from_json` is `cls(**json.loads(data))`: a required
    # field here makes every Scorecard blob written before the baseline became
    # choosable unreadable. Also load-bearing for display -- when the baseline
    # is the *same* engine with different settings (pretrained vs not), the two
    # engine ids are identical and this is the only thing distinguishing them.
    baseline_conditions: dict[str, Any] = field(default_factory=dict)
```

Replace `:359-361` with:

```python
        # Resolved at enqueue for new Runs; `None` only on a row written before
        # the baseline became choosable, where the registry default is correct.
        baseline = (
            self._engines.get(command.baseline_engine_id)
            if command.baseline_engine_id
            else self._engines.baseline()
        )
        baseline_manifest = baseline.manifest()
        if task not in baseline_manifest.tasks:
            raise ValidationError(
                f"Baseline engine '{baseline_manifest.id}' cannot train a {task.value} "
                f"model; it supports {', '.join(t.value for t in baseline_manifest.tasks)}"
            )
        baseline_conditions = validate_conditions(
            baseline_manifest, command.baseline_conditions
        )
```

Add one line to the `ScorecardInputs(...)` construction after `baseline_engine_id=baseline_manifest.id` (`:433`):

```python
            baseline_conditions=baseline_conditions,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/ -v && uv run mypy src`
Expected: PASS, 343+ tests

- [ ] **Step 6: Commit**

```bash
git add backend/src/daikonstudio/application/execution/train_protocol.py backend/tests/integration/test_train_protocol.py
git commit -m "feat: RunTraining fits the requested baseline and records its conditions"
```

---

### Task 5: Carry `baseline_conditions` to the API

**Files:**
- Modify: `backend/src/daikonstudio/domain/execution/scorecard.py`
- Modify: `backend/src/daikonstudio/application/execution/build_scorecard.py`
- Modify: `backend/src/daikonstudio/application/catalog/get_scorecard.py:62-65` (the `build_scorecard` call site)
- Modify: `backend/src/daikonstudio/interface/routes/protocols.py:170-229` (`ScorecardResponse`)
- Test: `backend/tests/unit/execution/test_scorecard.py` (extend)

**Interfaces:**
- Consumes: `ScorecardInputs.baseline_conditions` (Task 4).
- Produces: `build_scorecard(..., baseline_conditions: dict[str, Any] = {})`, `Scorecard.baseline_conditions`, `ScorecardResponse.baseline_conditions`. Consumed by Task 11.

Note the real shape before editing: `build_scorecard` is **keyword-only over flat arguments** (`build_scorecard.py:29-49`), not a function of `ScorecardInputs`. `GetScorecard` reads the blob and spreads it into that call. `ScorecardResponse` renames two fields on the way out (`target_unit`→`unit`, `target_direction`→`direction`), so it is a real mapping, not a passthrough.

The new argument **must have a default of `{}`**: `test_scorecard.py` calls `build_scorecard(...)` directly in several places (`:57`, `:121`, `:153`) besides the `regression_card(**overrides)` helper at `:13-34`, and a required argument would break every one of them.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/unit/execution/test_scorecard.py`:

```python
def test_the_baseline_conditions_reach_the_rendered_scorecard():
    """Without this the page cannot tell a pretrained model from an untrained one:
    both sides carry the same engine id, and only the conditions differ."""
    card = regression_card(
        baseline_engine_id="chemprop-dmpnn",
        baseline_conditions={"pretrained": "none"},
    )
    assert card.baseline_conditions == {"pretrained": "none"}


def test_baseline_conditions_default_to_empty_for_a_scorecard_written_earlier():
    """Every existing call site omits them, including scorecards read back off
    blobs that predate the field."""
    assert regression_card().baseline_conditions == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/execution/test_scorecard.py -v`
Expected: FAIL with `TypeError: build_scorecard() got an unexpected keyword argument 'baseline_conditions'`

- [ ] **Step 3: Write minimal implementation**

Four edits, in order:

1. `domain/execution/scorecard.py` — add `baseline_conditions: dict[str, Any]` to the `Scorecard` dataclass beside `baseline_engine_id`. Give it a default of `field(default_factory=dict)` if the dataclass is not keyword-only, so field ordering stays legal.
2. `application/execution/build_scorecard.py:29-49` — add `baseline_conditions: dict[str, Any] | None = None` to the keyword-only signature, and set `baseline_conditions=baseline_conditions or {}` on the constructed `Scorecard`.
3. `application/catalog/get_scorecard.py:62-65` — pass `baseline_conditions=inputs.baseline_conditions` into the call.
4. `interface/routes/protocols.py:170-229` — add `baseline_conditions: dict[str, Any]` to `ScorecardResponse` and to its domain mapping.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/ -v && uv run mypy src`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/domain/execution/scorecard.py backend/src/daikonstudio/application/execution/build_scorecard.py backend/src/daikonstudio/application/catalog/get_scorecard.py backend/src/daikonstudio/interface/routes/protocols.py backend/tests/unit/execution/test_scorecard.py
git commit -m "feat: expose the baseline's conditions on the scorecard"
```

---

### Task 6: Fetch and cache foundation-model weights

**Files:**
- Create: `backend/src/daikonstudio/infrastructure/engines/_pretrained.py`
- Modify: `backend/src/daikonstudio/settings.py:17` (add after `blob_storage_options`)
- Test: `backend/tests/unit/engines/test_pretrained.py` (create)

**Interfaces:**
- Produces: `WEIGHT_SETS: dict[str, WeightSet]` and `weights_path(name: str, directory: str) -> Path`, consumed by Task 7. `WeightSet` is a frozen dataclass with `url: str`, `md5: str`, `filename: str`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/engines/test_pretrained.py`:

```python
"""Caching weights fetched from the internet. No network in these tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._pretrained import (
    WEIGHT_SETS,
    WeightSet,
    weights_path,
)


def test_chemeleon_is_registered_with_its_published_checksum():
    weight_set = WEIGHT_SETS["CheMeleon"]
    assert weight_set.url == "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    assert weight_set.md5 == "6a80b54fdb7de37ef0374d302f01e8ce"


def test_an_already_cached_file_is_returned_without_a_download(tmp_path, monkeypatch):
    payload = b"pretend weights"
    cached = tmp_path / "fake.pt"
    cached.write_bytes(payload)
    # setitem, not setattr: WeightSet is frozen, so swap the whole entry.
    monkeypatch.setitem(
        WEIGHT_SETS,
        "CheMeleon",
        WeightSet(url="https://unused", md5=hashlib.md5(payload).hexdigest(), filename="fake.pt"),
    )

    def _explode(*args, **kwargs):
        raise AssertionError("downloaded despite a valid cached file")

    monkeypatch.setattr(
        "daikonstudio.infrastructure.engines._pretrained.urlretrieve", _explode
    )
    assert weights_path("CheMeleon", str(tmp_path)) == cached


def test_a_corrupt_download_is_rejected_and_not_left_in_the_cache(tmp_path, monkeypatch):
    """A truncated download otherwise surfaces as an unreadable-tensor error with
    no hint that the network was the cause."""

    def _write_garbage(url, filename):
        Path(filename).write_bytes(b"truncated")

    monkeypatch.setattr(
        "daikonstudio.infrastructure.engines._pretrained.urlretrieve", _write_garbage
    )
    with pytest.raises(ValidationError, match="checksum"):
        weights_path("CheMeleon", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_weight_set_names_what_is_available(tmp_path):
    with pytest.raises(ValidationError, match="CheMeleon"):
        weights_path("NotAModel", str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/engines/test_pretrained.py -v`
Expected: FAIL with `ModuleNotFoundError: daikonstudio.infrastructure.engines._pretrained`

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/daikonstudio/infrastructure/engines/_pretrained.py`:

```python
"""Fetch, verify and cache pretrained foundation-model weights.

Its own module rather than more lines in `chemprop_dmpnn.py`: downloading and
checksumming a file has nothing to do with message passing, and a second engine
wanting a foundation model should not import a D-MPNN to get it.

No torch import here. This module deals in bytes on disk, which is what lets it
be tested on a worker that has no gpu extra installed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from daikonstudio.domain.shared.errors import ValidationError

_CHUNK = 1024 * 1024


@dataclass(frozen=True, kw_only=True)
class WeightSet:
    url: str
    md5: str
    filename: str


# The published artifacts, keyed by the string an engine's `pretrained` condition
# offers. Adding a set is one entry -- but note a chemprop v1-format checkpoint
# would also need `chemprop convert` and the V1 atom featurizer, which this
# structure cannot express. That is a reason to extend it then, not now.
WEIGHT_SETS: dict[str, WeightSet] = {
    "CheMeleon": WeightSet(
        url="https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        md5="6a80b54fdb7de37ef0374d302f01e8ce",
        filename="chemeleon_mp.pt",
    ),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def weights_path(name: str, directory: str) -> Path:
    """The cached weights for `name`, downloading them once if absent.

    Bytes from the internet become model weights, so the checksum is a trust
    boundary and not ceremony: a truncated download otherwise surfaces as an
    unreadable-tensor error deep inside torch with no hint that the network was
    the cause.

    Download-to-temp-then-rename because two workers can share one machine, and
    `os.replace` is atomic within a filesystem -- so a second worker either sees
    no file or sees a whole one, never a half-written one.
    """
    try:
        weight_set = WEIGHT_SETS[name]
    except KeyError:
        raise ValidationError(
            f"Unknown pretrained weight set '{name}'. "
            f"Available: {', '.join(sorted(WEIGHT_SETS))}."
        ) from None

    cache = Path(directory).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / weight_set.filename
    if target.exists() and _md5(target) == weight_set.md5:
        return target

    handle, staging = tempfile.mkstemp(dir=cache, suffix=".partial")
    os.close(handle)
    staged = Path(staging)
    try:
        urlretrieve(weight_set.url, staged)  # noqa: S310 - a pinned https literal
        actual = _md5(staged)
        if actual != weight_set.md5:
            raise ValidationError(
                f"Downloaded weights for '{name}' failed their checksum "
                f"(expected {weight_set.md5}, got {actual}). The download from "
                f"{weight_set.url} was corrupt or truncated; nothing was cached."
            )
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target
```

Add to `settings.py` after `blob_storage_options` (`:17`):

```python
    # Where pretrained foundation-model weights are cached. Downloaded once on
    # first use; Dockerfile.gpu bakes them in so a production worker never
    # reaches the network, and an air-gapped deployment works.
    pretrained_weights_dir: str = "~/.cache/daikon-studio/weights"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/engines/test_pretrained.py -v && uv run mypy src`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/infrastructure/engines/_pretrained.py backend/src/daikonstudio/settings.py backend/tests/unit/engines/test_pretrained.py
git commit -m "feat: cache pretrained weight sets with a checksum on first use"
```

---

### Task 7: chemprop starts from CheMeleon

**Files:**
- Modify: `backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py:52-93` (conditions) and `:178-235` (`train`)
- Modify: `backend/Dockerfile.gpu`
- Test: `backend/tests/unit/engines/test_chemprop_dmpnn.py` (extend)

**Interfaces:**
- Consumes: `weights_path`, `WEIGHT_SETS` (Task 6).
- Produces: a `pretrained` enum condition on the `chemprop-dmpnn` manifest with options `("none", "CheMeleon")`, default `"none"`.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/unit/engines/test_chemprop_dmpnn.py`:

```python
def test_the_manifest_offers_pretrained_weights():
    spec = next(c for c in ChempropDMPNN.manifest().conditions if c.key == "pretrained")
    assert spec.type is ConditionType.ENUM
    assert spec.default == "none"
    assert spec.options == ("none", "CheMeleon")


@pytest.mark.skipif(
    not (Path(os.environ.get("STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights"))
         .expanduser() / "chemeleon_mp.pt").exists(),
    reason="CheMeleon weights not cached; CI does not download 35 MB",
)
def test_chemeleon_builds_a_network_sized_by_the_checkpoint_not_the_conditions():
    """CheMeleon pins d_h=2048. The FFN's input_dim must follow the checkpoint,
    not the message_hidden_dim condition, or the first layer is built for the
    wrong width and the fit dies on a shape mismatch."""
    from daikonstudio.infrastructure.engines.chemprop_dmpnn import _build_model

    model = _build_model(
        pretrained="CheMeleon",
        weights_dir=os.environ.get(
            "STUDIO_PRETRAINED_WEIGHTS_DIR", "~/.cache/daikon-studio/weights"
        ),
        hidden=300,
        depth=3,
        is_classification=False,
        output_transform=None,
    )
    assert model.message_passing.output_dim == 2048
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/engines/test_chemprop_dmpnn.py -v`
Expected: FAIL — no `pretrained` condition, no `_build_model`

- [ ] **Step 3: Write minimal implementation**

Add to `_MANIFEST.conditions`, after `batch_size` (`:92`):

```python
        ConditionSpec(
            key="pretrained",
            label="Pretrained weights",
            type=ConditionType.ENUM,
            default="none",
            options=("none", "CheMeleon"),
            help="Start from a foundation model's learned representation instead of "
            "random weights. CheMeleon was pretrained on ~1M PubChem molecules against "
            "classical descriptors; it fixes the hidden size at 2048 and the message "
            "passing steps at 6, so those two settings are ignored when it is selected.",
        ),
```

Extract the model construction at `:223-235` into a module-level function so it is testable without a fit, and make it honour the checkpoint:

```python
def _build_model(
    *,
    pretrained: str,
    weights_dir: str,
    hidden: int,
    depth: int,
    is_classification: bool,
    output_transform: Any,
) -> Any:
    """The network, before any data touches it.

    Under `pretrained`, the architecture comes from the checkpoint's own saved
    hyperparameters rather than from the conditions -- a request that bypassed
    the form must not be able to build a network the weights do not fit. The
    form pins and disables the two inert conditions so the stored record still
    matches what ran; this function is what makes that safe rather than trusted.
    """
    import torch
    from chemprop.models import MPNN
    from chemprop.nn import (
        BinaryClassificationFFN,
        BondMessagePassing,
        MeanAggregation,
        RegressionFFN,
    )

    from daikonstudio.infrastructure.engines._pretrained import weights_path

    if pretrained == "none":
        message_passing = BondMessagePassing(d_h=hidden, depth=depth)
        # An untrained encoder benefits from batch norm on the graph embedding.
        batch_norm = True
    else:
        # NOT MPNN.load_from_checkpoint: this file is not a Lightning checkpoint.
        # It holds exactly two keys -- `hyper_parameters` and `state_dict` -- for
        # the message-passing block alone, with no predictor head, and
        # load_from_checkpoint raises KeyError('metrics') on it.
        checkpoint = torch.load(
            weights_path(pretrained, weights_dir), weights_only=True
        )
        message_passing = BondMessagePassing(**checkpoint["hyper_parameters"])
        message_passing.load_state_dict(checkpoint["state_dict"])
        # False to match chemprop's own chemeleon_foundation_finetuning notebook.
        batch_norm = False

    # Must equal the message passing's output width, which under a pretrained
    # encoder is the checkpoint's d_h (2048 for CheMeleon), not `hidden`.
    input_dim = message_passing.output_dim
    predictor = (
        BinaryClassificationFFN(input_dim=input_dim)
        if is_classification
        else RegressionFFN(input_dim=input_dim, output_transform=output_transform)
    )
    return MPNN(
        message_passing=message_passing,
        # CheMeleon requires mean aggregation, which is also what this engine
        # has always used -- so there is nothing to branch on.
        agg=MeanAggregation(),
        predictor=predictor,
        batch_norm=batch_norm,
    )
```

In `train()`, read the new condition beside the others (`:182-185`) and replace the `predictor = ...` / `model = MPNN(...)` block at `:223-235` with a call to `_build_model`. The engine needs the weights directory: read it from `Settings()` inside `train()` (an import-time settings read would break the no-infrastructure-at-import rule this module lives by).

Then add to `backend/Dockerfile.gpu`, after the dependency sync and before the app copy:

```dockerfile
# Bake CheMeleon in so a production worker never reaches Zenodo, and an
# air-gapped deployment works. 34,859,448 bytes, MIT licensed.
ENV STUDIO_PRETRAINED_WEIGHTS_DIR=/opt/weights
RUN mkdir -p /opt/weights \
    && curl -fsSL -o /opt/weights/chemeleon_mp.pt \
       https://zenodo.org/records/15460715/files/chemeleon_mp.pt \
    && echo "6a80b54fdb7de37ef0374d302f01e8ce  /opt/weights/chemeleon_mp.pt" | md5sum -c -
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/engines/ -v && uv run mypy src`
Expected: PASS. `test_the_registry_loads_with_no_gpu_extra_installed` must still pass — if it fails, an import escaped to module scope.

- [ ] **Step 5: Commit**

```bash
git add backend/src/daikonstudio/infrastructure/engines/chemprop_dmpnn.py backend/Dockerfile.gpu backend/tests/unit/engines/test_chemprop_dmpnn.py
git commit -m "feat: chemprop can start from CheMeleon pretrained weights"
```

---

### Task 8: Regenerate the API contract

**Files:**
- Modify: `frontend/openapi.json`, `frontend/src/shared/lib/api/model/*`

- [ ] **Step 1: Regenerate**

Run from the repo root: `make generate-api`

- [ ] **Step 2: Verify the diff shows exactly the expected fields**

Run: `git diff --stat frontend/openapi.json frontend/src/shared/lib/api/model/`
Expected: `trainProtocolBody.ts` gains `baseline_engine_id` and `baseline_conditions`; `scorecardResponse.ts` gains `baseline_conditions`; `conditionResponse.ts` unchanged. Nothing else. Unrelated churn means the committed snapshot had drifted — review before committing.

- [ ] **Step 3: Verify the frontend still typechecks**

Run: `cd frontend && pnpm typecheck && pnpm lint`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/openapi.json frontend/src/shared/lib/api/
git commit -m "chore: regenerate the api contract for the choosable baseline"
```

---

### Task 9: The training form picks a baseline

**Files:**
- Modify: `frontend/src/features/protocols/components/train-protocol-form.tsx`
- Modify: `frontend/src/features/protocols/components/condition-fields.tsx`
- Modify: `frontend/src/features/engines/types/index.ts`
- Test: `frontend/src/features/protocols/components/train-protocol-form.test.tsx` (create if the feature has no test file yet; follow the setup in `lib/verdict.test.ts`)

**Interfaces:**
- Consumes: `TrainProtocolBody.baseline_engine_id` / `.baseline_conditions` (Task 8).
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it } from "vitest";
import { resolveConditions, comparesAgainstItself } from "./train-protocol-form";

const SPECS = [
  { key: "n_estimators", label: "Trees", type: "integer", default: 500, options: [] },
];

describe("comparesAgainstItself", () => {
  it("is true when both sides fall back to the same defaults", () => {
    expect(comparesAgainstItself("rf", {}, "rf", {}, SPECS, SPECS)).toBe(true);
  });

  it("is true when one side sets a value the other takes as its default", () => {
    // The trap: comparing raw form state would call these different, and the
    // warning would go missing on a run the server treats as a self-comparison.
    expect(
      comparesAgainstItself("rf", { n_estimators: 500 }, "rf", {}, SPECS, SPECS),
    ).toBe(true);
  });

  it("is false for the same engine with genuinely different settings", () => {
    expect(
      comparesAgainstItself("rf", { n_estimators: 100 }, "rf", {}, SPECS, SPECS),
    ).toBe(false);
  });

  it("is false for different engines", () => {
    expect(comparesAgainstItself("rf", {}, "xgb", {}, SPECS, SPECS)).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm test train-protocol-form`
Expected: FAIL — no such exports

- [ ] **Step 3: Write minimal implementation**

Export both helpers from `train-protocol-form.tsx`:

```tsx
/**
 * Conditions as the server will see them: form state over manifest defaults.
 * Comparing raw form state instead would call `{n_estimators: 500}` different
 * from `{}` even though `validate_conditions` resolves both to the same dict.
 */
export function resolveConditions(
  specs: EngineCondition[],
  values: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    specs.map((spec) => [spec.key, values[spec.key] ?? spec.default]),
  );
}

/** The client-side mirror of the server's `baseline_is_self`. */
export function comparesAgainstItself(
  engineId: string,
  conditions: Record<string, unknown>,
  baselineEngineId: string,
  baselineConditions: Record<string, unknown>,
  specs: EngineCondition[],
  baselineSpecs: EngineCondition[],
): boolean {
  if (!engineId || engineId !== baselineEngineId) return false;
  return (
    JSON.stringify(resolveConditions(specs, conditions)) ===
    JSON.stringify(resolveConditions(baselineSpecs, baselineConditions))
  );
}
```

Both manifests are the same engine whenever this can return true, so `specs` order is identical and `JSON.stringify` is a sound comparison.

Then in the component:

1. Add state: `const [baselineEngineId, setBaselineEngineId] = useState("");` and `const [baselineConditions, setBaselineConditions] = useState<Record<string, unknown>>({});`
2. Default the baseline once engines load, and reset it alongside the engine in the existing `useEffect` at `:46-51`:

```tsx
useEffect(() => {
  if (!baselineEngineId && eligible.length > 0) {
    const flagged = eligible.find((candidate) => candidate.is_baseline);
    if (flagged) setBaselineEngineId(flagged.id);
  }
}, [baselineEngineId, eligible]);
```

3. `const baselineEngine = eligible.find((c) => c.id === baselineEngineId);`
4. Add a **Compare against** `<Select>` block below the Engine block (after `:180`), listing `eligible` exactly as the engine select does.
5. Replace the `engine?.is_baseline` warning at `:174-179` with one driven by `comparesAgainstItself(...)`, moved below the baseline select:

```tsx
{comparesAgainstItself(
  engineId, conditions, baselineEngineId, baselineConditions,
  engine?.conditions ?? [], baselineEngine?.conditions ?? [],
) && (
  <p className="text-xs text-muted-foreground">
    This is the same engine with the same settings on both sides, so there is
    nothing to compare. Its scorecard will say so rather than showing a
    comparison that never happened — change a setting, or pick a different
    engine to measure against.
  </p>
)}
```

6. Add a collapsed **Baseline settings** block mirroring the existing `{engine && ...}` block at `:192-203`, rendering `<ConditionFields>` against `baselineEngine.conditions` into `baselineConditions`. Use the kit's `collapsible.tsx`.
7. Send both in `submit()`: `baseline_engine_id: baselineEngineId, baseline_conditions: baselineConditions`.
8. Update the page copy at `:119-122` — "It is scored against a fingerprint baseline automatically" is no longer accurate now that the baseline is chosen. Replace "a fingerprint baseline" with "a baseline you choose".

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && pnpm test && pnpm typecheck && pnpm lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/protocols/components/ frontend/src/features/engines/types/index.ts
git commit -m "feat: choose what a protocol is measured against in the training form"
```

---

### Task 10: Pin CheMeleon's fixed settings in the form

**Files:**
- Modify: `frontend/src/features/engines/types/index.ts`
- Modify: `frontend/src/features/protocols/components/condition-fields.tsx`
- Modify: `frontend/src/features/protocols/components/train-protocol-form.tsx`
- Test: `frontend/src/features/engines/types/index.test.ts` (extend or create)

**Interfaces:**
- Consumes: the `pretrained` condition (Task 7).
- Produces: `PINNED_BY_PRETRAINED: Record<string, Record<string, number>>`; `ConditionFields` gains an optional `pinned?: Record<string, unknown>` prop.

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it } from "vitest";
import { PINNED_BY_PRETRAINED } from "./index";

describe("PINNED_BY_PRETRAINED", () => {
  it("pins the two settings CheMeleon's checkpoint fixes", () => {
    expect(PINNED_BY_PRETRAINED.CheMeleon).toEqual({
      message_hidden_dim: 2048,
      depth: 6,
    });
  });

  it("pins values that sit inside the manifest's declared bounds", () => {
    // Why this matters: the form submits these, so an out-of-bounds pin would
    // be rejected by validate_conditions on the server.
    expect(PINNED_BY_PRETRAINED.CheMeleon.depth).toBeLessThanOrEqual(6);
    expect(PINNED_BY_PRETRAINED.CheMeleon.message_hidden_dim).toBeLessThanOrEqual(2400);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm test PINNED`
Expected: FAIL — no such export

- [ ] **Step 3: Write minimal implementation**

Add to `frontend/src/features/engines/types/index.ts`, beside `TASK_FOR_TARGET_KIND`:

```ts
/**
 * What each pretrained weight set fixes, mirroring its checkpoint's own saved
 * `hyper_parameters`.
 *
 * The second piece of hardcoded engine knowledge in this app, and it is here
 * for the same reason as TASK_FOR_TARGET_KIND: the manifest has no way to say
 * "this condition makes those two inert". Without it the form would accept a
 * `depth` the fit silently ignores, and the Scorecard would then report a
 * setting the model never used — the exact dishonesty the Scorecard exists to
 * prevent. The form submits these values, so the record stays true.
 *
 * ponytail: two constants for one weight set. If a second one lands, move this
 * onto ConditionSpec as a `pinned_by` field so the catalogue stays
 * self-describing.
 */
export const PINNED_BY_PRETRAINED: Record<string, Record<string, number>> = {
  CheMeleon: { message_hidden_dim: 2048, depth: 6 },
};
```

Give `ConditionFields` an optional `pinned?: Record<string, unknown>` prop. For any key present in `pinned`, render the input `disabled`, show the pinned value rather than form state, and render the caption "Fixed by the pretrained weights you selected."

In `train-protocol-form.tsx`, derive `const pinned = PINNED_BY_PRETRAINED[String(conditions.pretrained ?? "none")] ?? {};` and pass it to the chosen engine's `<ConditionFields>`. In a `useEffect`, merge the pinned values into `conditions` state whenever `conditions.pretrained` changes, so what is submitted matches what is displayed. Do the same for the baseline block with `baselineConditions`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && pnpm test && pnpm typecheck && pnpm lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/engines/types/ frontend/src/features/protocols/components/
git commit -m "feat: pin and disable the settings CheMeleon's checkpoint fixes"
```

---

### Task 11: The scorecard names the baseline it actually used

**Files:**
- Modify: `frontend/src/features/protocols/components/scorecard-view.tsx:122-127` and `:160-164`
- Test: `frontend/src/features/protocols/lib/verdict.test.ts` (extend)

**Interfaces:**
- Consumes: `ScorecardResponse.baseline_conditions` (Task 8).

- [ ] **Step 1: Write the failing test**

Add to `verdict.test.ts`:

```ts
it("describes a same-engine comparison by what differs, not by the engine id", () => {
  // Both sides are chemprop-dmpnn. Naming only the engine would render
  // "chemprop-dmpnn versus chemprop-dmpnn", which explains nothing.
  expect(
    describeBaseline({
      baseline_engine_id: "chemprop-dmpnn",
      engine_id: "chemprop-dmpnn",
      conditions: { pretrained: "CheMeleon", epochs: 50 },
      baseline_conditions: { pretrained: "none", epochs: 50 },
    }),
  ).toContain("pretrained");
});

it("names the engine when the two sides are different engines", () => {
  expect(
    describeBaseline({
      baseline_engine_id: "ecfp4-randomforest",
      engine_id: "chemprop-dmpnn",
      conditions: {},
      baseline_conditions: {},
    }),
  ).toContain("ecfp4-randomforest");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && pnpm test verdict`
Expected: FAIL — `describeBaseline` is not exported

- [ ] **Step 3: Write minimal implementation**

Add `describeBaseline` to `verdict.ts`: when `baseline_engine_id !== engine_id`, return the engine id; otherwise return a phrase listing the condition keys whose values differ between `conditions` and `baseline_conditions` — e.g. `the same engine with pretrained = none`.

In `scorecard-view.tsx`:
- `:160-164` — use `describeBaseline(scorecard)` in place of the bare `scorecard.baseline_engine_id`. Keep the existing "In published benchmarks a fingerprint baseline places mid-field…" sentence **only when the baseline is a fingerprint engine**; it is false about a chemprop baseline.
- `:122-127` — replace the hardcoded "You trained ECFP4 + RandomForest" with `scorecard.engine_id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && pnpm test && pnpm typecheck && pnpm lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/protocols/
git commit -m "fix: scorecard copy names the baseline that actually ran"
```

---

### Task 12: End-to-end verification

**Files:** none modified — this task produces evidence, not code.

- [ ] **Step 1: Full backend suite and linters**

```bash
cd backend && uv run pytest -v && uv run ruff check && uv run ruff format --check && uv run mypy src && uv run lint-imports
```
Expected: all pass, 343+ tests, 3 import-linter contracts kept.

- [ ] **Step 2: Frontend suite**

```bash
cd frontend && pnpm test && pnpm typecheck && pnpm lint
```

- [ ] **Step 3: Run the real loop**

Start the stack with `make dev` (backend `:8002`, frontend `:3003`, both worker lanes). Confirm `STUDIO_INLINE_JOBS=0` in `backend/.env` first, and check `lsof -ti:8002` for a stale backend from an older session — `make stop` has lost that race before (handoff §3).

Train BBBP with engine `chemprop-dmpnn{pretrained: CheMeleon}` against baseline `chemprop-dmpnn{pretrained: none}`.

Expected: the run routes to the gpu lane, both fits run, and the Scorecard's verdict band distinguishes the two sides by their pretraining rather than printing one engine id twice. Record the wall-clock and the head-to-head numbers in the PR description — the prior BBBP reference is 167.8s for a D-MPNN plus baseline, MCC 0.515 against the ECFP4 baseline's 0.632.

- [ ] **Step 4: Confirm the slim image still builds without torch**

```bash
docker build -f backend/Dockerfile -t studio-api-check backend/
docker run --rm studio-api-check python -c "
from daikonstudio.infrastructure.engines.registry import default_registry
import sys
assert len(default_registry().manifests()) == 3
assert 'torch' not in sys.modules
print('manifest serves without torch')"
```
Expected: prints the confirmation. This is the check that catches an import escaping to module scope in Task 7.

- [ ] **Step 5: Commit any fixes and open the PR**

Do **not** attempt `docker build -f backend/Dockerfile.gpu` on this laptop — it is arm64 and torch's cu124 wheels are amd64-only; the emulated build reaches "Prepared 127 packages" and stalls. Build it on x86 with `docker --context ned build`. Note in the PR that the CheMeleon `RUN curl` layer is unverified until that happens.

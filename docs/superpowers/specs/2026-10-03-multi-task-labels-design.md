# Multi-task labels

**Date:** 2026-10-03
**Status:** awaiting review

## Goal

A scientist uploads one CSV carrying several measured endpoints and trains on all of them
at once. Chemprop and MoLFormer-XL learn them jointly, from one shared representation.
Every other engine in the roster accepts the same dataset and trains one model per
endpoint, so the two can be compared on identical rows and an identical split.

This is "Ranked next #2" in `docs/roadmap.md`, and the roadmap's own description of the
blast radius is accurate: it touches the Dataset schema, `TargetSpec`, the training
context and the Scorecard.

## What exists today

- `Dataset` holds exactly one `target: TargetSpec` (`domain/data/dataset.py:43`), stored
  as a single JSONB object (`infrastructure/persistence/sqlalchemy/data/models.py:23`).
  The aggregate is deliberately immutable and has no update route.
- A run cannot override the target. `runs.params` carries no target field; the executor
  re-reads `dataset.target.column` at `application/execution/train_protocol.py:755`.
- `TrainContext` carries `target_column: str` and a single `task: TaskType`
  (`application/engines/context.py:55,53`). `TrainResult.metrics` is one flat
  `dict[str, float]`, as is `validation_metrics` (`context.py:84-85`).
- `Engine.predict` returns `row_id, value, uncertainty` — one scalar per row
  (`application/engines/protocol.py:36-38`).
- `derive_readouts` is 1:1 from `TargetSpec`: one readout named `{column}` for
  regression, two named `{column}_probability` and `{column}` for classification
  (`application/catalog/derive_readouts.py:34-62`).
- The Scorecard is **computed on read** by `get_scorecard` from a `ScorecardInputs` blob,
  never stored in a table. `runs.metrics` JSONB holds only the denormalised headline used
  for ranking (`infrastructure/persistence/sqlalchemy/execution/models.py:42-43`).
- chemprop supports `n_tasks > 1` natively but is hard-pinned to one:
  `y=np.array([float(target)])` (`infrastructure/engines/chemprop_dmpnn.py:134-147`) and
  `_forward`'s `reshape(-1)` (`:150-168`).
- `content_hash` covers data and split but not the target spec, and
  `uq_datasets_workspace_content_hash` is unique per workspace. **Uploading the same CSV
  a second time to predict a different column is rejected as a duplicate**
  (`application/data/create_dataset.py:208-215`). There is no route to multiple endpoints
  from one file today, by any means.

## Decisions

Made by the user on 2026-10-03:

1. **Engines that cannot learn several endpoints jointly are not excluded.** They train
   one model per endpoint behind a single shared adapter, so every engine works with
   every multi-label dataset.
2. **`supports_multitask` is declared on the engine manifest.** It selects the training
   path and labels the result; it does not gate which engines a dataset may use.
3. **Chemprop first, MoLFormer-XL second.** Chemprop is mostly unpinning existing code;
   MoLFormer needs a new multi-head with a masked loss.
4. **No `content_hash` change.** Fan-out removes the reason to upload one file twice: a
   single multi-label dataset backs both the joint model and the per-endpoint baselines.

Defaults set while writing this spec (not asked):

- **All targets in a dataset must share one `TargetKind`.** Mixed regression and
  classification in one model is real (ADMET-AI does it) but complicates the loss and the
  scorecard far more than it buys here. Validated at creation, with a `ponytail:` comment
  naming the ceiling.
- **A run states which kind of training happened.** A scorecard reading "XGBoost, 4
  endpoints" must not be mistakable for joint learning when it was four separate fits.

## Design

### Domain

`Dataset.target: TargetSpec` becomes `targets: tuple[TargetSpec, ...]`, ordered as the
scientist picked them.

**`Dataset.target` is deleted outright, with no compatibility property.** mypy strict is
a CI gate, so removing the attribute turns all ~12 call sites into type errors — a
complete, mechanical worklist. A shim returning `targets[0]` would instead train silently
on endpoint #1 and drop the rest, which is the exact failure this feature exists to end.

New invariants, all checked at creation:

- Targets are distinct, and each passes the existing `RESERVED_TARGET_COLUMNS` check.
  `target` joins that set, for the same defensive reason `row_id` is already in it: it is
  now a column name engine output carries, and nothing stops a future caller persisting
  it beside the scientist's own columns.
- All targets share one `TargetKind`.
- **Derived readout names do not collide.** Classification derives
  `{column}_probability` and `{column}`, so choosing both `foo` and `foo_probability` as
  targets produces two readouts named `foo_probability`. This is a new collision class
  that `RESERVED_TARGET_COLUMNS` does not cover, and it must be rejected by name.
- The degenerate-partition check (`create_dataset.py:233-305`) runs per target; the
  dataset is refused if *any* target is degenerate in train, and the error names the
  column.

The domain still carries only column names and specs, never a label matrix, so the
domain-purity contract (no numpy) is untouched.

### Persistence

Migration `013`: `datasets.target` JSONB object becomes `datasets.targets` JSONB array,
back-filling every existing row as a single-element list. Reversible.

`runs.metrics` becomes nested, keyed by target column: `{"reactive": {"mcc": 0.41, …}}`.
A single-target run has exactly one key, so the shape is uniform. This is a JSONB column
with no schema change; existing rows are read through a reader that wraps a flat dict in
the dataset's single target name.

No scorecard schema change — scorecards are computed on read.

### Engine contract

`TrainContext` gains `target_columns: tuple[str, ...]` as the authoritative field, and
keeps `target_column` as a **property that raises when the context carries more than
one**:

```python
@property
def target_column(self) -> str:
    """The single target. Raises when this context carries several: an engine that
    does not declare `supports_multitask` is never handed a multi-target context --
    the adapter splits it first. Loud, not silent."""
```

This is the opposite choice from `Dataset.target` above, deliberately. There, a
first-element shim would *silently* drop endpoints. Here, the adapter guarantees a
single-target context for exactly the engines that read this property, and the property
raises if that guarantee is ever broken. One is a silent wrong answer; the other is a
loud invariant.

`TrainResult.metrics` and `validation_metrics` become `dict[str, dict[str, float]]`,
keyed by target column. This is one line per engine (seven of them), which mypy finds;
uniform beats introducing a second result type.

`Engine.predict`'s documented return is unchanged — `row_id, value, uncertainty` — with
one extension: **an engine declaring `supports_multitask` also returns a `target`
column**, giving one row per (compound, endpoint). The adapter adds that column for every
other engine, so everything downstream of the adapter sees one long-format shape.

`EngineManifest` gains `supports_multitask: bool = False`.

### The fan-out adapter

One new module in `application/engines/`. It wraps a protocol-typed `Engine`, so it sits
in the application layer and the layering and bounded-context contracts hold.

Given a context with N targets it:

- trains the inner engine N times, once per single-target context;
- merges the N flat metric dicts into the nested shape;
- on predict, calls the inner engine per target, tags each frame with its `target` column
  and concatenates.

**Artifacts: the container appears only at N > 1.** At N = 1 the stored artifact is the
inner engine's bytes verbatim. This is not a style choice — every protocol already trained
on this platform has a bare engine artifact in blob storage, and an adapter that always
wrapped would expect a container where none exists, breaking prediction for every existing
protocol. At N > 1 the N artifacts are packed into one zip with an entry per target
column: boring, inspectable, no pickle.

The `target` column is added at every N, including 1, so the long format downstream has
one shape. Only the storage container is conditional.

Two details that are easy to get wrong:

- **Progress.** `ctx.report(fraction, phase)` is documented as progress within *this
  fit*, 0.0 to 1.0. The adapter must map each sub-fit's fraction onto its own slice —
  target *i* of N reports `(i + f) / N` — or the bar resets to zero four times.
- **`RunInterrupted` must not be caught.** `context.py:13-33` is explicit that swallowing
  it turns a cancelled run into one that keeps burning a GPU. The adapter wraps N fits,
  which makes it exactly the place someone would be tempted to catch and continue.

The adapter wraps non-multitask engines **always, including at N=1**, so there is one
code path and the long format is produced in one place.

### Native multi-task engines

**chemprop** (first): remove the `n_tasks=1` pin at `chemprop_dmpnn.py:134-147` and the
`reshape(-1)` at `:150-168`. chemprop's own loss already masks NaN targets, so sparse
label matrices work without new code.

**MoLFormer-XL** (second): the head becomes `Linear(d, n_tasks)` with a masked loss.

### Readouts and prediction results

`derive_readouts` goes from one `TargetSpec` to the ordered tuple, concatenating per
target. Four binary targets give eight readouts. Because readout names are already
namespaced by column, **prediction results stay wide** — one row per compound, with a
column per readout — and the results schema needs no new concept.

`predict_with_protocol.py:382-388` currently branches on `len(protocol.readouts) == 1`
and then destructures exactly two:

```python
probability_readout, class_readout = protocol.readouts
```

With four binary targets that is eight readouts and this raises. It becomes a loop over
targets, pairing each target's values with its own readout pair. The fixed 0.5 threshold
and its existing `ponytail:` note carry over unchanged, per target.

### Scorecard

`ScorecardInputs` becomes per-target; `get_scorecard` calls `build_scorecard` once per
target and returns N scorecards. Every existing guarantee holds per target: the headline
metric (`mcc`/`rmse`), the bootstrap CI with its 20-row minimum, the NaN discipline where
all four classification metrics go undefined together, and `metrics_undefined` carrying
the reason. `target_unit`/`target_direction` are already per-target fields on the
Scorecard, so they simply come from that target's own spec.

Run-level fields (`split_strategy`, `engine_id`, `conditions`, `baseline_is_self`,
`applicability_coverage`) are repeated across the N scorecards rather than hoisted into a
new envelope type. Duplicating five scalars is cheaper than splitting the aggregate and
rewriting every consumer of it.

The mandatory baseline needs no special handling: it is an engine like any other, so a
non-multitask baseline is fanned out by the same adapter and produces its own N metric
sets, which is exactly what the per-target head-to-head needs.

### API and frontend

- Columns step: "Value to predict" becomes a multi-select.
- Target step: one kind radio for all targets, plus a per-column unit and direction row
  shown only for numeric targets. An all-binary dataset sees a single radio, as today.
- `TargetBody` on the datasets route becomes a list; `DatasetResponse` carries `targets`.
- The train form at `train-protocol-form.tsx:278-288` shows all endpoints rather than
  one, and states the training kind for the selected engine: one joint model, or N
  separate models.
- The run page shows N scorecards.
- Sweep ranking needs a target selector. `runs.metrics` no longer holds a single
  rankable scalar, and averaging the per-target primaries would invent a number. The
  sweep table ranks by a chosen target, defaulting to the first.

### Job timeout

Fan-out multiplies wall-clock by the target count against a fixed
`worker_job_timeout = 1800`s. Four XGBoost fits on 324k rows at 2048-bit fingerprints can
cross it. The timeout scales with the number of targets for a fan-out run. The
lease (`runner_lease_seconds = 600`) is already renewed by the progress callback, so only
the hard job deadline needs the multiply.

## Build order

Two phases, because MoLFormer-XL is genuinely separable and shipping one joint engine
proves the capability end to end.

**Phase 1** — everything above except the MoLFormer head: domain and migration, the
engine contract, the adapter, chemprop unpinned, readouts and prediction, scorecards,
frontend, timeout. At the end of it every engine accepts a multi-label dataset and
chemprop learns one jointly.

**Phase 2** — the MoLFormer-XL multi-head with masked loss. No contract changes; it only
flips its own `supports_multitask` and implements the extended `predict`.

Phase 1 is one implementation plan. Phase 2 is small enough to fold into the end of it or
take separately.

## Out of scope

Deliberately untouched, all pre-existing and orthogonal:

- **Stratified splitting.** Multi-label stratification is its own problem, and the
  existing random and scaffold strategies do not look at the target at all.
- **Class weights** and the fixed 0.5 decision threshold.
- **Mixed-kind targets** in one dataset (see Decisions).
- **Changing `content_hash`.** Revisit only if someone wants two datasets over one file
  with different target subsets.

## Testing

- Dataset creation: multiple targets round-trip; duplicate target rejected; mixed kinds
  rejected; a target colliding with a reserved column rejected; `foo` + `foo_probability`
  rejected by readout-name collision; a dataset where one of several targets is
  degenerate in train is refused and the error names that column.
- Migration 013 up and down against a seeded single-target row.
- Adapter: N artifacts round-trip through the zip; metrics nest per target; progress is
  monotonic across sub-fits and spans 0..1 once, not N times; `RunInterrupted` raised by
  the second of four fits propagates rather than being swallowed.
- `TrainContext.target_column` raises on a multi-target context.
- chemprop trains a 2-target frame and returns two metric sets; a NaN label in one target
  does not contribute to the loss.
- Prediction: a 4-binary-target protocol writes 8 readout columns with the right names,
  and the old 1-target and 2-readout paths still produce identical output.
- Scorecard: N scorecards; a degenerate target reports `metrics_undefined` while its
  siblings still score.

Gates that must pass, per `CLAUDE.md` and prior branches: `ruff check` and `format`,
`mypy` strict, `lint-imports` (3 contracts), and `OMP_NUM_THREADS=1 pytest`.

## Verification on real data

The nuisance dataset at `~/Documents/Cage-Fusion-Paper/Dataset/train.csv`: 324,803 rows,
four binary endpoints (`aggregator` 45,281 positives, `luciferase_inhibitor` 20,714,
`reactive` 8,392, `promiscuous` 8,981).

**Expect no multi-task lift on this data, and treat that as a pass.** The four labels come
from four separate source collections (`Data Sources.pptx`: ChemFH and Lies & Liability
for aggregation, CovalentInDB for reactivity, Hit Dexter 2.0 for promiscuity). They are
near-disjoint by construction: 96% of flagged compounds carry exactly one flag, 25 of
324,803 carry three, none carry four, and every pairwise lift is ≤ 1.0 —
`promiscuous → aggregator` is 0.3×, where the chemistry predicts strongly positive, since
colloidal aggregation is a leading cause of promiscuity. A `0` largely means "absent from
that source list", not "tested and clean".

So this dataset is a correctness fixture and a negative control, not evidence the feature
works as science. A real lift needs a sparse label matrix over endpoints measured on
overlapping compounds.

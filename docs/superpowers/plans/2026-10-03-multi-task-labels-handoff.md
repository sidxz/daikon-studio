# Handoff: multi-task labels

**Date:** 2026-10-03
**Branch:** `multi-task-labels` (off `main`, unpushed)
**Commits:** `017ff12` spec, `a70a411` spec amendments
**Status:** design approved, **nothing implemented**

## Start here

Read `docs/superpowers/specs/2026-10-03-multi-task-labels-design.md` in full. It is
approved and current; both amendments are folded in. This file is orientation and traps,
not a summary of it.

Next action: invoke the `superpowers:writing-plans` skill to turn the spec into an
implementation plan under `docs/superpowers/plans/`. Do not start editing code first.

## What the feature is

A dataset currently carries exactly one label column and can train exactly one model
against it. This makes a dataset carry an ordered tuple of label columns.

Two engines (`chemprop_dmpnn`, `molformer_xl`) learn all labels jointly from one shared
representation. The other five (`ecfp4_randomforest`, `ecfp4_xgboost`,
`descriptors_xgboost`, `ecfp4_lightgbm`, `tanimoto_gp`) cannot, so a single shared
**fan-out adapter** trains one model per label behind the same interface. Every engine
therefore accepts every multi-label dataset, and a joint model can be compared against
per-label models on identical rows and an identical split.

`EngineManifest.supports_multitask` selects the path and labels the result. It does not
gate which engines a dataset may use.

## Architecture constraints (enforced in CI by lint-imports)

Three contracts in `backend/pyproject.toml`, all currently passing:

- **Layers:** `interface` → `infrastructure` → `application` → `domain`.
- **Domain purity:** `daikonstudio.domain` may not import `numpy`, `polars`, `sklearn`,
  `sqlalchemy`, `fastapi`, or any outer layer. The label *matrix* therefore cannot live
  in the domain — only column names and specs do.
- **Bounded-context independence:** `domain.catalog`, `domain.data` and
  `domain.execution` may not import each other.

The fan-out adapter belongs in `application/engines/`. It wraps a protocol-typed
`Engine`, so it satisfies all three.

## Traps found while writing the spec

These are the things that will bite an implementer who skims.

1. **Existing stored artifacts are bare bytes.** If the adapter always packed artifacts
   into a container, every already-trained protocol would fail on predict. The container
   appears only at N > 1; at N = 1 the artifact is the inner engine's bytes verbatim.
   The `target` column is added at every N, so only the storage container is conditional.

2. **`predict_with_protocol.py:383` destructures exactly two readouts**
   (`probability_readout, class_readout = protocol.readouts`). Four binary labels produce
   eight readouts and this raises. It becomes a per-label loop. This is the concrete
   breakage point, and the one most likely to be missed.

3. **A new name-collision class.** Classification derives `{column}_probability` and
   `{column}`, so choosing both `foo` and `foo_probability` as labels yields two readouts
   named `foo_probability`. `RESERVED_TARGET_COLUMNS` does not catch this; it needs its
   own check. `target` also joins that set, defensively, for the reason `row_id` is in it.

4. **The adapter must not swallow `RunInterrupted`.** `application/engines/context.py:13-33`
   is explicit: swallowing it turns a cancelled run into one that keeps burning a GPU.
   The adapter wraps N fits, which makes it precisely where someone would catch and
   continue.

5. **Progress must be mapped, not forwarded.** `ctx.report(fraction, phase)` is progress
   within *this fit*, 0.0 to 1.0. Label *i* of N must report `(i + f) / N`, or the bar
   resets to zero N times.

6. **`Dataset.target` is deleted with no shim, but `TrainContext.target_column` is kept
   as a raising property.** This looks inconsistent and is not. A `targets[0]` shim on
   the Dataset would silently train on label #1 and drop the rest. The `TrainContext`
   property is a guarded invariant: the adapter guarantees single-label contexts for
   exactly the engines that read it, and the property raises loudly if that is ever
   violated. Keep both as specified.

7. **The Scorecard is computed on read**, from a `ScorecardInputs` blob, and is never
   stored in a table. There is no scorecard schema change. The roadmap entry implies a
   larger blast radius than actually exists here.

## Decisions already made — do not relitigate

- Fan-out rather than restricting multi-label datasets to the two joint engines. This is
  what removes the need to change `content_hash`.
- `content_hash` is **not** changed.
- A dataset **may** mix label kinds (numeric and binary together). Only the joint engines
  refuse a mixed-kind dataset, at command validation. The fan-out path handles mixed
  kinds for free, since each sub-model sees one label of one kind.
- The sweep table shows every label as its own sortable column. No averaged score, no
  worst-case score, no default "rank by" dropdown.
- `chemprop_dmpnn` first (mostly unpinning existing code), `molformer_xl` second (a new
  multi-head with a masked loss). Phase 2 is separable.
- `TrainContext.task` stays a single `TaskType`; `_task_for` becomes per-label.
- Run-level scorecard fields are duplicated across the N scorecards rather than hoisted
  into a new envelope type.

## Environment notes

- Dev DB is at migration **012**. This feature's migration is **013**.
- **The runner agent does not hot-reload.** Run `make dev-worker` after any engine or
  training change, or you will debug a fix that never loaded.
- Gates that must pass before any merge: `ruff check`, `ruff format`, `mypy` strict,
  `lint-imports` (3 contracts), and `OMP_NUM_THREADS=1 pytest`. Skipping mypy has turned
  `main` red before.
- Verify on a real run, not only fixtures. A green suite has shipped a bug here before
  that one real run caught immediately.
- Sample dataset for end-to-end verification: 324,803 rows, four binary label columns,
  at `~/Documents/Cage-Fusion-Paper/Dataset/train.csv`. Useful mainly because its size
  exercises the job-timeout multiply — four sequential fits on that many rows can cross
  the fixed 1800s `worker_job_timeout`.

## Queued behind this

A separate, already-reviewed fix on `main`, not part of this feature:

`read_compound_ids` (`application/data/compound_ids.py:37-52`) builds a whole 400k-entry
map per call, from `get_chemical_space.py:192-197` and `get_scorecard.py:107-111`. About
80 ms and 80 MB per map hover, and hovers are not debounced. Fix is to pass the wanted
keys and filter before building the dict — measured at 9 ms. The review that found it
reported no Critical findings overall; its other items are in the memory note for the
identifier-column work.

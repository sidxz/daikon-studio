# Fan-out sweeps: design

**Date:** 2026-08-05
**Status:** Approved pending user review

## Problem

A scientist choosing between engines or hyperparameters today submits one
training run, waits, reads the Scorecard, goes back to the form, and does it
again. Nothing in the product holds "these twelve attempts are one question I
am asking", so the comparison happens in a notebook, or in someone's head, or
not at all. The runs list interleaves them with every prediction anyone else
ran that afternoon.

The queue can already execute twelve trainings concurrently — N runner agents
pull one Postgres queue with `SKIP LOCKED`, and the per-workspace fairness cap
stops one user monopolising the fleet. What is missing is not throughput. It is
a name for the group and a screen that ranks it.

## What this is not

The self-hosted-runners spec's phase 2 was Temporal, orchestrating multi-step
work on trusted infra. **That stays on hold.** A sweep has no dependencies:
submit N configs, watch them together, pick a winner. Durable recovery of
*dependent* state is what Temporal sells, and nothing here is dependent.

The trigger that would revisit it is named and unchanged: *the first time step
N+1 must consume step N's output and survive a crash in between.* Two of the
decisions below were made specifically to stay on the near side of that line —
the human picks the winner, and each run pays for its own baseline. Both
alternatives would have put a cross-run dependency inside the first feature
built after the decision to defer.

The `JobEnqueuer` port remains the seam a workflow engine would plug into, so
none of this is wasted if that day comes.

## Decisions taken during brainstorming

1. **A sweep is a list of complete configs, built by the client.** The request
   carries `configs: [{engine_id, conditions}]`. Varying conditions within one
   engine and comparing two engines are the same request shape, so there is no
   server-side grid expansion to write or validate. Rejected: a
   `{key: [values]}` cartesian product, which is fewer bytes and cannot express
   "chemprop versus ECFP4" at all.
2. **The human picks the winner.** The sweep detail page ranks the runs and the
   scientist publishes one, exactly as they do today from a Scorecard. Rejected:
   auto-registering the best, which needs a declared criterion and a step that
   consumes N runs' outputs — the Temporal trigger above.
3. **Cancelling a sweep cascades** to its pending and running rows, tolerating
   any that reached a terminal status in the meantime. Individual runs stay
   individually cancellable. Rejected: a sweep as a pure label, which makes a
   twenty-run mistake twenty clicks to stop while the pending rows keep
   consuming the fairness cap.
4. **Each run pays for its own baseline.** Every child run stays self-contained
   and independently citable. This is wasteful when twenty configs share one
   baseline — negligible for ECFP4, real for a chemprop baseline — and the fix
   is the already-deferred fit-result cache keyed on
   `(content_hash, engine_id, conditions)`, not a sweep-level special case.
   Rejected: one shared baseline run the others wait on, which is the Temporal
   trigger again.
5. **A `sweep_id` column, not a `sweeps` table.** See below.
6. **A persisted per-run metric summary**, rather than reading N Scorecards.
   See below.

## Data model

Migration 010 adds two nullable columns to `runs` and nothing else.

**`sweep_id UUID NULL`**, indexed `(workspace_id, sweep_id)`. Listing sweeps is
then one `GROUP BY` over `runs`, and the cancel cascade is a filter. `NULL`
means an ordinary solo run, which is what nearly every row will be.

The alternative was a `sweeps` table with its own aggregate, repository, and
use cases, which is what this codebase's layering would normally imply. It buys
a label and a group-by for roughly four hundred lines across five layers, plus
a second aggregate inside the execution context that owns no invariant the Run
does not already own. The cost of the column is that a sweep has no rename and
no lifecycle of its own; nothing needs either.

The sweep's display name lives in each child's `params` as `sweep_name`. That is
legitimate there — it is part of what was asked for, which is exactly what
`params` holds — and `params` being write-once by convention is the same reason
rename is off the table rather than a problem to solve. The **id** does not go
in `params`, for the reason the handoff gives: a grouping that can never be
corrected, and a filter that cannot use an index.

**`metrics JSONB NULL`**, written by `RunTraining` on the same update that
already persists `result_uri` and `protocol_id`:

```json
{"primary_metric": "mcc", "value": 0.603, "baseline_value": 0.632}
```

The sweep page's entire purpose is comparing N runs, and the only per-run number
today lives inside `GET /protocols/{id}/scorecard`, which recomputes RDKit
Tanimoto similarity over train×test on every call. Fetching that N times per
page load is cheap to write and expensive forever. This column follows the
precedent `Run.protocol_id`'s own comment sets: an outcome is not an
instruction, so it needs a field that survives an `update()`. It stays `NULL`
for prediction runs and for training runs that never reached `ready`.

`primary_metric` is `"mcc"` for classification and `"rmse"` for regression,
which `build_scorecard` decides today at line 130. That choice moves into a
shared `primary_metric_for(task)` helper called from both places, so the sweep
ranking and the Scorecard can never disagree about which number is the headline.

## Backend

**`TrainProtocol.__call__` gains one optional keyword, `sweep_id`,** which it
stamps on the Run it already builds. Every other line is untouched, so a sweep
child and a solo training run are the same object produced by the same code —
same cache key, same baseline resolution, same lane.

**`SubmitSweep`** validates the entire request before creating anything: the
dataset exists and belongs to the workspace, and every config's engine and
baseline engine are in the registry. Only then does it generate one `sweep_id`
and loop `TrainProtocol`. The pre-flight is the point — a failure discovered
halfway through the loop would otherwise leave a half-submitted sweep that the
user never asked for and cannot tell apart from a complete one. Conditions stay
unvalidated here, matching `TrainProtocol`'s existing argument: an invalid
hyperparameter fails its own Run visibly, and only the engine's manifest can
resolve a condition's default.

**`CancelSweep`** loads the group and calls the existing `Run.cancel()` on each
non-terminal row, swallowing the `ConflictError` from any that finished between
the read and the write. Cancellation semantics are unchanged and unextended: a
pending run never starts, and a running one stops at its next checkpoint,
because the row is the channel.

**`RunRepository`** gains `list_by_sweep(workspace_id, sweep_id)` and a grouped
summary read for the list page returning, per sweep: id, name, dataset id,
creation time, total, and counts by status.

**Routes:** `POST /sweeps`, `GET /sweeps`, `GET /sweeps/{id}`,
`POST /sweeps/{id}/cancel`.

Request body:

```json
{
  "name": "BBBP engine comparison",
  "dataset_id": "…",
  "baseline_engine_id": "ecfp4-randomforest",
  "baseline_conditions": {},
  "configs": [{"engine_id": "chemprop-dmpnn", "conditions": {"depth": 4}}]
}
```

`dataset_id` and the baseline are declared once at sweep level: configs measured
against different baselines, or trained on different data, are not a comparison.
Each child Protocol is named `{name} #{i+1}` — the index never collides, and the
config itself is visible on the row.

**The queue, the runner protocol, and the runner agents learn nothing about
sweeps.** `_CLAIM` sees an ordinary pending row with a lane, and no runner-facing
endpoint gains a field. That machinery is inside the adversarially-reviewed
security boundary and every line added there is a line to re-review.

## Frontend

A `features/sweeps/` folder mirroring `features/runs/`, and one nav entry under
Curate, where training already lives.

**Submit form.** Dataset, name, and baseline once; then a list of config rows,
each an engine select plus its conditions. It reuses the engine picker and
`ConditionFields` from `train-protocol-form.tsx` rather than reimplementing
condition rendering, which is the bulk of that form.

**Detail page.** A ranked table: config index, engine, status and progress,
primary metric, delta versus that run's own baseline, and a link to the
Scorecard. Sorting is direction-aware — MCC ranks high-first, RMSE low-first —
which is why `primary_metric` is stored rather than inferred from the metric
dict's keys. Unfinished runs sort last rather than as zero.

The page states that a workspace runs ten at a time. A twenty-run sweep does not
run twenty-wide, and without saying so the throttle reads as the product being
stuck instead of the fairness predicate doing its job.

## Testing

**Backend.** `SubmitSweep` creates N runs sharing one `sweep_id`; an unknown
engine in the last config creates *no* runs; `CancelSweep` cancels the
non-terminal rows and tolerates a concurrently-finished one; the grouped summary
returns correct per-status counts; `RunTraining` writes `metrics` on success.
Any test touching `runs` rows follows the existing rule about cleanup — the
session-scoped NullPool engine has no per-test rollback.

**Frontend.** A unit test for direction-aware ranking, including unfinished runs
and a run whose primary metric is undefined. Component coverage for the config
list matching the existing `train-protocol-form.test.tsx` pattern.

## Deferred, deliberately

- No auto-registration of a winner, no shared baseline, no grid expansion, no
  sweep rename, no sweep-level retry.
- The fit-result cache that would make decision 4 cheap is a pre-existing
  deferred item and stays one.
- Per-lane job timeouts still do not exist; a sweep of chemprop configs is
  bounded by the same single server-wide deadline as everything else.

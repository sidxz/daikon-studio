# Future-Seams Alignment — Design

**Date:** 2026-07-30
**Status:** Approved
**Scope:** Keep four future capabilities — generation runs, GPU/long jobs (and an eventual Temporal migration), out-of-process engines, queue priorities — landable without rewrites. Build almost nothing now: one bug fix, one tripwire test, and this document as the binding rules for future code.

## Context

Phase 1 shipped with deliberate seams: `_HANDLERS` in `worker.py` for new run kinds, `JobEnqueuer` as the queue boundary, an `EngineManifest` free of infrastructure imports, a structural `Engine` protocol. An audit (2026-07-30) confirmed every seam is intact and every `RunKind` consumer fails closed. Two things came out of it:

1. One piece of code actively contradicts "long jobs without rewrites": the stuck-RUNNING redelivery gap documented at `worker.py:127`. arq is at-least-once; after a worker crash the job redelivers onto a row already in RUNNING, `Run.start()` raises outside the try/except, and the run is stuck RUNNING forever. With 30-minute CPU jobs the crash window is small. On a GPU box running multi-hour jobs, worker death (OOM, CUDA errors) is the common case, not the corner case.
2. The claim that manifests "serialize to HTTP unchanged when engines move out of process" lives only in a docstring. Nothing fails if someone adds a non-serializable field.

Nearest concrete need (from the user): a cage_fusion/chemprop engine running as a **dedicated worker on a GPU box pulling a dedicated queue** — same codebase, out-of-band placement. Nothing else is scheduled.

## What we build now

### 1. Redelivery fix

- `Run.start()` (`domain/execution/run.py`) additionally allows RUNNING → RUNNING. An at-least-once queue redelivering after a crash is a legitimate restart; with no checkpoints, restart-from-zero is the designed behavior. Progress resets with the restart. Terminal states (READY/FAILED/CANCELLED) still raise `ConflictError`.
- `run_job` (`infrastructure/worker.py`) treats `ConflictError` from `start()` as "run went terminal while queued" (e.g. cancelled): drop the job — return without re-raising so arq stops retrying, and without touching the row. Replaces the known-gap comment at `worker.py:125-135`.
- Tests: redelivery onto a RUNNING row completes the run; delivery onto a CANCELLED row is a no-op that leaves the row CANCELLED.

### 2. Manifest tripwire test

One unit test: every manifest in `default_registry()` survives `json.loads(json.dumps(dataclasses.asdict(m)))`. Any future manifest field that isn't plain data (an infrastructure object, a callable, a non-serializable type) fails the suite. This mechanically enforces the Phase 5 exit.

### 3. This document

The seam rules below are binding on future work. They are the "alignment" — the point is that everything else stays cheap *because* these rules hold.

## Seam rules

### Generation runs (Phase 4)

**Seam:** `_HANDLERS` in `worker.py` — a new `RunKind`, not a wider `Engine`.

**Touch-list when it lands:** `RunKind` member; `_HANDLERS` entry; one Alembic migration replacing the `ck_runs_kind` CHECK constraint; a new endpoint (kind stays implied by endpoint, matching train/predict); orval regen for the frontend enum plus the hand-written union in `frontend/src/features/runs/hooks/use-runs.ts`.

**Never:**
- Never widen `Engine`, `TrainContext`/`TrainResult`, or `TaskType` to express generation. A generator has no input rows and no per-row float; the supervised contract has no shape for it.
- Never route a generation run through `RunTraining`'s scorecard machinery (baseline fit, optimism gap, `ScorecardInputs`). A scorecard built from empty actual/predicted arrays lies by construction. Generation gets its own honesty story (novelty, synthesizability, distance from training data) when it exists.
- The existing `is not RunKind.PREDICTION` guards (`predict_with_protocol.py`, `create_collection.py`) stay. A generation handler writes structures through its **own** Collection construction site with its own `GenerationMethod` value — it does not reuse the prediction results parquet path.

### GPU worker / long jobs (Phase 2)

**Seam:** arq named queues behind the existing `JobEnqueuer` protocol.

**Touch-list when it lands:** a second `WorkerSettings` with its own `queue_name` and `job_timeout` (timeouts are per-worker-class, not per-job); `ArqEnqueuer.enqueue` gains a queue argument at that point, routed from an optional field added to `EngineManifest` then (with a default, so existing engines are untouched); the mandatory random-split comparison becomes opt-out for expensive engines (already flagged at `train_protocol.py:37-39`).

**Never:** never bolt checkpoint/resume onto arq. The moment a job needs to survive worker death *without* restarting from zero, that is the Temporal trigger — not a longer timeout, not a resume hack.

### Temporal migration (when triggered)

**Seam:** the `JobEnqueuer` protocol (`application/execution/enqueue.py`).

**The load-bearing invariant: the queue message carries only `run_id`. All job state lives on the Run row.** Any orchestrator — arq today, Temporal later — can be swapped behind `JobEnqueuer` precisely because the message is a bare UUID and the worker rebuilds everything from Postgres. Never put params, engine ids, dataset refs, or any payload into the queue message. A queue argument (Phase 2, above) is routing metadata, not state, and is the one permitted addition.

### Out-of-process / third-party engines (Phase 5)

**Seam:** `EngineManifest` as the HTTP envelope; the registry keyed by `manifest().id`.

**Rules:**
- The manifest stays free of infrastructure imports. The tripwire test enforces this; if the test fails, fix the field, not the test.
- `Engine.train`/`predict` stay synchronous. A remote engine's adapter makes its HTTP calls inside sync `train`/`predict` on the worker thread — the contract does not go async to accommodate it.
- `TrainResult.artifact` stays `bytes` until a real checkpoint measurably hurts (memory pressure or predict-path latency). The upgrade then is bytes → URI into the blob store, touching only the engines and two call sites (`train_protocol.py` write, `predict_with_protocol.py` read). Do not pre-build streaming.
- Registration stays the in-tree `_ENGINES` tuple until a second team actually exists; the smallest honest step then is extending `_ENGINES` from entry points — a few lines, per the registry docstring.

## Out of scope, deliberately

Queue routing plumbing, manifest resource fields, the kind CHECK-constraint migration, artifact streaming, a staleness reaper, per-user fairness. Each is a sub-hour change at the time of need, and building it early means guessing topology (which queue names? which resource axes?) before the hardware or the second team exists.

## Testing

- Two redelivery tests (RUNNING restart, terminal drop) alongside the existing worker/run tests.
- One manifest round-trip test alongside the engine contract tests.
- Full existing suite stays green — the `start()` relaxation must not break any current transition test.

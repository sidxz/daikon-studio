# Retry a failed run

**Date:** 2026-08-04
**Status:** approved, not implemented
**Follows:** `docs/superpowers/HANDOFF-chemprop-and-baselines.md` §4.2
**Sibling spec:** `2026-08-04-choosable-baseline-and-pretrained-weights-design.md`

## What "re-run" means here

The handoff names three candidate meanings and says a fresh session must settle which.
Settled: **re-execute the same Run**. Not a new protocol, not a new protocol version.

The two rejected readings, and why:

- **A prefilled "train another"** deep link into `/protocols/new` is a different feature —
  it exists to act on a bad verdict, not to recover from a failure. It stays on the
  deferred list (handoff §5).
- **`InSilicoProtocol.new_version()`** copies `dataset_id`, `engine_id` and `conditions`
  from a *published* parent, so retraining through it on a frozen dataset reproduces the
  same answer. It earns its keep only when a version may change its inputs, which is not
  what "the run failed, run it again" asks for. It keeps its zero production callers.

## Why this is cheap

`Run.params` is write-once by convention — `SqlAlchemyRunRepository.update()` never
persists it (`domain/execution/run.py:88-93`, enforced at
`infrastructure/persistence/sqlalchemy/execution/repository.py:108-119`). A retried run
therefore re-executes its original instructions *by construction*. There is nothing to
copy, rebuild, or keep in sync. That single property is what makes retry-in-place a
smaller change than creating a new Run.

## The domain change

`Run`'s status is a one-way lattice and `_TERMINAL` gates every mutator uniformly
(`run.py:43`), so there is no path out of a terminal status today. This adds exactly one
method:

```python
def retry(self) -> None:
    """`failed -> pending` and `cancelled -> pending`.

    The only edge that leaves _TERMINAL, and deliberately narrow. `params` is
    write-once, so the re-enqueued job re-reads the same instructions -- there is
    nothing to rebuild. Clearing `error_message` matters: a stale one would render
    on a run that is queued again and has not failed this time.

    `running` is excluded on purpose. A crashed worker leaves a run RUNNING with no
    error recorded, and permitting a retry from there would start a second fit
    beside one that may still be alive -- the failure mode the handoff warns about,
    where arq's timeout cannot kill an OS thread. Cancel first: that flips the row,
    which is the channel the live worker checkpoints against, and then retry.
    """
    if self.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
        raise ConflictError(f"Cannot retry run '{self.id}' in status '{self.status}'")
    self.status = RunStatus.PENDING
    self.progress = 0.0
    self.phase = None
    self.error_message = None
    self._touch()
```

Allowing `cancelled` is what makes a worker-crash recoverable without a staleness reaper.
The run is stuck RUNNING with a frozen progress bar; the user cancels it — already legal
from RUNNING (`run.py:177-196`) — and then retries. The reaper stays deferred.

## The use case and endpoint

`POST /api/v1/runs/{run_id}/retry` → **204**, mirroring `/cancel` at
`interface/routes/runs.py:253`. A `RetryRun` use case in `application/execution/`:
authorise as editor, load the run, call `retry()`, `update()`, then re-enqueue on the
correct lane.

**Resolving the lane** is the only non-obvious step, because the enqueue side differs by
kind and neither path stores the lane on the row:

- **Training** — the engine id is in `run.params["engine_id"]`; the lane is
  `registry.get(engine_id).manifest().lane`. Once the sibling spec lands, it is the union
  of the chosen and baseline manifests' lanes, and the retry path must use the *same*
  resolution function, not a second copy of the rule. Extracting that into one helper both
  call sites share is part of this work.
- **Prediction** — resolved from the protocol, exactly as
  `predict_with_protocol.py:218` already does:
  `self._engines.get(protocol.engine_id).manifest().lane`.

Ordering is update-then-enqueue. Enqueueing first would let a worker pick up a run whose
row still reads FAILED, and `run_job` drops redeliveries for terminal runs
(`worker.py:138-141`), so the retry would vanish silently.

## Frontend

`run-detail.tsx` gains a **Retry** button beside the existing failure block (`:112-117`),
shown when status is `failed` or `cancelled`. It sits next to the error message, which is
where a user looking at a failure already is. A `useRetryRun` mutation follows
`useCancelRun` (`use-runs.ts:80-90`) — same shape, same toast pattern — and invalidates
the run query so the existing poll (`RUN_POLL_MS`, 2s) resumes on its own once the status
goes back to `pending`.

No confirm dialog, matching Cancel. Retrying is not destructive and not expensive to undo.

The training form (`train-protocol-form.tsx`) is left alone: it toasts and resets on
failure, and the run remains reachable at `/runs/{id}`. Adding a second retry affordance
there is duplicated surface for the same action.

## Testing

| Level | What it locks |
|---|---|
| Unit, `Run` | `failed → pending` and `cancelled → pending` succeed and clear `error_message`, `progress` and `phase`. |
| Unit, `Run` | `pending`, `running` and `ready` all raise `ConflictError`. The `running` case is the important one — it is the guard against a double fit. |
| Integration | A retried training run is re-enqueued on its engine's lane, and on the union lane when a baseline pulls it to `gpu`. |
| Integration | The retried run re-executes the original `params`; the row's `cache_key` is unchanged, because the inputs are. |
| Frontend | Retry renders for `failed` and `cancelled`, and for no other status. |

## Accepted risks

- **A stale arq redelivery can race a retry.** The original job may still be queued when
  the row returns to PENDING; both it and the retry-enqueued job would then find a
  non-terminal run and `start()` it. This is the existing at-least-once hazard the
  redelivery design already lives with (`run.py:133-147`), not one this feature
  introduces, and the outcome is a duplicated fit rather than a corrupted result. Worth a
  `ponytail:` comment naming it; worth a job-id-per-attempt fix only if it is ever
  observed.
- **Cancel-then-retry on an engine that never calls `ctx.report`** — both ECFP4 engines —
  stops the old fit only when it returns, so the two can briefly overlap. Bounded by the
  fit's own duration, and user-initiated rather than automatic.
- **Nothing reaps a run left RUNNING by a crash.** Retry makes it *recoverable* by hand;
  it does not make it self-healing. The staleness reaper stays on the deferred list.

# Resumable training runs

**Status:** design approved in conversation 2026-10-03; this document awaits review.
**Branch:** `resumable-runs`, cut from `training-options`.

## Why

A CheMeleon chemprop run on the 10k nuisance sample hit the 1800 s job limit at 87.5% and lost everything:
- training runs in a temporary directory with Lightning checkpointing off;
- the three fits (model, baseline, random-split comparison) keep their results only in memory;
- scores are computed only at the very end.

On the full 324k-row dataset a single neural fit takes hours. A training run that stops must continue from where it stopped, not start over. It can stop on its time limit, on a runner crash or restart, or on a cancel.

## Decisions (taken with the user)

| Question | Decision |
|---|---|
| Granularity | Completed fits **and** in-progress neural training state |
| Time limit hit | The run stops and fails as today. Progress is saved and the person clicks **Resume**. No automatic resume: the limit still stops runaway or hung runs. |
| Retry button | Training runs show **Resume** (primary) and **Start over** (discards saved progress), on the run page and on failed or cancelled sweep rows |
| Retention | Saved progress is deleted when the run succeeds, when the person starts over, or when its dataset is deleted. No age-based purge. |

## What already exists (and stays)

- **Retry in place.** `RetryRun` and `Run.retry()` move a failed or cancelled run back to pending, keeping the same row, params and cache key. The queue resets `attempts`.
- **Runner-death recovery.** The lease sweep (`SqlAlchemyRunQueue.sweep`) requeues a run whose runner stopped responding, up to `runner_max_attempts`.
- **Workspace-confined blob access.** A runner reads and writes blobs only inside its run's workspace, through the runner API: `GET/PUT /runner/runs/{id}/blobs/{key}`. Writes require holding the run's active claim (`ClaimedRunWrite`), so a runner whose lease expired cannot write.
- **Shared Lightning helpers.** chemprop and MoLFormer both train through Lightning and share `infrastructure/engines/_lightning.py` (`keep_best_by_validation_loss`).
- **Verified on Lightning 2.6.5 (the installed version).** A checkpoint saved from a callback's `on_train_epoch_end`, then passed as `trainer.fit(..., ckpt_path=...)`, resumes at the next epoch: a fit stopped after epoch 2 of 6 ran exactly epochs 3, 4 and 5. A callback's `state_dict()` / `load_state_dict()` round-trips through it.

## Design

### 1. A checkpoint store for a run

A new application-layer class wraps the existing `BlobStore` port. There is one store per training run, rooted at:

```
{workspace}/datasets/{dataset_id}/runs/{run_id}/checkpoints/
```

The root is under the dataset folder **on purpose**: `DeleteDataset` already calls `delete_prefix(dataset_folder)`, so deleting a dataset removes its runs' saved progress with no new code.

Interface (names final in the plan):

- `load(name) -> bytes | None`
- `save(name, data: bytes) -> None`
- `scoped(prefix) -> store`, a sub-store under `{root}/{prefix}/`
- `clear() -> None`, which deletes the whole root
- `interval_seconds`: how often in-progress training saves (§3)

**Integrity.** `save` writes the data blob first, then a small marker `{name}.json` holding:
- the fingerprint;
- the byte length;
- the sha256 of the data.

`load` returns `None` in four cases:
- the marker is missing;
- the marker's fingerprint differs from the current one;
- the length or sha256 differs, for example after a write interrupted mid-file;
- the store raises.

**Fingerprint.**
- **For a fit result:** the engine id, the engine manifest `version`, and a result-format version constant.
- **For in-progress neural state:** the same, plus the installed `torch`, `lightning` and engine-library (`chemprop` / `transformers`) versions.

A mismatch is treated as "nothing saved", and the next save overwrites it.

> Ceiling: a code change that keeps the engine's version string and its tensor shapes is not detected. Bump the engine's manifest `version` when training semantics change. A load that fails outright (shape mismatch, unpickling error) is caught and treated as "nothing saved" (§5).

**Best effort.** A failed `save` never fails the run. Causes include the network, the runner upload cap (HTTP 413) and a lost claim. The failure is logged, and resume then falls back to the previous save.

### 2. Completed fits are saved and skipped

`RunTraining` gives each of its three fits its own scope: `model`, `baseline` and `random-split`. `baseline_is_self` reuses the model fit as it does today.

**Before a fit:** if its scope holds a saved result, load it and skip the fit. The progress phase reads "Restored the {engine} fit from saved progress".

**After a fit:** save its `TrainResult`, packed into one blob:
- a JSON header with `metrics`, `validation_metrics`, `cutoffs` and the artifact length;
- the artifact bytes.

`TrainContext` gains `checkpoints: <store> | None = None`, and the fit receives its scope. `None` means "no saving". That covers tests, prediction and every caller that has no run.

**`FanOut`** does the same per target. Each target's sub-fit uses scope `target-{column}` inside the stage's scope. Its result is saved when it completes and skipped on resume. A four-target random forest on the full dataset therefore resumes target by target.

**On success**, after the Protocol row exists, `RunTraining` calls `clear()`. Cleanup is best effort; a failed cleanup is logged.

### 3. Neural fits save their training state

A new shared callback in `_lightning.py`, used by chemprop and MoLFormer alongside `keep_best_by_validation_loss`:
- **Periodic save.** At an epoch's end (`on_train_epoch_end`), when `interval_seconds` have passed since the fit began or since the last save, it calls `trainer.save_checkpoint` to a temp file and passes the bytes to `save("fit")`.
- **Save at a time-limit stop.** In `on_exception`, when the exception is `RunInterrupted` with `cancelled=False`, it saves once more. Both engines call `ctx.report` only at an epoch's end, so this save also lands on an epoch boundary.
- **No save on cancel** (`cancelled=True`). The run row is already cancelled and the runner API refuses the write. Resume after a cancel continues from the last periodic save, at most `interval_seconds` old.

**`keep_best_by_validation_loss`** gains `state_dict()` and `load_state_dict()` covering `best_loss` and `best_state`. Best-epoch selection then survives a resume.

**On resume**, before `trainer.fit`, the engine:
1. loads `fit` from its scope;
2. writes it to a temp file;
3. passes `ckpt_path`;
4. reports the phase "Resuming {engine} from epoch {n} of {epochs}".

Everything derived from the data is recomputed deterministically, exactly as in the first attempt:
- the split;
- descriptor fill and scaling;
- positive weights;
- the model's construction.

> Not bit-identical: data-loader shuffling after a resume differs from an uninterrupted run. chemprop on MPS is not run-to-run reproducible anyway, so this changes nothing a user can rely on.

**Interval.** A runner setting, `STUDIO_CHECKPOINT_INTERVAL_SECONDS`, defaults to 600. It is read where the runner builds `RunTraining` and carried on the store.

**Expected save sizes:**

| Engine | Weights + optimizer state + best-epoch weights |
|---|---|
| Plain chemprop | ~5 MB |
| CheMeleon | ~150 MB |
| MoLFormer-XL | up to ~750 MB, under the 1 GiB runner upload cap |

A save above the cap fails like any other: it is logged, and resume falls back to the fit level.

**Tree engines and the GP** save nothing mid-fit. Their unit of saved work is the completed fit (§2).

### 4. Resume, Start over, and recovery

- **`RetryRunCommand.fresh: bool = False`.**
  - When `fresh` is set, `RetryRun` deletes the run's checkpoint root, via `BlobStore.delete_prefix` on the API side, before requeueing.
  - Without `fresh`, the requeued run resumes.
  - Prediction runs ignore the flag; they keep no saved progress.
- **HTTP.** `POST /runs/{id}/retry` accepts an optional JSON body `{"fresh": bool}`. No body means `false`, so existing clients are unchanged.
- **Runner-side cleanup.** The store's `clear()` calls `BlobStore.delete_prefix(root)`. `HttpBlobStore.delete_prefix` is implemented through one new runner route, `DELETE /runner/runs/{id}/checkpoints`. It deletes exactly that run's checkpoint root and nothing else, and requires the active claim. The client refuses, before any request, a prefix that is not that root, and the server derives the root itself from the run, never from the request.
- **UI, run page.** For a failed or cancelled training run, the Retry button becomes **Resume** (primary), next to **Start over**. Prediction runs keep **Retry**.
- **UI, sweep table.** A failed or cancelled row shows a **Resume** action.
- **Time-limit message.** When the run had a checkpoint store, the failure message adds:
  > Its progress is saved: Resume continues from where it stopped.

  The message names both settings: `STUDIO_WORKER_JOB_TIMEOUT` and `STUDIO_WORKER_JOB_TIMEOUT_BY_LANE`.
- **Runner crash.** The lease sweep requeues the run as today. The next claim finds the saved progress and resumes, with no new code on that path.

### 5. Failure handling

| Situation | Behavior |
|---|---|
| A save fails (network, 413, claim lost) | Logged. The run continues. Resume uses the previous save. |
| A saved item fails integrity or fingerprint checks | Treated as nothing saved, and overwritten by the next save |
| A saved item passes the checks but fails to load (shape mismatch, unpickling error) | Logged and discarded. That piece starts over, and the phase says so. |
| Two runners on one run | Impossible to write. A runner without the active claim is refused by `ClaimedRunWrite`. |
| Start over while saved progress exists | Root deleted before the requeue |
| Dataset deleted | Root deleted with the dataset folder |

## Testing

- **Store.**
  - Save then load round-trips.
  - A fingerprint mismatch, a truncated blob and a missing marker each load as `None`.
  - Scoping nests keys correctly.
  - `clear()` removes the root.
  - A failing `save` does not raise.
- **`RunTraining`.** A fake engine counts its fits.
  - With saved `model` and `baseline` results, a run performs only the random-split fit, and its Scorecard equals an uninterrupted run's.
  - `clear()` runs on success.
- **`FanOut`.** With one of two targets saved, only the other target is fitted.
- **chemprop (real, tiny).**
  - A `report` raising `RunInterrupted(cancelled=False)` after epoch 2 leaves a saved `fit`.
  - A second `train` with the same store resumes at epoch 3. Spy on the epochs that run.
  - `keep_best`'s best loss carries over.
  - The result predicts.
- **MoLFormer.** The same resume test, if it fits the existing tiny-model fixtures. Otherwise one test that the callback is wired.
- **Retry.**
  - `fresh=true` deletes the root.
  - `fresh=false` keeps it.
  - The HTTP body is optional.
- **Runner route.** `DELETE /runner/runs/{id}/checkpoints` deletes only that root, and refuses without the claim.
- **Frontend.**
  - Resume and Start over on a failed training run.
  - Retry on a failed prediction run.
  - Resume on a failed sweep row.
- **Live check.**
  1. Set the GPU lane limit to 300 s.
  2. Start a CheMeleon run and let the limit stop it.
  3. Click Resume.
  4. It finishes without repeating completed fits or epochs, and the phase text says what was restored.

## Out of scope

- Prediction-run checkpoints.
- Saves in the middle of an epoch.
- Automatic resume after a time limit.
- Purging saved progress by age.
- Resuming one sweep row from another row's progress.

# Remote Engines & Chemprop — Design

**Date:** 2026-07-30
**Status:** Approved
**Scope:** Run an engine where its hardware is, without any code knowing where that is. Adds Chemprop D-MPNN as the first such engine, arq lane routing, a cooperative progress/cancellation channel, network-addressable blob storage, and container images.

## Thesis

**A daikon-studio worker is stateless and location-independent.** Four environment
variables are required; every other setting introduced below has a working default.

| Variable | What it is |
|---|---|
| `STUDIO_DATABASE_URL` | Postgres, anywhere reachable |
| `STUDIO_REDIS_URL` | Valkey/Redis, anywhere reachable |
| `STUDIO_BLOB_BASE_URL` | object storage (`s3://`, `abfs://`, `file://`) |
| `STUDIO_WORKER_LANE` | which queue this process pulls |

No volumes. No shared filesystem. No inbound network. Anything that can satisfy those
four can host a worker: a Docker Swarm node, an AKS pod, a spot VM, a laptop. Placement
is deployment configuration, not code. Nothing in this repository names a host.

Everything below follows from that sentence.

## Context

Phase 1 shipped two CPU engines (ECFP4 + RandomForest, ECFP4 + XGBoost) fitting in
seconds inside a single arq worker on the developer's machine. The
[future-seams alignment](2026-07-30-future-seams-alignment-design.md) named the next need
precisely: *a chemprop engine running as a dedicated worker on a GPU box pulling a
dedicated queue — same codebase, out-of-band placement.* This design is that work.

The concrete deployment is `atlantic`, a GPU node in the `ned` Docker Swarm. That is one
example of the contract above, not a target the code knows about. Some organisations will
have a multi-node GPU cluster; that requires no extra code, because N workers on N nodes
all pull the same lane's queue and Redis distributes between them.

## The finding that shapes this design

At Phase 1's job durations arq's timeout behaviour is a curiosity. At GPU durations it is
the default failure mode, and it is worth stating exactly because three decisions below
exist only to prevent it.

`arq/worker.py:599` runs a job as `await asyncio.wait_for(task, timeout_s)`. On timeout it
cancels the *coroutine*. `RunTraining` calls engines through `asyncio.to_thread`, and
**Python cannot kill a thread** — so the fit keeps running and keeps holding the GPU.
From there:

1. `run_job` catches `(Exception, SystemExit)`. `asyncio.CancelledError` derives from
   `BaseException`, so it passes straight through and `run.fail()` never fires. The row
   is left `RUNNING`.
2. arq defaults to `retry_jobs=True`, `max_tries=5`, and `worker.py:624` treats
   `CancelledError` as *"cancelled, will be run again"*. It re-queues.
3. The redelivery reaches `Run.start()`, which the future-seams work relaxed to permit
   `RUNNING → RUNNING`, so it restarts — while the first thread is still training.

One mis-set timeout therefore yields up to five concurrent, unkillable training threads on
one GPU and a Run stuck `RUNNING` forever.

**The fix is not a larger timeout.** The only mechanism in Python that can stop that thread
is cooperative: the engine must periodically call back into code that can raise inside the
training thread. This is also true under Temporal, whose activity cancellation is likewise
heartbeat-driven — so this is not a limitation arq imposes, and swapping orchestrators
would not avoid it.

## What we build

### 1. Lanes — routing, not orchestration

An engine declares *what it needs*; a deployment decides *where that runs*.

```python
# application/engines/manifest.py
@dataclass(frozen=True, kw_only=True)
class EngineManifest:
    ...
    lane: str = "default"
```

```python
# infrastructure/worker.py
def queue_for(lane: str) -> str:
    """arq's own default queue for the default lane, so this change needs no
    drain-and-migrate of jobs already queued under the old name."""
    return default_queue_name if lane == "default" else f"{default_queue_name}:{lane}"
```

- `JobEnqueuer.enqueue(run_id, lane="default")` — the port gains one keyword argument.
  This is the single addition the future-seams document permits to the queue boundary:
  routing metadata, never state. **The queue message stays a bare `run_id`.**
- `ArqEnqueuer` passes `_queue_name=queue_for(lane)` to `enqueue_job`. `InlineEnqueuer`
  ignores the lane, because in-process execution has no queue.
- `TrainProtocol` already resolves the engine to reject unknown ids, so it reads
  `manifest.lane` from the instance it is holding.
- `PredictWithProtocol` gains an `EngineRegistry` collaborator to map
  `protocol.engine_id` to its lane. One constructor parameter, one container line.
- `WorkerSettings` derives `queue_name` from `STUDIO_WORKER_LANE` and `max_jobs` from
  `STUDIO_WORKER_MAX_JOBS` (default 10). Its `job_timeout` is **derived, not configured** —
  see §2, where `STUDIO_WORKER_JOB_TIMEOUT` is the soft deadline and arq's hard timeout is
  that plus a margin.

Deployment consequences, all configuration:

- A GPU worker runs with `STUDIO_WORKER_LANE=gpu` and `STUDIO_WORKER_MAX_JOBS=1`.
  Without the latter, arq's default of 10 concurrent jobs exhausts GPU memory.
- Multiple GPUs on one node: one worker process per GPU, each with
  `CUDA_VISIBLE_DEVICES=i` and `max_jobs=1`, all on the same lane.
- Multiple GPU nodes: more workers on the same lane. Redis is the distributor.
- No `gpu`-lane worker anywhere means chemprop runs sit `PENDING`. Visible and
  diagnosable, which is the correct failure for a missing deployment.

**A training run is never split across lanes.** `RunTraining` performs the chosen fit,
the mandatory baseline, and the optimism-gap re-fit in one job. The three must see a
bit-identical frame and split or the comparison means nothing, and shipping the frame to
a CPU worker to save the baseline's few seconds would be a net loss.

### 2. Cooperative progress and cancellation

`TrainContext` gains one field. `ProgressReporter`, the `_no_op` default, and the
`RunInterrupted` exception all live in `application/engines/context.py` beside it — the
engine contract is the only thing that needs them, and putting `RunInterrupted` in
`domain/shared/errors.py` would imply a domain rule it does not express (it is a control
signal from the worker to the engine, not a violated invariant).

```python
report: ProgressReporter = _no_op   # Callable[[float, str], None]
```

The contract, which the engine protocol docstring will state:

- An engine **should** call `ctx.report(fraction, phase)` periodically during long work.
  `fraction` is progress *within this fit*, 0.0 to 1.0; the worker maps it onto the run.
- `report` **may raise `RunInterrupted`**. An engine must not catch it.
- Ignoring `report` entirely is legal. Such an engine is simply not interruptible — which
  is the honest description of both ECFP4 engines, where a single `.fit()` call offers no
  yield point.

`RunTraining` builds the reporter. It executes on the worker thread inside
`asyncio.to_thread`, so it marshals to the event loop with
`asyncio.run_coroutine_threadsafe` and blocks on the result. Each call, in order:

1. **Deadline check**, from `time.monotonic()`. Free, so it runs on every call. Exceeding
   `STUDIO_WORKER_JOB_TIMEOUT` raises `RunInterrupted("deadline exceeded")`.
2. **Throttle**, to at most one database round-trip every 10 seconds. Cancellation
   latency is therefore bounded by that interval, which is immaterial against a fit
   measured in minutes.
3. **Re-read the Run's status.** If it went `CANCELLED`, raise `RunInterrupted`.
   Otherwise write progress and phase.

`run_job` catches `RunInterrupted` ahead of its existing handler and, critically, does
**not** re-raise — a re-raise makes arq retry work that was deliberately stopped:

- cancelled: the row is already `CANCELLED` (that is why the reporter raised), so return
  without touching it.
- deadline: `run.fail("exceeded the configured deadline …")` and return.

`run_job` also re-reads the Run immediately before `succeed()`, so a cancellation landing
inside the final throttle window is not overwritten with `READY`.

arq's own `job_timeout` is set to the soft deadline plus a ten-minute margin. It becomes
a backstop that should never fire, rather than the primary mechanism — which is the whole
point, given what firing costs.

This resolves the standing `ponytail:` comment on `Run.cancel()`, which admits it "only
flips the row's status; it does not signal the worker."

### 3. Chemprop D-MPNN engine

`infrastructure/engines/chemprop_dmpnn.py`, a third implementation of the existing
structural `Engine` protocol. No protocol changes beyond `report` above.

```python
EngineManifest(
    id="chemprop-dmpnn",
    version="1.0.0",
    name="Chemprop D-MPNN",
    description=(
        "A directed message-passing neural network over the molecular graph. Learns its "
        "own representation instead of using a fixed fingerprint. Trains on a GPU and "
        "takes minutes to hours depending on dataset size."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(epochs, depth, message_hidden_dim, batch_size),
    lane="gpu",
)
```

Four conditions, deliberately: `epochs` (50), `depth` (3), `message_hidden_dim` (300),
`batch_size` (64). Chemprop exposes dozens; these are the ones that change the answer.

**`torch` and `chemprop` are imported inside `train` and `predict`, never at module
scope.** This is what lets the API tier and the default-lane worker install without CUDA
while still serving the manifest, validating conditions and rendering the engine picker.
The registry instantiates the class at import time, so the constructor must not import
them either. A missing installation raises a `ValidationError` naming the `gpu` extra,
rather than a bare `ModuleNotFoundError`.

Training:

1. `lightning.pytorch.seed_everything(ctx.seed, workers=True)`.
2. Build `MoleculeDatapoint`s from the `train`/`validation`/`test` partitions of
   `ctx.frame`. Validation rows drive early stopping; an empty validation partition
   simply means no validation loader.
3. For regression, normalise targets on the training set and attach the resulting
   `UnscaleTransform` to the predictor, so predictions return in the target's own unit.
   The Scorecard compares them against `actual` in real units.
4. `Trainer(accelerator="auto", devices=1, enable_checkpointing=False, logger=False,
   enable_progress_bar=False, callbacks=[...])`. `accelerator="auto"` means the same code
   runs on a laptop CPU, slowly, which is what makes it testable without a GPU.
5. A Lightning callback calls `ctx.report(epoch / max_epochs, "training chemprop-dmpnn")`
   on `on_train_epoch_end`. This is the interruption point that makes a GPU fit stoppable.
6. Score on the test partition through the shared metric helpers below.
7. `trainer.save_checkpoint()` into a scratch directory, read back as bytes.

Prediction writes the artifact bytes to a scratch file, loads with
`MPNN.load_from_checkpoint`, and returns `row_id`/`value`/`uncertainty` with the same
explicit `Int64`/`Float64`/`Float64` dtypes `_predict_with_tree_ensemble` uses — an
all-null `uncertainty` column would otherwise infer as polars' `Null` dtype and make the
engines' outputs schema-incompatible. Classification returns P(class=1), matching the
existing contract. Uncertainty is null: chemprop can produce it via an MVE head or an
ensemble, and a fabricated number is worse than an admitted absent one.

**Reproducibility is honest, not exact.** `seed_everything` fixes Python, NumPy and torch
seeds, but GPU kernels are not bit-deterministic. The optimism-gap comparison assumes the
split strategy is the only variable between its two numbers; for this engine there is
also kernel-level noise. The comparison stays directionally valid and the engine's
docstring says so.

### 4. Shared metrics

`_scoring.py`'s `_score` currently couples ECFP4 featurisation, sklearn prediction and
metric computation. Chemprop must report metrics under *identical* names and definitions
or the Scorecard's central claim — this model versus the mandatory baseline — compares
unlike things.

Two pure functions are extracted, and `_score` becomes a thin caller so existing
behaviour is bit-identical:

```python
def regression_metrics(y_true, predicted) -> dict[str, float]                    # rmse, mae, r2
def classification_metrics(y_true, labels, probabilities, *,
                           train_has_both_classes: bool) -> dict[str, float]      # mcc, balanced_accuracy, auroc, auprc
```

The single-class guard generalises from sklearn's `model.classes_` to a plain boolean, so
an engine with no `classes_` attribute can supply it. The rule that plain accuracy is
never computed, not even as an unused local, is preserved.

### 5. Blob storage — swappable, and copied deliberately

`FsspecBlobStore` already anticipates this ("file:// in dev, s3:// in prod"). The change
is a dependency and a setting, not code:

```toml
[project.optional-dependencies]
s3    = ["s3fs>=2024.10"]      # MinIO, AWS S3, any S3-compatible endpoint
azure = ["adlfs>=2024.7"]      # Azure Blob Storage
gpu   = ["chemprop>=2.2,<3", "torch>=2.6"]
```

`Settings` gains `blob_storage_options: dict[str, Any] = {}`, forwarded to
`fsspec.core.url_to_fs`. This carries `endpoint_url` for MinIO, `key`/`secret` for S3, or
`account_name`/`connection_string` for Azure, from one JSON environment variable — rather
than depending on whether a particular botocore version honours `AWS_ENDPOINT_URL`.

The `ned` swarm already runs MinIO (`docker-swarm-core/116-minio/minio.yml`), exposed
through Traefik at `https://minio-api.snet.biobio.tamu.edu`. Reusing it adds no
infrastructure. Because it is reached by public hostname rather than an overlay alias, the
same `STUDIO_BLOB_BASE_URL` resolves identically from a swarm container, a laptop and an
Azure VM — which is the property this design wants. Should that traffic ever matter,
attaching MinIO to the application's overlay network is a one-line stack change.

**On copying to local SSD.** The instinct is right and the measurement says the work is
already done. Per training run the worker performs *one* `get_bytes` of the dataset
snapshot; the resulting frame is held in memory and reused by all three fits, and chemprop
iterates that in-memory frame for every epoch. RAM is faster than any NVMe, so staging to
local disk would be a strictly slower copy. The volumes involved:

| Object | Size | Transfers per run |
|---|---|---|
| Dataset snapshot (Parquet) | ~4 MB at 50k rows, ~60 MB at 1M | 1 read |
| Chemprop checkpoint | 5–10 MB | 1 write |
| ScorecardInputs (JSON) | ~5 MB at 50k rows | 1 write |

Roughly 5 MB in and 15 MB out per run. The baseline's RandomForest artifact — the one
genuinely large object the system can produce — is never stored, per `train_protocol.py`'s
existing "one artifact, not three" decision.

The one place a real filesystem path is unavoidable is chemprop's checkpoint round-trip,
since Lightning saves to a path. That uses a plain `tempfile.TemporaryDirectory()`, which
honours the standard `TMPDIR` environment variable — so a deployment points scratch at a
fast local NVMe by setting `TMPDIR`, with no setting of our own to invent. The GPU image
sets it to `/scratch`. Everything written there is transient and removed with the
directory.

`Dataset` — content-hashed, immutably snapshotted, carrying its validation report and
split spec — already *is* the central dataset registry. It was only ever missing a
networked backend. Reaching for DVC or LakeFS would duplicate it.

### 6. Container images

daikon-studio has no Dockerfile today. Two are added, both in `backend/`.

`Dockerfile` — `python:3.13-slim`, `uv sync --extra s3`. One image serving the API and
the default-lane worker, differing only by command.

`Dockerfile.gpu` — `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, `uv python install
3.13`, then `uv sync --extra gpu --extra s3` with torch from the cu124 index.

The obvious base, `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, is **ruled out**: it
ships Python 3.11, and this codebase uses PEP 695 generics (`class PageResult[T]` in
`domain/shared/pagination.py`, `def result_to_response[T]` in `interface/error_handlers.py`)
which require 3.12+. Lowering `requires-python` to accommodate a base image would be the
tail wagging the dog. uv installs its own interpreter, so the CUDA runtime image is the
cleaner floor.

Both images run as a non-root user and declare no volumes, per the thesis.

The swarm stack file is **out of scope here** and lands in `snet2-infrastructure`
alongside every other daikon service's deploy configuration. What it must provide:
`node.labels.gpu == true` placement, `NVIDIA_VISIBLE_DEVICES=all`, the four environment
variables, and no volumes.

## Testing

- **Lane routing.** `queue_for` for both cases; `ArqEnqueuer` passes `_queue_name`;
  `TrainProtocol` and `PredictWithProtocol` each route by the manifest's lane.
- **Cooperative cancellation.** A fake engine whose `train` calls `ctx.report` in a loop:
  cancelling the row mid-flight ends the run `CANCELLED`, returns the thread, and does
  **not** re-raise (asserting arq would not retry).
- **Deadline.** The same fake engine under a one-second deadline ends `FAILED` with a
  message naming the deadline.
- **Default `report` is a no-op**, so an engine that ignores it still trains — added to
  the existing engine contract test.
- **Metric extraction is behaviour-preserving**: the existing ECFP4 engine tests must pass
  unchanged, which is the assertion that matters.
- **Manifest tripwire** already covers chemprop's manifest once it is registered: every
  manifest in `default_registry()` must survive a JSON round-trip.
- **Chemprop engine**, skipped unless `chemprop` imports: a ~50-molecule, 2-epoch CPU fit
  asserting the predict schema, the dtypes, and the exact metric key names for both tasks.

## Deferred, with the reason each is safe to defer

- **Fit-result caching** — storing each completed fit under
  `(content_hash, engine_id, conditions, split)` so a restart skips finished work, and
  repeat trainings of a dataset get the baseline free. About 25 lines and the cheap
  alternative to Temporal for crash resilience. Deferred until a crash actually costs
  something measurable; restart-from-zero remains the designed recovery.
- **`TrainResult.artifact` as bytes** — unchanged until a checkpoint measurably hurts, per
  the future-seams rule.
- **Chemprop uncertainty** (MVE head or ensembling) — null is honest; a number is not.
- **Per-engine optimism-gap opt-out** — explicitly rejected for now. The gap matters *most*
  for a large neural network on a scaffold split, which is exactly where the flattering
  number comes from. Two GPU fits per training run is the price of the product's central
  claim.
- **Temporal** — its trigger is unchanged and unmet: a job that must survive worker death
  *without* restarting from zero. Note that Temporal would not have avoided the
  orphaned-thread problem above, since its cancellation is cooperative too.

## Out of scope

The swarm stack file (lands in `snet2-infrastructure`), any frontend change (the engine's
`description` carries the "trains on a GPU, takes minutes to hours" message, and the
existing Run polling renders the new per-epoch progress with no code change), exposing
`lane` over HTTP (nothing outside the enqueuer needs it), and a queue-depth or
worker-health dashboard.

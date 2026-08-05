# Handoff — daikon-studio, after the Chemprop engine landed

**Written:** 2026-08-04, at the end of the remote-engines/chemprop build.
**For:** a fresh session picking up three requested features.
**This is not a plan.** It is the state of the world, the traps that cost real time to
find, and the three asks as stated. Write the design yourself — brainstorm first, then
`writing-plans`. Read `HANDOFF-frontend-phase-2.md` for the UI's own history.

---

## 1. Where things are

Branch **`remote-engines-chemprop`**, **21 commits ahead of `main`, none pushed**. The
entire feature exists only on this laptop. Push it before doing anything else.

One commit on the branch is **not** part of this work: `c184283 feat: animate login mark…`
is the user's own frontend commit, authored in parallel and landed here because this was
the checked-out branch. It probably belongs on `main` separately.

Working tree clean. Backend suite green: **343 passed**, import-linter 3 contracts kept,
ruff clean, mypy clean on 106 files.

### Running locally (all of it — nothing is deployed anywhere)

| | |
|---|---|
| Backend `:8002`, Frontend `:3003` | native, via `make dev` |
| Worker — default lane | pulls `arq:queue`, runs the ECFP4 engines |
| Worker — gpu lane | pulls `arq:queue:gpu`, runs chemprop **on CPU** |
| Postgres `:5435`, Valkey `:6381` | docker compose |
| Blobs | `.blobs/` at the repo root |
| Sentinel | remote, `sentinel.orca-03.biobio.tamu.edu`, service name `daikon-studio-dev` |

**Nothing is deployed to ned or atlantic.** No stack file exists, no images have been
built or pushed. All the swarm discussion in the session was planning only. "gpu lane" is
a routing label, not a location — both workers are local.

The full loop was exercised for real: BBBP (1,965 compounds, scaffold split) → chemprop
D-MPNN → Scorecard. It took 167.8s and produced draft protocol `BBBP-DMPNN`, which lost
to the ECFP4 baseline on MCC (0.515 vs 0.632) with an optimism gap of 0.195 and 78%
applicability. The honesty layer works.

---

## 2. What landed

Design: `docs/superpowers/specs/2026-07-30-remote-engines-chemprop-design.md`.
Plan: `docs/superpowers/plans/2026-07-30-remote-engines-chemprop.md`. Both are accurate
except where §3 below says otherwise.

- **Lanes.** `EngineManifest.lane` (default `"default"`); `queue_for(lane)` in
  `infrastructure/worker.py`; `JobEnqueuer.enqueue(run_id, lane=…)`. One worker serves one
  lane, chosen by `STUDIO_WORKER_LANE`. Multi-node needs no code — more workers, same queue.
- **Cooperative progress + cancellation.** `TrainContext.report`, `RunInterrupted` and
  `ProgressReporter` in `application/engines/context.py`; the reporter is built in
  `RunTraining._reporter`. This is the *only* way to stop a fit already running.
- **Chemprop D-MPNN engine.** `infrastructure/engines/chemprop_dmpnn.py`, lane `"gpu"`.
  Every torch/lightning/chemprop import is inside a method so the API tier installs
  without CUDA — verified in the slim image, `torch` genuinely absent while the manifest
  still serves.
- **Shared metrics.** `regression_metrics` / `classification_metrics` extracted into
  `infrastructure/engines/_scoring.py` so chemprop and the baseline are measured by
  literally the same code.
- **Swappable blob backends.** `Settings.blob_storage_options` → `fsspec.url_to_fs`.
  `s3`/`azure`/`gpu` extras in `pyproject.toml`.
- **Container images.** `backend/Dockerfile` (slim, verified) and `backend/Dockerfile.gpu`
  (**never successfully built** — see §3).

---

## 3. Traps — read this section before touching anything

**`OMP_NUM_THREADS=1` is load-bearing, not tuning.** torch and scikit-learn each ship
their own `libomp`, and a training job loads both by design — `RunTraining` fits the
chosen engine and the mandatory baseline in one process. Three OpenMP runtimes in one
process segfaults: `EXC_BAD_ACCESS` at `0x30` in `__kmp_fork_barrier`, ~76s into a real
BBBP fit, killing the worker outright. Verified by reproduction: crashes unset, crashes
with the accelerator forced to `cpu` (so **MPS is not the trigger**), completes in 167.8s
with the var set. `KMP_DUPLICATE_LIB_OK` was already `True` in the shell throughout and
does **not** help. Set in the Makefile and both Dockerfiles. Do not "clean it up".

**Never put the blob store under `/tmp`.** This machine sweeps aged files out of `/tmp`
with no reboot (41 days uptime when it happened), leaving directories behind. Six days
after upload the store held 20 directories and zero files, and every dataset snapshot,
artifact and scorecard Postgres pointed at was gone. `.env.example` documents this.

**`accelerator="auto"` selects MPS on Apple Silicon, and MPS is slower here.** Measured:
100 molecules 7.15s MPS vs 0.54s CPU; 2,000 molecules 11.25s vs 7.97s. The engine
docstring in `chemprop_dmpnn.py` still claims auto "means the same code runs on a laptop
CPU" — **that sentence is wrong** and was left unfixed.

**uv extras are not additive across syncs.** `uv sync --extra gpu` uninstalls what
`--extra s3` installed. Both Dockerfiles pass them in one command; do the same locally.

**arq's `job_timeout` cancels the coroutine, not the `to_thread` OS thread.** Python
cannot kill a thread. On timeout arq also re-queues (`retry_jobs=True`, `max_tries=5`)
onto a row `Run.start()` now permits restarting — so a mis-set timeout produces multiple
unkillable fits on one device. `run_job` must **never** re-raise `RunInterrupted`. Note
Temporal would not avoid this; its activity cancellation is cooperative too.

**A crashed worker leaves the run `RUNNING` with no error recorded.** The UI then shows a
frozen progress bar rather than a failure. arq redelivers after `job_timeout` (currently
`worker_job_timeout + 600` = 40 min), so it self-heals eventually, but there is no
staleness reaper. The future-seams doc parked one as out of scope; this session is the
first evidence it's worth building.

*(Update, self-hosted-runners phase: arq/Valkey are gone. The Postgres-backed runner
queue now self-heals a crashed runner via `RunQueue.sweep`'s lease expiry, not an arq
hard timeout -- `runner_lease_seconds` (default 600s), independently heartbeated by the
agent (`infrastructure/runner/agent.py`) so a healthy long job doesn't trip it. The
staleness-reaper gap this paragraph called out is the one this mechanism was built to
close; see `docs/superpowers/specs/2026-08-04-self-hosted-runners-design.md`.)*

**`Dockerfile.gpu` has never been built successfully.** This laptop is arm64 and torch's
cu124 wheels are amd64-only; under emulation the build reached "Prepared 127 packages"
and was stopped. It must be built on x86 — `docker --context ned build` is cheaper than
standing up CI. Also unresolved: the locked torch pulls **CUDA 13** vendored libraries
under a `nvidia/cuda:12.4.1` base, so the host driver on the GPU node must be new enough
for CUDA 13. Check `nvidia-smi` before deploying; that is a go/no-go.

**`make dev`'s stop can lose a race.** A stale backend from a previous session held
`:8002` for six days and survived `make stop`; the new one silently failed with
`Address already in use`. If the backend seems wedged, check `lsof -ti:8002` for a process
older than your session.

**`STUDIO_INLINE_JOBS` must be `0`** or the API runs fits inside the HTTP request and the
browser times out. Currently `0` in `backend/.env` (gitignored).

**Reproducing a job without the UI**: build the ctx `_on_startup` builds
(`sessions`, `store`, `job_deadline_seconds`) and call `worker.run_job(ctx, run_id)`
directly. `Run.start()` allows `RUNNING → RUNNING`, so a row left running by a crash
restarts cleanly. About 20 lines, and it exercises the real reporter, RDKit normalisation
and scaffold split — a synthetic engine-only harness did **not** reproduce the segfault.

---

## 4. The three asks

Stated as the user stated them. Deliberately not designed here.

### 4.1 Choose which baseline model to compare against

Today the baseline is fixed at the registry level, not per-run:

- `EngineManifest.is_baseline` — `application/engines/manifest.py:55`
- `Ecfp4RandomForest` sets it `True` — `infrastructure/engines/ecfp4_randomforest.py:50`
- `EngineRegistry.baseline()` raises unless **exactly one** engine is flagged —
  `application/engines/registry.py:54`
- `RunTraining` calls `self._engines.baseline()` — `application/execution/train_protocol.py:359`
- Results land in `ScorecardInputs.baseline_engine_id` / `baseline_metrics` /
  `baseline_is_self`; the last one short-circuits the second fit when the chosen engine
  *is* the baseline on default conditions.

Constraint worth respecting: the design doc treats a baseline comparison as **mandatory,
never opt-out**. "Choosable" is not the same as "skippable", and the frontend's verdict
copy ("No better than the baseline") assumes a comparison always exists.

### 4.2 Ability to re-run

There is no re-run or retrain affordance anywhere in the UI. The only two entry points
both create a *new* protocol:

- `features/datasets/components/dataset-detail.tsx:63` → `/protocols/new?dataset=<id>`
- `features/protocols/components/protocol-list.tsx:25` → `/protocols/new`

Relevant existing machinery the UI never reaches: `InSilicoProtocol.new_version()` at
`domain/catalog/protocol.py:97`, which bumps `protocol_version` and is already accounted
for in `predict_with_protocol.py`'s artifact-reading comments. Training runs are **not**
cached — `compute_cache_key` is written for training runs but only predictions reuse a
cached Run — so an identical re-submission redoes the work.

Open question the fresh session must settle: is "re-run" a new protocol, a new *version*
of the same protocol, or a re-execution of the same Run? Those are three different
features and the UI copy, the Scorecard's identity, and any citation of a published
protocol all depend on which.

### 4.3 CheMeleon / pretrained weights for chemprop

The user wants, in their words: a **universal way to start from pretrained weights** —
open question whether that is an upload, or standard preconfigured weight sets (CheMeleon
being the first), or both — and to use **no-pretrained vs pretrained as a baseline
comparison**.

This interacts directly with 4.1: "pretrained vs not" as a comparison is a second axis of
baseline, distinct from "which engine".

Nothing has been researched about CheMeleon in this session — do not trust any
recollection of its API, checkpoint format, licence or size without checking the source.
Seams that will matter:

- `EngineManifest.conditions` is the only declared-parameter mechanism, and it is plain
  data that must survive the JSON round-trip tripwire in `tests/unit/engines/test_engine_contract.py`.
- `TrainResult.artifact` is `bytes`; the future-seams doc says keep it that way until a
  checkpoint measurably hurts, at which point it becomes a blob URI.
- Uploads already have a home: `StoreUpload` / `upload_key` in `application/data/create_dataset.py`.
- chemprop's `MPNN.load_from_checkpoint` reconstructs architecture from the checkpoint's
  saved hyperparameters — relevant if pretrained weights arrive as a `.ckpt`.
- Chemprop version installed resolves to **2.3.0** (the `gpu` extra pins `>=2.2,<3`).

---

## 5. Deferred, each with evidence behind it

- **Staleness reaper** — a crashed worker's run shows a frozen progress bar for 40 minutes.
- **Error message quality** — the worker records `repr(exc)`, and `FileNotFoundError`'s
  repr drops the filename, so the first failure of the day read
  `FileNotFoundError(2, 'No such file or directory')` with no path and no dataset name.
  `_require_structure_column` in `train_protocol.py` is the convention to copy.
- **The wrong docstring** in `chemprop_dmpnn.py` about `accelerator="auto"` (§3).
- **Skipping MPS deliberately** — select CUDA when present, else CPU. Measured faster and
  avoids an untested code path.
- **"Train another on this dataset" link on the Scorecard** — the page delivers the
  verdict "No better than the baseline" and offers no way to act on it.
- **Fit-result caching** keyed on `(content_hash, engine_id, conditions, split)` — the
  cheap alternative to Temporal for crash resilience, and it would make the baseline free
  on repeat trainings.
- From the future-seams doc, still untouched: artifact bytes → URI, per-engine
  optimism-gap opt-out, Temporal (its trigger remains unmet).

---

## 6. Deployment, for whenever it happens

Settled: **all services on ned**, orca only for the existing Sentinel. Blobs in **MinIO**
(`minio-api.snet.biobio.tamu.edu`, verified reachable from the laptop and from anywhere
because it is fronted by Traefik on a public hostname). NFS was floated and **rejected** —
it contradicts the spec's own "no volumes, no shared filesystem" thesis and does not
survive ephemeral GPU containers.

There is no shared Postgres or Redis on the cluster to reuse; every one of the 13 Postgres
instances is per-app and overlay-only. Sentinel is the only genuinely shared dependency,
and local dev already uses it — that matches every sibling project.

Not started: the stack file (directory `145`, subnet `172.16.145.0/24`, both free), x86
image builds, a MinIO bucket and service-account key, and the CUDA driver check. The
hostname `studio.snet.biobio.tamu.edu` is currently taken by the legacy `daikon-ai-studio`
in `115-daikon2`; the user's stated intent is to retire that ned instance since
`studio.orca-03.biobio.tamu.edu` already serves it. A hostname change there would need
redirect URIs updated in Keycloak, Google **and** GitHub first — a `cf.snet` rename was
prepared and then fully reverted, so `snet2-infrastructure` is untouched.

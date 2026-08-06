# Handoff — daikon-studio, after the first roster addition and the first GPU image

**Written:** 2026-08-06, at the end of the session that added `descriptors-xgboost` and
got `Dockerfile.gpu` building.
**For:** whoever continues the roster expansion.
**This is not a plan.** The plan is `docs/superpowers/2026-08-06-high-impact-plan.md` —
read it second. This file is the state of the world, what was verified by running it and
what was not, and the traps that are still there.

Prior eras: `HANDOFF-fanout-sweeps.md`, then `HANDOFF-chemprop-and-baselines.md`.

---

## 1. Where things are

Branch **`descriptors-engine-and-gpu-image`**, **unpushed**, cut from `main` at `93e308b`.
**Do not push or merge without the user asking** — that standing instruction from the
previous handoffs still holds.

The user's current direction, set explicitly this session: **expand capability by adding
models.** Sweeps were a proof of concept and are *not* the focus. The measurement work
(selection barrier, paired bootstrap, replication) is deferred by decision — §6 of the plan
records what that costs so it stays visible rather than forgotten.

### Gates at HEAD

Backend, all run live this session, not remembered:

- `uv run pytest` — **535 passed**
- `uv run ruff check` / `ruff format --check` — clean, 208 files
- `uv run mypy src` — clean, 135 source files
- `uv run lint-imports` — **3 contracts kept, 0 broken**

Frontend was **not touched** this session, but its gates were run to confirm that: biome
clean (131 files), **69 tests passed** across 9 files — unchanged from the sweeps era.
Adding an engine changes the *data* returned by `GET /api/v1/engines`, not the OpenAPI
schema, so **no `make generate-api` is needed** — verify that assumption yourself if you
add an engine with a condition type that does not already exist.

---

## 2. What landed

### `descriptors-xgboost` — a new engine, default lane

`infrastructure/engines/descriptors_xgboost.py`, plus `rdkit_descriptors()` in
`infrastructure/chem/featurize.py`. All 217 `Descriptors.descList`, no new dependency.

Four things in it are load-bearing and will look like arbitrary choices if you skim:

1. **NaN, not 0.0, for anything unmeasurable.** This is the *opposite* of `ecfp4`'s
   all-zero-row convention in the same file, deliberately. A zero bit means "substructure
   absent", which is true; a zero molecular weight is a measurement that never happened.
   `chem/descriptors.py` already made the same call for the same reason.
2. **Boosting, not a forest — because of (1).** XGBoost treats NaN as missing and learns a
   split direction for it. sklearn's `RandomForestRegressor` raises on NaN outright. The
   representation and the estimator were chosen together; you cannot swap in RF without
   also changing the missing-value policy.
3. **Infinities are swept to NaN.** `Ipc` overflows on larger molecules, and XGBoost
   *accepts* NaN but *rejects* inf ("Input data contains `inf`"). Without the sweep the one
   value that reaches the estimator is the one that kills it.
4. **A feature-drift guard** (`_require_matching_features` in `_scoring.py`).
   `Descriptors.descList` is a property of the installed RDKit, not a constant — it grows
   between releases. Upgrading RDKit under a stored model shifts every column, and no
   estimator can detect that: XGBoost sees the right number of floats and returns confident
   nonsense. The artifact persists its descriptor names and `predict` refuses a mismatch.
   **This is the only silent failure mode in the predict path.** Do not remove it as
   ceremony.

Measured on a diverse 600-molecule set (metals, salts, a 60-carbon chain, a fused
polycyclic, zwitterions): **3.76 ms/mol** single-threaded, no infinities, **0.49% NaN
cells**, no all-NaN rows, no descriptor that is always NaN.

### `_scoring.py` now takes a featurizer

`_score`, `_score_validation` and `_predict_with_tree_ensemble` no longer hard-code
`ecfp4`. They default to it, so both ECFP4 engines and **every artifact written before this
session** read unchanged — `predict` uses `bundle.get("featurizer", "ecfp4")`, not
`bundle["featurizer"]`, precisely so old artifacts keep predicting. There is a test for
that; keep it.

### The gpu lane, corrected

**The gpu lane already runs on the Mac GPU.** `lightning`'s `accelerator="auto"` resolves
to `MPSAccelerator` / `mps:0`, and a full chemprop fit *and* predict both complete there —
no `PYTORCH_ENABLE_MPS_FALLBACK`, no unimplemented-op errors, 7.3s for three epochs on 40
molecules. Every comment in the repo claiming "locally there is no GPU and chemprop falls
back to CPU" was **stale and wrong**, and has been corrected in the Makefile.

**Do not build a `Dockerfile.mps`.** Docker Desktop on macOS runs a Linux VM that cannot
see Metal; there is no Mac equivalent of `--gpus all`. A Mac "GPU image" would silently
train on CPU inside the container, strictly worse than the native path that already works.
The shape is **one CUDA image plus a native Mac runner**, and that is now written into
`Dockerfile.gpu`'s header so the idea does not get re-proposed.

Runs now name their device in the progress phase (`training chemprop-dmpnn on mps:0`).
Nothing validates that a runner registered for the gpu lane actually owns a GPU, and cpu /
mps / cuda do not produce identical numbers, so "some device" was the only honest thing
anyone could previously say about a finished run.

### `Dockerfile.gpu` — built for the first time, and it was broken

`daikon-runner:gpu` now exists on the **`ned`** docker context (linux/amd64, docker 28.0.4),
**22.5GB**. First successful build in the project's history.

The first build also proved the file was wrong. It built completely green, then died on
`import chemprop` with `ImportError: libXrender.so.1`. chemprop 2.3's featurizer imports
`cuik_molmaker`, whose bundled `libcuik_molmaker_core.so` links **libXrender, libXext and
libexpat**, and `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` ships none of them. The
file's "no apt step on purpose" comment was actively harmful and is gone.

The package list was resolved by `ldd`-ing the shared object in a throwaway container and
installing until the import succeeded, not guessed. If a future chemprop bumps
`cuik_molmaker`, re-run:

```
ldd /app/.venv/lib/python3.13/site-packages/cuik_molmaker/lib/libcuik_molmaker_core.so | grep 'not found'
```

**Verified by running the final image**, not by reading it: chemprop 2.3.0 imports, the
baked CheMeleon checkpoint loads to a 2048-wide encoder, all four engines register.

Build command — note the **absolute paths**. A relative `backend` context silently resolves
against the shell's cwd, and `docker build ... | tail` masks the non-zero exit, which cost
one wasted cycle this session:

```
docker --context ned build -f /abs/path/backend/Dockerfile.gpu -t daikon-runner:gpu /abs/path/backend
```

---

## 3. Traps still there

- **The locked torch is `2.13.0+cu130` — CUDA 13.0 — while the base image tag says 12.4.1.**
  Fine in itself (PyPI linux torch wheels vendor their own CUDA userspace), but the **host
  driver must be new enough for CUDA 13**, not 12.4, and the image tag is misleading about
  which. Nothing has run on a real CUDA GPU yet.
- **A green `docker build` says nothing about whether the image works.** That is how the
  libXrender bug survived. Run the image.
- **Per-lane job timeouts still do not exist.** The deadline is served from the server's
  single setting to every runner regardless of lane. Chemprop as both engine and baseline
  on a scaffold split is three fits in one job against a 1800s default. Unchanged this
  session.
- **`uncertainty` still means four different things across engines** and the triage grid
  sorts across them. `descriptors-xgboost` adds a fifth engine with a null one (XGBoost has
  no ensemble spread), consistent with `ecfp4-xgboost`. Plan §6 has the full analysis.
- **The scorecard's green "Beats the baseline, +0.123" badge is unsupported at n=197** —
  the baseline's 0.514 sits inside the bootstrap CI of [0.487, 0.764]. Independent of
  sweeps. Adding engines multiplies where it appears.
- **The `ned` and `orca` docker contexts are remote university machines over SSH.** Builds
  there are real resource use on someone else's infrastructure. `ned` was used this session
  because the user's own notes named it as the intended build host.

---

## 4. Next

From the plan's ranked list, the next item is the **Tanimoto-kernel Gaussian Process**:
small-n regime, posterior variance as principled uncertainty rather than a bolted-on head,
13th of 66 at the Polaris antiviral challenge ahead of most deep learning, ~30 lines of an
sklearn `Kernel` subclass, no new dependency. Fingerprints not descriptors — the kernel is
defined on binary vectors. Declare the O(n³) ceiling in the manifest help text.

After that: frozen CheMeleon + TabPFN (**check first** whether it really needs `lane="gpu"`
— the embeddings were verified CPU-only, and the lane was assigned for the chemprop
*dependency*, not the hardware), then descriptors into the chemprop predictor, then
MoLFormer-XL.

Two capabilities bigger than any of those, both named in the plan and neither started:
**multi-task training** (chemprop supports `n_tasks>1`; the platform hard-codes 1) and
**transfer from a prior run's stored artifact**.

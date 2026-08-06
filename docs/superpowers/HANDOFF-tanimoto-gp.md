# Handoff — daikon-studio, after the Gaussian process and a bug the last session shipped

**Written:** 2026-08-06, at the end of the session that added `tanimoto-gp` and fixed the
float32 overflow in `rdkit_descriptors`.
**For:** whoever continues the roster expansion.
**This is not a plan.** The plan is `docs/superpowers/2026-08-06-high-impact-plan.md`,
now with items 1, 1b and 2 marked shipped — read it second.

Prior eras: `HANDOFF-fanout-sweeps.md`, `HANDOFF-chemprop-and-baselines.md`, then
`HANDOFF-descriptors-engine-and-gpu-image.md`, which this one continues directly.

---

## 1. Where things are

Branch **`descriptors-engine-and-gpu-image`**, **unpushed**, now three commits past where
the previous handoff left it. **Do not push or merge without the user asking** — that
standing instruction from every previous handoff still holds.

Direction unchanged: **expand capability by adding models.** Sweeps are not the focus.
The measurement work (selection barrier, paired bootstrap, replication) stays deferred by
decision; plan §"Deferred" records what that costs.

### Gates at HEAD

All run live this session, not remembered:

- `uv run pytest` — **551 passed** (535 at the previous handoff, +16)
- `uv run ruff check src tests` / `ruff format --check src tests` — clean, 210 files
- `uv run mypy src` — clean, 136 source files
- `uv run lint-imports` — **3 contracts kept, 0 broken**

Frontend **not touched**, gates run to confirm: biome clean (131 files), **69 tests
passed** across 9 files. No `make generate-api` needed — adding an engine changes the
*data* from `GET /api/v1/engines`, not the schema, and the new engine's only condition is
`INTEGER`, a type that already existed. Verified separately that the frontend hardcodes no
engine ids outside test fixtures, so the roster renders from API data alone.

**Note the gate's scope:** it is `ruff check src tests`, not bare `ruff check`. The latter
reports 8 pre-existing import-order errors in the alembic migrations, which are outside
the project's gate and were outside it before this session too. Do not "fix" them thinking
you regressed something.

---

## 2. What landed

### `tanimoto-gp` — a new engine, default lane

`infrastructure/engines/tanimoto_gp.py`, containing both the `TanimotoKernel` and the
engine. sklearn's `GaussianProcessRegressor` / `GaussianProcessClassifier`, no new
dependency, one condition (`n_restarts_optimizer`, default 2).

**Read the measured numbers before you repeat the pitch.** On BBBP, scaffold splits,
three seeds:

| n | baseline MCC | descriptors MCC | GP MCC | GP fit |
|---|---|---|---|---|
| 200 | 0.527 | 0.440 | 0.394 | 0.1s |
| 500 | 0.460 | 0.550 | 0.450 | 0.4s |
| 1000 | 0.580 | 0.611 | 0.547 | 1.2s |

The GP **does not beat the baseline on accuracy**, and every gap there is inside the
±0.14 MCC bootstrap noise the plan documents. It wins on cost (fastest at every size) and
on uncertainty quality — its most-confident half scores 0.889 against 0.794 overall, where
the forest manages 0.841. The engine's value is the regime and the uncertainty, which is
what the capability-first framing said it would be. Do not let it get written up as an
accuracy win.

Five things in it are load-bearing and will look arbitrary if you skim:

1. **The float64 cast at the top of the kernel.** `ecfp4` returns uint8 and sklearn's
   `dtype="numeric"` validation *preserves* it all the way into the kernel — verified by
   instrumenting a real fit, not assumed. Without the cast, `x_norm + y_norm` wraps past
   255 once a pair of molecules sets ~128 bits each, and the divide cannot write float
   output into a uint8 `out`. The second failure is loud; **the first is not.**
2. **`__init__` exists despite taking no arguments.** sklearn's `Kernel.get_params`
   introspects the signature and rejects the inherited varargs `object.__init__`. The
   failure lands inside `clone()` during `fit`, not at construction, so it does not look
   like a kernel problem when it happens.
3. **A hard 10,000-row training ceiling**, refused with a message naming the reason.
   Memory is O(n²) and the fit O(n³); past it the failure mode is an OOM-killed worker
   with no message, which is strictly worse for a scientist than a refusal. The manifest
   tells users 5,000, where it is still comfortable. The check runs *before*
   featurization, so it is cheap.
4. **Single-class training splits are refused explicitly.** The tree engines fit happily
   on one class and let `_score` report undefined metrics. `GaussianProcessClassifier`
   raises instead, and its own message names neither the engine nor the fix — so without
   this the run fails on a bare sklearn `ValueError`. `train_protocol.py`'s optimism-gap
   leg already names "a random partition that leaves one class in the training rows" as a
   thing that happens, so this is not hypothetical.
5. **`WhiteKernel` is not decoration.** It is the fitted observation noise, and it is also
   what keeps the Cholesky from failing on duplicate structures, which give identical rows
   and a singular matrix. Duplicate measurements of one compound are ordinary in an assay;
   there is a test for it.

**Classification gets no posterior variance.** GPC's Laplace approximation exposes no
latent variance, so that path reports the same distance-from-the-boundary the forest does.
This engine adds a **sixth** meaning to `uncertainty`, it does not resolve the existing
five. The triage grid still sorts across all of them.

### `_scoring.py` — two shared helpers, two predict paths

`_load_bundle` (unpickle + drift guard + featurize) and `_prediction_frame` (the row_id /
value / uncertainty schema with its explicit dtypes) are now shared. `_predict_with_tree_
ensemble` and `_predict_with_gaussian_process` sit on top and differ only in where
uncertainty comes from.

They were deliberately **not** collapsed into one function. Routing a GP through the tree
path would find no `estimators_` and report `None`, discarding the one property the engine
exists for. The uncertainty source is exactly what distinguishes the engines.

### `rdkit_descriptors` swept the wrong ceiling — a live bug from the previous session

`descriptors-xgboost` shipped last session and **crashed on the first real dataset it
met**, with `Input data contains inf` — precisely the error its featurizer's sweep was
written to prevent.

RDKit's `Ipc` returns **6.5e39** on molecules like ivermectin. That is an ordinary
float64, so `np.isfinite` passed it through untouched; XGBoost stores features as float32
(max 3.4e38) and *its* cast produced the inf. The binding ceiling was never float64's.
Two of three seeds at n=1000 crashed before the fix; all three train after.

The fix is one comparison, `np.abs(rows) > _FLOAT32_MAX`, which covers all three cases
together: inf compares greater, NaN compares false and stays NaN, and a merely huge
float64 is swept. Two regression tests carry the actual offending molecule.

**The generalisable lesson.** The original tests used synthetic shapes — `"C"*60`, a fused
polycyclic — which overflow *float64* and were caught correctly. The molecules that
actually break it sit in the band **between** the two ceilings, and only a real dataset
contains those. This is the same shape as the previous session's libXrender bug: a green
suite said nothing, and running the thing for real is what found it.

---

## 3. Traps still there

Everything in the previous handoff's §3 still applies. Unchanged:

- **The locked torch is `2.13.0+cu130` — CUDA 13.0 — while the base image tag says
  12.4.1.** The host driver must be new enough for CUDA 13. Nothing has run on a real
  CUDA GPU yet.
- **A green `docker build` says nothing about whether the image works.** Run it. See
  above for the second instance of this same lesson in two sessions.
- **Per-lane job timeouts still do not exist.** One server-wide deadline serves every
  runner regardless of lane.
- **`uncertainty` now means six different things across engines** and the triage grid
  sorts across them. The GP's is the only principled one, and only on the regression path.
- **The scorecard's green "Beats the baseline, +0.123" badge is unsupported at n=197.** A
  fifth engine multiplies where it appears. Still the one measurement item that is live
  independent of sweeps.
- **`ned` and `orca` are remote university machines over SSH.** Builds there are real
  resource use on someone else's infrastructure.

New this session:

- **`tanimoto-gp` gets slower as data grows**, alone in the roster. The 10,000-row refusal
  is a real user-visible failure for anyone who picks it on a large dataset. That is the
  intended behaviour, but nothing in the UI warns *before* they submit — the manifest
  description is the only signal, and it is prose.
- **A `ConvergenceWarning` from sklearn fires when the amplitude hits its bound**, which
  is what happens on data with no signal. It is sklearn telling the truth and was left
  alone, but it will appear in production logs and is not a bug.

---

## 4. Next

From the plan's ranked list, the next item is **frozen CheMeleon + TabPFN** (#3). The
open question from the previous handoff still needs answering **first**: the research
verified frozen CheMeleon embeddings run CPU-only, and the engine was assigned `lane="gpu"`
for the chemprop *dependency*, not the hardware. If the chemprop extra installs on a
default-lane runner, this ships without touching the GPU image at all.

After that: descriptors into the chemprop predictor (#4, shares its featurizer with #1, so
mostly wiring), then MoLFormer-XL (#5).

Two capabilities bigger than any of those, both named in the plan and neither started:
**multi-task training** (chemprop supports `n_tasks>1`; the platform hard-codes 1) and
**transfer from a prior run's stored artifact**. With items 1 and 2 shipped, multi-task is
now the largest remaining gap in the plan's "holes in the roster" list.

**One process note worth carrying forward.** Both of the last two sessions found their
real bug by running the thing on real input — a real image, a real dataset — after the
whole test suite was green. The benchmark script that found this session's bug is
throwaway, but writing one for each new engine is cheap and has now paid twice.

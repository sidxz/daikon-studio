# Roster expansion plan — capability first

**Date:** 2026-08-06 · **Status:** plan only, nothing implemented · **Reranks**
`2026-08-04-engine-roster-research.md` §3 (its evidence stands; its ordering does not)

Direction set by the user on 2026-08-06: sweeps were a proof of concept and are not the
focus; the goal is **expanding what the platform can do by bringing in more models**. This
document ranks additions by *what a user can do after each one that they could not before*,
not by expected accuracy delta.

---

## The gpu lane is not blocked, and there is no Mac image to build

Corrected 2026-08-06 by measurement, replacing the assumption that ran through the
earlier drafts.

**The gpu lane already runs on the Mac GPU.** `lightning`'s `accelerator="auto"` resolves
to `MPSAccelerator` / `mps:0` on this machine, and a full chemprop fit *and* predict both
complete there — no `PYTORCH_ENABLE_MPS_FALLBACK`, no unimplemented-op errors, 7.3s for
three epochs. Every comment claiming "locally there is no GPU and chemprop falls back to
CPU" was stale and has been corrected in the Makefile.

**Two GPU images is the wrong shape; it should be one image plus a native path.** Docker
Desktop on macOS runs a Linux VM that cannot see Metal — there is no Mac equivalent of
`--gpus all`. A "Mac GPU image" would silently train on CPU inside the container, which is
strictly worse than what already works natively. So:

| Backend | How it runs | Status |
|---|---|---|
| Apple Silicon / MPS | native runner agent, `make dev-worker-gpu` | works today, verified |
| CUDA | `Dockerfile.gpu`, x86_64 | **first successful build 2026-08-06** on `ned`, 22.5GB |

**`Dockerfile.gpu` built for the first time — and the first build proved it was broken.**
The image built cleanly, then `import chemprop` failed inside it with `ImportError:
libXrender.so.1: cannot open shared object file`. chemprop 2.3's featurizer imports
`cuik_molmaker`, whose bundled `libcuik_molmaker_core.so` links libXrender, libXext and
libexpat, and `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` ships none of them. The file's
"no apt step on purpose" comment was therefore wrong, and a GPU image that cannot import
the only engine it exists to run is worse than no image.

The package list was resolved by `ldd`-ing the shared object and installing until `import
chemprop` succeeded, not guessed — `libxrender1 libxext6 libexpat1`, with the nvidia apt
list removed first so the build still avoids the NVIDIA mirror that broke it originally.
**Verified in the built image:** chemprop 2.3.0 imports, the baked CheMeleon checkpoint
loads to a 2048-wide encoder, and all four engines register.

The lesson worth keeping: a green `docker build` said nothing about any of this. It was
found by *running* the image, which is the only thing that would have found it short of a
real GPU run failing hours in.

Worth knowing for deployment: the locked torch is **2.13.0+cu130**, i.e. CUDA 13.0, while
the base image tag says 12.4.1. That is fine — PyPI's linux torch wheels vendor their own
CUDA userspace — but the *host driver* must be new enough for CUDA 13, not 12.4, and the
base image tag is misleading about which.

Consequence for the roster: **GPU-lane engines are testable today.** They were never
gated on the CUDA image, only on somebody checking. Items 3–5 below can be developed and
exercised locally, with the CUDA image needed for production throughput rather than for
correctness.

Runs now name the device they trained on in their progress phase (`training
chemprop-dmpnn on mps:0`). Nothing validates that a runner registered for the gpu lane
owns a GPU, the three devices do not produce identical numbers, and until this the only
honest thing anyone could say about a finished run was "some device".

## What that reframe changes

**Adding engines is much cheaper than modifying the baseline.** The earlier draft of this
plan put "strengthen the mandatory ECFP4 baseline" first, and that item drags provenance in
with it — mutating the baseline reinterprets every historical scorecard, which forces
`EngineManifest.version` persistence and a code SHA before it is safe. A **new** engine
alongside the existing ones orphans nothing. Same featurization work, none of the blocker.
The capability framing is the cheaper path to the same code.

**Accuracy-delta arguments stop being the point.** The roster research established that no
candidate reliably beats a tuned descriptor baseline at 10²–10⁴ compounds, and that
blind-challenge top tiers are statistically indistinguishable. Under a capability goal that
is no longer an objection — the value of an engine is the regime it opens, not the third
decimal place it wins.

**An engine is ~110 lines.** `ecfp4_randomforest.py` is 115, `ecfp4_xgboost.py` is 111. The
`Engine` protocol is `manifest()` / `train(ctx)` / `predict(ctx)`, shared metrics live in
`_scoring.py`, and featurizers live in `infrastructure/chem/featurize.py` next to `ecfp4()`.
Adding three CPU engines is a realistic near-term goal, not a quarter of work.

---

## The holes in the current roster

As written 2026-08-06, before items 1 and 2 shipped. Struck items are now closed; the
list is left intact because the ones remaining are the argument for what comes next.

Then: ECFP4+RandomForest (baseline, CPU), ECFP4+XGBoost (CPU), Chemprop D-MPNN ±CheMeleon
(GPU). Every engine saw either hashed Morgan bits or the raw graph. `TaskType` is
`REGRESSION` and `BINARY_CLASSIFICATION`.

What a user could not do at all:

1. ~~Run a **descriptor-based** model.~~ **Closed by #1.** Every reproducible top entry on
   the TDC audit — CaliciBoost, MapLight, MapLight+GNN — is descriptors plus boosting.
2. ~~Model a **small dataset** well.~~ **Addressed by #2**, with the caveat measured there:
   the GP opens the regime and is by far the cheapest engine in it, but did not beat the
   baseline on accuracy at n=200–1000 on BBBP.
3. ~~Get **principled uncertainty** from any engine.~~ **Partly closed by #2** — posterior
   variance, but on the regression path only. Classification still has no principled
   spread from any engine.
4. Use a **sequence** model. Everything is graph or fingerprint. Still open (#5).
5. Train **one model across several endpoints**. Chemprop supports `n_tasks > 1` natively;
   the platform hard-codes 1. Still open, and now the largest remaining gap.

---

## Ranked additions

### 1. Descriptors + gradient boosting — **SHIPPED 2026-08-06**

`descriptors-xgboost`, on the default lane. The single biggest capability gap: until this,
every engine saw either 2048 hashed Morgan bits or the raw graph, so the platform could not
express the recipe behind every TDC entry that survived the Feb 2026 audit.

What landed:

- `rdkit_descriptors()` in `chem/featurize.py` — all 217 `Descriptors.descList`, no new
  dependency. Measured 3.76 ms/mol single-threaded over a diverse 600-molecule set, 0.49%
  NaN cells, no all-NaN rows, no descriptor that is always NaN.
- **NaN, not 0.0, for anything unmeasurable** — the opposite of `ecfp4`'s convention and
  the right one here, since a zero bit means "substructure absent" (true) while a zero
  molecular weight is a measurement that never happened. XGBoost treats NaN as missing and
  learns a split direction; sklearn's RandomForest raises on it. That is why the pairing is
  boosting, not a forest — representation and estimator were chosen together.
- **Infinities swept to NaN.** `Ipc` overflows on larger molecules and XGBoost rejects inf
  outright while accepting NaN, so the one value that would fail loudly is normalised.
- No scaling, deliberately — boosting is scale-invariant.
- `_score` / `_score_validation` / `_predict_with_tree_ensemble` now take the featurizer
  instead of hard-coding `ecfp4`, defaulting to it so existing artifacts and both ECFP4
  engines are untouched.
- **A feature-drift guard.** `Descriptors.descList` is a property of the installed RDKit,
  not a constant — it grows between releases. Upgrading RDKit under a stored model shifts
  every column, which no estimator can detect: XGBoost sees the right number of floats and
  returns confident nonsense. The artifact persists its descriptor names and `predict`
  refuses a mismatch. This is the only silent failure in the predict path.

11 new tests; full suite green (238 unit, 190 api), ruff/mypy/import-linter clean.

### 2. Tanimoto-kernel Gaussian Process — **SHIPPED 2026-08-06**

`tanimoto-gp`, on the default lane. `GaussianProcessRegressor` / `GaussianProcessClassifier`
over a ~40-line `Kernel` subclass, no new dependency, one condition (`n_restarts_optimizer`).

What landed, and what measurement changed about the pitch below:

- **The uncertainty claim holds; the accuracy claim was never made and should not be.**
  On BBBP with scaffold splits over three seeds, MCC was 0.394 (n=200), 0.450 (n=500),
  0.547 (n=1000) against the baseline's 0.527 / 0.460 / 0.580. The GP does **not** beat
  the baseline here, and every gap in that table is inside the ±0.14 MCC bootstrap noise
  §"Deferred" already documents. What it does win: it was the fastest engine at every
  size (0.1–1.2s vs the forest's 0.5–0.9s and descriptors' 1.3–6.9s), and its uncertainty
  ranks best — restricting to its most-confident half lifts accuracy 0.794 → 0.889,
  against the forest's 0.794 → 0.841.
- **Posterior variance is regression-only.** `GaussianProcessClassifier` uses a Laplace
  approximation that exposes no latent variance, so the classification path reports the
  same distance-from-the-boundary the forest does. The genuine posterior std — in the
  target's units, via `normalize_y` — exists only on the regression path. The engine adds
  a *sixth* meaning to `uncertainty`, not a resolution of the existing five.
- **The float32 cast in the kernel is load-bearing.** `ecfp4` returns uint8 and sklearn's
  `dtype="numeric"` validation preserves it into the kernel (verified by instrumenting a
  real fit). Without the cast, `x_norm + y_norm` wraps past 255 for a pair of large
  molecules — silently — and the divide cannot write float output into a uint8 `out`.
- **A hard training-size ceiling**, not just documentation: 10,000 rows, refused with a
  message naming the reason. Past that the failure mode is an OOM-killed worker with no
  message. The manifest tells users 5,000, where it is still comfortable.
- **Single-class training splits are refused explicitly.** The tree engines fit happily
  on one class and let `_score` report undefined metrics; `GaussianProcessClassifier`
  raises, and its own message names neither the engine nor the fix.

The predict path did not collapse into `_predict_with_tree_ensemble`: the two share the
bundle shape and output schema (now `_load_bundle` and `_prediction_frame`) and differ on
exactly the thing that distinguishes the engines. Routing a GP through the tree function
would find no `estimators_` and report `None` — discarding the one property it exists for.

Original rationale, which still stands:

- **Opens the small-data regime.** GPs are the classic n<1000 method, which is where most
  in-house assays live.
- **Its uncertainty is the posterior variance** — principled, in target units, no head to
  bolt on and no calibration story required to make it honest. Hirschfeld et al. (*JCIM*
  2020) ranked MPNN GP second by NLL across their whole comparison.
- **It has competition evidence.** At the Polaris/ASAP antiviral challenge a Gaussian
  process with a Tanimoto kernel placed 13th of 66, ahead of most deep-learning entries
  (Inductive Bio's post-mortem).
- **No new dependency.** `GaussianProcessRegressor` / `GaussianProcessClassifier` plus a
  ~30-line `Kernel` subclass. The Tanimoto kernel k(x,y) = ⟨x,y⟩ / (‖x‖² + ‖y‖² − ⟨x,y⟩) is
  PSD on binary vectors, so it is a legitimate kernel rather than a similarity hack.

Fingerprints here, not descriptors — the kernel is defined on binary vectors and handles
the representation natively, where a descriptor GP would need scaling and a different kernel.

Declare the ceiling in the manifest help text: the fit is O(n³) and memory O(n²), so this
engine is honest up to roughly n≈5,000 and should say so rather than quietly thrashing.

### 1b. `rdkit_descriptors` swept the wrong overflow ceiling — **FIXED 2026-08-06**

Not a plan item; a live bug in #1, found the first time that engine ran on a real dataset
rather than on test fixtures. `descriptors-xgboost` died on BBBP with `Input data contains
inf` — precisely the error the featurizer's sweep was written to prevent.

RDKit's `Ipc` returns **6.5e39** on molecules like ivermectin. That is an ordinary
float64, so `np.isfinite` passed it through; XGBoost stores features as float32, whose
maximum is 3.4e38, and *its* cast produced the `inf`. The binding ceiling was never
float64's. One line in `chem/featurize.py` (`np.abs(rows) > _FLOAT32_MAX`, which catches
inf, NaN and the gap together), plus two regression tests carrying the actual molecule.
Two of three seeds at n=1000 crashed before; all three train after.

Worth generalising: the synthetic shapes the original tests used (`"C"*60`, a fused
polycyclic) overflow float64 and were caught. The molecules that actually break it sit in
the band between the two ceilings, and only real data contains them.

### 3. Frozen CheMeleon + TabPFN — foundation-model representation at small n

Replaces a fine-tune with a forward pass, and beat *fine-tuned* CheMeleon 86.2% to 41.4%
across 58 tasks. Verified: `BondMessagePassing(**ckpt["hyper_parameters"])` plus
`MeanAggregation` yields 2048-d vectors with no backprop. Designed for ≤10k rows, which
matches the same small-data regime as #2 but with a pretrained representation instead of a
kernel.

- New dependency `tabpfn` (8.2.0, py3.14 OK), with a **required** "Built with
  PriorLabs-TabPFN" attribution in the interface.
- **Open question worth answering before scheduling this:** the research verified frozen
  CheMeleon embeddings run **CPU-only**. The engine was assigned `lane="gpu"` because it
  needs the chemprop *dependency*, not because it needs the *hardware*. If the chemprop
  extra can be installed on a default-lane runner, this engine ships before
  `Dockerfile.gpu` too — which would move it up next to #1 and #2.

### 4. Descriptors into the chemprop predictor — the GPU-lane version of #1

Verified compatible with CheMeleon: molecule-level `x_d` descriptors concatenate *after*
aggregation, so encoder `output_dim` stays 2048 and the pretrained weights are untouched.
This is the ADMET-AI / MapLight+GNN configuration. Shares its featurizer with #1, so doing
#1 first makes this mostly wiring.

Gated on the GPU image.

### 5. Fine-tuned MoLFormer-XL — the missing architectural family

Sequence rather than graph, on a platform whose pitch is honest cross-engine comparison.
Apache-2.0, 187 MB, `transformers`-native. Expect no average accuracy gain; the capability
is the family itself, plus the low-label and structure-separated regimes where it helps.
Another full fine-tune, so it goes last of the model additions.

---

## Bigger than any single engine

Both of these expand capability more than #4 or #5 and are worth naming now even though
neither is an engine addition.

**Multi-task / multi-endpoint training.** One model across many assays, borrowing strength
between them — ADMET-AI's entire thesis, and natively supported by chemprop's `n_tasks`.
The platform hard-codes one target column per dataset (`y=np.array([float(target)])`, and
`_forward`'s `reshape(-1)`), so this is a domain and schema change rather than a new file.
Biggest capability on the horizon, largest blast radius.

**Transfer from a prior run's artifact.** "Start from the model I trained on my other
assay." Artifacts are already stored and addressable; this is a condition pointing at a
previous run rather than new science. Nothing else in this space offers it, and it is worth
more at n=200 than any architecture choice on this list.

---

## Deferred, with the reason it will come back

The measurement work — selection barrier, paired bootstrap, replication over split seeds —
is deferred by decision, since sweeps are not the current focus. Recording the boundary so
the tradeoff stays visible rather than forgotten:

- Measured noise: test-set composition gives a bootstrap CI half-width of ≈0.14 MCC at
  n=197; training seed sd 0.018; split seed sd 0.038 scaffold, 0.075 random.
- Consequence for this plan: the platform can *offer* these engines but cannot yet tell a
  user **which one to pick for their dataset**, because per-engine differences are smaller
  than split-seed noise. That is acceptable while the goal is breadth. It becomes the
  blocking problem the first time someone asks the platform to recommend rather than
  enumerate.
- One item stays live regardless of sweeps: the scorecard's green "Beats the baseline,
  +0.123" badge is unsupported at n=197 — the baseline's 0.514 sits inside the CI of
  [0.487, 0.764]. That is a false claim on screen for single runs, independent of sweeps,
  and adding five engines multiplies where it appears.

---

## Explicitly not doing

- **`EvidentialFFN`** — four peer-reviewed papers against the epistemic component (Meinert,
  AAAI 2023; Bengs, NeurIPS 2022; Juergens, ICML 2024; Shen, NeurIPS 2024). Chemprop's docs
  carry no warning and will let you select it silently.
- **A second CheMeleon-format checkpoint** — closed question, roster research §4.
- **MolE, MolGPS, MiniMol, Uni-Mol2, GROVER, MolCLR** — licence, missing weights, audited
  leakage, or refutation. Reasons in roster research §4.
- **Chemprop ensembles** — best blind-challenge evidence, but N× the fits on a lane already
  running three per job against a global 1800s deadline with per-lane timeouts removed.
- **Citing TDC leaderboard positions** — report against MapLight/CaliciBoost specifically.

# Roadmap — capability first

**Last updated:** 2026-10-04 (chemprop ensembles; earlier, traps and deferred items
revised after the beta-readiness pass, `docs/plans/2026-10-02-beta-readiness.md`). Companions: `engine-research.md`, which holds the external
evidence (licences, refuted models, what not to add) and is not re-derived here; and
`study-replication.md`, which measures the platform against one real completed study
(Mtb ERA → SAC3) and ranks what it would take to host one.

The goal is **expanding what the platform can do by bringing in more models**, ranked by
what a scientist can do after each one that they could not before — not by expected
accuracy delta. The roster research established that no candidate reliably beats a tuned
descriptor baseline at 10²–10⁴ compounds and that blind-challenge top tiers are
statistically indistinguishable, so under a capability goal the third decimal place is
not the argument. The regime an engine opens is.

---

## Hardware: three backends, all real

| Backend | How it runs | Status |
|---|---|---|
| Apple Silicon / MPS | native runner agent, `make dev-worker-gpu` | works, verified |
| CUDA | `Dockerfile.gpu`, x86_64, built on `ned` | built and verified 2026-08-06 |
| CPU | fallback only | last resort, not a target |

**The gpu lane is not a special case and nothing is blocked on it.** `lightning`'s
`accelerator="auto"` resolves to `MPSAccelerator` / `mps:0` on this machine and a full
chemprop fit *and* predict both complete there. GPU-lane engines are developed and
exercised locally on real GPU hardware; the CUDA image is for production throughput, not
for correctness.

**Measured 2026-08-06, against the old claim that MPS is slower:** a frozen CheMeleon
encoder pass runs at **1.7 ms/molecule on `mps:0` against 6.0 ms/molecule on CPU** — the
GPU is 3.5× faster for this workload. The earlier "MPS is slower here" note came from
100-molecule batches where per-op overhead dominated, and it should not be generalised to
real workloads without re-measuring.

An earlier version of this document ranked engines partly by how little GPU they needed,
and `engine-research.md` §5 went further and called the lane "unproven". Both were wrong,
both were inherited from a phase-1 assumption that no GPU existed anywhere, and the
correction is recorded rather than quietly applied so the reasoning does not come back.
Runs name the device they trained on in their progress phase (`training chemprop-dmpnn on
mps:0`), because nothing validates that a runner registered for the gpu lane owns a GPU
and the three devices do not produce identical numbers.

---

## The roster

| Engine | Lane | Representation | Opens |
|---|---|---|---|
| `ecfp4-randomforest` | default | 2048 Morgan bits | the mandatory baseline |
| `ecfp4-xgboost` | default | 2048 Morgan bits | boosting on fingerprints |
| `ecfp4-lightgbm` | default | 2048 Morgan bits | leaf-wise boosting, sparse-feature bundling, the fastest fit here |
| `descriptors-xgboost` | default | 217 RDKit descriptors | the recipe behind every reproducible TDC entry |
| `tanimoto-gp` | default | 2048 Morgan bits | small n, posterior variance (regression only) |
| `chemprop-dmpnn` | gpu | learned graph | learned representations, ±CheMeleon, ensembles whose spread is the uncertainty |
| `molformer-xl` | gpu | SMILES tokens | the sequence family, frozen or fine-tuned |

Added 2026-08-06 on request, to replicate the Mtb ERA → SAC3 study — see
`study-replication.md` for what that study needed and what is still missing.
`ecfp4-lightgbm` stands in for its LightGBM member; `molformer-xl` is the same
architecture family as its MolFormer-XL member. Note that study's own re-analysis found
MoLFormer-XL had the **best** global AUROC (0.556) and the **worst** top-100 behaviour
(20 hits against LightGBM's 37) — it raises the number in the paper while lowering the
number that matters. There is no `descriptors-lightgbm`: on dense descriptors LightGBM
and XGBoost converge to near-identical models, and the sparse fingerprint is where the
two libraries actually diverge.

### Ensembles

`chemprop-dmpnn`'s `ensemble_size` (1–10, default 1) trains that many models, each from
its own seed, predicts their mean and reports their standard deviation as the
uncertainty. Shipped 2026-10-04. It sat in "not doing" until per-lane deadlines and
resumable runs removed the reason: the run's time limit now scales with the ensemble
size, and a stopped ensemble resumes in the model it stopped in.

Measured through the engine on real data, scaffold splits, seeds 0–2, one model against
five (MPS, default settings):

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| ESOL test RMSE | 0.651 → 0.603 | 0.733 → 0.651 | 0.851 → 0.803 |
| ESOL RMSE, confident half / uncertain half | 0.49 / 0.70 | 0.49 / 0.78 | 0.66 / 0.92 |
| BBBP test AUROC | 0.925 → 0.923 | 0.869 → 0.882 | 0.855 → 0.882 |
| BBBP error-ranking AUROC, spread / boundary | 0.805 / 0.837 | 0.709 / 0.748 | 0.730 / 0.776 |

- **Regression: the spread is informative.** The confident half's RMSE is 28–38% below
  the uncertain half's in every seed, and five models beat one in every seed.
- **Classification: the spread is not better than the proxy.** It mostly mirrors the
  distance from the boundary and ranks misclassifications slightly worse. The same held
  on all four labels of the 10k nuisance sample with three models (error-ranking AUROC
  0.76–0.85 for the spread against 0.80–0.88 for the boundary), and there the spread did
  not track Tanimoto similarity to the nearest training compound either (|ρ| < 0.05).
  Do not present it as having closed the classification gap.
- **CheMeleon members agree about three times more closely** (mean spread 0.075 against
  0.209 on ESOL seed 0), as the setting's help text says, and five gave no gain over one
  (RMSE 0.559 against 0.550). The spread still ranked errors: confident half 0.41,
  uncertain half 0.68.
- **Cost is linear.** About 4.8–4.9× the wall time for five models, pretrained or not;
  three models on the 10k four-label sample took 421 s. A model in an ensemble is stored
  without optimizer state, so it is a third of a one-model checkpoint: 1.3 MB against
  3.85 MB at default size, 37.3 MB against 112 MB with CheMeleon.
- Single runs per seed, and three seeds. Inside the split-seed noise recorded below, so
  this is a direction, not a claim.

### What the roster still cannot do

1. ~~One model across several endpoints.~~ Shipped 2026-10-03 (multi-task labels):
   chemprop and MoLFormer learn every target jointly, other engines fit one per target.
2. **Principled uncertainty on the classification path.** The GP gives a real posterior
   for regression. Since 2026-10-04 a chemprop ensemble reports its models' spread for
   classification too, but measured on BBBP it is not better than the
   distance-from-the-boundary proxy every other classifier reports: it mostly mirrors it
   (rank correlation 0.87–0.96) and ranks errors slightly worse in all three seeds. See
   "Ensembles" below. Still open.
3. **Start from a prior run's artifact.** "Train from the model I fitted on my other
   assay." Artifacts are already stored and addressable, so this is a condition pointing at
   a previous run rather than new science — and it is worth more at n=200 than any
   architecture choice on this list. (Retrying a *failed or cancelled* run in place
   shipped 2026-10-02 — `POST /runs/{id}/retry` — but that re-executes the same
   instructions; it is not transfer.)
4. **Combine two engines' predictions.** No averaging, no stacking, no meta-learner
   anywhere. The roster can now cover a study's members but not its ensemble; see
   `study-replication.md` for why the fan-out comparison is worth more than the stacker.

---

## Ranked next

### 1. Transfer from a prior run's artifact

The cheapest large capability left. No new science, no new dependency. The OpenADMET ×
ExpansionRx organizers' first lesson was that more data won, and on this platform reusing
a model fitted on another assay is how a run gets more data.

*(Shipped from this list: fine-tuned MoLFormer-XL on 2026-08-06; descriptors into the
chemprop predictor and multi-task training on 2026-10-03, as training options and
multi-task labels.)*

---

## Traps

Carried forward deliberately — each of these has already cost a session.

- **A green `docker build` says nothing about whether the image works.** The first
  successful `Dockerfile.gpu` build died on `import chemprop` with a missing
  `libXrender.so.1`; the fix was found by *running* the image. Run it.
- **A green test suite says nothing about whether an engine works on real data.** Two
  separate bugs shipped past a full green suite and were caught by the first real dataset:
  the libXrender import, and `rdkit_descriptors` sweeping float64's ceiling when XGBoost's
  float32 cast was the one that bound (RDKit's `Ipc` returns 6.5e39 on molecules like
  ivermectin — an ordinary float64, an infinity once cast). Synthetic test molecules sit
  outside the band that breaks things. Run every new engine on a real dataset.
- **The locked torch is `2.13.0+cu130` — CUDA 13.0 — while the base image tag says
  12.4.1.** PyPI's linux wheels vendor their own CUDA userspace so this is fine, but the
  *host driver* must be new enough for CUDA 13, not 12.4, and the tag misleads.
- **Per-lane job timeouts exist since 2026-10-02:** `STUDIO_WORKER_JOB_TIMEOUT_BY_LANE`
  (JSON, e.g. `{"gpu": 7200}`) overrides the server-wide `STUDIO_WORKER_JOB_TIMEOUT`
  per lane in the claim response, and the deadline is now also checked *between*
  fits, which is the only check the tree and GP engines ever reach. Predictions
  have no cooperative deadline, but the runner agent now hard-kills: past the
  deadline plus a 5-minute grace it records the failure and exits the process (the
  only way to stop a thread), and the container restarts clean.
- **A green `docker build` of the CPU image said nothing either.** LightGBM's wheel
  links `libgomp.so.1`, which `python:3.13-slim` does not ship; the API image built
  clean for two months and crashed at import. Found 2026-10-02 by booting the image.
  `make image-smoke` and CI now import the app and every engine inside every image
  they build. The same lesson as the GPU image's X11 libraries, one bullet up.
- **`uncertainty` means five different things across the roster** and the triage grid sorts
  across all of them. Only the GP's regression path is a true posterior spread.
- **The Scorecard's green "Beats the baseline, +0.123" badge was unsupported at n=197** —
  the baseline's 0.514 sat inside a CI of [0.487, 0.764]. Since 2026-10-02 the
  Scorecard carries a 95 % bootstrap interval on the primary metric
  (`primary_metric_ci`, 1000 resamples of the test set, fixed seed) and the verdict
  reads "within this test set's sampling noise" whenever the baseline's number falls
  inside it. The interval is **unpaired** — the baseline's per-compound predictions
  are still not persisted — so a paired test of the difference remains open
  (`publication-grade-gaps`, item 3).
- **The whole test suite segfaults locally once the gpu extra is installed.** Exit 139,
  in `tests/unit/engines`, and it is the dual-OpenMP collision `Dockerfile.gpu` already
  sets `OMP_NUM_THREADS=1` for — RDKit, sklearn, LightGBM and torch each vendor their
  own libomp on macOS. Verified pre-existing on 2026-08-06 by stashing every change and
  reproducing it on a clean checkout, so it is **not** caused by any one engine, and it
  is not memory (the largest model here is 182 MB against 16 GB, and macOS OOM sends
  SIGKILL/137, not SIGSEGV/139). `OMP_NUM_THREADS=1` pushes the crash later but does not
  remove it. Until someone fixes it properly, verify engine work by running each test
  file in its own process; they all pass that way.
- **MoLFormer's default configuration is not reproducible.** Its linear attention draws
  random Fourier features and `MolformerFeatureMap.forward` redraws them on *every*
  forward pass unless `deterministic_eval=True` is passed to `from_pretrained` — so the
  same molecule scores differently on each call, silently. The flag is set and there is
  a test that fails if it is dropped.
- **`molformer-xl` executes code downloaded from the Hugging Face Hub.** The architecture
  is not in `transformers`; `trust_remote_code=True` is mandatory. `_REVISION` pins it to
  one immutable commit so what runs on a worker cannot change under a published Protocol.
  Bump it having read the diff, never to a branch name.
- **`ned` and `orca` are remote university machines over SSH.** Builds there are real
  resource use on someone else's infrastructure.
- **The runner agent does not hot-reload.** `make dev-worker` / `make dev-worker-gpu`
  after any engine or training change, or you debug a fix that never loaded.

---

## Deferred, with the reason it will come back

The measurement work — selection barrier, paired bootstrap, replication over split seeds —
is deferred by decision while the goal is breadth. Recording the boundary so the tradeoff
stays visible:

- Measured noise: test-set composition gives a bootstrap CI half-width of ≈0.14 MCC at
  n=197; training seed sd 0.018; split seed sd 0.038 scaffold, 0.075 random.
- **Consequence:** the platform can *offer* these engines but cannot yet tell a user
  **which one to pick for their dataset**, because per-engine differences on a
  BBBP-sized set are smaller than split-seed noise. Acceptable while the goal is breadth.
  It becomes the blocking problem the first time someone asks the platform to recommend
  rather than enumerate.
- **Benchmark on datasets that can carry the claim. BBBP cannot.** It is 2,039 molecules
  at 75% positive, small enough and skewed enough that most engines look alike on it, and
  every "these engines are within noise of each other" conclusion above came from it.
  Measured 2026-08-06 on HIV (41,127 molecules, 3.5% positive, scaffold split, n=1000):
  `ecfp4-randomforest` scored **MCC 0.000** and `tanimoto-gp` **0.000** — both predict
  all-negative — while `descriptors-xgboost` scored **0.465**. The engines are not
  interchangeable; BBBP just could not tell. New engines get measured on an imbalanced
  set of real size before any claim is made about them.

---

## Explicitly not doing

- **`EvidentialFFN`** — four peer-reviewed papers against the epistemic component
  (Meinert AAAI 2023; Bengs NeurIPS 2022; Juergens ICML 2024; Shen NeurIPS 2024).
  Chemprop's docs carry no warning and will let you select it silently.
- **A second CheMeleon-format checkpoint** — none exists; closed question, research §4.
- **MolE, MolGPS, MiniMol, Uni-Mol2, GROVER, MolCLR** — licence, missing weights, audited
  leakage, or refutation. Reasons in research §4.
- **TabPFN, in any pairing — including frozen CheMeleon + TabPFN.** Built and then removed
  on 2026-08-06. `engine-research.md` §3.2 still carries the evidence for it; this entry is
  the reason it is not being acted on. Three findings, in order of how much they cost:
  1. **Inference is too slow for this platform, and by an unknown factor.** A working
     engine stalled for minutes on a 750-molecule training set inside the real `train()`
     path, against a global 1800s deadline. An isolated probe at 450 molecules had
     suggested ~10s, so the cost does not scale the way a spot check implies — and nobody
     measured where it actually lands.
  2. **The current model versions are licence-gated.** v2.5, v2.6 and v3 need a Prior Labs
     account and a `TABPFN_TOKEN` before their weights download at all. Only v2 is
     ungated, so the engine would ship pinned to a superseded release.
  3. **Its licence has an attribution clause** (Prior Labs License §10): "Built with
     PriorLabs-TabPFN" must appear in the interface, and any model name must begin with
     "TabPFN". Satisfiable, but it is a permanent constraint on naming and UI copy in
     exchange for one engine.
- **Citing TDC leaderboard positions** — report against MapLight/CaliciBoost specifically.

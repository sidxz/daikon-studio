# Roadmap — capability first

**Last updated:** 2026-08-06. Companion: `engine-research.md`, which holds the external
evidence (licences, refuted models, what not to add) and is not re-derived here.

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
| `descriptors-xgboost` | default | 217 RDKit descriptors | the recipe behind every reproducible TDC entry |
| `tanimoto-gp` | default | 2048 Morgan bits | small n, posterior variance (regression only) |
| `chemprop-dmpnn` | gpu | learned graph | learned representations, ±CheMeleon |

### What the roster still cannot do

1. **A sequence model.** Everything is graph, fingerprint or descriptor. Open — MoLFormer-XL.
2. **One model across several endpoints.** chemprop supports `n_tasks > 1` natively; the
   platform hard-codes one target column (`y=np.array([float(target)])`, and `_forward`'s
   `reshape(-1)`). Largest remaining gap, and a domain/schema change rather than a new file.
3. **Principled uncertainty on the classification path.** The GP gives a real posterior
   for regression. Every classification engine reports distance-from-the-boundary, which
   is a proxy.
4. **Start from a prior run's artifact.** "Train from the model I fitted on my other
   assay." Artifacts are already stored and addressable, so this is a condition pointing at
   a previous run rather than new science — and it is worth more at n=200 than any
   architecture choice on this list.

---

## Ranked next

### 1. Descriptors into the chemprop predictor

Molecule-level `x_d` descriptors concatenate *after* aggregation, so the encoder's
`output_dim` stays 2048 and pretrained CheMeleon weights are untouched — verified. This is
the ADMET-AI / MapLight+GNN configuration, and it shares its featurizer with
`descriptors-xgboost`, so most of the work is wiring.

### 2. Multi-task training

See "what the roster still cannot do" #2. Bigger than any single engine: one model across
many assays, borrowing strength between them, which is ADMET-AI's whole thesis. Largest
blast radius on this list — it touches the Dataset schema, `TargetSpec`, the training
context and the Scorecard.

### 3. Transfer from a prior run's artifact

The cheapest large capability left. No new science, no new dependency.

### 4. Fine-tuned MoLFormer-XL

The missing architectural family on a platform whose pitch is honest cross-engine
comparison. Apache-2.0, 187 MB, `transformers`-native. Expect no average accuracy gain;
the capability is the family itself, plus the low-label and structure-separated regimes.

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
- **Per-lane job timeouts do not exist.** One server-wide `STUDIO_WORKER_JOB_TIMEOUT`
  serves every runner regardless of lane, which is why no sweep has yet included a
  chemprop config. The fix is per-lane timeouts, not avoiding the engine.
- **`uncertainty` means five different things across the roster** and the triage grid sorts
  across all of them. Only the GP's regression path is a true posterior spread.
- **The Scorecard's green "Beats the baseline, +0.123" badge is unsupported at n=197** —
  the baseline's 0.514 sits inside a CI of [0.487, 0.764]. A false claim on screen for
  single runs, and every added engine multiplies where it appears. This is the one
  measurement item that is live independently of sweeps.
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
- **Chemprop ensembles** — best blind-challenge evidence, but N× the fits against a global
  deadline with no per-lane timeout.
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

# Engine roster research — what to add after CheMeleon

**Date:** 2026-08-04 · **Branch:** `baselines-and-pretrained` · **Status:** research only, nothing implemented

Question asked: which recent models are beating benchmarks or widely used enough to add
to the engine roster? Current roster is ECFP4+RandomForest (the mandatory baseline),
ECFP4+XGBoost, and Chemprop D-MPNN with optional CheMeleon pretrained weights.

The short answer is that the interesting additions are **conditions on the existing GPU
engine**, not new engines — and that the roster's weakest link is the baseline's
representation, not the absence of a foundation model.

---

## 1. Verified locally (not taken from the literature)

These were run against the repo's own `.venv` and the cached CheMeleon checkpoint at
`~/.cache/daikon-studio/weights/chemeleon_mp.pt`. Re-runnable; each was a throwaway
script, so re-derive rather than trusting these numbers blindly after any dependency bump.

| Claim | Result |
|---|---|
| RDKit descriptor count, no new dependency | **217** via `Descriptors.descList` |
| Descriptor featurization cost, `OMP_NUM_THREADS=1` | **3.41 ms/mol** → 34s per 10k mols, 171s per 50k |
| Frozen CheMeleon embeddings without fine-tuning | **(n, 2048)** float32, distinct per molecule, CPU-only, no new code |
| Molecule-level `x_d` descriptors vs CheMeleon | encoder `output_dim` stays **2048**; predictor widens to 2048+d. **Weights untouched** |
| `MveFFN` output shape | **(n, 1, 2)** vs `RegressionFFN`'s **(n, 1)** |
| TabPFN 8.2.0 | `requires_python >=3.10`, classifiers include **3.14** (this venv runs 3.14.4) |
| TabPFN license | Apache-2.0 **+ Section 10**: mandatory "Built with PriorLabs-TabPFN" attribution. Not a commercial restriction |
| chemprop version in venv | **2.3.0** |

Two of these are load-bearing and worth restating:

**Molecule-level descriptors compose with CheMeleon.** They concatenate *after*
aggregation, so they never touch the message-passing block. This is what makes the
"Chemprop + RDKit descriptors + pretrained encoder" recipe possible here — it is the
configuration behind ADMET-AI and, in substance, MapLight+GNN.

**`MveFFN` will silently corrupt predictions unless `_forward` is fixed first.**
`chemprop_dmpnn.py:166` does `torch.cat(batches).cpu().numpy().reshape(-1)`, which
assumes one target per molecule. With `MveFFN` (`n_targets = 2`) that flattens means and
variances into one interleaved vector — 8 values for 4 molecules — with no error raised.

---

## 2. The evidence base

### The headline finding

Across every 2024–2026 study that tuned both sides with a **matched budget**, pretrained
molecular foundation models do not reliably beat well-built descriptor baselines at
typical dataset sizes (10²–10⁴ compounds). Every study reporting a decisive
foundation-model win compared against a *leaderboard*, not against a baseline the authors
tuned themselves. That asymmetry is the single most consistent signal in the research.

- **Praski, Adamczyk & Czech**, [arXiv:2508.06199](https://arxiv.org/abs/2508.06199) —
  25 pretrained models × 25 datasets, scaffold splits, hierarchical Bayesian
  Bradley-Terry with a ROPE. Only **CLAMP** (itself fingerprint-based) was statistically
  superior to count-ECFP. MoLFormer scored 79.80% mean AUROC vs ECFP's **79.89%**.
  *Caveat that matters for us: frozen embeddings, so this understates fine-tunable models.*
- **Deng et al.**, [Nat Commun 14, 6395 (2023)](https://www.nature.com/articles/s41467-023-41948-6)
  — 62,820 models, 30 seeds. RF on RDKit2D beat MolBERT and GROVER on BACE, BBBP, ESOL, Lipop.
- **Dias, Bustillo & Rodrigues**, [Nat Commun 14, 6394 (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575963/)
  — deep learning competitive only above ~1,000 training examples.
- **Jiang et al.**, [J Cheminform 13, 12 (2021)](https://link.springer.com/article/10.1186/s13321-020-00479-8)
  — equal TPE budget both sides; descriptor models took **24/33 (73%)** of top-three slots.
- **Sun, Dai & Yu**, [NeurIPS 2022](https://arxiv.org/abs/2207.06010) — SSL graph
  pretraining has no statistically significant advantage; *hyperparameters mattered more
  than the pretraining objective*. Also: GROVER's non-deterministic "balanced scaffold
  split" inflates BBBP by 5–20%.

### Representation hierarchy

From Ben Hicham, Rittig, Grohe & Mitsos, [arXiv:2604.16123](https://arxiv.org/abs/2604.16123)
(58 tasks across Polaris + MoleculeACE), win rate by representation feeding TabPFN:

| Representation | Win rate |
|---|---|
| Frozen CheMeleon | **86.2%** |
| Mordred descriptors | 67.2% |
| RDKit2D descriptors | 56.9% |
| **Fine-tuned CheMeleon** | **41.4%** |
| Morgan fingerprints | **22.4%** |

**Rich descriptors ≫ hashed fingerprints.** An ECFP4-only baseline is a weak baseline by
2026 standards — which contradicts the stated intent in `ecfp4_randomforest.py`'s
docstring ("a genuinely competitive model, not a strawman").

Note also that frozen-plus-TabPFN beat *fine-tuned* CheMeleon by more than 2×. This is
an accuracy result, not a cost result.

### Blind challenges (the closest thing to ground truth)

- **Polaris/ASAP antiviral, 2025** (66 participants) — potency won by MolE at 0.509 MAE.
  But [Inductive Bio's post-mortem](https://www.inductive.bio/blog/lessons-from-the-polaris-admet-competition)
  reports the gap between the winner and their own fingerprint baseline was **1.7%**.
  A Gaussian process with a Tanimoto kernel placed 13th, ahead of most deep learning.
  MolE placed **10th** on ADMET. Assay noise floor for pIC50 is ~0.3 log units against a
  winning MAE of 0.509 — very little headroom remains.
- **OpenADMET × ExpansionRx, 2026** (370+ groups, 4,000+ submissions) — top 10 dominated
  by Chemprop and Chemprop hybrids; **all of the top five were ensembles or multi-model
  selections**; CheMeleon "widely adopted". Organizers: hyperparameter tuning gave
  "minimal performance gain"; 4 of top 5 used proprietary data.
- **OpenADMET PXR, 2026** — the top tier contains **28 statistically equivalent entries**
  (BH-FDR 5%). 1st-to-5th spans 0.006 MAE against a ±0.028 CI. The "winner" is a coin flip.

The axis that separated entries was **task-relevant data and ensembling**, not
architecture or pretraining scale.

### Benchmark integrity — relevant to publication

**Citing TDC leaderboard positions is now a liability.**
[bioRxiv 10.64898/2026.02.26.708193](https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1)
(Feb 2026) audited the top three entries on all 22 ADMET endpoints for reproducibility,
leakage and HPO hygiene. Only **CaliciBoost, MapLight and MapLight+GNN** passed every
check — all three descriptor+boosting. **Data leakage was identified in MiniMol**, which
holds 7 of the 8 pretrained-model first places. The board has not been updated since
2025-07-13, and the `Bioavailability_Ma` page currently serves corrupted data grafted
from `Pgp_Broccatelli`.

**MoleculeNet is not a valid basis for a 2026 novelty claim.** Its benchmark repo has had
no commits since April 2021. Under the recommended random split, 76% of ESOL test
molecules have a training neighbour at ECFP4 Tanimoto > 0.4; HIV leaks on 56% *under
scaffold split*. BACE has undefined stereocentres on 71% of molecules; BBBP has 59
duplicate structures with 10 conflicting labels.

If we report against TDC, report against MapLight/CaliciBoost specifically and say why.
Relevant to the open publication-grade gaps.

---

## 3. Ranked additions — GPU-first

The platform targets GPU (`Dockerfile.gpu`, CUDA 12.4, CheMeleon baked into the image).
CPU-friendliness is **not** a ranking criterion. The right axis is **GPU-seconds added
per job**, because a chemprop-vs-chemprop run is already *three fits in one job*
(chosen, baseline, random-split) against a 1800s default timeout, with CheMeleon's
2048-wide/depth-6 encoder rather than 300/3.

Anything that adds a **fit** is expensive. Anything that adds only a **forward pass** is
nearly free.

### 3.1 RDKit descriptors into the chemprop predictor — *free, best evidence*

A condition on the existing engine, not a new engine. Verified compatible with CheMeleon
(§1). No new dependency — RDKit is already a core dep. Costs 3.41 ms/mol, single-threaded.
This is the ADMET-AI / MapLight+GNN recipe and the highest-evidence configuration found.

### 3.2 Frozen CheMeleon + TabPFN — *removes a fit*

No backprop at all: one encoder forward pass, then in-context inference. It **replaces**
a fine-tune rather than adding one, while beating fine-tuned CheMeleon 86.2% to 41.4%
(§2). For a lane whose known failure mode is running out of time, an arm that makes runs
cheaper *and* more accurate is unusually well-matched.

- New dependency: `tabpfn` (8.2.0, py3.14 OK).
- Requires chemprop for the encoder, so `lane="gpu"`.
- **Licence obligation:** "Built with PriorLabs-TabPFN" must appear in the interface.
- Designed for ≤10k rows — fine for ADMET, wrong for very large sets.

### 3.3 `MveFFN` uncertainty head — *free, closes a known gap*

One head swap, no extra fit. Fills the `uncertainty` column that `predict()` currently
hard-codes to null, with an existing `ponytail:` comment naming exactly this upgrade path.
**Blocked on** fixing `_forward`'s `reshape(-1)` first (§1) — otherwise predictions come
back silently interleaved.

### 3.4 Chemprop ensembles — *best evidence, worst current fit*

Every one of the OpenADMET top five was an ensemble. But N=5 multiplies the exact
resource already identified as the most likely first-run killer. **Defer** until the
timeout is raised and the GPU image has actually been built.

### 3.5 Fine-tuned MoLFormer-XL — *architectural contrast, later*

Worth having as a genuinely different family (sequence, not graph) on a platform whose
value proposition is honest engine comparison. Apache-2.0, 187 MB, `transformers`-native,
280K monthly HF downloads. Expect no average gain — but it helps in the low-label and
structure-separated regimes. Another full fine-tune, so it belongs after the lane is proven.

**Do not rank this on the frozen-embedding benchmarks.** Those measure a CPU-regime use
of the model and say nothing about fine-tuning it.

---

## 4. Skip list, with reasons

Evidence-driven, so unaffected by the GPU premise:

| Model | Why not |
|---|---|
| **MolE** | CC-BY-NC-4.0 (non-commercial) **and** the repo's "Download pretrained models" section still says `TODO`. Zenodo deposit is code only. Not integrable at any price. |
| **MolGPS** | No public weights. Graphium unpushed since May 2025. |
| **MiniMol** | Flagged for **data leakage** in the Feb 2026 TDC audit. Also drags in `torch-scatter`/`torch-sparse`/`torch-cluster`, which is worse under CUDA, not better. |
| **Uni-Mol2** | 0 of 22 first places in the only independent head-to-head. Requires per-molecule ETKDG conformers — real cost, real failure modes. *Softened: "GPU recommended" was never a fair objection here, and it placed 6th at OpenADMET.* |
| **GROVER, MolCLR** | Refuted at NeurIPS 2022. GROVER additionally cannot be converted to chemprop v2 — the converter hard-codes v1 D-MPNN key names. |
| **Boltz-2 as an FM** | The most interesting result in the space (best non-ensemble on 9/22 TDC), but 16 GitHub stars and **no licence**. Watch, don't adopt. |
| **ADMET-AI** | Use as a comparator only. Gave **negative R²** on Caco-2 and HLM CLint against novel program chemistry. Never present its output as zero-shot truth. |

### Closed question: a second CheMeleon-format checkpoint

**None exists.** Chemprop v2.3.1's `FoundationModels` enum has exactly one member, and no
second has been added across the six releases since foundation support shipped in v2.2.0.
The `--from-foundation <path>` branch expects a full `MPNN.load_from_file`, not our
two-key MP-block dict — so our loading pattern is CheMeleon-specific by construction. Our
pinned MD5 already matches the current Zenodo record (15460715); the other record
(15426601) is a superseded upload, not a sibling model. There is nothing to add and
nothing to update. **Do not re-research this.**

Watch out for two false positives: `hspark1212/chemeleon2-checkpoints` on HF is an
unrelated *crystal-structure* diffusion model that happens to share the name, and
`openadmet/*-chemeleon-*` are downstream fine-tunes, not foundation weights.

---

## 5. Sequencing — read before implementing any of the above

Everything here lands on a lane that **has never executed**. Per the open remaining-work
notes:

1. `Dockerfile.gpu` **has never been built once.** Needs one real x86 build
   (`docker --context ned build`); this laptop is arm64 and the emulated build stalls.
2. `STUDIO_WORKER_JOB_TIMEOUT` needs raising on the gpu lane before a real BBBP run —
   the branch made the three-fit case the default one.
3. There is no CI building the GPU image.

Adding engines to an unproven lane stacks unverified on unverified. The first move is
not an engine.

Suggested order once unblocked: **3.1** (free, best evidence) → **3.3** (free, closes a
product gap, needs the `_forward` fix) → **3.2** (removes a fit) → raise timeout, build
image → **3.4** → **3.5**.

---

## 6. Not verified

- Full text of the bioRxiv TDC audit — biorxiv.org and ResearchGate both 403 automated
  fetches. Authors, DOI and abstract-level findings confirmed across two independent
  retrievals; per-model re-evaluated rankings and exact leakage mechanisms are not.
- Pseudonym → organization mappings for the OpenADMET challenges (`pebble`,
  `matcha-croissant`) rest on Inductive Bio's own press releases.
- CIKM 2025 exact tables — ACM CAPTCHA-walled; numbers came from indexed PDF text.
- Whether `MveFFN` interacts correctly with the `UnscaleTransform` output transform for
  regression. The variance channel's scaling was **not** checked and must be before use.
- No benchmark of any recommendation was run on this platform's own data. Every accuracy
  number here is from the literature.

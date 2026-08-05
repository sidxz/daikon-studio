# Dataset & Protocol diagnostics — design

Date: 2026-08-04
Status: approved, in implementation

## Problem

The Dataset detail page and the Protocol detail page both under-display data the
system has already computed. Neither page has a single chart; the frontend has no
charting dependency at all.

Three concrete instances of "computed then discarded":

- `application/execution/build_scorecard.py:62` computes a nearest-neighbour
  Tanimoto for every test compound, then collapses the whole distribution to one
  fraction for a progress bar.
- `application/data/assign_split.py:_scaffold_labels` computes a Murcko scaffold
  for every row while splitting and throws them away.
- The `ScorecardInputs` blob holds `actual[]`/`predicted[]` for the entire test
  set; only the twenty worst residuals reach the API.

Three defects found while mapping:

- `split_strategy` is on the wire and never rendered, though `scorecard.py`'s own
  docstring calls which split produced a number "the single most important fact
  about how flattering it is allowed to be."
- `worst_rows[].similarity` is on the wire and unused, though `WorstRow`'s
  docstring says it exists so out-of-distribution compounds can be flagged.
- `METRIC_LABELS` defines `roc_auc`/`pr_auc`; `_scoring.py` emits `auroc`/`auprc`,
  so classification scorecards render raw metric keys.

## Approach

Add two read-only diagnostic surfaces and the charts that render them. No change
to what training persists, so no backfill: every new protocol number is derived
from the existing immutable blob at request time, and the dataset profile is
derived from the existing immutable snapshot Parquet.

### Dataset profile

A new `DatasetProfile` domain artifact, computed from the frozen snapshot and
cached as a JSON blob beside it.

Computed **lazily on first request**, not at freeze time. Datasets are immutable
and content-addressed, so a cached profile can never go stale and there is
nothing to invalidate. Lazy also keeps a multi-second RDKit pass off the upload
request and makes the feature work for datasets that already exist, which
freeze-time computation would not.

Contents, all keyed to a decision a scientist actually makes:

| Section | Answers |
|---|---|
| Target distribution (train/test overlaid, or class balance per split) | Is the metric meaningful? Is the dynamic range wide enough? Did the split skew the classes? |
| Nearest-neighbour Tanimoto, test → train | Is this split a real generalization test or theatre? |
| Near-duplicate leakage (test rows at NN ≥ 0.95) | Is the score inflated by compounds the model has effectively already seen? |
| Scaffold diversity: unique/singleton counts, cumulative coverage, top scaffolds as structures, cross-split overlap | Congeneric series or diverse deck? Did the scaffold split actually separate scaffolds? |
| Physicochemical descriptors, train vs test | Is this drug-like space? Are the partitions matched? |
| Descriptor–target rank correlation | Would one trivial descriptor explain most of the target? |
| Activity cliffs | Which near-identical pairs disagree, and what ceiling does that put on accuracy? |

### Scorecard diagnostics

New fields on `Scorecard`, computed inside `build_scorecard` from data already
passed to it:

- `parity` — per-test-row `(actual, predicted, similarity)`, downsampled with an
  explicit `sampled_from` count. Serves the parity plot for regression and the
  probability-by-true-class view for classification.
- `residual_histogram` — signed residuals, regression only.
- `error_by_similarity` — mean absolute error binned by nearest-neighbour
  Tanimoto. Turns the applicability claim from an assertion into evidence.
- `scaffold_errors` — median absolute error per Murcko scaffold. Promotes the
  worst-twenty section from anecdote to systematic.
- `calibration` — predicted-probability bins vs observed positive rate,
  classification only.

### Run history

`runs.protocol_id` has an index created specifically for this
(`007_run_protocol_link.py:56`) and no query uses it. Add a `protocol_id` filter
through `RunRepository.list` → `ListRuns` → `GET /api/v1/runs`, and a run-history
section on the protocol page.

### Charting

`@observablehq/plot`, one dependency, wrapped in a single `PlotFigure` client
component. Chosen over Recharts (three to four lines per chart instead of twenty)
and over hand-rolled SVG (which stops being the smaller diff around the fourth
chart type). Histogram bins are computed on the backend so payloads stay small
and the chart layer stays dumb.

## Layering

`application` may not import `infrastructure`, so RDKit reaches the profile
through `StructureNormalizer`, which grows `descriptors` and
`high_similarity_pairs` — the same rationale that already put `murcko_scaffold`
and `nearest_neighbour_tanimoto` on that port rather than splitting off
single-method ports.

Domain modules (`domain/data/profile.py`, additions to
`domain/execution/scorecard.py`) stay plain dataclasses with no polars, numpy or
RDKit, matching `domain/data/validation.py`.

## Known ceilings

- Activity-cliff detection is O(n²) in fingerprint comparisons. Capped at a
  deterministic subsample with the cap reported in the payload, never silently
  truncated.
- The profile is computed in one pass over the snapshot held in memory, matching
  what training and prediction results already do.

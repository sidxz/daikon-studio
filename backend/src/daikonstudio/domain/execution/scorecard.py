"""The Scorecard -- the screen where a scientist decides whether to trust a model.

Every honesty property the platform claims is expressed here or nowhere: MCC
(never plain accuracy) leads for classification, the mandatory baseline sits
beside the primary result, and the worst twenty predictions are structures a
chemist can look at, not a single R^2 they cannot act on.

Plain dataclasses only -- no polars/numpy/rdkit, matching `domain/data/validation.py`.
Fields are typed `str`, not the `application`-layer `TaskType` enum, because a
domain module may not import `application` (Domain purity contract); the string
values (`"rmse"`/`"mcc"`, `"value"`/`"probability"`) are themselves the self-describing
contract a consumer reads, so nothing is lost by not carrying the enum.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class WorstRow:
    """One of the twenty largest absolute residuals on the test split.

    `residual` is always `abs(actual - predicted)`. For a regression Scorecard
    that is a residual in the target's own unit; for a classification one,
    `actual` is a 0/1 label and `predicted` is P(class=1), so it is a probability
    residual instead -- a different quantity with the same name. Rather than
    silently change what the field means, the owning `Scorecard.prediction_kind`
    says which one it is; a consumer must read that field to know, not guess from
    context.

    `similarity` is this structure's own nearest-neighbour Tanimoto to the
    training set, so the triage grid can flag individual out-of-distribution
    compounds. `None` when the training set was empty and no similarity could be
    computed at all -- never a fabricated `0.0`, which would misread as "confirmed
    far from anything trained on" rather than "unknown".
    """

    structure: str
    actual: float
    predicted: float
    residual: float
    scaffold: str
    similarity: float | None


@dataclass(frozen=True, kw_only=True)
class Scorecard:
    """The head-to-head: this model, the mandatory baseline, and where it fails.

    `primary_metric` is `"rmse"` for regression and `"mcc"` for classification --
    never `"accuracy"`, which is not computed anywhere upstream and so cannot
    appear here by mistake. `metrics_undefined` (keyed by metric name) is why a
    metric reads `None` instead of a number, so a scientist never sees a bare
    blank where a reason belongs.

    `baseline_is_self` is `True` when the chosen engine and the baseline are the
    identical fit (same engine, same conditions) -- a consumer must render "this
    model is the baseline" rather than a head-to-head that never happened.

    `noise_floor` is the Dataset's own duplicate-spread: the honest floor for
    model error, since a model cannot be more accurate than the assay it was
    trained on. `None` for classification, which has no equivalent, regardless of
    what was measured upstream.

    `applicability_coverage` is the fraction of test structures within 0.3
    Tanimoto of the training set. `None` when the training set was empty and
    coverage could not be computed -- never a fabricated `0.0`, which would read
    as "every compound is out of distribution" rather than "unmeasurable".

    `random_split_metrics_undefined` is `metrics_undefined`'s counterpart for
    `random_split_metrics`, scored on a different partition of the same rows.
    The two can disagree about which metrics are undefined and why -- a class
    that survives the Dataset's own test split can still collapse to one class
    under the random reshuffle, or vice versa -- so a consumer must not reuse
    `metrics_undefined` to explain a `random_split_metrics` null: that would
    either attribute the wrong partition's reason, or (when `metrics_undefined`
    is `None`) leave the null unexplained.

    `target_unit`/`target_direction` are the Dataset's own `TargetSpec.unit`/
    `.direction` (`"nM"`/`"low"`, or either `None`) -- without them `rmse: 0.61`
    reads identically whether the target was nM or uM, and a bare number
    carries no answer to "is higher better here?" Every other place a
    predicted number reaches a consumer (prediction results, both export
    formats) already carries its unit and direction; this is the last one.

    `split_strategy` is the Dataset's own `SplitSpec.strategy` (`"random"` or
    `"scaffold"`, matching `domain.data.split.SplitStrategy`'s own string
    values -- plain `str` here, not that enum, because the bounded-context
    independence contract forbids `domain.execution` from importing
    `domain.data`). Without it, a consumer can only *infer* "this metric came
    from a random split" from `random_split_metrics is None and
    random_split_unavailable is None` -- a two-null inference that silently
    breaks the moment either field's meaning changes, and which split
    produced a number is the single most important fact about how flattering
    it is allowed to be.
    """

    primary_metric: str
    prediction_kind: str
    metrics: dict[str, float | None]
    metrics_undefined: dict[str, str] | None
    baseline_engine_id: str
    baseline_metrics: dict[str, float | None]
    baseline_is_self: bool
    random_split_metrics: dict[str, float | None] | None
    random_split_unavailable: str | None
    random_split_metrics_undefined: dict[str, str] | None
    noise_floor: float | None
    worst_rows: list[WorstRow]
    applicability_coverage: float | None
    target_unit: str | None
    target_direction: str | None
    split_strategy: str

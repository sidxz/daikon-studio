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
from typing import Any


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
    # The compound's own ID when its dataset names an identifier column, looked
    # up when the scorecard is read (`GetScorecard`), never stored with it.
    compound_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class ParityPoint:
    """One test compound's measured value against its predicted one.

    The whole test set, not the worst twenty: a scatter of every point is the
    only view that shows *regression to the mean* -- a model that predicts
    everything near the dataset average, scores a respectable R^2, and is
    useless for ranking compounds. No aggregate metric on this card can express
    that, and the twenty worst residuals actively hide it.

    `similarity` is the compound's own nearest-neighbour Tanimoto to the
    training set, carried per point so the scatter can be coloured by it: the
    claim "errors grow as compounds get less like the training set" becomes
    visible rather than asserted. `None` when the training set was empty.
    """

    actual: float
    predicted: float
    similarity: float | None


@dataclass(frozen=True, kw_only=True)
class Bin:
    """A half-open interval and what was measured in it.

    `count` is not decoration: a bin holding three compounds and a bin holding
    three hundred are drawn the same width, and without the count a reader has
    no way to tell a trend from a coincidence.
    """

    lower: float
    upper: float
    count: int
    value: float


@dataclass(frozen=True, kw_only=True)
class Histogram:
    """`edges` is one longer than `counts`.

    Deliberately a second definition of the same shape that lives in
    `domain/data/profile.py`, not a shared import: `domain.execution` may not
    import `domain.data` (the bounded-context independence contract that already
    forces `split_strategy` to be a bare `str` here rather than the `SplitStrategy`
    enum). Six lines duplicated is the price of that contract, and it is the
    contract's own answer, not an oversight.
    """

    edges: list[float]
    counts: list[int]


@dataclass(frozen=True, kw_only=True)
class ScaffoldError:
    """How the model does on one Murcko scaffold family.

    The systematic version of `worst_rows`: twenty bad predictions grouped by
    scaffold say "these twenty were bad", while a median error per family across
    the whole test set says "it is worse on the sulfonamides than on anything
    else", which is a statement a chemist can act on.
    """

    scaffold: str
    count: int
    median_error: float


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

    `engine_id`/`conditions` are this model's own -- the baseline is choosable
    now, so `baseline_engine_id` no longer implies what actually trained the
    model under test. A renderer that names the baseline without naming this
    would tell the truth about only one side of the comparison.

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
    #: The same engine scored on the validation partition. This is the number a
    #: scientist is meant to tune conditions against; `metrics` is the one they
    #: are judged by. Keeping both visible, and saying which is which, is what
    #: stops the test partition from being consumed one retrain at a time --
    #: before this existed the test score was the only feedback available, so
    #: every hyperparameter decision was made by looking at it. `None` when the
    #: split declared no validation partition, or when the run predates the
    #: measurement.
    validation_metrics: dict[str, float | None] | None
    metrics_undefined: dict[str, str] | None
    engine_id: str
    conditions: dict[str, Any]
    baseline_engine_id: str
    baseline_conditions: dict[str, Any]
    baseline_metrics: dict[str, float | None]
    baseline_is_self: bool
    random_split_metrics: dict[str, float | None] | None
    random_split_unavailable: str | None
    random_split_metrics_undefined: dict[str, str] | None
    noise_floor: float | None
    worst_rows: list[WorstRow]
    applicability_coverage: float | None
    # 95 % bootstrap interval over the test set for `metrics[primary_metric]`,
    # unpaired (see `build_scorecard.primary_metric_ci`). None when the test
    # set is too small or too skewed to support one. A verdict that beats the
    # baseline by less than this interval's width is a verdict about noise.
    primary_metric_ci: tuple[float, float] | None = None
    target_unit: str | None
    target_direction: str | None
    split_strategy: str

    # --- Diagnostics -------------------------------------------------------
    # Everything below is derived from the same `actual`/`predicted`/
    # `structures` the fields above are derived from -- nothing new is measured
    # and nothing extra is persisted. They exist because the summary above
    # answers "how good is this number" and a scientist deciding whether to run
    # a model needs "where, and on what, is it wrong".

    #: Every test compound, subsampled if there are more than the cap.
    parity: list[ParityPoint]
    #: The population `parity` was drawn from, when it was subsampled; `None`
    #: when every test compound is present. A scatter that silently drops half
    #: its points reads as the whole test set, so the drop is stated.
    parity_sampled_from: int | None
    #: Signed `predicted - actual`, regression only. Centred away from zero is
    #: bias -- a model that is uniformly optimistic, which RMSE cannot show.
    residual_histogram: Histogram | None
    #: Mean absolute error against nearest-neighbour Tanimoto, in equal-count
    #: bins. This is what turns `applicability_coverage` from an assertion into
    #: evidence: if error does not rise as similarity falls, the applicability
    #: domain is not buying this model anything and a reader should know.
    error_by_similarity: list[Bin]
    #: Median absolute error per scaffold family, worst first.
    scaffold_errors: list[ScaffoldError]
    #: Predicted probability against observed positive rate, classification
    #: only. A model can rank compounds well and still be badly calibrated, and
    #: a probability that is not calibrated must not be read as one.
    calibration: list[Bin]

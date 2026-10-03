"""Turns Task 14's raw measurements into the Scorecard a scientist reads.

`build_scorecard` does not compute metrics -- Task 14's engines already produced
them via `_score`, on the model's own code path and the baseline's. Recomputing
either here, or computing the model's while passing the baseline's through
unchanged, would put the two halves of the head-to-head on different code paths,
and they would silently diverge the moment a third engine exists. `actual` and
`predicted` are used only for the worst-20 residual list and the applicability
distribution below.

Chemistry (Murcko scaffolds, nearest-neighbour Tanimoto) reaches this module
through the `StructureNormalizer` port, never `infrastructure.chem` directly --
`application` may not import `infrastructure` (Task 9's layers contract, no
exemptions). `nearest_neighbour_tanimoto` was added to that port here for the
same reason `murcko_scaffold` was added to it in Task 10: a second single-method
port for one more RDKit function would be a needless abstraction split.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.execution.scorecard import (
    Bin,
    Histogram,
    ParityPoint,
    ScaffoldError,
    Scorecard,
    WorstRow,
)

_WORST_ROWS_LIMIT = 20
_APPLICABILITY_THRESHOLD = 0.3

#: Points carried to the client for the parity scatter. Above this the scatter
#: is over-plotted anyway and the payload stops being free, so it is subsampled
#: on a deterministic stride and `parity_sampled_from` says so.
_PARITY_LIMIT = 4000

_RESIDUAL_BINS = 30

#: Equal-*count* bins, not equal-width: ECFP4 similarities cluster low, so
#: fixed-width bins put nearly every compound in one or two of them and leave
#: the rest holding a handful of points whose mean error is noise. Equal-count
#: bins give every point on the curve the same weight of evidence.
_SIMILARITY_BINS = 8

#: Upper bound. The actual count scales with the test set, because ten equal-
#: width bins over a 197-compound test set leaves several holding one or two
#: compounds, and a calibration curve drawn through those reads as wild
#: miscalibration when it is really just sampling noise.
_CALIBRATION_BINS = 10
_MIN_PER_CALIBRATION_BIN = 25

#: A "median error" over one or two compounds is not a median. Families smaller
#: than this are left out rather than shown with a number that cannot support
#: the reading the section invites.
_MIN_SCAFFOLD_GROUP = 3
_SCAFFOLD_GROUP_LIMIT = 12


def primary_metric_for(task: TaskType) -> str:
    """The one metric a Scorecard leads with, and the one a sweep ranks on.

    Shared so those two can never disagree about which number is the headline
    -- a ranked list ordered by a metric the Scorecard does not show is a
    silent lie about which model won.
    """
    return "mcc" if task is TaskType.BINARY_CLASSIFICATION else "rmse"


#: Below this many test rows a bootstrap interval is mostly a statement about
#: the resampling, not the model; the card then shows no interval at all.
_CI_MIN_ROWS = 20
_CI_RESAMPLES = 1000


def _mcc(actual: np.ndarray, predicted_positive: np.ndarray) -> float | None:
    """Matthews correlation from 0/1 labels and a boolean prediction; None when
    a resample holds one class on either side and the metric is undefined."""
    positive = actual >= 0.5
    tp = float(np.sum(positive & predicted_positive))
    tn = float(np.sum(~positive & ~predicted_positive))
    fp = float(np.sum(~positive & predicted_positive))
    fn = float(np.sum(positive & ~predicted_positive))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return None if denominator == 0 else (tp * tn - fp * fn) / denominator


def primary_metric_ci(
    task: TaskType,
    actual: list[float],
    predicted: list[float],
    *,
    resamples: int = _CI_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float] | None:
    """A 95 % bootstrap interval over the test set for the headline metric.

    This is what stops "+0.12 over the baseline" at n=197 reading as a win when
    the baseline's number sits inside [0.49, 0.76] (docs/roadmap.md, Traps). It
    is *unpaired*, and the UI says so: the baseline's per-compound predictions
    are not persisted, so this is the sampling noise of this one number, not a
    paired test of the difference. Still the honest floor under the verdict.

    Recomputed from `actual`/`predicted` with the metric's own definition (MCC
    at the 0.5 threshold, RMSE), not by re-running the engines' `_score`: the
    point estimate stays theirs, the interval is ours, and a fixed seed makes it
    the same on every page load.
    """
    n = len(actual)
    if n < _CI_MIN_ROWS or n != len(predicted):
        return None
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if task is TaskType.BINARY_CLASSIFICATION:
            value = _mcc(a[idx], p[idx] >= 0.5)
            if value is not None:
                values.append(value)
        else:
            values.append(float(np.sqrt(np.mean((a[idx] - p[idx]) ** 2))))
    if len(values) < resamples // 2:
        # Most resamples were single-class: the test set is too skewed for an
        # interval to mean anything, which the undefined-metric reason already says.
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def build_scorecard(
    *,
    task: TaskType,
    metrics: dict[str, float | None],
    validation_metrics: dict[str, float | None] | None = None,
    engine_id: str,
    conditions: dict[str, Any],
    baseline_engine_id: str,
    baseline_metrics: dict[str, float | None],
    baseline_is_self: bool,
    actual: list[float],
    predicted: list[float],
    structures: list[str],
    train_structures: list[str],
    normalizer: StructureNormalizer,
    target_unit: str | None,
    target_direction: str | None,
    split_strategy: str,
    random_split_metrics: dict[str, float | None] | None = None,
    random_split_unavailable: str | None = None,
    random_split_metrics_undefined: dict[str, str] | None = None,
    metrics_undefined: dict[str, str] | None = None,
    duplicate_spread: float | None = None,
    baseline_conditions: dict[str, Any] | None = None,
) -> Scorecard:
    is_classification = task is TaskType.BINARY_CLASSIFICATION

    # `nearest_neighbour_tanimoto([], []) -> []` returns zeros for an empty
    # reference set rather than raising, which would silently read as "every test
    # structure is confirmed 0.0 similar to nothing" -- a fabricated answer, not a
    # measured one. Skipping the call whenever either side is empty is what makes
    # `None` (not `0.0`) the honest result of "there was nothing to compare."
    similarities: list[float] | None = (
        normalizer.nearest_neighbour_tanimoto(structures, train_structures)
        if structures and train_structures
        else None
    )
    applicability_coverage = (
        sum(1 for s in similarities if s >= _APPLICABILITY_THRESHOLD) / len(similarities)
        if similarities
        else None
    )

    residuals = [abs(a - p) for a, p in zip(actual, predicted, strict=True)]
    worst_order = sorted(range(len(residuals)), key=lambda i: residuals[i], reverse=True)

    # Once for every test structure, then reused by both the worst-rows list and
    # the per-family error breakdown. Computing them twice would double the
    # RDKit cost of the most-viewed screen in the product to produce the same
    # strings.
    scaffolds = [normalizer.murcko_scaffold(structure) for structure in structures]

    worst_rows = [
        WorstRow(
            structure=structures[i],
            actual=actual[i],
            predicted=predicted[i],
            residual=residuals[i],
            scaffold=scaffolds[i],
            similarity=similarities[i] if similarities is not None else None,
        )
        for i in worst_order[:_WORST_ROWS_LIMIT]
    ]

    return Scorecard(
        primary_metric=primary_metric_for(task),
        primary_metric_ci=primary_metric_ci(task, actual, predicted),
        prediction_kind="probability" if is_classification else "value",
        metrics=metrics,
        validation_metrics=validation_metrics,
        metrics_undefined=metrics_undefined,
        engine_id=engine_id,
        conditions=conditions,
        baseline_engine_id=baseline_engine_id,
        baseline_conditions=baseline_conditions or {},
        baseline_metrics=baseline_metrics,
        baseline_is_self=baseline_is_self,
        random_split_metrics=random_split_metrics,
        random_split_unavailable=random_split_unavailable,
        random_split_metrics_undefined=random_split_metrics_undefined,
        # Binary targets have no duplicate-spread equivalent; forced here rather
        # than trusted from the caller so a stale or mistaken `duplicate_spread`
        # can never present a meaningless floor as though it meant something.
        noise_floor=None if is_classification else duplicate_spread,
        worst_rows=worst_rows,
        applicability_coverage=applicability_coverage,
        target_unit=target_unit,
        target_direction=target_direction,
        split_strategy=split_strategy,
        parity=_parity(actual, predicted, similarities),
        parity_sampled_from=len(actual) if len(actual) > _PARITY_LIMIT else None,
        # Signed residuals are a regression reading. For classification `actual`
        # is a 0/1 label and `predicted` a probability, so their difference is
        # bounded by construction and its histogram is bimodal by construction
        # too -- a shape that says nothing about the model. The calibration
        # curve below is the classification counterpart.
        residual_histogram=None if is_classification else _residual_histogram(actual, predicted),
        error_by_similarity=_error_by_similarity(residuals, similarities),
        scaffold_errors=_scaffold_errors(residuals, scaffolds),
        calibration=_calibration(actual, predicted) if is_classification else [],
    )


def _parity(
    actual: list[float], predicted: list[float], similarities: list[float] | None
) -> list[ParityPoint]:
    # A stride, not a head slice: predictions arrive in test-set order, which for
    # a scaffold split is grouped by chemical family, so the first N points would
    # be one corner of the chemistry presented as the whole scatter.
    stride = max(1, -(-len(actual) // _PARITY_LIMIT))
    return [
        ParityPoint(
            actual=actual[i],
            predicted=predicted[i],
            similarity=similarities[i] if similarities is not None else None,
        )
        for i in range(0, len(actual), stride)
    ]


def _histogram(values: list[float], bins: int) -> Histogram:
    """Plain-Python binning.

    numpy is imported above for the bootstrap, where a thousand resamples earn
    it; one histogram over a few thousand floats does not, so this stays as it
    was written.
    """
    if not values:
        return Histogram(edges=[], counts=[])
    low, high = min(values), max(values)
    if low == high:
        # A degenerate range would make every edge identical and every value land
        # in no bin. One unit either side keeps the single spike drawable.
        low, high = low - 0.5, high + 0.5
    width = (high - low) / bins
    edges = [low + width * i for i in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / width), bins - 1)
        counts[index] += 1
    return Histogram(edges=edges, counts=counts)


def _residual_histogram(actual: list[float], predicted: list[float]) -> Histogram | None:
    if not actual:
        return None
    # Signed, and predicted minus actual rather than the reverse, so a histogram
    # sitting right of zero reads as "the model predicts high" in the target's
    # own direction.
    return _histogram([p - a for a, p in zip(actual, predicted, strict=True)], _RESIDUAL_BINS)


def _error_by_similarity(residuals: list[float], similarities: list[float] | None) -> list[Bin]:
    if similarities is None or len(residuals) < _SIMILARITY_BINS * 2:
        # Fewer than two compounds per bin is not a curve. Empty, so the
        # consumer omits the section rather than drawing eight noisy points.
        return []
    order = sorted(range(len(residuals)), key=lambda i: similarities[i])
    size = len(order) / _SIMILARITY_BINS

    bins = []
    for index in range(_SIMILARITY_BINS):
        group = order[int(index * size) : int((index + 1) * size)]
        if not group:
            continue
        bins.append(
            Bin(
                lower=similarities[group[0]],
                upper=similarities[group[-1]],
                count=len(group),
                value=sum(residuals[i] for i in group) / len(group),
            )
        )
    return bins


def _scaffold_errors(residuals: list[float], scaffolds: list[str]) -> list[ScaffoldError]:
    grouped: dict[str, list[float]] = {}
    for scaffold, residual in zip(scaffolds, residuals, strict=True):
        grouped.setdefault(scaffold, []).append(residual)

    families = [
        ScaffoldError(
            scaffold=scaffold,
            count=len(errors),
            median_error=statistics.median(errors),
        )
        for scaffold, errors in grouped.items()
        if len(errors) >= _MIN_SCAFFOLD_GROUP
    ]
    if len(families) < 2:
        # One family is not a comparison, and this section exists only to say
        # which families are worse than which.
        return []
    families.sort(key=lambda family: family.median_error, reverse=True)
    return families[:_SCAFFOLD_GROUP_LIMIT]


def _calibration(actual: list[float], predicted: list[float]) -> list[Bin]:
    if not actual:
        return []
    count = max(3, min(_CALIBRATION_BINS, len(actual) // _MIN_PER_CALIBRATION_BIN))
    bins = []
    for index in range(count):
        lower = index / count
        upper = (index + 1) / count
        # The last bin closes at 1.0 inclusive, so a confident P=1.0 prediction
        # is counted rather than dropped.
        group = [
            label
            for label, probability in zip(actual, predicted, strict=True)
            if lower <= probability < upper or (index == count - 1 and probability == 1.0)
        ]
        if not group:
            # Omitted, not zero-filled: no compound was predicted in this band,
            # which is not the same as "every compound in this band was negative".
            continue
        bins.append(
            Bin(
                lower=lower,
                upper=upper,
                count=len(group),
                value=sum(1 for label in group if label > 0.5) / len(group),
            )
        )
    return bins

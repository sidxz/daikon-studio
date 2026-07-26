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

from daikonstudio.application.engines.manifest import TaskType
from daikonstudio.application.ports.structure_normalizer import StructureNormalizer
from daikonstudio.domain.execution.scorecard import Scorecard, WorstRow

_WORST_ROWS_LIMIT = 20
_APPLICABILITY_THRESHOLD = 0.3


def build_scorecard(
    *,
    task: TaskType,
    metrics: dict[str, float | None],
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
    worst_rows = [
        WorstRow(
            structure=structures[i],
            actual=actual[i],
            predicted=predicted[i],
            residual=residuals[i],
            scaffold=normalizer.murcko_scaffold(structures[i]),
            similarity=similarities[i] if similarities is not None else None,
        )
        for i in worst_order[:_WORST_ROWS_LIMIT]
    ]

    return Scorecard(
        primary_metric="mcc" if is_classification else "rmse",
        prediction_kind="probability" if is_classification else "value",
        metrics=metrics,
        metrics_undefined=metrics_undefined,
        baseline_engine_id=baseline_engine_id,
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
    )

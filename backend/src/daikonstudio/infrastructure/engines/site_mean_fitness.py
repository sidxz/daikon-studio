"""Site-mean fitness: the floor a variant-effect model actually has to clear.

Predicts the mean training label of the variants that mutate the same position, and the
global training mean for a position the fit never saw. No learning, no features, no
hyperparameters.

It exists because the published record says this floor is hard to beat. Across 41 viral
and 33 cellular deep-mutational-scanning sets, "a simple baseline model that predicted
site mean fitness matched or outperformed supervised models on many datasets", and
protein language models "barely outperform a naive predictor relying entirely on mean
fitness values at individual sites" (Vieira, Lin & Wilke, bioRxiv
10.64898/2026.03.08.710389). Our own mandatory sequence baseline cannot stand in for it:
`protein_descriptors_randomforest` reads amino-acid composition, and on a single-parent
series every sequence differs by one residue in a few hundred, so composition captures
*which* residue was introduced while ignoring *where* -- the orthogonal half, and the
weaker one.

**Under a POSITION holdout this engine is a constant, and that is the point.** Every test
position is unseen by construction, so there is nothing to look up and it returns the
training mean. Read it as a pair: strong under a random split and constant under a
position holdout means the model it was measured against was memorising positions, and
the difference is how much. A Scorecard that shows a variant-effect model beating only
the composition baseline has not yet established that it learned anything about the
protein.

No torch and no fit, so this stays on the default lane and costs milliseconds.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl

from daikonstudio.application.data.assign_split import consensus, mutated_positions
from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._scoring import (
    SEQUENCE_FEATURIZER,
    _predict_with_tree_ensemble,
    _scored,
    _sequences,
    bundle_features,
)

_MANIFEST = EngineManifest(
    id="site-mean-fitness",
    version="1.0.0",
    name="Site-mean fitness",
    description=(
        "Predicts the average measured value of the training variants that mutate the "
        "same residue, ignoring which residue replaced it. The floor a variant-effect "
        "model has to clear to have learned anything beyond which positions matter — "
        "published models often do not. Under a mutated-position holdout it reduces to "
        "the training average, which is the honest comparison."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
    conditions=(),
    # Exactly one engine may be flagged and it is still the descriptor forest. This is
    # an extra floor a user opts into for a variant series, not a replacement -- it is
    # meaningless on any dataset that is not a single-parent series, where the
    # composition baseline still works.
    is_baseline=False,
    structure_kinds=(StructureKind.SEQUENCE,),
)


class _SiteMean:
    """The fitted lookup. sklearn-shaped -- `predict`, `predict_proba`, `classes_` --
    because `_scoring` takes any model with that surface, so this reuses the whole
    metric, cutoff and prediction path without touching it.
    """

    def __init__(
        self,
        parent: str,
        means: dict[int, float],
        fallback: float,
        classes: np.ndarray | None = None,
    ) -> None:
        self.parent = parent
        self.means = means
        self.fallback = fallback
        # Set only for a classification fit, so a regression model has no `classes_` at
        # all, exactly as an sklearn regressor does not.
        if classes is not None:
            self.classes_ = classes

    def _site_mean(self, sequence: str) -> float:
        if len(sequence) != len(self.parent):
            raise ValidationError(
                f"This model was fitted on a variant series of {len(self.parent)} "
                f"residues, but was asked to predict a sequence of {len(sequence)}. A "
                f"position only means the same site in sequences of the same length."
            )
        known = [
            self.means[position]
            for position in mutated_positions(sequence, self.parent)
            if position in self.means
        ]
        # The wild type and any unseen position land on the fallback together: both
        # genuinely carry no site information, and inventing a difference between them
        # would be inventing a prediction.
        return float(np.mean(known)) if known else self.fallback

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.array([self._site_mean(str(s)) for s in x], dtype=float)
        if not hasattr(self, "classes_"):
            return values
        # Hard labels at the natural cutoff; `_metrics_on` substitutes a tuned one when
        # the run asks for it, reading `predict_proba` instead.
        return (values >= 0.5).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        positive = np.array([self._site_mean(str(s)) for s in x], dtype=float)
        return np.column_stack([1.0 - positive, positive])


class SiteMeanFitness:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        validate_conditions(_MANIFEST, ctx.conditions)
        sequences = [str(s) for s in ctx.frame[ctx.structure_column].to_list()]
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) > 1:
            raise ValidationError(
                f"Site-mean fitness needs an aligned single-parent variant series: every "
                f"sequence must have the same length, but these range from {min(lengths)} "
                f"to {max(lengths)} residues. Align the series to a common parent, or use "
                f"the protein descriptor baseline for sequences from different proteins."
            )

        # The parent is inferred from every row, not from the training partition, so that
        # this agrees exactly with the POSITION holdout that produced the split. Nothing
        # leaks: the modal residue at a site is a property of the sequences, and no label
        # is read here.
        parent = consensus(sequences)

        train_rows = ctx.frame.filter(pl.col("split") == "train")
        labels = train_rows[ctx.target_column].to_numpy().astype(float)
        totals: dict[int, list[float]] = {}
        for sequence, label in zip(
            (str(s) for s in train_rows[ctx.structure_column].to_list()), labels, strict=True
        ):
            for position in mutated_positions(sequence, parent):
                totals.setdefault(position, []).append(float(label))

        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION
        model = _SiteMean(
            parent=parent,
            means={position: float(np.mean(values)) for position, values in totals.items()},
            # `float(np.mean([]))` is nan with a warning, so an empty training partition
            # gets an explicit 0.0 rather than a nan that would silently poison the
            # metrics instead of failing.
            fallback=float(np.mean(labels)) if labels.size else 0.0,
            classes=np.unique(labels.astype(int)) if is_classification else None,
        )

        artifact = pickle.dumps(
            {
                "model": model,
                "is_classification": is_classification,
                **bundle_features(SEQUENCE_FEATURIZER),
            }
        )
        metrics, validation_metrics, cutoffs = _scored(model, ctx, is_classification, _sequences)
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        # The tree path, which reports no uncertainty for a model with no `estimators_`.
        # Correct here: a site mean has a spread, but it is the spread of the training
        # labels at that site, not of an ensemble's disagreement, and reporting one in a
        # column the Scorecard plots as model confidence would misdescribe it.
        return _predict_with_tree_ensemble(ctx)

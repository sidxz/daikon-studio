"""Gaussian process over frozen ESM-2 embeddings -- the head the literature actually favours.

`esm2_xgboost.py` says in its own docstring that the GP head is the follow-up and to
expect it to beat the tree. This is it: Kermut tops supervised ProteinGym as a GP over
these same 1280-dimensional vectors, in about ten minutes where ProteinNPT needs upwards
of four hundred hours.

Two reasons it is a separate engine rather than a kernel swap on `tanimoto_gp.py`. That
kernel is defined on binary vectors and is positive semi-definite only there, so it
cannot take an embedding; and this one needs torch, so it belongs on the `gpu` lane. What
it does reuse from that module is the shape of the thing -- a `GaussianProcessRegressor`
with `normalize_y`, a fitted `WhiteKernel`, a training-size ceiling, and `return_std`
wired to the Scorecard's uncertainty column.

Measured on the TEM-1 pilot under a position holdout: **0.7680 Spearman / 0.5769 r2 on
VESM2 with the site block and the likelihood**, against the tree's 0.7167 / 0.5100. Two of
those differences survive a paired bootstrap of 10,000 redraws of the test rows, which is
the only reason to state them -- this engine over the tree on identical features
(+0.040, interval +0.004 to +0.077) and the site block over no site block here (+0.036,
+0.019 to +0.053, the tightest result in that whole comparison). So this is the engine to
reach for on single-parent variant data, not only the small-n fallback.

The site block earning more here than in a tree is what an isotropic RBF reading a
per-site delta through `||a - b||` should do, where a tree has to spend split candidates
on 2,560 extra columns. Note the asymmetry in what that is worth as evidence: the *gain*
here is measured, while the corresponding *loss* in the tree is not -- both tree-side
site-block differences came out inside the noise, so the mechanism's positive half is
established and its negative half is only a story that fits.

**Regression only, and that is a statement about uncertainty rather than a gap.** This
engine exists to report a posterior standard deviation in the target's own units.
`GaussianProcessClassifier`'s Laplace approximation exposes no latent variance, so a
classification branch here would report the same distance-from-the-boundary proxy every
other engine already reports, from a model that costs cubically more to fit.
`esm2-xgboost` classifies over the same embeddings. `train_protocol` refuses a mismatched
task by manifest with a message naming the supported ones, and the picker filters on it.

Three traps, all of them load-bearing.

**The length scale, which is why "a GP cannot read ESM-2" is a live claim.** ALDE reports
flatly that ESM-2 encodings "cannot be used by GPs, likely because they are too high
dimensional", while Kermut does exactly that successfully. The difference is the length
scale. sklearn's `RBF` starts at 1.0 with bounds (1e-5, 1e5); standardized 1280-dimensional
vectors sit about sqrt(2 * 1280) ~ 50 apart, so at a length scale of 1 every off-diagonal
kernel entry underflows to zero, the GP learns nothing and predicts its prior mean for
every row. That is a *plausible* model: it fits, it scores, and it looks like the
dimensionality problem ALDE describes. So the scale is initialised from the data by the
median pairwise distance and the bounds are set around that, never left to the default.

**The prior mean is half the value and sklearn has no hook for it.** Kermut's ablations
attribute roughly half its advantage to using ESM-2's zero-shot LLR as the GP's prior
mean -- -0.036 contiguous and -0.032 modulo when it is removed, against -0.008 random. So
it lands on the hard splits, which is where a variant-effect model is judged. sklearn
offers no mean function, so the standard trick: least-squares a line in the LLR, fit the
GP to the residual, add the line back at predict. Measured here on the TEM-1 pilot under
a position holdout, the LLR was worth +0.053 Spearman to the tree engine -- the largest
single effect in that comparison.

**Standardize, do not whiten, do not PCA.** Standardization is necessary for the kernel
above to mean anything. Whitening degraded 34 of 36 published configurations, and PCA
costs accuracy; neither is here and neither should be added for accuracy's sake.

**Do not market this as active learning.** ALDE's wet-lab campaign used one-hot plus a DNN
ensemble plus Thompson sampling, found frequentist ensembles beat GPs for uncertainty
quality, and found optimisation performance uncorrelated with calibration. A posterior
standard deviation is an honest number about this model's own confidence; it is not a
demonstrated design loop.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import polars as pl
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import (  # type: ignore[import-untyped]
    RBF,
    ConstantKernel,
    Kernel,
    WhiteKernel,
)
from sklearn.metrics.pairwise import euclidean_distances  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.engines._options import PROTEIN_MODEL
from daikonstudio.infrastructure.engines._scoring import (
    _predict_with_gaussian_process,
    _scored,
    bundle_features,
    esm2_featurizer,
)

#: Refuse rather than thrash, as in `tanimoto_gp`: the kernel matrix is n x n float64 and
#: the fit factorises it once per marginal-likelihood step. The practical number is lower
#: here than for fingerprints, because every row also costs a 650M-parameter forward pass
#: before the GP sees it -- and because the datasets this engine is for are small by
#: nature. Every honest stability set in the roster research was n = 19-700.
_MAX_TRAINING_ROWS = 10_000

#: How many rows the median pairwise distance is estimated from. The estimate feeds an
#: initial value and a pair of bounds spanning four orders of magnitude, so it needs to be
#: roughly right rather than exact, and a full n x n distance matrix on 10,000 rows would
#: cost more than the estimate is worth.
_SCALE_SAMPLE = 1_000

_MANIFEST = EngineManifest(
    id="esm2-gp",
    version="1.0.0",
    name="ESM-2 embeddings + Gaussian process",
    description=(
        "A Gaussian process on frozen ESM-2 650M protein embeddings, suited to the small "
        "datasets protein engineering actually produces (tens to a few thousand "
        "sequences). Its uncertainty is a posterior standard deviation in the target's "
        "own units rather than a spread across an ensemble. The protein model is used as "
        "a fixed featurizer and is not retrained. Cost grows cubically with training-set "
        "size: practical to about 2,000 sequences, with a hard limit of 10,000."
    ),
    tasks=(TaskType.REGRESSION,),
    conditions=(
        ConditionSpec(
            key="n_restarts_optimizer",
            label="Tuning restarts",
            type=ConditionType.INTEGER,
            default=2,
            minimum=0,
            maximum=10,
            help="Number of additional hyperparameter optimizations from random starting "
            "points. More restarts take proportionally longer but are less likely to end "
            "in a poor local optimum. 0 uses the initial values only.",
        ),
        ConditionSpec(
            key="likelihood_feature",
            label="Start from ESM-2's opinion of each substitution",
            type=ConditionType.BOOL,
            default=False,
            help="Scores every variant against the series' parent sequence before "
            "training, asking the model how likely the new residue is at that site "
            "compared with the one it replaced, and uses that as the starting point the "
            "measurements correct. This is usually the single largest improvement "
            "available here. Only meaningful for variants of a single parent protein.",
        ),
        PROTEIN_MODEL,
        ConditionSpec(
            key="site_features",
            label="Tell the model where each mutation is",
            type=ConditionType.BOOL,
            default=False,
            help="Adds what the model reads at the mutated site, and how that site "
            "differs from the parent sequence's. Switching it on is the recommended "
            "starting point for this engine: on the one dataset we have measured it "
            "improved rank correlation by 0.036, and of every setting we tested it is "
            "the one whose benefit was most clearly larger than the measurement noise. "
            "Only meaningful for variants of a single parent protein, and it needs "
            "every sequence to be the same length.",
        ),
    ),
    lane="gpu",
    is_baseline=False,
    structure_kinds=(StructureKind.SEQUENCE,),
)


def _median_distance(scaled: np.ndarray, seed: int) -> float:
    """The median pairwise Euclidean distance over a sample of rows.

    The median heuristic, which is the standard answer to "where should an RBF length
    scale start" and the specific thing that keeps this engine out of the failure the
    module docstring describes. Zero distances are dropped before the median: a dataset
    of variants of one parent holds near-duplicate rows, and duplicates are not evidence
    about the scale on which the function varies.
    """
    if scaled.shape[0] > _SCALE_SAMPLE:
        rows = np.random.default_rng(seed).choice(scaled.shape[0], _SCALE_SAMPLE, replace=False)
        scaled = scaled[rows]
    distances = euclidean_distances(scaled)
    spread = distances[np.triu_indices_from(distances, k=1)]
    positive = spread[spread > 0.0]
    # 1.0 only when every training row is identical, which the WhiteKernel then absorbs.
    return float(np.median(positive)) if positive.size else 1.0


def _kernel(scaled: np.ndarray, seed: int) -> Kernel:
    """Amplitude x RBF + noise, with the length scale initialised from the data.

    Kermut's own kernel on the embedding half is a textbook isotropic squared
    exponential, so this is that and not a reimplementation of its full structure.
    Isotropic rather than ARD on purpose: one length scale is three hyperparameters in
    total, where ARD would be 1,281 fitted by marginal likelihood on possibly a few
    hundred rows.
    """
    scale = _median_distance(scaled, seed)
    return (
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(scale, (scale / 1e2, scale * 1e2))
        # The observation noise, fitted rather than assumed, and what keeps the Cholesky
        # from failing on the duplicate rows a single-parent series produces.
        + WhiteKernel(1e-2, (1e-6, 1e1))
    )


class _PriorMeanGP:
    """A fitted GP over standardized embeddings, plus the line in the LLR it corrects.

    Exists so that `_scored` and `_predict_with_gaussian_process` work unchanged: both
    ask the artifact for `predict(x)` and `predict(x, return_std=True)` and neither needs
    to know that the mean function is not zero. The alternative was threading a prior
    through two shared paths that every other engine uses.

    `prior` is `None` when the LLR is off, in which case this is a scaler and a GP.
    """

    def __init__(
        self,
        gp: Any,
        scaler: Any,
        prior: tuple[float, float] | None,
    ) -> None:
        self._gp = gp
        self._scaler = scaler
        self._prior = prior

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | float]:
        """Features for the kernel, and the prior mean to add back.

        The LLR is the featurizer's final column (see `_scoring._esm2_with_likelihood`),
        and it is deliberately kept *out* of the kernel: it enters as the mean, which is
        what Kermut does with it and what its ablations measure. Feeding it to the RBF as
        well would make one column of 1,281 compete on equal footing with the embedding.
        """
        if self._prior is None:
            return x, 0.0
        slope, intercept = self._prior
        return x[:, :-1], slope * x[:, -1] + intercept

    def predict(self, x: np.ndarray, return_std: bool = False) -> Any:
        if x.shape[0] == 0:
            # `StandardScaler.transform` raises on zero rows where the tree engines
            # return an empty array, and an empty partition is legitimate here: a sparse
            # target in a fanned-out leg can have no measured test row, which
            # `regression_metrics` is written to report as undefined rather than fail.
            # Without this the whole run dies on a sklearn message naming neither cause.
            empty = np.zeros(0)
            return (empty, np.zeros(0)) if return_std else empty
        features, mean = self._split(x)
        scaled = self._scaler.transform(features)
        if return_std:
            value, deviation = self._gp.predict(scaled, return_std=True)
            return value + mean, deviation
        return self._gp.predict(scaled) + mean


class Esm2GP:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")

        if train_rows.height > _MAX_TRAINING_ROWS:
            raise ValidationError(
                f"The ESM-2 Gaussian process accepts at most {_MAX_TRAINING_ROWS:,} "
                f"training sequences; this training set has {train_rows.height:,}. Use "
                "the ESM-2 tree engine for datasets of this size."
            )

        # The whole frame's column, as in `esm2_xgboost`: this is where the parent is
        # settled for every partition and for every later predict.
        featurizer_key, featurizer, parent = esm2_featurizer(
            conditions, ctx.frame[ctx.structure_column].to_list()
        )
        ctx.report(0.0, "Embedding sequences with ESM-2")
        x_train = featurizer(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()
        ctx.report(0.5, "Fitting")

        prior: tuple[float, float] | None = None
        features, residual = x_train, y_train
        if conditions.get("likelihood_feature"):
            # Least squares on [LLR, 1], then the GP models what the line leaves over.
            # A rank-deficient design -- every LLR identical, which happens when no row
            # differs from the parent -- gives lstsq's minimum-norm answer, a zero slope
            # and the mean as the intercept. That is the right degenerate behaviour and
            # needs no guard of its own.
            likelihood = x_train[:, -1]
            design = np.column_stack([likelihood, np.ones_like(likelihood)])
            slope, intercept = np.linalg.lstsq(design, y_train, rcond=None)[0]
            prior = (float(slope), float(intercept))
            features = x_train[:, :-1]
            residual = y_train - (slope * likelihood + intercept)

        scaler = StandardScaler().fit(features)
        scaled = scaler.transform(features)
        gp = GaussianProcessRegressor(
            kernel=_kernel(scaled, ctx.seed),
            normalize_y=True,
            # Numerical jitter only; the noise model is the WhiteKernel in `_kernel`.
            alpha=1e-8,
            n_restarts_optimizer=conditions["n_restarts_optimizer"],
            random_state=ctx.seed,
        )
        gp.fit(scaled, residual)
        model = _PriorMeanGP(gp, scaler, prior)

        artifact = pickle.dumps(
            {
                "model": model,
                "is_classification": False,
                **bundle_features(featurizer_key, parent),
            }
        )
        metrics, validation_metrics, cutoffs = _scored(model, ctx, False, featurizer)
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_gaussian_process(ctx)

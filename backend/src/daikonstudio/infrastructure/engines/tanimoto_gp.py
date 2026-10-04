"""Tanimoto-kernel Gaussian process -- the small-data engine, with uncertainty by construction.

Every other engine reports uncertainty as a proxy: spread across a random forest's
trees, distance from a decision boundary, or nothing at all. This one reports a
posterior standard deviation in the units of the measurement, because that is what the
model *is* -- a distribution over functions rather than a point estimate with a
confidence heuristic bolted on. Hirschfeld et al. (JCIM 2020) ranked GP uncertainty
second by NLL across their whole comparison, and a Tanimoto-kernel GP placed 13th of 66
at the Polaris/ASAP antiviral challenge, ahead of most deep-learning entries.

The regime it opens is small n. Gaussian processes are the classic n<1000 method, which
is where most in-house assays actually live, and where chemprop is hopeless and a forest
is mediocre. It is also the only engine here that gets *worse* as data grows, which is
why the training-size ceiling below is enforced rather than documented.

Fingerprints rather than descriptors, deliberately: the Tanimoto kernel is defined on
binary vectors and is positive semi-definite there, so it is a legitimate kernel rather
than a similarity function pressed into service. A descriptor GP would need scaling and
a different kernel entirely, and would be a different engine.

Runs on the default lane: sklearn and RDKit, no new dependency, no GPU.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import polars as pl
from sklearn.gaussian_process import (  # type: ignore[import-untyped]
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process.kernels import (  # type: ignore[import-untyped]
    ConstantKernel,
    Kernel,
    WhiteKernel,
)

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import (
    ConditionSpec,
    ConditionType,
    EngineManifest,
    TaskType,
    validate_conditions,
)
from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.engines._scoring import _predict_with_gaussian_process, _scored

#: Refuse rather than thrash. The kernel matrix is n x n float64 and the fit factorises
#: it once per marginal-likelihood step, so 10,000 rows is 800 MB per copy and minutes
#: per iteration. Past this the failure mode is an OOM-killed worker with no message,
#: which is strictly worse for a scientist than a refusal that names the reason. The
#: manifest's user-facing number is 5,000, where it is still comfortable; between the
#: two it works and is slow, which is the user's call to make.
_MAX_TRAINING_ROWS = 10_000


class TanimotoKernel(Kernel):  # type: ignore[misc]
    """k(x, y) = <x,y> / (||x||^2 + ||y||^2 - <x,y>) -- the Jaccard index on binary vectors.

    Positive semi-definite on binary input (Gower & Legendre), which is what makes this
    a kernel and not just a similarity measure a GP has been talked into accepting.

    Parameter-free, which is the whole reason it composes: signal amplitude comes from a
    `ConstantKernel` factor and observation noise from a `WhiteKernel` term, both
    sklearn's own and both fitted by marginal likelihood. So `hyperparameters` is empty
    and `eval_gradient` returns a zero-width gradient rather than raising.
    """

    def __init__(self) -> None:
        # Required despite taking no arguments. sklearn's `Kernel.get_params`
        # introspects `__init__`'s signature and rejects varargs outright
        # ("kernels should always specify their parameters in the signature of
        # their __init__"), which is what the inherited `object.__init__` is. The
        # failure lands inside `clone()` during `fit`, not at construction.
        pass

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # float64 first, and not for tidiness: `ecfp4` returns uint8 and sklearn's
        # `dtype="numeric"` validation preserves it, so without this the accumulations
        # below stay uint8. Two failures follow -- `x_norm + y_norm` wraps past 255 once
        # a pair of large molecules sets ~128 bits each, and the divide cannot write its
        # float output into a uint8 `out` at all. The second is loud; the first is not.
        X = np.asarray(X, dtype=np.float64)
        Y = X if Y is None else np.asarray(Y, dtype=np.float64)

        intersection: np.ndarray = X @ Y.T
        x_norm: np.ndarray = np.einsum("ij,ij->i", X, X)
        y_norm: np.ndarray = np.einsum("ij,ij->i", Y, Y)
        union = x_norm[:, np.newaxis] + y_norm[np.newaxis, :] - intersection

        # union == 0 only when both rows are all-zero, which `ecfp4` emits for a SMILES
        # RDKit could not parse. Zero similarity is the honest reading of that, and it
        # keeps a NaN out of the kernel matrix, which would poison the whole fit.
        similarity: np.ndarray = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union != 0.0
        )
        if eval_gradient:
            return similarity, np.empty((X.shape[0], X.shape[0], 0))
        return similarity

    def diag(self, X: np.ndarray) -> np.ndarray:
        """1 for a molecule with any bit set, 0 for an unparseable one.

        k(x, x) is <x,x>/<x,x>, so the only question per row is whether it is all-zero.
        Computing the full matrix to read its diagonal would be O(n^2) work for an O(n)
        answer, and sklearn calls this on the *prediction* set, which can be large.
        """
        X = np.asarray(X, dtype=np.float64)
        norms: np.ndarray = np.einsum("ij,ij->i", X, X)
        return (norms > 0.0).astype(np.float64)

    def is_stationary(self) -> bool:
        return False


_MANIFEST = EngineManifest(
    id="tanimoto-gp",
    version="1.0.0",
    name="Tanimoto Gaussian Process",
    description=(
        "A Gaussian process with a Tanimoto kernel on ECFP4 fingerprints, suited to "
        "small datasets (a few hundred to a few thousand compounds). For regression, it "
        "is the only engine here whose uncertainty is a posterior standard deviation in "
        "the target's units rather than an ensemble proxy. Cost scales cubically with "
        "training-set size: practical up to about 5,000 compounds, with a hard limit of "
        "10,000."
    ),
    tasks=(TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION),
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
    ),
    is_baseline=False,
)


def _base_kernel() -> Kernel:
    """Amplitude times similarity. The bounds are wide enough not to bind in practice
    and narrow enough that the optimiser does not wander into a degenerate fit."""
    return ConstantKernel(1.0, (1e-3, 1e3)) * TanimotoKernel()


class TanimotoGP:
    @staticmethod
    def manifest() -> EngineManifest:
        return _MANIFEST

    def train(self, ctx: TrainContext) -> TrainResult:
        conditions = validate_conditions(_MANIFEST, ctx.conditions)
        train_rows = ctx.frame.filter(pl.col("split") == "train")
        is_classification = ctx.task is TaskType.BINARY_CLASSIFICATION

        if train_rows.height > _MAX_TRAINING_ROWS:
            raise ValidationError(
                f"The Tanimoto Gaussian process accepts at most {_MAX_TRAINING_ROWS:,} "
                f"training compounds; this training set has {train_rows.height:,}. Use a "
                "tree-based or graph engine for datasets of this size."
            )

        x_train = ecfp4(train_rows[ctx.structure_column].to_list())
        y_train = train_rows[ctx.target_column].to_numpy()

        if is_classification and len(np.unique(y_train)) < 2:
            # The tree engines fit happily on one class and let `_scored` report
            # undefined metrics. GaussianProcessClassifier raises instead, and its own
            # message says nothing about which split is at fault -- so the run would
            # fail on a bare sklearn ValueError naming neither the engine nor the fix.
            raise ValidationError(
                "Every training compound has the same label, so there is nothing for a "
                "Gaussian process classifier to separate. Check the split fractions, or "
                "the class balance of the dataset itself."
            )

        model = self._model(is_classification, conditions["n_restarts_optimizer"], ctx.seed)
        model.fit(x_train, y_train)

        artifact = pickle.dumps(
            {
                "model": model,
                "is_classification": is_classification,
                # Named rather than assumed, matching the descriptor engine. No
                # `feature_names`: ECFP4's 2048 hashed bits have no names to drift.
                "featurizer": "ecfp4",
            }
        )
        metrics, validation_metrics, cutoffs = _scored(model, ctx, is_classification)
        return TrainResult(
            artifact=artifact,
            metrics=metrics,
            validation_metrics=validation_metrics,
            cutoffs=cutoffs,
        )

    @staticmethod
    def _model(is_classification: bool, restarts: object, seed: int) -> Any:
        # `Any`, as `model` is in `ecfp4_randomforest` and `_scoring`: sklearn ships no
        # py.typed marker, so both constructors already resolve to Any and a union of
        # the two would just be a union of Any, which mypy rejects as a type.
        if is_classification:
            # No noise term in the kernel. For a Bernoulli likelihood the observation
            # noise lives in the likelihood, which the Laplace approximation handles;
            # adding a WhiteKernel here would be modelling it twice.
            return GaussianProcessClassifier(
                kernel=_base_kernel(),
                n_restarts_optimizer=restarts,
                random_state=seed,
            )
        return GaussianProcessRegressor(
            # WhiteKernel is the actual observation noise, fitted by marginal likelihood
            # rather than assumed. It is also what keeps the Cholesky from failing on
            # duplicate structures, which produce identical rows and a singular matrix.
            kernel=_base_kernel() + WhiteKernel(1e-2, (1e-6, 1e1)),
            # The GP prior mean is 0 and assay values are not -- a logP dataset centred
            # on 3 would be shrunk toward zero without this. sklearn un-scales both the
            # mean and the standard deviation on the way out, so `return_std` stays in
            # the target's own units, which is the point of reporting it at all.
            normalize_y=True,
            # Numerical jitter only, not a noise model; that is WhiteKernel's job above.
            alpha=1e-8,
            n_restarts_optimizer=restarts,
            random_state=seed,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        return _predict_with_gaussian_process(ctx)

"""Training settings more than one engine declares, and the arithmetic behind them.

Declared once so every engine that supports a setting offers it with the same key, the
same options and the same words -- a sweep comparing engines compares like with like.
"""

from __future__ import annotations

import math

import numpy as np

from daikonstudio.application.engines.manifest import ConditionSpec, ConditionType, TaskType

POSITIVE_WEIGHTING = ConditionSpec(
    key="positive_weighting",
    label="Positive-class weighting",
    type=ConditionType.ENUM,
    default="none",
    options=("none", "balanced", "sqrt_balanced"),
    tasks=(TaskType.BINARY_CLASSIFICATION,),
    help=(
        "Up-weights active compounds in the loss so a rare label is not drowned out. "
        "Balanced weights each label's actives by its inactive-to-active ratio in the "
        "training set; square-root balanced uses the square root of that ratio, a gentler "
        "choice for very rare labels. Weighting inflates predicted probabilities, so tune "
        "decision cutoffs alongside it. Applies to active/inactive targets only."
    ),
)

RDKIT_DESCRIPTORS = ConditionSpec(
    key="rdkit_descriptors",
    label="Add RDKit descriptors",
    type=ConditionType.BOOL,
    default=False,
    help=(
        "Adds RDKit's 2D descriptor set (about 200 descriptors, such as molecular weight, "
        "logP, TPSA and ring counts) to the model's input."
    ),
)


def positive_weight(y_train: np.ndarray, mode: str) -> float | None:
    """The loss weight on one binary label's positives, from its training labels only.

    `None` when weighting is off, or when the training rows hold no positives -- possible
    only in the random-split comparison's reshuffle, since dataset creation refuses a
    single-class training partition.
    """
    if mode == "none":
        return None
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0:
        return None
    ratio = negatives / positives
    return ratio if mode == "balanced" else math.sqrt(ratio)

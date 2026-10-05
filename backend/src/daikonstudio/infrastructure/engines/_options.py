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
    option_labels=("None", "Balanced", "Square-root balanced"),
    tasks=(TaskType.BINARY_CLASSIFICATION,),
    help=(
        "Up-weights active compounds in the loss so a rare label is not drowned out. "
        "Balanced weights each label's actives by its inactive-to-active ratio in the "
        "training set; square-root balanced uses the square root of that ratio, a gentler "
        "choice for very rare labels. Weighting inflates predicted probabilities, so tune "
        "decision cutoffs alongside it. Applies to active/inactive targets only."
    ),
)

EPOCH_SELECTION = ConditionSpec(
    key="epoch_selection",
    label="Keep the epoch with",
    type=ConditionType.ENUM,
    default="auprc",
    options=("auprc", "auroc", "loss"),
    option_labels=("Best PR AUC", "Best AUROC", "Lowest validation loss"),
    tasks=(TaskType.BINARY_CLASSIFICATION,),
    help=(
        "Training runs every epoch requested, scores the model on the validation "
        "compounds after each one, and keeps the best. Best PR AUC keeps the epoch that "
        "ranks actives highest among all compounds, the measure that matters when actives "
        "are rare; it is the default. Best AUROC also judges ranking but weighs every "
        "active-inactive pair equally, so it suits labels where actives are common. "
        "Lowest validation loss also rewards probabilities that match observed rates. "
        "Loss often starts to rise while ranking is still improving, so this choice tends "
        "to keep an earlier, less accurate epoch, and with positive-class weighting a few "
        "actives decide it. With several targets, each is scored separately and the "
        "scores are averaged. Applies to active/inactive targets only; continuous targets "
        "keep the epoch with the lowest validation loss."
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

    `None` when weighting is off, or when the training rows hold no positives or no
    negatives (the ratio is then undefined or zero) -- possible only in the random-split
    comparison's reshuffle, since dataset creation refuses a single-class training
    partition.
    """
    if mode == "none":
        return None
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    if positives == 0 or negatives == 0:
        return None
    ratio = negatives / positives
    return ratio if mode == "balanced" else math.sqrt(ratio)

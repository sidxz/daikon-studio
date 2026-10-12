"""Training settings more than one engine declares, and the arithmetic behind them.

Declared once so every engine that supports a setting offers it with the same key, the
same options and the same words -- a sweep comparing engines compares like with like.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

from daikonstudio.application.engines.manifest import ConditionSpec, ConditionType, TaskType
from daikonstudio.infrastructure.protein.embed import (
    DEFAULT_VARIANT,
    PROTEIN_MODEL_LABELS,
    PROTEIN_MODELS,
)

PROTEIN_MODEL = ConditionSpec(
    key="protein_model",
    label="Protein model",
    type=ConditionType.ENUM,
    default=DEFAULT_VARIANT,
    options=PROTEIN_MODELS,
    option_labels=PROTEIN_MODEL_LABELS,
    help=(
        "Which pretrained protein model reads the sequences. ESM-2 650M is the "
        "best-established choice and the default. VESM2 650M is the same model with "
        "published refinements aimed at judging single substitutions, and it is worth "
        "trying for variants of a single protein. On the one dataset we have measured "
        "ourselves it scored a little higher than ESM-2, but not by enough to rule out "
        "chance, so treat the two as comparable until your own data says otherwise. "
        "ESM-2 150M is "
        "about four times faster and uses far less memory, at a cost that depends on the "
        "split: close to nothing when the test sequences resemble the training ones, but "
        "measurably worse -- up to 0.16 in rank correlation -- when they do not, which "
        "is exactly what a grouped or held-out-position split is for. Prefer the larger "
        "model whenever the split is a hard one. "
        "ESM C 600M is a more recent model, and the reasons to pick it are its speed and "
        "its reach: it reads sequences of up to 2,048 residues, where the ESM-2 models "
        "stop at 1,022 and anything longer is cut off. It is not the more accurate "
        "choice on the comparisons published so far, where it scores slightly below "
        "ESM-2 650M, and we have not measured the two against each other on our own "
        "data. Its opinion of a substitution cannot be read yet, so the likelihood "
        "setting is unavailable with it."
    ),
)

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


COUNT_FINGERPRINTS = ConditionSpec(
    key="count_fingerprints",
    label="Count repeated substructures",
    type=ConditionType.BOOL,
    default=False,
    help=(
        "Records how many times each substructure occurs instead of only whether it is "
        "present. Usually a small gain, and largest where a structure repeats itself — "
        "peptides, polymers and long chains."
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


def tree_threads() -> int:
    """Threads for a boosted-tree fit (XGBoost on the CPU, LightGBM): up to eight, or one
    when torch is loaded in this process.

    torch and the tree libraries each bring their own OpenMP runtime, and two runtimes
    both running thread teams in one process is undefined behaviour. On macOS it
    segfaults, reproduced 2026-10-05 after one torch op: LightGBM at 8 threads, and
    XGBoost at 4. That is why both images set OMP_NUM_THREADS=1. A process that never
    loaded torch has no second runtime, so its trees may use threads: the CPU image has
    no torch, and a runner on the GPU image loads it only for a neural fit.

    Eight, because there was no gain past it at 80k rows (LightGBM on atlantic: 1.8 s
    at 8, 1.7 at 16) and one thread per logical CPU stalled a LightGBM fit 220-fold on
    a host with two cores busy elsewhere. Results do not depend on the count: both
    libraries are built deterministic here (see their engines).
    """
    if "torch" in sys.modules:
        return 1
    return min(8, os.process_cpu_count() or 1)

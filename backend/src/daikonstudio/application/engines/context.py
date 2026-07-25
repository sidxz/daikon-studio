from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from daikonstudio.application.engines.manifest import TaskType


@dataclass(frozen=True, kw_only=True)
class TrainContext:
    """`frame` carries the dataset columns plus a `split` column of train/validation/test.

    `task` is passed explicitly and is authoritative. An engine must NEVER infer
    regression-vs-classification from the target values: a regression target whose
    values happen to all be 0.0 or 1.0 would silently train a classifier. The
    Dataset's TargetSpec is the only source of truth for what is being predicted.
    """

    frame: pl.DataFrame
    task: TaskType
    structure_column: str
    target_column: str
    conditions: dict[str, object]
    seed: int


@dataclass(frozen=True, kw_only=True)
class TrainResult:
    artifact: bytes
    metrics: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class PredictContext:
    frame: pl.DataFrame
    structure_column: str
    artifact: bytes
    conditions: dict[str, object]

"""One model per target, behind the single-engine contract.

Five engines in the roster fit exactly one target: a random forest, a boosted tree
or a Gaussian process has one output. This adapter is what lets every one of them
accept a Dataset with several targets anyway -- it trains the inner engine once per
target, on the same rows and the same split, so a per-target baseline and a joint
model are compared on identical ground. The registry wraps every engine that does
not declare `supports_multitask`, at every target count including one, so the
long-format output below is produced in exactly one place.

Three rules this file exists to keep:

- **`RunInterrupted` propagates.** Nothing here catches anything. Swallowing a
  cancellation raised in fit two of four would keep burning a runner for fits
  three and four (see `context.py`).
- **Progress is mapped, not forwarded.** `ctx.report` is progress within one fit,
  0.0 to 1.0; target i of N reports into its own slice, (i + f) / N, or the bar
  would reset to zero N times.
- **The container appears only at N > 1.** At one target the artifact is the inner
  engine's bytes verbatim: every Protocol trained before targets could be several
  has a bare artifact in blob storage, and a container there would break prediction
  for all of them. At N > 1 the artifacts are packed into one zip -- boring,
  inspectable, no pickle -- under index names, never column names, since a CSV
  header can hold a slash or a leading `/` that a zip entry name would rewrite.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import polars as pl

from daikonstudio.application.engines.context import (
    PredictContext,
    ProgressReporter,
    TrainContext,
    TrainResult,
)
from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine

_TARGETS_ENTRY = "targets.json"


class FanOut:
    def __init__(self, inner: Engine) -> None:
        self._inner = inner

    def manifest(self) -> EngineManifest:
        return self._inner.manifest()

    def train(self, ctx: TrainContext) -> TrainResult:
        columns = ctx.target_columns
        count = len(columns)
        results: list[TrainResult] = []
        for index, (column, task) in enumerate(ctx.targets.items()):
            if count > 1:
                # Before each sub-fit, not only inside it: tree and GP engines never
                # report on their own, and this checkpoint is what lets a cancel or a
                # deadline stop the run between targets.
                ctx.report(index / count, f"Training on {column} ({index + 1} of {count})")
            results.append(
                self._inner.train(
                    replace(ctx, targets={column: task}, report=_slice(ctx.report, index, count))
                )
            )
        validation = {
            key: value
            for result in results
            if result.validation_metrics is not None
            for key, value in result.validation_metrics.items()
        }
        cutoffs = {
            key: value
            for result in results
            if result.cutoffs is not None
            for key, value in result.cutoffs.items()
        }
        return TrainResult(
            artifact=results[0].artifact if count == 1 else _pack(columns, results),
            metrics={key: value for result in results for key, value in result.metrics.items()},
            validation_metrics=validation or None,
            cutoffs=cutoffs or None,
        )

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        columns = ctx.target_columns
        artifacts = [ctx.artifact] if len(columns) == 1 else _unpack(ctx.artifact, columns)
        return pl.concat(
            [
                self._inner.predict(
                    replace(ctx, artifact=artifact, target_columns=(column,))
                ).with_columns(pl.lit(column, dtype=pl.String).alias("target"))
                for column, artifact in zip(columns, artifacts, strict=True)
            ]
        )


def _slice(report: ProgressReporter, index: int, count: int) -> ProgressReporter:
    if count == 1:
        return report

    def sliced(fraction: float, phase: str) -> None:
        report((index + min(max(fraction, 0.0), 1.0)) / count, phase)

    return sliced


def _pack(columns: tuple[str, ...], results: list[TrainResult]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_TARGETS_ENTRY, json.dumps(list(columns)))
        for index, result in enumerate(results):
            archive.writestr(f"{index}.bin", result.artifact)
    return buffer.getvalue()


def _unpack(artifact: bytes, columns: tuple[str, ...]) -> list[bytes]:
    with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
        trained = tuple(json.loads(archive.read(_TARGETS_ENTRY)))
        if trained != columns:
            raise ValueError(
                f"This artifact was trained on {list(trained)}, not on the requested "
                f"targets {list(columns)}."
            )
        return [archive.read(f"{index}.bin") for index in range(len(columns))]

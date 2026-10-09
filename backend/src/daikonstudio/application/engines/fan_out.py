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

Sparse labels are handled here and nowhere else for these engines. A Dataset may be
measured for some targets and blank for others; each sub-fit is handed only the rows
carrying a measurement for its own target, so no inner engine -- and none of the five
scoring sites inside them -- ever sees a null. On a dense Dataset the filter removes
nothing. The cost is that the mask is invisible: an engine declaring
`supports_multitask` opts out of it silently, which is what
`tests/unit/engines/test_multitask_nulls.py` exists to catch.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import replace

import polars as pl

from daikonstudio.application.engines.checkpoints import Checkpoints, unpack_result
from daikonstudio.application.engines.context import (
    EpochPoint,
    EpochRecorder,
    PredictContext,
    ProgressReporter,
    TrainContext,
    TrainResult,
)
from daikonstudio.application.engines.manifest import EngineManifest
from daikonstudio.application.engines.protocol import Engine

logger = logging.getLogger(__name__)

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
            # A multi-target fit saves each target's result as it completes, scoped by
            # position (a column name could be anything) with the column in the
            # fingerprint, so a resumed run refits only the targets it had not finished.
            saved_scope = (
                ctx.checkpoints.scoped(f"target-{index}", column=column)
                if ctx.checkpoints is not None and count > 1
                else None
            )
            restored = _restore(saved_scope)
            if restored is not None:
                ctx.report(
                    (index + 1) / count, f"Restored the fit for {column} from saved progress"
                )
                results.append(restored)
                continue
            result = self._inner.train(
                replace(
                    ctx,
                    # Filtered on *this* target's nulls only. A row measured for this
                    # target is kept even where another target is blank on it --
                    # filtering on any null would rebuild the intersection the
                    # ingestion gate exists to avoid.
                    #
                    # Guarded on the column existing because this adapter should not
                    # invent a failure: a frame without the target column is a caller
                    # error, and the inner engine reports it in the terms of whatever
                    # it was trying to read. Raising a polars `ColumnNotFoundError`
                    # from here would replace that with a worse message.
                    frame=(
                        ctx.frame.filter(pl.col(column).is_not_null())
                        if column in ctx.frame.columns
                        else ctx.frame
                    ),
                    targets={column: task},
                    report=_slice(ctx.report, index, count, column),
                    record_epoch=_tagged(ctx.record_epoch, column),
                    checkpoints=saved_scope if saved_scope is not None else ctx.checkpoints,
                )
            )
            if saved_scope is not None:
                saved_scope.save_result(result)
            results.append(result)
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


def _restore(scope: Checkpoints | None) -> TrainResult | None:
    if scope is None:
        return None
    data = scope.load("result")
    if data is None:
        return None
    try:
        return unpack_result(data)
    except Exception:
        logger.warning(
            "A saved fit in %s did not unpack; refitting it.", scope.root, exc_info=True
        )
        return None


def _slice(report: ProgressReporter, index: int, count: int, column: str) -> ProgressReporter:
    if count == 1:
        return report

    def sliced(fraction: float, phase: str) -> None:
        # Named, because the engine's own phase cannot say which target it is fitting.
        report((index + min(max(fraction, 0.0), 1.0)) / count, f"{column}: {phase}")

    return sliced


def _tagged(record: EpochRecorder, column: str) -> EpochRecorder:
    def tagged(point: EpochPoint) -> None:
        record(replace(point, target=column))

    return tagged


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

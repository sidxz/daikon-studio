"""The Engine contract's behavioural half.

`train`/`predict` are SYNCHRONOUS on purpose, unlike the async plugin protocol used
elsewhere in the project family. Engines are CPU-bound (fitting a model, scoring a
batch); making them `async def` would either block the event loop or force every
engine author to think about threads. Instead the contract stays plain sync code,
and the worker offloads the call with `asyncio.to_thread`.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

from daikonstudio.application.engines.context import PredictContext, TrainContext, TrainResult
from daikonstudio.application.engines.manifest import EngineManifest


class Engine(Protocol):
    """Structural — an engine matches this shape; no base class, no registration."""

    @staticmethod
    def manifest() -> EngineManifest: ...

    def train(self, ctx: TrainContext) -> TrainResult: ...

    def predict(self, ctx: PredictContext) -> pl.DataFrame:
        """Returns columns: row_id (int), value (float), uncertainty (float | null)."""
        ...

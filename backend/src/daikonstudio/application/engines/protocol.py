"""The Engine contract's behavioural half.

`train`/`predict` are SYNCHRONOUS on purpose, unlike the async plugin protocol used
elsewhere in the project family. Engines do long blocking work -- fitting a model,
scoring a batch -- and they do it on whatever device they resolve to, CPU, MPS or
CUDA alike; the contract is about blocking, not about which processor blocks. Making
them `async def` would either block the event loop or force every engine author to
think about threads. Instead the contract stays plain sync code, and the worker
offloads the call with `asyncio.to_thread`.

An engine that does long work SHOULD call `ctx.report(fraction, phase)` periodically.
That single callback is what publishes progress to the Run row a client is polling,
and -- because it may raise `RunInterrupted` -- is the only thing that can stop a fit
already running on a worker thread. An engine that never calls it still works; it is
simply not interruptible, and its progress bar will not move.
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

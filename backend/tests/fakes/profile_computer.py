"""An in-process `ProfileComputer` for the API tests: the real profile, no child process.

What the endpoint does -- 202 while a profile computes, one computation per key however
many readers ask, a cached read once it is saved, 409 after a failure -- is the same
wherever `build_profile` runs, and spawning a fresh interpreter (which imports polars and
RDKit) for every one of a twenty-compound fixture's profiles would add seconds per test.
The real child process is covered directly in `tests/unit/chem/test_profile_process.py`,
including that it returns the same dict this does.

The hooks exist because the old tests monkeypatched `build_profile` to stall or raise,
and a function patched in this process is not the one the child imports.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import polars as pl

from daikonstudio.application.data.build_profile import build_profile
from daikonstudio.domain.data.profile import profile_to_dict
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.data.target import TargetSpec
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer


class InProcessProfileComputer:
    def __init__(self) -> None:
        self.calls = 0
        self.running = 0
        #: The most computations ever in flight at once: 1 unless serialization broke.
        self.peak = 0
        #: Set it to hold every computation until the test releases it.
        self.release: asyncio.Event | None = None
        #: Set it to fail instead of computing.
        self.error: Exception | None = None

    async def compute(
        self,
        *,
        raw: bytes,
        structure_column: str,
        target: TargetSpec,
        structure_kind: StructureKind,
    ) -> dict[str, Any]:
        self.calls += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.release is not None:
                await asyncio.wait_for(self.release.wait(), timeout=10)
            if self.error is not None:
                raise self.error
            return profile_to_dict(
                build_profile(
                    frame=pl.read_parquet(io.BytesIO(raw)),
                    structure_column=structure_column,
                    target=target,
                    structure_kind=structure_kind,
                    normalizer=RdkitStructureNormalizer(),
                )
            )
        finally:
            self.running -= 1

"""The concrete `ProfileComputer` (application/ports): `build_profile` in a child process.

Profiling holds the calling process's GIL for as long as it runs, and a thread does not
help -- `asyncio.to_thread` keeps the work off the event loop but not off the GIL. The
pass is a nested Python loop over every molecule and ~200 RDKit descriptors, so a
404k-compound dataset is ~81 million interpreter iterations, about 23 minutes at this
project's measured 3.41 ms/mol, with the GIL held nearly throughout. In the API that
starves every other request in the process: while one such profile ran, a Scorecard
request budgeted at ~10 s had not answered after an hour (prod, 2026-10-20). It is the
same failure as the Tanimoto search's, documented in `normalizer.py::_search`, and it
takes the same cure -- a child with its own GIL, spawned rather than forked so the fresh
interpreter carries no torch, no event loop and no lock held by another thread.

The whole function goes in the child, not the descriptor pass alone: the child must
return the *binned* profile. Piping back the descriptor columns instead would move
gigabytes of Python floats through a pickle, which is worse than the stall being fixed.
"""

from __future__ import annotations

import asyncio
import io
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import polars as pl
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from daikonstudio.application.data.build_profile import build_profile
from daikonstudio.domain.data.profile import profile_to_dict
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.data.target import TargetSpec
from daikonstudio.infrastructure.chem import normalizer as normalizer_module
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer


class ChildProcessProfileComputer:
    async def compute(
        self,
        *,
        raw: bytes,
        structure_column: str,
        target: TargetSpec,
        structure_kind: StructureKind,
    ) -> dict[str, Any]:
        spawn = multiprocessing.get_context("spawn")
        # A pool per call, as `_search` does, so the child exits with its gigabytes
        # rather than idling in a pool holding them until the next profile.
        with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
            # Awaited, not `.result()`: blocking here would hold the event loop for
            # the whole computation, which is the stall this class exists to remove.
            return await asyncio.wrap_future(
                pool.submit(_profile_in_child, raw, structure_column, target, structure_kind)
            )


def _profile_in_child(
    raw: bytes,
    structure_column: str,
    target: TargetSpec,
    structure_kind: StructureKind,
) -> dict[str, Any]:
    """The child's half. Module-level because `spawn` re-imports this module to find
    it, which also means nothing here may run at import time.

    The BLAS may use several threads, for the same reason `_search_in_child` lets it:
    the images pin OMP_NUM_THREADS=1 so torch's OpenMP cannot collide with another
    runtime, and this fresh process has not loaded torch.
    """
    # The profile's own Tanimoto search must stay in this process: it would otherwise
    # see a large search and spawn a grandchild, and a child that already owns its GIL
    # gains nothing from that. Set before `build_profile`, read by `_search`.
    normalizer_module.PROFILING_IN_CHILD = True
    with threadpool_limits(limits=min(8, os.process_cpu_count() or 1), user_api="blas"):
        profile = build_profile(
            frame=pl.read_parquet(io.BytesIO(raw)),
            structure_column=structure_column,
            target=target,
            normalizer=RdkitStructureNormalizer(),
            structure_kind=structure_kind,
        )
    return profile_to_dict(profile)

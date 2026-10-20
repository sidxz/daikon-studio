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
        pool = ProcessPoolExecutor(
            max_workers=1, mp_context=spawn, initializer=_mark_profiling_in_child
        )
        try:
            # Awaited, not `.result()`: blocking here would hold the event loop for
            # the whole computation, which is the stall this class exists to remove.
            profile = await asyncio.wrap_future(
                pool.submit(_profile_in_child, raw, structure_column, target, structure_kind)
            )
        except BaseException:
            # Cancelled (an API shutdown, or the caller's timeout) or failed. The worker
            # is killed, because `shutdown(wait=False)` does not kill it: it stops
            # *scheduling*, and a work item already running runs to completion.
            # `concurrent.futures.process._python_exit`, registered with
            # `threading._register_atexit`, then joins the manager thread at interpreter
            # exit and that join waits for the running work item -- so a SIGTERM
            # mid-profile hangs until the container's grace period kills the API, and
            # Ctrl-C under `make dev` hangs the same way. Worse, the caller releases its
            # one-profile-at-a-time lock as this returns, so the orphan's core and 3-4 GB
            # would still be held when the next profile's child starts beside it:
            # two at once against an 8 GiB container, which is the overrun that lock
            # exists to prevent. Terminate first, and the shutdown has nothing to join.
            for process in (pool._processes or {}).values():
                if process is not None:
                    process.terminate()
            pool.shutdown(wait=False)
            raise
        # The success path waits, and the two differ for one reason: here the worker is
        # already idle, so the join costs an interpreter exit rather than a computation,
        # and it is what releases the child's gigabytes *before* `compute` returns and
        # the caller drops the lock that keeps profiles from overlapping.
        pool.shutdown(wait=True)
        return profile


def _mark_profiling_in_child() -> None:
    """The pool's `initializer`, and the only place the flag is set: an initializer runs
    in a worker process and nowhere else.

    The profile's own Tanimoto search must stay in the child: it would otherwise see a
    large search and spawn a grandchild, and a process that already owns its GIL gains
    nothing from that. Read by `_search`.

    Deliberately not set inside `_profile_in_child`. That is an ordinary module-level
    function, so a caller in the API process -- a test avoiding the cost of a spawn, a
    future "small datasets in-process" fallback -- would set the flag for the rest of
    the interpreter's life, after which every large Tanimoto search runs on the
    event-loop thread: verbatim the outage `normalizer.py::_search` documents.
    """
    normalizer_module.PROFILING_IN_CHILD = True


def _profile_in_child(
    raw: bytes,
    structure_column: str,
    target: TargetSpec,
    structure_kind: StructureKind,
) -> dict[str, Any]:
    """The child's half. Module-level because `spawn` re-imports this module to find
    it, which also means nothing here may run at import time.

    Sets nothing process-wide: `PROFILING_IN_CHILD` belongs to the pool's initializer
    (`_mark_profiling_in_child`), so calling this function in the API process computes a
    profile slowly and leaves nothing behind.

    The BLAS may use several threads, for the same reason `_search_in_child` lets it:
    the images pin OMP_NUM_THREADS=1 so torch's OpenMP cannot collide with another
    runtime, and this fresh process has not loaded torch.
    """
    with threadpool_limits(limits=min(8, os.process_cpu_count() or 1), user_api="blas"):
        profile = build_profile(
            frame=pl.read_parquet(io.BytesIO(raw)),
            structure_column=structure_column,
            target=target,
            normalizer=RdkitStructureNormalizer(),
            structure_kind=structure_kind,
        )
    return profile_to_dict(profile)

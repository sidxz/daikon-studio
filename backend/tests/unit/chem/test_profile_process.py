"""Profiling a dataset in a child process must compute the same profile, off the GIL.

The fix this covers: `build_profile` is a nested Python loop over every molecule and
~200 descriptors, so it holds the calling process's GIL for as long as it runs -- 23
minutes on a 404k-compound dataset, during which a Scorecard request budgeted at ~10 s
did not answer at all (prod, 2026-10-20). Moving it to a spawned child is only a fix if
the profile that comes back is identical and the loop is genuinely free meanwhile, so
both are asserted here against a real child process.
"""

from __future__ import annotations

import asyncio
import io

import polars as pl
import pytest

from daikonstudio.application.data.build_profile import build_profile
from daikonstudio.domain.data.profile import profile_to_dict
from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.infrastructure.chem import normalizer as normalizer_module
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer
from daikonstudio.infrastructure.chem.profile import ChildProcessProfileComputer

_STRUCTURES = [
    "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "CCc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "Clc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F",
    "CCO",
    "c1ccncc1",
    "CC(=O)O",
]
_TARGET = TargetSpec(column="value", kind=TargetKind.NUMERIC)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "smiles": _STRUCTURES,
            "value": [1.0, 5.0, 2.5, 0.5, 3.0, 4.0],
            "split": ["train", "train", "train", "validation", "test", "test"],
        }
    )


async def test_a_profile_computed_off_process_matches_one_computed_here() -> None:
    frame = _frame()
    buffer = io.BytesIO()
    frame.write_parquet(buffer)

    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(tick())
    try:
        off_process = await ChildProcessProfileComputer().compute(
            raw=buffer.getvalue(),
            structure_column="smiles",
            target=_TARGET,
            structure_kind=StructureKind.MOLECULE,
        )
    finally:
        ticker.cancel()

    # The point of the change: the event loop kept running while the child worked.
    assert ticks > 1

    here = profile_to_dict(
        build_profile(
            frame=frame,
            structure_column="smiles",
            target=_TARGET,
            normalizer=RdkitStructureNormalizer(),
            structure_kind=StructureKind.MOLECULE,
        )
    )
    assert off_process == here


async def test_a_child_that_fails_fails_the_await() -> None:
    """A dead child must raise, not hang: `GetDatasetProfile` turns the exception into
    the 409 that tells a reader to retry, and a reader who is never told polls forever.
    An unreadable snapshot stands in for the real cases -- a killed or crashed child."""
    with pytest.raises(Exception):  # noqa: B017 -- polars' own error type is not the point
        await ChildProcessProfileComputer().compute(
            raw=b"not a parquet file",
            structure_column="smiles",
            target=_TARGET,
            structure_kind=StructureKind.MOLECULE,
        )


class _NoPool:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("spawned a grandchild process")


def test_the_profile_child_keeps_its_tanimoto_search_in_process(monkeypatch) -> None:
    """Inside the profile child, `_search` must not spawn again: the child already has
    its own GIL, so a grandchild buys nothing and costs an interpreter plus a copy of
    the fingerprints. `LARGE_SEARCH_PAIRS` is lowered so a two-molecule search counts
    as large -- a genuinely large one would take minutes to prove the same branch."""
    monkeypatch.setattr(normalizer_module, "LARGE_SEARCH_PAIRS", 1)
    monkeypatch.setattr(normalizer_module, "ProcessPoolExecutor", _NoPool)
    monkeypatch.setattr(
        normalizer_module,
        "nearest_neighbours_tanimoto",
        lambda query, reference, k: ("in-process", k),
    )

    # Not in the child: a large search still goes to a process pool.
    with pytest.raises(RuntimeError, match="grandchild"):
        normalizer_module._search(["CCO"], ["CCN"], 1)

    monkeypatch.setattr(normalizer_module, "PROFILING_IN_CHILD", True)
    assert normalizer_module._search(["CCO"], ["CCN"], 1) == ("in-process", 1)

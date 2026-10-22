"""The agent may only say what a step's result supports, and may see only part of it.

Everything here is about the check and the look, because they are the only reason it
is safe to put a model next to a result a scientist is about to act on. A claim that
reads well and cites a residue the pocket does not have is exactly the failure this
exists to catch, and it is the one a reader cannot catch for themselves.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from daikonstudio.application.studies.readers import (
    LOOK_ROWS_MAX,
    _check_pockets,
    check_table,
    checkable,
    look_table,
    reader,
    summarize_table,
)
from daikonstudio.application.studies.schemas import Claim, Look
from daikonstudio.application.tools import codecs


def _pocket(rank: int, residues: list[tuple[str, int]], **extra: Any) -> dict[str, Any]:
    return {
        "rank": rank,
        "residues": [{"chain": "A", "resname": n, "resseq": s} for n, s in residues],
        "ligand_overlap": extra.get("ligand_overlap", []),
        "annotated": extra.get("annotated", []),
        "interface": extra.get("interface", False),
    }


DATA = {
    "meta": {},
    "pockets": [
        _pocket(1, [("ALA", 128), ("LYS", 132)]),
        _pocket(
            10,
            [("HIS", 94), ("HIS", 96), ("THR", 200)],
            ligand_overlap=[{"comp_id": "ZN", "shared": 3, "of": 3, "additive": False}],
        ),
        _pocket(11, [("ASP", 25), ("ILE", 50)], interface=True),
    ],
}


def _claim(**kw: Any) -> Claim:
    return Claim(
        step=1,
        candidates=kw.get("candidates", [10]),
        label=kw.get("label", "the zinc cone"),
        evidence=kw.get("evidence", "lining_residues"),
        cites=kw.get("cites", ["HIS94", "HIS96"]),
    )


def test_a_grounded_claim_survives_and_reads_back() -> None:
    points, dropped = _check_pockets([_claim()], DATA)
    assert dropped == 0
    assert points[0].at == "rank 10"
    assert points[0].label == "the zinc cone"
    assert points[0].cites == ("HIS94", "HIS96")


def test_a_residue_the_candidate_does_not_have_costs_the_claim() -> None:
    """The whole claim goes, not just the bad residue: a reading with an invented
    residue in it was not read off this result."""
    points, dropped = _check_pockets([_claim(cites=["HIS94", "TYR158"])], DATA)
    assert points == []
    assert dropped == 1


def test_residues_are_matched_however_they_are_spelled() -> None:
    points, _ = _check_pockets([_claim(cites=["A:His 94", "his96"])], DATA)
    assert len(points) == 1


def test_a_rank_that_is_not_there_costs_the_claim() -> None:
    points, dropped = _check_pockets([_claim(candidates=[10, 99])], DATA)
    assert points == []
    assert dropped == 1


def test_a_claim_with_nothing_cited_is_an_opinion_and_is_dropped() -> None:
    points, dropped = _check_pockets([_claim(cites=[])], DATA)
    assert points == []
    assert dropped == 1


def test_evidence_must_be_in_the_result_not_just_asserted() -> None:
    """Calling candidate 10 an interface pocket is a claim about a column, and
    that column says no."""
    points, dropped = _check_pockets([_claim(evidence="interface")], DATA)
    assert points == []
    assert dropped == 1
    kept, _ = _check_pockets(
        [_claim(candidates=[11], cites=["ASP25"], evidence="interface")], DATA
    )
    assert len(kept) == 1


def test_one_claim_may_join_candidates_the_tool_reported_apart() -> None:
    """The fragmentation case: residues are checked against the candidates
    together, because the point of the claim is that they are one site."""
    points, dropped = _check_pockets(
        [_claim(candidates=[1, 10], cites=["ALA128", "THR200"], label="one cleft")], DATA
    )
    assert dropped == 0
    assert points[0].at == "ranks 1, 10"


def test_a_pocket_claim_cannot_rest_on_table_evidence() -> None:
    """`Evidence` is one literal for every kind, so the pocket check has to refuse the
    words that belong to tables rather than wave them through."""
    points, dropped = _check_pockets([_claim(evidence="rows")], DATA)
    assert points == []
    assert dropped == 1


ROWS = [
    {"ID": "a", "SMILES": "CCO", "pIC50": 6.1},
    {"ID": "b", "SMILES": "CCN", "pIC50": 7.4},
]


def _cell(**kw: Any) -> Claim:
    return Claim(
        step=1,
        label=kw.get("label", "potency spans a log"),
        evidence=kw.get("evidence", "column"),
        cites=kw.get("cites", ["pIC50"]),
    )


def test_a_table_claim_must_cite_a_column_or_an_identifier() -> None:
    kept, dropped = check_table([_cell(cites=["pIC50", "a"])], ROWS)
    assert dropped == 0
    assert kept[0].at == "column pIC50"
    assert kept[0].cites == ("pIC50", "a")
    assert kept[0].step is None  # the agent knows which step it was about, the reader does not

    rows, _ = check_table([_cell(evidence="rows", cites=["a", "b"])], ROWS)
    assert rows[0].at == "rows a, b"

    # Spelled exactly, nothing the table does not hold, and nothing is an opinion.
    for cites in (["potency"], ["PIC50"], ["A"], []):
        assert check_table([_cell(cites=cites)], ROWS) == ([], 1)
    # A pocket's words are not a table's.
    assert check_table([_cell(evidence="lining_residues")], ROWS) == ([], 1)


def test_a_table_look_is_capped_and_elides_structures() -> None:
    rows = [
        {"ID": f"c{i}", "SMILES": f"CC(=O)N{'C' * i}c1ccccc1", "pIC50": 5 + i / 100}
        for i in range(200)
    ]
    seen = look_table(rows, Look(step=1, rows=10_000, columns=["SMILES", "pIC50", "nope"]))
    lines = seen.splitlines()
    assert len(lines) == LOOK_ROWS_MAX
    assert lines[0] == "c0: SMILES=<structure>, pIC50=5.0"
    assert "nope" not in seen
    assert not any(row["SMILES"] in seen for row in rows)

    # No columns named means all of them; a count below the table's start is not a slice
    # from its end.
    assert look_table(rows, Look(step=1, rows=1)) == "c0: ID=c0, SMILES=<structure>, pIC50=5.0"
    assert len(look_table(rows, Look(step=1, rows=-1)).splitlines()) <= LOOK_ROWS_MAX

    # A table with no identifier column is named by its first, which here is a structure.
    assert "CCO" not in look_table([{"SMILES": "CCO", "pIC50": 6.1}], Look(step=1))


def test_a_table_summary_never_quotes_a_row() -> None:
    smiles = ["CC(=O)Nc1ccc(O)cc1", "c1ccccc1O", "CCN(CC)CC"]
    frame = pl.DataFrame({"ID": ["a", "b", "c"], "SMILES": smiles, "pIC50": [6.1, None, 7.4]})
    _, summary = codecs.encode("compound_table", frame)
    told = summarize_table(None, summary)
    assert not any(s in told for s in smiles)
    assert "Rows: 3" in told
    assert "SMILES" in told
    assert "pIC50" in told


def test_a_kind_without_a_reader_summarizes_its_meta_and_cannot_be_checked() -> None:
    assert reader("model_fit").summarize(None, {"r2": 0.5}) == '{"r2": 0.5}'
    structure = {
        "id": "4TRO",
        "chains": ["A", "B"],
        "ligands": [{"comp_id": "ZN", "chain": "A", "resseq": 101, "contacts": [{"chain": "A"}]}],
    }
    told = reader("structure").summarize(None, structure)
    assert "4TRO" in told
    assert "ZN A101" in told
    for kind in ("model_fit", "structure"):
        assert reader(kind).look is None
        assert reader(kind).check is None


def test_checkable_lists_pockets_and_every_table_kind() -> None:
    assert checkable() == sorted({"pocket_candidates", *codecs.TABLE_KINDS})

"""The agent may only say what a step's result supports, and may see only part of it.

Everything here is about the check and the look, because they are the only reason it
is safe to put a model next to a result a scientist is about to act on. A claim that
reads well and cites a residue the pocket does not have is exactly the failure this
exists to catch, and it is the one a reader cannot catch for themselves.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from daikonstudio.application.studies.read_value import decode_view
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


def _table(columns: dict[str, list[Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A table as the agent meets it: its decoded rows and the summary stored beside them."""
    blob, summary = codecs.encode("compound_table", pl.DataFrame(columns))
    return decode_view("compound_table", blob, limit=None), summary


def _table_claim(**kw: Any) -> Claim:
    return Claim(
        step=1,
        label=kw.get("label", "potency spans a log"),
        evidence=kw.get("evidence", "column"),
        cites=kw.get("cites", ["pIC50"]),
    )


def test_a_table_claim_must_cite_a_column_or_an_identifier() -> None:
    # The identifier is second, so it is found by what it is called and not by where it sits.
    rows, summary = _table({"SMILES": ["CCO", "CCN"], "ID": ["a", "b"], "pIC50": [6.1, 7.4]})

    kept, dropped = check_table([_table_claim(cites=["pIC50", "a"])], rows, summary)
    assert dropped == 0
    assert kept[0].at == "column pIC50"
    assert kept[0].cites == ("pIC50", "a")
    assert kept[0].step is None  # the agent knows which step it was about, the reader does not

    kept, _ = check_table([_table_claim(evidence="rows", cites=["a", "b"])], rows, summary)
    assert kept[0].at == "rows a, b"

    # Spelled exactly, nothing the table does not hold, and nothing is an opinion.
    for evidence, cites in (
        ("column", ["potency"]),
        ("column", ["PIC50"]),
        ("column", ["A"]),
        ("rows", ["A"]),
        ("column", []),
    ):
        claim = _table_claim(evidence=evidence, cites=cites)
        assert check_table([claim], rows, summary) == ([], 1)
    # An identifier cited as a column, and a column cited as rows, each point at the wrong thing.
    assert check_table([_table_claim(cites=["a"])], rows, summary) == ([], 1)
    assert check_table([_table_claim(evidence="rows")], rows, summary) == ([], 1)
    # A pocket's words are not a table's.
    assert check_table([_table_claim(evidence="lining_residues")], rows, summary) == ([], 1)

    # A table that opens with a structure is named by the next column, so its rows can still
    # be cited, though not by the structures, which the agent was never shown.
    labelled, summary = _table({"SMILES": ["CCO", "CCN"], "label": ["x1", "x2"]})
    kept, _ = check_table([_table_claim(evidence="rows", cites=["x1"])], labelled, summary)
    assert kept[0].at == "rows x1"
    by_structure = _table_claim(evidence="rows", cites=["CCO"])
    assert check_table([by_structure], labelled, summary) == ([], 1)

    # An identifier is cited as the look showed it, on one line.
    untidy, summary = _table({"ID": ["a ", "b\nc"], "pIC50": [6.1, 7.4]})
    kept, _ = check_table([_table_claim(evidence="rows", cites=["a", "b c"])], untidy, summary)
    assert kept[0].at == "rows a, b c"


def test_a_table_look_is_capped_and_elides_structures() -> None:
    smiles = [f"CC(=O)N{'C' * i}c1ccccc1" for i in range(200)]
    peptides = [f"MKVLAAGIVGLLLAQ{'A' * i}KRRQPSTW" for i in range(200)]
    rows, summary = _table(
        {
            "ID": [f"c{i}" for i in range(200)],
            "SMILES": smiles,
            "Molecule": smiles,
            # Headers no rule lists: found by what the column holds.
            "Ligand": smiles,
            "Peptide": peptides,
            # Quotes the SMILES it could not read, in a sentence no pattern would take for one.
            "parse_error": [f"Cannot parse '{s}'" for s in smiles],
            "pIC50": [5 + i / 100 for i in range(200)],
        }
    )
    seen = look_table(rows, Look(step=1, rows=10_000), summary)
    lines = seen.splitlines()
    assert len(lines) == LOOK_ROWS_MAX
    assert lines[0] == (
        "c0: ID=c0, SMILES=<structure>, Molecule=<structure>, Ligand=<structure>, "
        "Peptide=<structure>, parse_error=<structure>, pIC50=5.0"
    )
    assert not any(s in seen for s in [*smiles, *peptides])

    asked = look_table(
        rows, Look(step=1, rows=10_000, columns=["SMILES", "pIC50", "nope"]), summary
    )
    assert len(asked.splitlines()) == LOOK_ROWS_MAX
    assert asked.splitlines()[0] == "c0: SMILES=<structure>, pIC50=5.0"
    assert "nope" not in asked

    # A count below the table's start is not a slice from its end.
    assert look_table(rows, Look(step=1, rows=-1), summary) == "(no rows)"

    # A table that opens with a structure is named by the next column, also for a summary
    # stored before columns were judged by what they hold; one of only structures, by none.
    labelled, summary = _table({"SMILES": ["CCO"], "label": ["x1"]})
    assert look_table(labelled, Look(step=1), summary) == "x1: SMILES=<structure>, label=x1"
    assert look_table(labelled, Look(step=1), {}) == "x1: SMILES=<structure>, label=x1"
    bare, summary = _table({"SMILES": ["CCO"], "Molecule": ["CCN"]})
    assert look_table(bare, Look(step=1), summary) == (
        "<structure>: SMILES=<structure>, Molecule=<structure>"
    )

    # An export that names its rows by `molecule_chembl_id` keeps its ids: only a header that
    # is just "molecule" or "structure" is taken for a structure.
    chembl, summary = _table(
        {
            "canonical_smiles": ["CC(=O)Oc1ccccc1C(=O)O"],
            "molecule_chembl_id": ["CHEMBL25"],
            "standard_value": [10.0],
        }
    )
    assert look_table(chembl, Look(step=1), summary) == (
        "CHEMBL25: canonical_smiles=<structure>, molecule_chembl_id=CHEMBL25, standard_value=10.0"
    )

    # A cell is one short line however it was written.
    noted, summary = _table({"ID": ["a"], "note": ["two\nlines   and " + "x" * 200]})
    assert look_table(noted, Look(step=1), summary) == (
        "a: ID=a, note=" + ("two lines and " + "x" * 200)[:77] + "..."
    )


def test_a_table_summary_never_quotes_a_row() -> None:
    smiles = ["CC(=O)Nc1ccc(O)cc1", "c1ccccc1O", "CCN(CC)CC"]
    frame = pl.DataFrame({"ID": ["a", "b", "c"], "SMILES": smiles, "pIC50": [6.1, None, 7.4]})
    _, summary = codecs.encode("compound_table", frame)
    told = summarize_table(None, summary)
    assert not any(s in told for s in smiles)
    assert "Rows: 3" in told
    assert "Rows are named by the values of ID." in told
    assert "SMILES" in told
    assert "pIC50" in told
    # With every column a structure there is nothing to name the rows by.
    _, summary = codecs.encode("compound_table", pl.DataFrame({"SMILES": smiles}))
    assert "Rows are named by" not in summarize_table(None, summary)


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

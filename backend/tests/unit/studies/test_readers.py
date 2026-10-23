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
    check_cores,
    check_fit,
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
    assert reader("citations").summarize(None, {"r2": 0.5}) == '{"r2": 0.5}'
    structure = {
        "id": "4TRO",
        "chains": ["A", "B"],
        "ligands": [{"comp_id": "ZN", "chain": "A", "resseq": 101, "contacts": [{"chain": "A"}]}],
    }
    told = reader("structure").summarize(None, structure)
    assert "4TRO" in told
    assert "ZN A101" in told
    # Prepared with nothing kept, the receptor says so: `[]` is "keep nothing", not "not given",
    # and the entry's own zinc no longer reads as still in place.
    told = reader("structure").summarize(None, {**structure, "kept_ligands": []})
    assert "kept_ligands: none" in told
    assert "ligands before preparation: ZN A101" in told
    # A correlation is read whole: every coefficient the card shows, so none is "not reported".
    meta = {
        "activity_column": "pIC50",
        "n": 26,
        "excluded": 2,
        "strongest": {"property": "heavy_atoms", "r": 0.4531},
    }
    properties = [
        {"property": "logp", "r": 0.1237, "ci_low": -0.2131, "ci_high": 0.41, "n": 26},
        {"property": "heavy_atoms", "r": 0.4531, "ci_low": 0.1, "ci_high": 0.7, "n": 26},
        {"property": "hbd", "r": None, "ci_low": None, "ci_high": None, "n": 1},
    ]
    assert reader("correlation").summarize({"meta": meta, "properties": properties}, meta) == (
        "Activity: pIC50. Readings left out because they are not numbers: 2.\n"
        "Strongest: heavy_atoms, r 0.453.\n"
        "Each property, as r (95% interval) and n:\n"
        "- logp: r 0.124 (-0.213 to 0.41), n 26\n"
        "- heavy_atoms: r 0.453 (0.1 to 0.7), n 26\n"
        "- hbd: r none (no interval), n 1"
    )
    for kind in ("citations", "structure", "correlation"):
        assert reader(kind).look is None
        assert reader(kind).check is None


def test_checkable_lists_pockets_and_every_table_kind() -> None:
    assert checkable() == sorted(
        {"pocket_candidates", "core_candidates", "model_fit", *codecs.TABLE_KINDS}
    )


def test_a_predicted_pocket_is_described_by_its_probability_and_what_was_ignored() -> None:
    data: dict[str, Any] = {
        "meta": {"receptor_ligands": [], "ignored_ligands": ["NAD"], "numbering": "not checked"},
        "pockets": [
            {
                "rank": 1,
                "probability": 0.81,
                "score": 17.9,
                "interface": False,
                "residues": [{"chain": "A", "resname": "TYR", "resseq": 158}],
                "ligand_overlap": [],
                "annotated": [],
            }
        ],
    }
    text = reader("pocket_candidates").summarize(data, data["meta"])
    assert text.startswith("Ignored by the predictor: NAD.")
    assert "Candidate 1: probability 0.81" in text
    assert "volume" not in text


CORES = {
    "meta": {
        "n_candidates": 2,
        "n_compounds": 26,
        "n_unparsed": 0,
        "notes": ["two presets searched"],
        "series": [],
    },
    "candidates": [
        {
            "rank": 1,
            "smarts": "[#6]",
            "svg": "<svg/>",
            "coverage": 1.0,
            "n_atoms": 13,
            "n_ring_atoms": 12,
            "n_variable": 1,
            "variable": {"X1": "O/S"},
            "n_rgroups": 3,
            "unmatched": [],
            "matched": ["CPD-001", "CPD-002"],
            "misses": [],
            "description": ["#1 ring-aromatic", "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1"],
            "preset_label": "ring aromatic",
            "derived": None,
            "timed_out": False,
            "series": None,
        },
        {
            "rank": 2,
            "smarts": "[#6]",
            "svg": "<svg/>",
            "coverage": 0.92,
            "n_atoms": 14,
            "n_ring_atoms": 12,
            "n_variable": 0,
            "variable": {},
            "n_rgroups": 3,
            "unmatched": ["CPD-004", "CPD-011"],
            "matched": ["CPD-001", "CPD-002"],
            "misses": [],
            "description": ["#2 strict", "[#6]"],
            "preset_label": "strict",
            "derived": None,
            "timed_out": False,
            "series": None,
        },
    ],
}


def _core_claim(**kw: Any) -> Claim:
    return Claim(
        step=1,
        candidates=kw.get("candidates", [2]),
        label=kw.get("label", "the amide-intact core"),
        evidence=kw.get("evidence", "unmatched"),
        cites=kw.get("cites", ["CPD-004"]),
    )


def test_a_core_claim_cites_what_the_candidate_missed_or_let_vary() -> None:
    kept, dropped = check_cores([_core_claim()], CORES)
    assert dropped == 0 and kept[0].at == "rank 2" and kept[0].cites == ("CPD-004",)
    kept, dropped = check_cores(
        [_core_claim(candidates=[1], evidence="variable", cites=["X1", "S"])], CORES
    )
    assert dropped == 0
    _, dropped = check_cores(
        [_core_claim(cites=["CPD-001"])], CORES
    )  # matched, so not citable as a miss
    assert dropped == 1
    _, dropped = check_cores([_core_claim(candidates=[3])], CORES)
    assert dropped == 1
    _, dropped = check_cores([_core_claim(cites=[])], CORES)
    assert dropped == 1


def test_a_core_summary_names_every_candidate_and_never_its_smarts() -> None:
    told = reader("core_candidates").summarize(CORES, CORES["meta"])
    assert "rank 1" in told and "rank 2" in told and "X1" in told and "CPD-004" in told
    assert "[#" not in told and "<svg" not in told


def test_a_full_coverage_core_is_claimed_through_the_compounds_it_covers() -> None:
    exact = {"candidates": [{"rank": 1, "coverage": 1.0, "matched": ["CPD-001", "CPD-002"]}]}
    kept, dropped = check_cores(
        [_core_claim(candidates=[1], evidence="coverage", cites=["CPD-001"])], exact
    )
    assert dropped == 0 and len(kept) == 1
    _, dropped = check_cores(
        [_core_claim(candidates=[1], evidence="coverage", cites=["CPD-009"])], exact
    )
    assert dropped == 1


def test_a_claim_joining_cores_needs_each_core_to_support_every_cite() -> None:
    _, dropped = check_cores([_core_claim(candidates=[1, 2], cites=["CPD-004"])], CORES)
    assert dropped == 1


FIT = {
    "meta": {
        "ok": True,
        "reason": None,
        "activity_column": "pIC50",
        "n_compounds": 8,
        "n_excluded_censored": 1,
        "n_terms": 4,
        "r2": 1.0,
        "q2": 1.0,
        "reference": {"R1": "a", "R2": "x"},
        "varying_positions": ["R1", "R2"],
    },
    "terms": [{"position": "R1", "group": "b", "n": 3, "coef": 1.0, "reference": False}],
    "fitted": [{"id": "ax", "fitted": 5.0}],
    "unmade": [{"groups": {"R1": "c", "R2": "z"}, "prediction": 5.0}],
}


def test_a_fit_claim_cites_terms_groups_or_compounds_the_fit_has() -> None:
    kept, dropped = check_fit(
        [Claim(step=1, label="R1 b adds a log", evidence="term", cites=["R1", "b"])], FIT
    )
    assert dropped == 0 and kept[0].at == "term R1, b"
    _, dropped = check_fit([Claim(step=1, label="", evidence="term", cites=["R3"])], FIT)
    assert dropped == 1
    told = reader("model_fit").summarize(FIT, FIT["meta"])
    assert "q2" in told.lower() or "q²" in told
    assert "R1 b" in told and "c" in told


def _fit_claim(evidence: str, *cites: str, label: str = "") -> Claim:
    return Claim(step=1, label=label, evidence=evidence, cites=list(cites))


def test_a_refused_fit_supports_no_claim() -> None:
    refused = {
        "meta": {"ok": False, "reason": "saturated", "reference": {"R1": "a"}},
        "terms": [],
        "fitted": [],
        "unmade": [],
    }
    kept, dropped = check_fit([_fit_claim("term", "R1", "a")], refused)
    assert kept == [] and dropped == 1


def test_a_term_claim_needs_its_position_and_group_in_one_term() -> None:
    two = {**FIT, "terms": [*FIT["terms"], {"position": "R2", "group": "Cl", "n": 2, "coef": 0.5}]}
    kept, dropped = check_fit([_fit_claim("term", "R1", "Cl")], two)
    assert kept == [] and dropped == 1


def test_an_unmade_claim_cites_groups_not_compounds() -> None:
    kept, dropped = check_fit([_fit_claim("unmade", "CPD-001")], FIT)
    assert kept == [] and dropped == 1


def test_a_fit_claim_cites_positions_and_not_fitted_compounds() -> None:
    kept, dropped = check_fit([_fit_claim("fit", "ax")], FIT)
    assert kept == [] and dropped == 1


def test_the_fit_notes_forbid_a_number_in_a_label() -> None:
    assert "no number in a claim's label" in reader("model_fit").notes

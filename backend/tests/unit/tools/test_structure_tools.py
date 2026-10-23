"""The preparer and the pocket finders.

The preparer runs on crambin. fpocket finds no pocket on crambin at its default
settings, so the pocket finder runs on ubiquitin (1UBQ, ATOM records only),
which has four; crambin's first three residues are the no-pocket case. p2rank
finds no ligandable point on ubiquitin at its defaults, so its positive cases run
on carbonic anhydrase II (1CA2, ATOM records only), whose active site is known.

PDBFixer and PROPKA live in the `chem-tools` extra and fpocket and p2rank are
binaries on that lane, so each test skips where its dependency is missing rather than
failing CI, which installs none of them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from daikonstudio.infrastructure.tools.impl import structure
from daikonstudio.infrastructure.tools.params import PrepareParams

FIXTURES = Path(__file__).parents[2] / "fixtures"
CRAMBIN = (FIXTURES / "1crn.pdb").read_text()
UBIQUITIN = (FIXTURES / "1ubq.pdb").read_text()
CARBONIC_ANHYDRASE = (FIXTURES / "1ca2.pdb").read_text()

needs_fpocket = pytest.mark.skipif(
    not (os.environ.get("STUDIO_FPOCKET_BIN") or shutil.which("fpocket")),
    reason="fpocket is not installed",
)
needs_prank = pytest.mark.skipif(
    not (os.environ.get("STUDIO_P2RANK_BIN") or shutil.which("prank")),
    reason="p2rank is not installed",
)


def _structure(text: str, ligands: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"format": "pdb", "text": text, "meta": {"id": "1CRN", "ligands": ligands or []}}


def _atoms(text: str) -> int:
    return sum(line.startswith("ATOM") for line in text.splitlines())


def _with_hetero(text: str) -> str:
    het = [
        "HETATM  328  C1  LIG A 100      10.000  10.000   5.000  1.00 20.00           C  ",
        "HETATM  329  O   HOH A 200      12.000  12.000   6.000  1.00 30.00           O  ",
        "HETATM  330  C1  GOL A 300      14.000  14.000   7.000  1.00 30.00           C  ",
    ]
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("TER")) + 1
    return "\n".join([*lines[:at], *het, *lines[at:]])


def _sequence(text: str) -> str:
    """The structure's own sequence, so a test that is not about numbering has
    numbering that agrees."""
    seen: dict[int, str] = {}
    for line in text.splitlines():
        if line.startswith("ATOM"):
            letter = structure._AA3.get(line[17:20].strip())
            if letter:
                seen[int(line[22:26])] = letter
    return "".join(seen.get(i + 1, "X") for i in range(max(seen)))


def _pockets(text: str, **kwargs: Any) -> dict[str, Any]:
    inputs: dict[str, object] = {"structure": _structure(text, kwargs.get("ligands"))}
    if "annotations" in kwargs:
        inputs["annotations"] = kwargs["annotations"]
    out = structure.find_pockets(None, inputs)["pockets"]
    assert isinstance(out, dict)
    return out


def _predicted(text: str, **kwargs: Any) -> dict[str, Any]:
    inputs: dict[str, object] = {"structure": _structure(text, kwargs.get("ligands"))}
    out = structure.predict_pockets(None, inputs)["pockets"]
    assert isinstance(out, dict)
    return out


class TestPrepare:
    def setup_method(self) -> None:
        pytest.importorskip("pdbfixer")
        pytest.importorskip("propka")

    def test_adds_hydrogens_and_estimates_pkas(self) -> None:
        out = structure.prepare_structure(PrepareParams(), {"structure": _structure(CRAMBIN)})
        prepared = out["structure"]
        assert isinstance(prepared, dict)
        assert _atoms(prepared["text"]) > _atoms(CRAMBIN)
        meta = prepared["meta"]
        assert meta["prepared"] is True
        assert meta["ph"] == 7.4
        assert meta["id"] == "1CRN"
        assert meta["ligands"] == []
        assert meta["atoms"] == _atoms(prepared["text"])
        assert len(meta["pkas"]) > 0
        assert set(meta["pkas"][0]) == {"residue", "number", "chain", "pka"}

    def test_keeps_the_cofactor_and_drops_the_drop(self) -> None:
        """A cofactor is part of what a ligand would have to bind alongside; a
        water and a glycerol are what the crystallographer added."""
        ligands = [{"comp_id": "LIG", "chain": "A", "resseq": 100, "contacts": []}]
        out = structure.prepare_structure(
            PrepareParams(ph=6.0), {"structure": _structure(_with_hetero(CRAMBIN), ligands)}
        )
        prepared = out["structure"]
        assert isinstance(prepared, dict)
        assert prepared["meta"]["removed"] == "waters and crystallisation additives"
        assert prepared["meta"]["kept_ligands"] == ["LIG"]
        assert prepared["meta"]["ph"] == 6.0
        assert prepared["meta"]["ligands"] == ligands
        kept = [line for line in prepared["text"].splitlines() if line.startswith("HETATM")]
        assert [line[17:20].strip() for line in kept] == ["LIG"]


@needs_fpocket
class TestPockets:
    def test_finds_pockets_on_ubiquitin(self) -> None:
        out = _pockets(UBIQUITIN)
        assert out["meta"]["count"] >= 1
        assert out["meta"]["count"] == len(out["pockets"])
        assert out["meta"]["numbering"]["checked"] is False
        assert out["meta"]["receptor_ligands"] == []
        assert out["meta"]["top"]["rank"] == 1
        assert out["meta"]["ligand_site_ranks"] == []
        for pocket in out["pockets"]:
            assert pocket["residues"]
            assert 0 <= pocket["druggability"] <= 1
            assert pocket["interface"] is False
            assert set(pocket["residues"][0]) == {"chain", "resname", "resseq"}
            assert pocket["lines_ligand"] == []

    def test_a_pocket_whose_only_ligand_came_from_the_drop_says_so(self) -> None:
        """Glycerol marks a hole something froze in, not a site worth aiming at,
        and it must not promote the pocket into the ligand-site list."""
        residues = _pockets(UBIQUITIN)["pockets"][0]["residues"]
        contacts = [
            {"chain": r["chain"], "resname": r["resname"], "resseq": r["resseq"]} for r in residues
        ]
        ligand = {"comp_id": "GOL", "chain": "A", "resseq": 300, "contacts": contacts}
        out = _pockets(UBIQUITIN, ligands=[ligand])
        assert out["pockets"][0]["ligand_overlap"][0]["additive"] is True
        assert out["meta"]["ligand_site_ranks"] == []

    def test_keep_ligands_accepts_the_text_a_planner_can_actually_send(self) -> None:
        """Structured output carries settings as name/value text, so a list
        arrives as a string. Without this the plan is rejected outright."""
        assert PrepareParams(ph=7.0, keep_ligands="").keep_ligands == []
        assert PrepareParams(ph=7.0, keep_ligands="[]").keep_ligands == []
        assert PrepareParams(ph=7.0, keep_ligands="NAD, HEM").keep_ligands == ["NAD", "HEM"]
        assert PrepareParams(ph=7.0).keep_ligands is None

    def test_keep_ligands_strips_a_bound_inhibitor(self) -> None:
        """Keeping a cofactor is right; keeping an inhibitor fills the site the
        study is asking about, so the plan has to be able to say which."""
        out = structure.prepare_structure(
            PrepareParams(ph=7.0, keep_ligands=[]),
            {"structure": _structure(_with_hetero(CRAMBIN), [])},
        )["structure"]
        assert isinstance(out, dict)
        assert out["meta"]["kept_ligands"] == []
        assert not any(line.startswith("HETATM") for line in out["text"].splitlines())

    def test_withholds_site_labels_when_the_numbering_disagrees(self) -> None:
        """The silent failure this guards: labels land on the wrong residues and
        read exactly like labels on the right ones."""
        site = {"type": "Binding site", "start": 1, "end": 76, "description": "whole chain"}
        shifted = "G" * 20 + _sequence(UBIQUITIN)  # as if numbered from a propeptide
        out = _pockets(
            UBIQUITIN,
            annotations={"meta": {}, "features": [site], "sequence": shifted},
        )
        assert out["meta"]["numbering"]["matches"] is False
        assert out["meta"]["numbering"]["offset"] == 20
        assert all(p["annotated"] == [] for p in out["pockets"])

    def test_labels_pockets_with_annotated_sites(self) -> None:
        site = {"type": "Binding site", "start": 1, "end": 76, "description": "whole chain"}
        out = _pockets(
            UBIQUITIN,
            annotations={"meta": {}, "features": [site], "sequence": _sequence(UBIQUITIN)},
        )
        assert out["meta"]["numbering"]["matches"] is True
        assert all(pocket["annotated"] == [site] for pocket in out["pockets"])

    def test_a_pocket_lined_by_a_ligands_contacts_names_it(self) -> None:
        first = _pockets(UBIQUITIN)["pockets"][0]
        contacts = [
            {"chain": r["chain"], "resname": r["resname"], "resseq": r["resseq"]}
            for r in first["residues"]
        ]
        ligand = {"comp_id": "LIG", "chain": "A", "resseq": 100, "contacts": contacts}
        out = _pockets(UBIQUITIN, ligands=[ligand])
        overlap = out["pockets"][0]["ligand_overlap"]
        assert overlap == [
            {
                "comp_id": "LIG",
                "chain": "A",
                "resseq": 100,
                "shared": len(contacts),
                "of": len(contacts),
                "additive": False,
            }
        ]
        assert 1 in out["meta"]["ligand_site_ranks"]

    def test_a_peptide_has_no_pockets_and_that_is_an_answer(self) -> None:
        peptide = "\n".join(
            line
            for line in CRAMBIN.splitlines()
            if line.startswith("ATOM") and int(line[22:26]) <= 3
        )
        out = _pockets(peptide + "\nEND\n")
        assert out["meta"]["count"] == 0
        assert out["meta"]["top"] is None
        assert out["pockets"] == []


def test_missing_fpocket_is_a_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STUDIO_FPOCKET_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="fpocket is not installed"):
        structure.fpocket_bin()


@needs_prank
class TestP2rank:
    def test_predicts_the_active_site_with_a_probability_not_a_druggability(self) -> None:
        """One run carries three facts. The zinc site comes back first; a cofactor
        is neither part of the receptor nor a pocket of its own, so the result says
        it was left out (water and glycerol came from the drop and are not named);
        and a ligand's contacts overlap a site as they do for fpocket."""
        # The zinc site: His94 binds the metal, His64 shuttles the proton, Thr200
        # gates the substrate.
        site = [
            {"chain": "A", "resname": "HIS", "resseq": 94},
            {"chain": "A", "resname": "HIS", "resseq": 64},
            {"chain": "A", "resname": "THR", "resseq": 200},
        ]
        ligand = {"comp_id": "LIG", "chain": "A", "resseq": 100, "contacts": site}
        out = _predicted(_with_hetero(CARBONIC_ANHYDRASE), ligands=[ligand])
        assert out["meta"]["count"] == len(out["pockets"]) >= 1
        assert out["meta"]["top"] == {"rank": 1, "probability": out["pockets"][0]["probability"]}
        assert out["meta"]["receptor_ligands"] == []
        assert out["meta"]["ignored_ligands"] == ["LIG"]
        for pocket in out["pockets"]:
            assert 0 <= pocket["probability"] <= 1
            assert "druggability" not in pocket and "volume" not in pocket
            assert pocket["residues"]
            assert set(pocket["residues"][0]) == {"chain", "resname", "resseq"}
            assert all(r["resname"] in structure._AA3 for r in pocket["residues"])
        lining = {(r["chain"], r["resname"], r["resseq"]) for r in out["pockets"][0]["residues"]}
        assert {("A", "HIS", 94), ("A", "HIS", 64), ("A", "THR", 200)} <= lining
        assert out["pockets"][0]["ligand_overlap"][0]["shared"] == len(site)
        assert 1 in out["meta"]["ligand_site_ranks"]

    def test_a_peptide_has_no_pockets_and_that_is_an_answer(self) -> None:
        peptide = "\n".join(
            line
            for line in CRAMBIN.splitlines()
            if line.startswith("ATOM") and int(line[22:26]) <= 3
        )
        out = _predicted(peptide + "\nEND\n")
        assert out["meta"]["count"] == 0
        assert out["meta"]["ignored_ligands"] == []
        assert out["meta"]["top"] is None
        assert out["pockets"] == []


def test_predictions_parse_padded_cells_and_insertion_codes() -> None:
    """Every cell in p2rank's CSVs is space-padded, the headers too, and a
    residue label can carry an insertion code; the residue keeps its number."""
    predictions = (
        "name     ,  rank,   score, probability, sas_points, surf_atoms,   center_x,   "
        "center_y,   center_z, residue_ids, surf_atom_ids\n"
        "pocket1  ,     1,   17.86,       0.810,         88,         41,     8.0527,   "
        "-34.4723,    11.2115, A_52A A_7, 1 2\n"
    )
    residues = (
        "chain, residue_label, residue_name, score, zscore, probability, pocket\n"
        "A,   52A, HIS,  0.5,  0.1,   0.4, 1\n"
        "A,     7, GLY,  0.0,  0.0,   0.0, 1\n"
    )
    assert structure._predictions(predictions, residues) == [
        {
            "rank": 1,
            "score": 17.86,
            "probability": 0.81,
            "residues": [
                {"chain": "A", "resname": "HIS", "resseq": 52},
                {"chain": "A", "resname": "GLY", "resseq": 7},
            ],
        }
    ]

"""The preparer (PDBFixer, PROPKA) and the pocket finder (fpocket).

PDBFixer, OpenMM and PROPKA are the `chem-tools` extra and fpocket is a binary on
that lane, so all of them are reached inside the function bodies: importing
this module costs nothing where they are absent.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from daikonstudio.infrastructure.tools.params import PrepareParams

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

# Crystallisation, cryoprotection and buffer components. A candidate whose only
# ligand is one of these is a hole something from the drop sat in, not a known
# site, and none of them are part of the receptor. Metals are deliberately
# absent: a zinc or a magnesium is as often catalytic as it is a salt.
ADDITIVES = frozenset(
    {
        "HOH",
        "DOD",
        "GOL",
        "EDO",
        "PEG",
        "PG4",
        "PGE",
        "1PE",
        "2PE",
        "MPD",
        "DMS",
        "SO4",
        "PO4",
        "ACT",
        "ACY",
        "CIT",
        "FLC",
        "TRS",
        "MES",
        "EPE",
        "IMD",
        "BME",
        "DTT",
        "FMT",
        "NO3",
        "AZI",
        "MLI",
        "TLA",
        "SCN",
        "IOD",
        "BR",
        "CL",
        "NA",
        "K",
    }
)

# For checking that the file's residue numbers mean what UniProt's mean.
_AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Preparation can leave a protonation state in the residue name.
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "CYX": "C",
    "CYM": "C",
    "ASH": "D",
    "GLH": "E",
    "LYN": "K",
}

_POCKET = re.compile(r"^Pocket (\d+) :")
_FIELDS = {
    "Score": "score",
    "Druggability Score": "druggability",
    "Volume": "volume",
    "Number of Alpha Spheres": "alpha_spheres",
}


def prepare_structure(
    params: BaseModel | None, inputs: Mapping[str, object]
) -> Mapping[str, object]:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer
    from propka.run import single

    ph = cast(PrepareParams, params).ph
    source = cast(dict[str, Any], inputs["structure"])
    text: str = source["text"]

    fixer = PDBFixer(pdbfile=io.StringIO(text))
    fixer.findMissingResidues()
    gaps = len(fixer.missingResidues)
    # Record the gaps and leave them open. Building an unobserved loop invents
    # coordinates, and a loop is often disordered precisely because it lines an
    # open site: on InhA (4TRN) modelling the one gap drops the substrate pocket
    # from the best candidate in the protein to the twelfth, and promotes a
    # surface groove in its place. Side chains of residues that are present are
    # still completed below — that is repair, this would be modelling.
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    replaced = sorted({residue.name for residue, _ in fixer.nonstandardResidues})
    fixer.replaceNonstandardResidues()
    # PDBFixer has no template to protonate a cofactor with, so the heterogens
    # come out here and the biological ones go back below as deposited.
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    buffer = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buffer, keepIds=True)
    prepared = buffer.getvalue()

    # Put the cofactors back, as deposited and unprotonated. A pocket finder
    # tessellates an empty cofactor cleft and reports it as the best site in the
    # protein, which answers "where does the cofactor go" when the question was
    # "where else could something bind". Waters and anything from the
    # crystallisation drop stay out.
    wanted = cast(PrepareParams, params).keep_ligands
    kept = [
        line
        for line in text.splitlines()
        if line.startswith("HETATM")
        and line[17:20].strip() not in ADDITIVES
        and (wanted is None or line[17:20].strip() in {w.strip().upper() for w in wanted})
    ]
    if kept:
        body = [line for line in prepared.splitlines() if line.strip() != "END"]
        # Renumber the serials. Preparation renumbers the protein from 1, so a
        # cofactor pasted back with its deposited serials collides with atoms
        # that already exist — and a reader that indexes by serial silently
        # keeps one of each pair. fpocket does, which is how a cofactor can be
        # in the file and absent from the tessellation at the same time.
        serial = max(
            (int(line[6:11]) for line in body if line.startswith(("ATOM", "HETATM"))),
            default=0,
        )
        renumbered = []
        for line in kept:
            serial += 1
            renumbered.append(f"{line[:6]}{serial:>5}{line[11:]}")
        prepared = "\n".join([*body, *renumbered, "END"]) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.pdb"
        path.write_text(text)
        molecule = single(str(path), optargs=["--quiet"], write_pka=False)
    pkas = [
        {
            "residue": group.atom.res_name,
            "number": group.atom.res_num,
            "chain": group.atom.chain_id,
            "pka": round(float(group.pka_value), 2),
        }
        for group in molecule.conformations["AVR"].groups
        if group.titratable
    ]

    meta = {
        **source.get("meta", {}),
        "prepared": True,
        "ph": ph,
        "removed": "waters and crystallisation additives",
        "kept_ligands": sorted({line[17:20].strip() for line in kept}),
        "missing_residues": gaps,
        "gaps_modelled": False,
        "nonstandard_replaced": replaced,
        "atoms": sum(line.startswith("ATOM") for line in prepared.splitlines()),
        "pkas": pkas,
    }
    return {"structure": {"format": "pdb", "text": prepared, "meta": meta}}


def tool_binary(env: str, name: str) -> str:
    found = os.environ.get(env) or shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} is not installed on this lane.")
    return found


def fpocket_bin() -> str:
    return tool_binary("STUDIO_FPOCKET_BIN", "fpocket")


def _info(text: str) -> list[dict[str, Any]]:
    """The `Pocket N :` blocks of fpocket's info file, by rank."""
    pockets: list[dict[str, Any]] = []
    for line in text.splitlines():
        if match := _POCKET.match(line):
            pockets.append({"rank": int(match.group(1))})
        elif pockets and ":" in line:
            name, _, value = line.partition(":")
            if (key := _FIELDS.get(name.strip())) is not None:
                pockets[-1][key] = float(value)
    for pocket in pockets:
        pocket["alpha_spheres"] = int(pocket.get("alpha_spheres", 0))
        # fpocket estimates volume by Monte Carlo (300 iterations by default),
        # so it moves a few percent between runs on the same file. Six decimals
        # of that is a measurement the number cannot support.
        if "volume" in pocket:
            pocket["volume"] = round(float(pocket["volume"]))
    return pockets


def _residues(text: str) -> list[dict[str, Any]]:
    seen = {
        (line[21], line[17:20].strip(), int(line[22:26]))
        for line in text.splitlines()
        if line.startswith("ATOM")
    }
    return [
        {"chain": chain, "resname": resname, "resseq": resseq}
        for chain, resname, resseq in sorted(seen, key=lambda r: (r[0], r[2]))
    ]


def check_numbering(residues: Mapping[int, str], sequence: str) -> dict[str, Any]:
    """Do the file's residue numbers mean what UniProt's mean?

    Every annotated-site label hangs on this. It holds for a bacterial enzyme
    deposited whole and breaks for a construct numbered from its own first
    residue, or for a chain whose signal or propeptide was cleaved before the
    mature protein was numbered — which is most of what anyone works on. The
    failure is silent: the labels still appear, against the wrong residues. So
    compare identities, and when they disagree look for the constant offset that
    would reconcile them, because naming it is more use than a bare refusal.
    """
    if not sequence:
        return {"checked": False, "reason": "no UniProt sequence to compare against"}

    def agreement(offset: int) -> tuple[int, int]:
        hits = total = 0
        for number, name in residues.items():
            letter = _AA3.get(name)
            index = number - 1 + offset
            if letter is None or not 0 <= index < len(sequence):
                continue
            total += 1
            hits += letter == sequence[index]
        return hits, total

    hits, total = agreement(0)
    if total and hits / total >= 0.95:
        return {"checked": True, "matches": True, "agree": hits, "of": total, "offset": 0}
    scored = [(agreement(offset), offset) for offset in range(-500, 501)]
    (best_hits, best_total), offset = max(scored, key=lambda t: (t[0][0], -abs(t[1])))
    reconciled = bool(best_total) and best_hits / best_total >= 0.95
    return {
        "checked": True,
        "matches": False,
        "agree": hits,
        "of": total,
        "offset": offset if reconciled else None,
    }


def evidence(
    pockets: list[dict[str, Any]],
    *,
    source: Mapping[str, Any],
    annotations: Mapping[str, Any] | None,
    on_receptor: set[str],
) -> dict[str, Any]:
    """What either finder's candidates are checked against: the deposited
    ligands, the numbering, and the annotated sites.

    Each pocket arrives with its rank and lining residues and leaves with the
    evidence columns added. The returned meta has no `top`, because each finder
    names its own score there.
    """
    ligands: list[dict[str, Any]] = source.get("meta", {}).get("ligands", [])
    features: list[dict[str, Any]] = (annotations or {}).get("features", [])

    numbering = check_numbering(
        {r["resseq"]: r["resname"] for p in pockets for r in p["residues"]},
        (annotations or {}).get("sequence", ""),
    )
    labelled = numbering.get("matches", False)

    for pocket in pockets:
        # A cofactor made part of the receptor comes back as a lining "residue".
        # It lines the pocket, which is worth saying, but it is not a residue of
        # the protein and does not belong in a residue list or a site label.
        pocket["lines_ligand"] = sorted(
            {r["resname"] for r in pocket["residues"] if r["resname"] in on_receptor}
        )
        pocket["residues"] = [r for r in pocket["residues"] if r["resname"] in _AA3]
        lining = {(r["chain"], r["resseq"]) for r in pocket["residues"]}
        numbers = {r["resseq"] for r in pocket["residues"]}
        pocket["ligand_overlap"] = [
            {
                "comp_id": ligand["comp_id"],
                "chain": ligand["chain"],
                "resseq": ligand["resseq"],
                "shared": shared,
                "of": len(contacts),
                "additive": ligand["comp_id"] in ADDITIVES,
            }
            for ligand in ligands
            if (contacts := {(c["chain"], c["resseq"]) for c in ligand.get("contacts", [])})
            and (shared := len(contacts & lining))
        ]
        # No site label unless the numbering was checked and agreed: a label on
        # the wrong residues is worse than no label.
        pocket["annotated"] = [
            {key: feature.get(key) for key in ("type", "start", "end", "description")}
            for feature in (features if labelled else [])
            if feature.get("start") is not None
            and feature.get("end") is not None
            and any(feature["start"] <= n <= feature["end"] for n in numbers)
        ]
        pocket["interface"] = len({r["chain"] for r in pocket["residues"]}) > 1

    return {
        "count": len(pockets),
        # The viewer on the result renders this entry, so the pocket value has to
        # carry which structure its residue numbers belong to.
        "structure_id": source.get("meta", {}).get("id"),
        "numbering": numbering,
        "receptor_ligands": sorted(on_receptor),
        "ligand_site_ranks": [
            p["rank"] for p in pockets if any(not o["additive"] for o in p["ligand_overlap"])
        ],
    }


def find_pockets(params: BaseModel | None, inputs: Mapping[str, object]) -> Mapping[str, object]:
    source = cast(dict[str, Any], inputs["structure"])
    annotations = cast(dict[str, Any] | None, inputs.get("annotations"))

    # fpocket tessellates ATOM records only, so a cofactor left as HETATM is a
    # void it will rank as a pocket. Rewriting it makes it part of the receptor.
    receptor = []
    on_receptor: set[str] = set()
    for line in cast(str, source["text"]).splitlines():
        if line.startswith("HETATM") and line[17:20].strip() not in ADDITIVES:
            on_receptor.add(line[17:20].strip())
            receptor.append("ATOM  " + line[6:])
        else:
            receptor.append(line)

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "in.pdb").write_text("\n".join(receptor) + "\n")
        subprocess.run(
            [fpocket_bin(), "-f", "in.pdb"], cwd=tmp, check=True, capture_output=True, timeout=600
        )
        out = Path(tmp) / "in_out"
        pockets = _info((out / "in_info.txt").read_text())
        for pocket in pockets:
            atoms = out / "pockets" / f"pocket{pocket['rank']}_atm.pdb"
            pocket["residues"] = _residues(atoms.read_text())

    meta = evidence(pockets, source=source, annotations=annotations, on_receptor=on_receptor)
    meta["top"] = (
        {"rank": pockets[0]["rank"], "druggability": pockets[0]["druggability"]}
        if pockets
        else None
    )
    return {"pockets": {"meta": meta, "pockets": pockets}}


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "prepare_structure": prepare_structure,
    "find_pockets": find_pockets,
}

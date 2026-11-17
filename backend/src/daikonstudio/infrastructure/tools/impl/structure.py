"""The preparer (PDBFixer, PROPKA) and the pocket finders (fpocket, p2rank).

PDBFixer, OpenMM and PROPKA are the `chem-tools` extra, and fpocket and p2rank
(a Java program) are binaries on that lane, so all of them are reached inside
the function bodies: importing this module costs nothing where they are absent.
"""

from __future__ import annotations

import csv
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

from daikonstudio.infrastructure.tools.params import (
    DockParams,
    InteractionParams,
    PrepareParams,
    ValidityParams,
)

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
    gaps = _unmodelled(text, len(fixer.missingResidues))
    # Record the gaps and leave them open. Building an unobserved loop invents
    # coordinates, and a loop is often disordered precisely because it lines an
    # open site, so a cavity found against a built wall is partly the builder's.
    # Side chains of residues that are present are still completed below — that
    # is repair, this would be modelling. This said that modelling 4TRN's one gap
    # moved the substrate pocket from first to twelfth; measured, it moves
    # nothing (same 14 pockets, same rank 12 at druggability 0.032), because that
    # gap is MET1. The reason above is the reason, not a ranking.
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
        # The pKa list is 50-odd records and reaches a reader as a count, here and in the
        # card. What a gate is for is the residues that came out against expectation, so
        # those are named and the sentence below says what the silence about the rest means.
        "protonation_shifted": shifted_protonation(pkas, ph),
        "protonation": (
            f"Assigned from estimated pKa at pH {ph}. Titratable groups not named above "
            f"took the state their kind usually has at this pH."
        ),
    }
    return {"structure": {"format": "pdb", "text": prepared, "meta": meta}}


# The pKa a titratable group has in free solution, and whether being protonated makes it
# charged. A group whose computed pKa puts it on the other side of the working pH from
# these is the only thing worth saying about protonation: the rest took the state anyone
# would have assumed. Values are the usual textbook set; PROPKA names the termini N+ and C-.
_USUAL_PKA = {
    "ASP": 3.8,
    "GLU": 4.5,
    "C-": 3.2,
    "CYS": 9.0,
    "TYR": 10.0,
    "HIS": 6.5,
    "LYS": 10.5,
    "ARG": 12.5,
    "N+": 8.0,
}
# Protonating these removes a charge rather than adding one, so the comparison flips.
_ACIDIC = frozenset({"ASP", "GLU", "C-", "CYS", "TYR"})


def _charged(residue: str, pka: float, ph: float) -> bool:
    """Whether the group carries a charge at this pH, given this pKa."""
    return ph > pka if residue in _ACIDIC else ph < pka


def shifted_protonation(pkas: list[dict[str, Any]], ph: float) -> list[str]:
    """The residues whose assigned state is not the one their kind usually takes at this pH.

    A preparation gate exists so a scientist can read the protonation before docking into
    it, and 56 pKa values is not something anyone reads. What changes a result is the
    residue that came out against expectation -- a buried aspartate left neutral, a
    histidine that came out charged -- so those are named and the rest are covered by
    saying what the others did.
    """
    told = []
    for group in pkas:
        residue = str(group["residue"])
        usual = _USUAL_PKA.get(residue)
        pka = float(group["pka"])
        if usual is None or _charged(residue, pka, ph) == _charged(residue, usual, ph):
            continue
        state = "charged" if _charged(residue, pka, ph) else "neutral"
        told.append(
            f"{residue}{group['number']} {group['chain']} pKa {pka}, {state} "
            f"(usually {'neutral' if state == 'charged' else 'charged'})"
        )
    return told


def _unmodelled(text: str, gaps: int) -> list[str]:
    """Which residues the deposition says were never located, as "MET A1".

    A count alone cannot be acted on: a gap at the N-terminus is harmless and one on a
    loop that borders a site makes every cavity there suspect, and told only "one residue
    is missing" a reader has to assume the worse of the two. PDBFixer knows where the gaps
    are but keys them by position within the chain, which is not a number anyone designs
    against; REMARK 465 is the deposited record and names them. A file without it, a
    predicted model or one already prepared, falls back to the count.
    """
    named: list[str] = []
    listing = False
    for line in text.splitlines():
        if not line.startswith("REMARK 465"):
            continue
        rest = line[10:].strip()
        if rest.startswith("M RES C SSSEQI"):
            listing = True
        elif listing and len(parts := rest.split()) >= 3 and parts[-1][:1].isdigit():
            # "MET A 1", or "1 MET A 1" when the entry has models. The number is last
            # either way, with an insertion code stuck to it when there is one.
            named.append(f"{parts[-3]} {parts[-2]}{parts[-1]}")
    # Gated on `gaps` so a file that was modelled elsewhere but kept its remarks does not
    # report residues it now holds.
    if not gaps:
        return []
    return named or [f"{gaps} not named in this entry"]


def tool_binary(env: str, name: str) -> str:
    found = os.environ.get(env) or shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} is not installed on this lane.")
    return found


def _run(name: str, command: list[str], cwd: str) -> None:
    """Run a finder; a failure says which tool failed and why, not the command line."""
    try:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True, timeout=600)
    except subprocess.CalledProcessError as error:
        output = (error.stderr or error.stdout or b"").decode(errors="replace").strip()
        raise RuntimeError(f"{name} failed: {output[-300:]}") from error


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
        _run("fpocket", [fpocket_bin(), "-f", "in.pdb"], cwd=tmp)
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


def prank_bin() -> str:
    return tool_binary("STUDIO_P2RANK_BIN", "prank")


def _rows(text: str) -> list[dict[str, str]]:
    # p2rank pads every cell with spaces, the header cells included.
    return [
        {key.strip(): value.strip() for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(text), skipinitialspace=True)
    ]


def _predictions(predictions_csv: str, residues_csv: str) -> list[dict[str, Any]]:
    """p2rank's predicted sites, by rank, with their lining residues.

    The predictions file names a residue only as chain and label (`A_52A`); its
    name comes from the residues file. A label can carry an insertion code, and
    the residue keeps the number in front of it.
    """
    names = {(r["chain"], r["residue_label"]): r["residue_name"] for r in _rows(residues_csv)}
    pockets = []
    for row in _rows(predictions_csv):
        residues = []
        for residue_id in row["residue_ids"].split():
            chain, _, label = residue_id.partition("_")
            if number := re.match(r"-?\d+", label):
                residues.append(
                    {"chain": chain, "resname": names[(chain, label)], "resseq": int(number[0])}
                )
        pockets.append(
            {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "probability": float(row["probability"]),
                "residues": residues,
            }
        )
    return sorted(pockets, key=lambda p: p["rank"])


def predict_pockets(
    params: BaseModel | None, inputs: Mapping[str, object]
) -> Mapping[str, object]:
    source = cast(dict[str, Any], inputs["structure"])
    annotations = cast(dict[str, Any] | None, inputs.get("annotations"))
    text = cast(str, source["text"])

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "in.pdb").write_text(text)
        command = [prank_bin(), "predict", "-f", "in.pdb", "-o", "out"]
        _run("p2rank", [*command, "-threads", "1", "-visualizations", "0"], cwd=tmp)
        out = Path(tmp) / "out"
        pockets = _predictions(
            (out / "in.pdb_predictions.csv").read_text(), (out / "in.pdb_residues.csv").read_text()
        )

    meta = evidence(pockets, source=source, annotations=annotations, on_receptor=set())
    # p2rank reads protein atoms only: a cofactor is ignored whether it is
    # written as HETATM or as ATOM, so it is neither part of the receptor nor a
    # pocket of its own. The result says which ones were left out, because the
    # site a cofactor occupies is then reported as open.
    meta["ignored_ligands"] = sorted(
        {
            line[17:20].strip()
            for line in text.splitlines()
            if line.startswith("HETATM") and line[17:20].strip() not in ADDITIVES
        }
    )
    meta["top"] = (
        {"rank": pockets[0]["rank"], "probability": pockets[0]["probability"]} if pockets else None
    )
    return {"pockets": {"meta": meta, "pockets": pockets}}


# The smallest box worth docking in, and the margin left around the lining atoms. A pocket's
# residues are its *lining*, so their atoms bound the wall rather than the cavity, and a box
# drawn tight to them clips poses that reach into the opening. The floor matters for a shallow
# site, where the extent in one axis can be a couple of angstroms and a ligand would not fit.
BOX_PADDING = 4.0
BOX_MINIMUM = 12.0


def box_for_pocket(
    text: str, residues: list[Mapping[str, Any]], *, padding: float = BOX_PADDING
) -> dict[str, Any]:
    """The search box around a pocket, in angstroms, from its lining residues' atoms.

    A pocket names residues -- chain, name and number -- and carries no coordinates, because
    what a scientist approves is a site and not a cube. So the box is derived here, from the
    same structure the pocket was found in, and both are reported: the centre and size a
    docking program needs, and the count of residues they came from, which is what tells a
    reader the box describes the site they approved.

    Raises rather than guesses when no atom matches. That means the pocket and the structure
    are not the same molecule -- a pocket found before the structure was re-prepared, most
    likely -- and a box computed from the few residues that happened to match would be
    somewhere arbitrary, which docking would not report as a failure.
    """
    wanted = {(str(r.get("chain", "")).strip(), int(r["resseq"])) for r in residues}
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for line in text.splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if (line[21].strip(), int(line[22:26])) not in wanted:
            continue
        xs.append(float(line[30:38]))
        ys.append(float(line[38:46]))
        zs.append(float(line[46:54]))
    if not xs:
        raise ValueError(
            f"None of the pocket's {len(wanted)} residues were found in this structure. "
            "The pocket and the structure are probably not the same molecule."
        )
    centre = tuple(round((min(a) + max(a)) / 2, 3) for a in (xs, ys, zs))
    size = tuple(round(max(max(a) - min(a) + 2 * padding, BOX_MINIMUM), 3) for a in (xs, ys, zs))
    return {
        "center_x": centre[0],
        "center_y": centre[1],
        "center_z": centre[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "n_residues": len(wanted),
        "n_atoms": len(xs),
        "padding": padding,
    }


def vina_bin() -> str:
    return tool_binary("STUDIO_VINA_BIN", "vina")


def _ligand_pdbqt(smiles: str, seed: int) -> str:
    """One compound as Vina input: a 3D conformer, minimised, written as PDBQT.

    Meeko and RDKit are imported here, not at module scope, for the reason this module
    already gives: the chem lane has them and nowhere else needs to.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore[import-untyped]
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("not a readable structure")
    mol = Chem.AddHs(mol)
    # A docking run with no conformer silently docks a flat molecule, so a failed embed is
    # an error for this compound rather than something to carry on from.
    # ignore[attr-defined]: rdkit-stubs omit these generated AllChem functions.
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:  # type: ignore[attr-defined]
        raise ValueError("no 3D conformer could be generated")
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
    text, ok, error = PDBQTWriterLegacy.write_string(MoleculePreparation()(mol)[0])
    if not ok:
        raise ValueError(str(error))
    return str(text)


def _scored(text: str) -> list[tuple[float, str]]:
    """Vina's output file split into poses, each with the affinity it was ranked by."""
    poses: list[tuple[float, str]] = []
    affinity: float | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            affinity, lines = None, []
        elif line.startswith("REMARK VINA RESULT:"):
            affinity = float(line.split()[3])
        elif line.startswith("ENDMDL"):
            if affinity is not None:
                poses.append((affinity, "\n".join(lines)))
        else:
            lines.append(line)
    return poses


def dock_compounds(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Dock a table of compounds into the approved pocket with AutoDock Vina.

    The box comes from the pocket rather than from the scientist, because the pocket is
    the thing they approved and a box typed by hand is the commonest way a docking run
    answers about the wrong site.

    What this returns is an ordering. A Vina affinity is a score, not a measured binding
    energy, and in the most recent blind assessment molecular weight correlated with the
    truth about as well as the best submitted method did. So the number is carried, and
    the manifest and the note below say what it is not.
    """
    import polars as pl

    settings = cast(DockParams, params or DockParams())
    structure = cast(dict[str, Any], inputs["structure"])
    pocket = cast(dict[str, Any], inputs["pocket"])
    frame = cast(pl.DataFrame, inputs["compounds"])
    text = cast(str, structure["text"])
    box = box_for_pocket(text, list(pocket.get("residues") or []))

    from daikonstudio.infrastructure.tools.impl.chem import structure_column

    column = structure_column(frame)
    # Rows are named by whichever identifier the table carries, falling back to the
    # structure column so a result always says which compound a pose belongs to.
    named = _name_column(frame) or column
    names = [str(v) for v in frame[named]]
    smiles = frame[column].to_list()
    wanted = min(settings.max_compounds, frame.height)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    # The receptor as it was actually docked into: ATOM records only, which is what the
    # preparer was given. Returned beside the poses so a pose can be drawn in the thing it
    # was docked into rather than beside the deposited entry, which still holds the ligand
    # and the waters this run stripped.
    kept_lines = [line for line in text.splitlines() if line.startswith("ATOM")]
    docked_into = "\n".join(kept_lines) + "\nEND\n"
    with tempfile.TemporaryDirectory() as tmp:
        receptor = Path(tmp) / "receptor.pdbqt"
        receptor.write_text(_receptor_pdbqt(text, tmp))
        for index in range(wanted):
            if smiles[index] is None:
                failures.append({"id": names[index], "why": "no structure"})
                continue
            try:
                ligand = Path(tmp) / "ligand.pdbqt"
                ligand.write_text(_ligand_pdbqt(str(smiles[index]), settings.seed))
                out = Path(tmp) / "out.pdbqt"
                _run(
                    "vina",
                    [
                        vina_bin(),
                        "--receptor",
                        str(receptor),
                        "--ligand",
                        str(ligand),
                        "--center_x",
                        str(box["center_x"]),
                        "--center_y",
                        str(box["center_y"]),
                        "--center_z",
                        str(box["center_z"]),
                        "--size_x",
                        str(box["size_x"]),
                        "--size_y",
                        str(box["size_y"]),
                        "--size_z",
                        str(box["size_z"]),
                        "--exhaustiveness",
                        str(settings.exhaustiveness),
                        "--num_modes",
                        str(settings.poses_per_compound),
                        "--seed",
                        str(settings.seed),
                        "--out",
                        str(out),
                    ],
                    tmp,
                )
                for rank, (affinity, geometry) in enumerate(_scored(out.read_text()), start=1):
                    rows.append(
                        {
                            "compound": names[index],
                            "smiles": smiles[index],
                            "pose_rank": rank,
                            "affinity_kcal_mol": affinity,
                            "pose_pdbqt": geometry,
                        }
                    )
            except (ValueError, RuntimeError) as error:
                failures.append({"id": names[index], "why": str(error)[:120]})

    poses = pl.DataFrame(
        rows,
        schema={
            "compound": pl.String,
            "smiles": pl.String,
            "pose_rank": pl.Int64,
            "affinity_kcal_mol": pl.Float64,
            "pose_pdbqt": pl.String,
        },
    )
    best = (
        poses.filter(pl.col("pose_rank") == 1).sort("affinity_kcal_mol").head(25)
        if poses.height
        else poses
    )
    return {
        "poses": poses,
        "receptor": {"text": docked_into, "meta": {"source": "prepared receptor, ATOM records"}},
        "meta": {
            "box": box,
            "n_compounds_docked": wanted - len(failures),
            "n_compounds_given": frame.height,
            "n_not_attempted": max(0, frame.height - wanted),
            "failures": failures[:25],
            # Named here because a pose table is long and the agent sees fifty rows of it.
            "best_by_score": [
                {"compound": c, "affinity_kcal_mol": a}
                for c, a in zip(
                    best["compound"].to_list(), best["affinity_kcal_mol"].to_list(), strict=True
                )
            ],
            "note": (
                "Affinities rank these compounds against each other in this box. They are "
                "not measured binding energies and should not be quoted as potency."
            ),
        },
    }


def _name_column(frame: Any) -> str | None:
    for candidate in ("compound_id", "id", "name"):
        if candidate in frame.columns:
            return candidate
    return None


def _receptor_pdbqt(text: str, tmp: str) -> str:
    """The receptor as Vina input, via Meeko's preparer.

    Run as a subprocess, the way this module already runs fpocket and p2rank: Meeko's
    entry point reads `sys.argv` and exits the process when it is unhappy, neither of
    which a tool running inside the runner can afford.

    Only ATOM records are passed. A pocket's residues were read from ATOM records, so a
    cofactor or a crystallisation additive left in would become part of the receptor and
    could block the very site being docked into.
    """
    source = Path(tmp) / "receptor.pdb"
    kept = [line for line in text.splitlines() if line.startswith("ATOM")]
    source.write_text("\n".join(kept) + "\nEND\n")
    base = Path(tmp) / "rec"
    _run(
        "mk_prepare_receptor",
        [
            tool_binary("STUDIO_MK_PREPARE_RECEPTOR_BIN", "mk_prepare_receptor.py"),
            "--read_pdb",
            str(source),
            "-o",
            str(base),
            "-p",
        ],
        tmp,
    )
    return base.with_suffix(".pdbqt").read_text()


# ProLIF converts the receptor through MDAnalysis, which requires every hydrogen to be
# explicit. Handed a structure without them it does not raise -- it segfaults, taking the
# runner process with it. So the hydrogens are counted first and the tool refuses.
_MIN_HYDROGEN_SHARE = 0.2


def _has_hydrogens(text: str) -> bool:
    atoms = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    if not atoms:
        return False
    # Element is columns 77-78; fall back to the atom name for files that omit it.
    hydrogens = sum(
        1
        for line in atoms
        if (line[76:78].strip() or line[12:16].strip()[:1]).upper().startswith("H")
    )
    return hydrogens / len(atoms) >= _MIN_HYDROGEN_SHARE


def _pose_molecule(pose: str, affinity: float, tmp: str) -> Any:
    """The docked ligand as RDKit sees it, with its bond orders back.

    A pose leaves Vina as PDBQT, which carries coordinates and atom types but no bond
    orders; read as a plain PDB the ligand comes back as a bag of single bonds and every
    interaction computed from it is wrong. Meeko rebuilds it, so the molecule analysed is
    the one that was docked.
    """
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    path = Path(tmp) / "pose.pdbqt"
    path.write_text(f"MODEL 1\nREMARK VINA RESULT: {affinity} 0.0 0.0\n{pose}\nENDMDL\n")
    built = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule.from_file(str(path), skip_typing=True))
    if not built or built[0] is None:
        raise ValueError("the pose could not be rebuilt as a molecule")
    # Non-polar hydrogens are absent from PDBQT and ProLIF needs them to find contacts.
    return Chem.AddHs(built[0], addCoords=True)


def pose_interactions(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Which residue each pose touches, and how.

    This is the half of a docking result that can be checked. A score can only be
    compared with another score, but a contact can be held against what is known about
    the site: a pose that scores well and never reaches the catalytic residue is a pose
    to disbelieve, and nothing in the ranking says so.
    """
    import polars as pl

    # ignore[attr-defined]: prolif ships a py.typed with no __all__, so mypy sees nothing.
    import prolif as plf

    settings = cast(InteractionParams, params or InteractionParams())
    receptor = cast(dict[str, Any], inputs["receptor"])
    frame = cast(pl.DataFrame, inputs["poses"])
    text = cast(str, receptor["text"])
    if not _has_hydrogens(text):
        raise ValueError(
            "This receptor has no hydrogens, and interactions cannot be read without "
            "them. Prepare the structure before docking into it."
        )

    wanted = frame.filter(pl.col("pose_rank") <= settings.poses_per_compound)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        import MDAnalysis as mda  # type: ignore[import-untyped]

        path = Path(tmp) / "receptor.pdb"
        path.write_text(text)
        protein = plf.Molecule.from_mda(  # type: ignore[attr-defined]
            mda.Universe(str(path)).select_atoms("protein")
        )
        for record in wanted.iter_rows(named=True):
            try:
                ligand = _pose_molecule(
                    str(record["pose_pdbqt"]), float(record["affinity_kcal_mol"]), tmp
                )
            except (ValueError, RuntimeError):
                continue
            fingerprint = plf.Fingerprint()  # type: ignore[attr-defined]
            fingerprint.run_from_iterable(
                [plf.Molecule(ligand)],  # type: ignore[attr-defined]
                protein,
                progress=False,
            )
            table = fingerprint.to_dataframe()
            for column in table.columns:
                if not table[column].any():
                    continue
                _, residue, kind = column
                name, _, chain = str(residue).partition(".")
                rows.append(
                    {
                        "compound": record["compound"],
                        "pose_rank": record["pose_rank"],
                        "residue": name,
                        "chain": chain,
                        "interaction": kind,
                    }
                )

    interactions = pl.DataFrame(
        rows,
        schema={
            "compound": pl.String,
            "pose_rank": pl.Int64,
            "residue": pl.String,
            "chain": pl.String,
            "interaction": pl.String,
        },
    )
    kinds = (
        interactions.group_by("interaction").len().sort("len", descending=True)
        if interactions.height
        else interactions
    )
    return {
        "interactions": interactions,
        "meta": {
            "n_poses": wanted.height,
            "n_interactions": interactions.height,
            # Named here because the table is long and a reader sees fifty rows of it.
            "by_kind": (
                {str(k): int(n) for k, n in zip(kinds["interaction"], kinds["len"], strict=True)}
                if interactions.height
                else {}
            ),
            "residues_contacted": sorted(
                {
                    f"{r}.{c}"
                    for r, c in zip(interactions["residue"], interactions["chain"], strict=True)
                }
            )[:40],
            "note": (
                "Contacts are geometric: they say the pose is placed to make an "
                "interaction, not that it does in reality."
            ),
        },
    }


def _verdict(report: Any) -> tuple[bool, str]:
    """One PoseBusters row as a pass and the names of what failed.

    Anything that is not `True` counts as a failure, including a check that could not
    run: a pose this cannot judge is not a pose to trust.
    """
    failed = [str(name) for name, value in report.items() if value is not True]
    return (not failed, ", ".join(failed))


def check_poses(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Which poses are physically possible, before anyone reads their scores.

    PoseBusters' docking checks: chemistry, bond geometry, ring and double-bond
    flatness, internal strain against the molecule's own conformer ensemble, and
    distance to and volume overlap with the receptor. A pose that fails is wrong, not
    weak, and nothing in a score says so -- the blind assessments this comes from found
    low-RMSD poses that no force field would accept.
    """
    import polars as pl
    from posebusters import PoseBusters

    settings = cast(ValidityParams, params or ValidityParams())
    receptor = cast(dict[str, Any], inputs["receptor"])
    frame = cast(pl.DataFrame, inputs["poses"])

    verdicts: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receptor.pdb"
        path.write_text(cast(str, receptor["text"]))
        # The receptor is loaded once per call; poses are judged one at a time so a pose
        # that cannot be rebuilt is a failure of that pose and not of the step.
        buster = PoseBusters(config="dock")
        for record in frame.iter_rows(named=True):
            try:
                ligand = _pose_molecule(
                    str(record["pose_pdbqt"]), float(record["affinity_kcal_mol"]), tmp
                )
                report = buster.bust(mol_pred=ligand, mol_cond=path)
            except (ValueError, RuntimeError) as error:
                verdicts.append((False, f"could not be rebuilt: {str(error)[:80]}"))
                continue
            verdicts.append(_verdict(report.iloc[0]))

    checked = frame.with_columns(
        pl.Series("valid", [ok for ok, _ in verdicts], dtype=pl.Boolean),
        pl.Series("failed_checks", [why for _, why in verdicts], dtype=pl.String),
    )
    invalid = checked.filter(~pl.col("valid"))
    by_check: dict[str, int] = {}
    for why in invalid["failed_checks"]:
        for name in str(why).split(", "):
            by_check[name] = by_check.get(name, 0) + 1
    return {
        "poses": checked if settings.keep_invalid else checked.filter(pl.col("valid")),
        # Passed through, not re-derived: a pose is only worth drawing inside the receptor
        # it was docked into, and a viewer looks for that among its own step's outputs.
        "receptor": receptor,
        "meta": {
            "n_poses": frame.height,
            "n_valid": frame.height - invalid.height,
            "n_invalid": invalid.height,
            "failures_by_check": dict(sorted(by_check.items(), key=lambda kv: -kv[1])),
            # Named here because the invalid poses are the answer and may sit past
            # the rows a reader sees.
            "invalid": [
                {"compound": c, "pose_rank": r, "failed_checks": w}
                for c, r, w in zip(
                    invalid["compound"].head(25),
                    invalid["pose_rank"].head(25),
                    invalid["failed_checks"].head(25),
                    strict=True,
                )
            ],
            "note": (
                "A pose that passes is physically possible, not correct. A pose that "
                "fails is wrong whatever its score."
            ),
        },
    }


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "prepare_structure": prepare_structure,
    "find_pockets": find_pockets,
    "predict_pockets": predict_pockets,
    "dock_compounds": dock_compounds,
    "pose_interactions": pose_interactions,
    "check_poses": check_poses,
}

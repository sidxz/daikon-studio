"""Reading a PDB file: its atoms, its chains, and the ligands touching the protein.

Shared by the fetch tools, which read what an archive sent, and by a structure the scientist
attaches, so a structure is described the same way wherever it came from.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CONTACT_ANGSTROM = 4.0


def parse_pdb(text: str) -> list[dict[str, Any]]:
    """ATOM and HETATM records by their fixed columns."""
    atoms: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line[:6].strip() not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        atoms.append(
            {
                "het": line.startswith("HETATM"),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resseq": int(line[22:26]),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "b": float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else 0.0,
            }
        )
    return atoms


def ca_chains(atoms: list[dict[str, Any]]) -> list[str]:
    return sorted({a["chain"] for a in atoms if a["name"] == "CA" and not a["het"]})


def ligands(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protein = [a for a in atoms if not a["het"]]
    if not protein:
        return []
    coords = np.array([a["xyz"] for a in protein])
    groups: dict[tuple[str, str, int], list[tuple[float, float, float]]] = {}
    for a in atoms:
        if a["het"] and a["resname"] != "HOH":
            groups.setdefault((a["resname"], a["chain"], a["resseq"]), []).append(a["xyz"])
    out = []
    for (comp, chain, resseq), xyz in groups.items():
        dist = np.linalg.norm(coords[:, None, :] - np.array(xyz)[None, :, :], axis=2).min(axis=1)
        near = {
            (p["chain"], p["resname"], p["resseq"])
            for p, d in zip(protein, dist, strict=True)
            if d <= CONTACT_ANGSTROM
        }
        contacts = [
            {"chain": c, "resname": n, "resseq": s}
            for c, n, s in sorted(near, key=lambda t: (t[0], t[2]))
        ]
        out.append({"comp_id": comp, "chain": chain, "resseq": resseq, "contacts": contacts})
    return out


def describe_pdb(text: str) -> dict[str, Any]:
    """The `meta` of an attached structure, in the shape the fetch tools give theirs.

    Only what the file itself says: there is no archive to name it, so no id and no title,
    and nothing about how it was solved is claimed. Raises `ValueError` for text with no
    atoms in it, or whose fixed columns do not hold numbers.
    """
    atoms = parse_pdb(text)
    if not atoms:
        raise ValueError("It has no ATOM or HETATM records.")
    return {"id": None, "source": "upload", "chains": ca_chains(atoms), "ligands": ligands(atoms)}

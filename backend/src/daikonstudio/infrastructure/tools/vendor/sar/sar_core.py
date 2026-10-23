"""
sar_core.py — chemistry engine for the KNIME SAR-table workflow.

Pure RDKit/stdlib. No KNIME imports, so it runs standalone (see selftest.py)
and is importable from a KNIME Python Script node.

Four jobs:
  1. mcs()            -> maximum common substructure as SMARTS
  2. decompose()      -> R-group decomposition against a core
  3. NameResolver     -> R-group SMILES -> word, via a 3-tier cascade
  4. DisplayPolicy    -> per R-group, word or drawing? (a name stops helping
                         once the fragment stops being a substituent and
                         becomes a molecule with substituents of its own)

Design note on the lookup table
-------------------------------
The CSV stores hand-written SMILES that only need to be VALID, not canonical.
Both table entries and incoming R-groups are pushed through the same
canon_key() before comparison, so there is no way for a hand-typed
non-canonical SMILES to silently fail to match.

Its `display` column (auto | name | picture) is the per-entry name-or-drawing
override. Keeping it in the same file you already curate means the decision
lives next to the name, and needs no extra infrastructure.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdDepictor, rdFMCS
from rdkit.Chem import rdRGroupDecomposition as rgd

RDLogger.DisableLog("rdApp.warning")


def use_coordgen(enable: bool = True) -> bool:
    """Turn on CoordGen layout. Returns whether it took effect.

    CoordGen draws scaffolds far better than the legacy depictor -- regular
    rings, substituents that don't overlap -- so it is on by default in the
    CLI and the notebook.

    It is a *function*, not an import-time side effect, deliberately. Import
    of this module must stay free of anything that can block or surprise:
    a module-level call gives you no way to skip it, no way to time it, and
    nowhere to put a breakpoint if it misbehaves. Note that CoordGen loads
    its ring-template database on the first *drawing* rather than here, so a
    cold first render can take a second or two -- see diagnose_imports.py.
    """
    try:
        rdDepictor.SetPreferCoordGen(bool(enable))
        return True
    except Exception:            # older RDKit without CoordGen
        return False

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# --------------------------------------------------------------------------
# 1. Canonical keys
# --------------------------------------------------------------------------

def canon_key(smiles: str) -> Optional[str]:
    """Normalize a fragment SMILES into a comparable key.

    Strips atom-map numbers and isotopes from dummy atoms so that
    '[*:1]CC', 'CC[*:1]' and '[1*]CC' all collapse to the same key.
    Returns None if the SMILES will not parse.
    """
    if smiles is None:
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        # Fragments with explicit valence oddities sometimes need a soft parse.
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            return None

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)

    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def n_attachments(smiles: str) -> int:
    """How many dummy atoms (attachment points) the fragment carries."""
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


def fragment_complexity(smiles: str) -> dict:
    """Size measures used to decide name-vs-picture.

    heavy_atoms is the primary metric: GetNumHeavyAtoms() counts atoms with
    atomic number > 1, so it excludes both hydrogens and the dummy attachment
    atom -- what's left is the substituent itself. On the two cases that
    prompted this: 2,4-difluorophenyl is 8, while
    1-(3-methylpiperazino)-2,4-difluoro-5-tert-butylphenyl is 19.

    n_ring_systems is a useful secondary signal, since a fragment carrying
    more than one ring system has usually stopped being a substituent and
    become a molecule with substituents of its own.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
        if mol is None:
            return {"heavy_atoms": 0, "n_rings": 0, "n_ring_systems": 0}
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)

    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()

    # Molecular weight breaks ties that heavy-atom count cannot: F, Cl, Br and
    # I are all one heavy atom, but a chemist expects them in that order.
    try:
        mw = Descriptors.MolWt(mol)
    except Exception:
        mw = float(sum(a.GetMass() for a in mol.GetAtoms()))

    # Count ring systems (fused rings share atoms and count once).
    systems: List[set] = []
    for ring in ri.AtomRings():
        s = set(ring)
        merged = [g for g in systems if g & s]
        for g in merged:
            systems.remove(g)
            s |= g
        systems.append(s)

    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_rings": n_rings,
        "n_ring_systems": len(systems),
        "mw": round(mw, 3),
    }


# Order a chemist reads down an R-group column in. Grouping by what the
# substituent bonds through, then by size within the group, reproduces the
# convention: H, Me, Et, Pr ... then rings, then halogens F<Cl<Br<I, then O,
# N, carbonyl, S, and aryl/heteroaryl last.
CLASS_ORDER = ("hydrogen", "alkyl", "cycloalkyl", "unsaturated", "halogen",
               "oxygen", "nitrogen", "carbonyl", "sulfur", "aryl",
               "heteroaryl", "heterocycloalkyl", "misc", "legacy", "unknown")
_CLASS_RANK = {c: i for i, c in enumerate(CLASS_ORDER)}


def sort_key(info: dict) -> str:
    """A sortable string putting fragments in the order a chemist expects.

    Returned as a zero-padded string rather than a tuple so it can travel in
    an HTML `data-sort` attribute and still sort correctly lexicographically
    in the browser -- no JavaScript comparator needed.

    Hydrogen is forced to the front regardless of its `class`, because "H
    first" is the one part of the convention nobody varies on.
    """
    name = (info.get("name") or "").lower()
    cls = (info.get("class") or "").lower()
    heavy = int(info.get("heavy_atoms") or 0)
    mw = float(info.get("mw") or 0.0)

    if heavy <= 0 and info.get("source") == "empty":
        return "99|9999|99999.99|"                       # blanks last
    if name in ("hydrogen", "h") or heavy == 0:
        rank = -1
    else:
        rank = _CLASS_RANK.get(cls, len(CLASS_ORDER))
    return f"{rank + 1:02d}|{heavy:04d}|{mw:08.2f}|{name}"


def cap_with_hydrogen(smiles: str) -> Optional[str]:
    """Replace every attachment point with H, giving a closed-shell molecule.

    '[*:1]CC' -> 'CC' (ethane). Used for the PubChem tier, which can only
    name real molecules, not radicals.
    """
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummies, reverse=True):
        rw.RemoveAtom(idx)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return None


# --------------------------------------------------------------------------
# 2. MCS
# --------------------------------------------------------------------------

def mcs(
    smiles_list: Sequence[str],
    threshold: float = 1.0,
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
    match_valences: bool = False,
    timeout: int = 60,
) -> dict:
    """Maximum common substructure of a set of SMILES.

    Defaults are deliberately not RDKit's defaults: ring_matches_ring_only and
    complete_rings_only are ON, because an MCS containing half a ring is
    useless as an R-group-decomposition core -- it produces R-groups that are
    ring fragments rather than recognizable substituents.

    Returns a dict with keys: smarts, num_atoms, num_bonds, timed_out,
    invalid_rows, error. `smarts` is None when there is no answer, and `error`
    then carries a readable reason.

    Degenerate inputs are handled here rather than left to RDKit:
      0 molecules  -> undefined; smarts is None
      1 molecule   -> that molecule (rdFMCS.FindMCS raises below 2 inputs)
    Both are reachable in practice, because unparseable rows are dropped
    first -- a 30-compound file with 29 bad SMILES lands on the 1-molecule
    case, and should say so rather than raise.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    bad = [i for i, m in enumerate(mols) if m is None]
    mols = [m for m in mols if m is not None]

    base = {"smarts": None, "num_atoms": 0, "num_bonds": 0,
            "timed_out": False, "invalid_rows": bad, "error": None}

    if not mols:
        return {**base,
                "error": "no parseable structures" if bad else "no structures"}

    if len(mols) == 1:
        m = mols[0]
        try:
            smarts = Chem.MolToSmarts(m)
        except Exception as e:            # pragma: no cover - defensive
            return {**base, "error": f"could not express the molecule as SMARTS: {e}"}
        return {**base, "smarts": smarts or None,
                "num_atoms": m.GetNumAtoms(), "num_bonds": m.GetNumBonds()}

    try:
        res = rdFMCS.FindMCS(
            mols,
            threshold=threshold,
            ringMatchesRingOnly=ring_matches_ring_only,
            completeRingsOnly=complete_rings_only,
            matchValences=match_valences,
            timeout=timeout,
        )
    except RuntimeError as e:
        # FindMCS reports invalid arguments this way. Returning it beats
        # crashing halfway through an interactive session.
        return {**base, "error": f"RDKit FindMCS: {e}"}

    return {
        "smarts": res.smartsString or None,
        "num_atoms": res.numAtoms,
        "num_bonds": res.numBonds,
        "timed_out": bool(res.canceled),
        "invalid_rows": bad,
        "error": None if res.smartsString else "no common substructure found",
    }


def mcs_coverage(smiles_list: Sequence[str], smarts: str) -> float:
    """Fraction of input molecules the candidate core actually matches.

    Worth showing the chemist next to the picture: a core that only hits 60%
    of the set will silently drop 40% of rows at decomposition time.
    """
    core = Chem.MolFromSmarts(smarts)
    if core is None:
        return 0.0
    _prepare_query_mol(core)
    hits = 0
    total = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        total += 1
        if m.HasSubstructMatch(core):
            hits += 1
    return hits / total if total else 0.0


_EMPTY_SVG = "<svg xmlns='http://www.w3.org/2000/svg'/>"


def _prepare_query_mol(mol: Chem.Mol) -> Chem.Mol:
    """Make a MolFromSmarts query mol drawable.

    MolFromSmarts does no sanitization, so the mol has no ring info and no
    valence cache. Drawing it directly raises "Can't kekulize mol", because
    the drawing prep step tries to kekulize aromatic query bonds -- and an
    RDKit MCS SMARTS is full of them ('[#6]1:[#6]:[#6]...'). Populating the
    property cache and ring info, then drawing with kekulize=False, is what
    makes the core renderable.
    """
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def render_core_svg(smarts: str, width: int = 420, height: int = 300) -> str:
    """SVG of the core itself, drawn from its SMARTS."""
    from rdkit.Chem.Draw import rdMolDraw2D

    core = Chem.MolFromSmarts(smarts)
    if core is None:
        return _EMPTY_SVG
    try:
        _prepare_query_mol(core)
        rdDepictor.Compute2DCoords(core)
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(d, core, kekulize=False)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return _EMPTY_SVG


def render_highlight_svg(
    smiles: str, smarts: str, width: int = 320, height: int = 240
) -> str:
    """SVG of a molecule with the core match highlighted.

    More informative for the approval step than the bare core: the chemist can
    see where the core lands and what is left hanging off it.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _EMPTY_SVG

    core = Chem.MolFromSmarts(smarts)
    if core is not None:
        _prepare_query_mol(core)

    match = mol.GetSubstructMatch(core) if core is not None else ()
    bonds = []
    if match:
        mset = set(match)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in mset and b.GetEndAtomIdx() in mset:
                bonds.append(b.GetIdx())

    try:
        rdDepictor.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol, highlightAtoms=list(match), highlightBonds=bonds
        )
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return _EMPTY_SVG


def _label_dummies(mol: Chem.Mol, fallback: str = "R") -> List[int]:
    """Relabel dummy atoms as R1/R2/... and return their indices.

    RDKit draws a dummy atom as a bare '*'. Relabelling it makes the drawing
    read as a substituent on a scaffold rather than a free radical, and the
    map number is the R-group number, so the label matches the table column.
    """
    idxs = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            n = a.GetAtomMapNum()
            a.SetProp("atomLabel", f"R{n}" if n else (fallback or "R"))
            a.SetAtomMapNum(0)
            idxs.append(a.GetIdx())
        else:
            a.SetAtomMapNum(0)
    return idxs


def render_fragment_svg(
    smiles: str,
    width: int = 200,
    height: int = 150,
    attachment_label: str = "R",
) -> str:
    """SVG of a single R-group fragment, attachment point labelled."""
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return _EMPTY_SVG

    try:
        _label_dummies(mol, attachment_label)
        rdDepictor.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return _EMPTY_SVG


# --------------------------------------------------------------------------
# the Markush core -- the scaffold drawn once, with R1/R2 where they attach
# --------------------------------------------------------------------------

def markush_core(rows: Sequence[dict], query_mol=None,
                 variable_labels: Optional[dict] = None) -> dict:
    """The labelled scaffold, taken from the decomposition itself.

    RDKit's R-group decomposition returns a `Core` per row carrying dummy
    atoms whose atom-map numbers are the R-group numbers -- so the core
    already knows that R1 belongs on *that* atom and R2 on *that* one. No
    re-matching or guessing needed; the answer came back with the data.

    Variable atoms are a different story and must be supplied. The Core that
    RGD hands back is a *concrete* scaffold -- whichever molecule matched
    first, so indole for a benzazole set -- and it carries no record that a
    position was allowed to vary. Pass the candidate's `query_mol` and its
    `variable_labels` (from sar_variable.find_core_candidates) and the query
    is matched onto the core to transfer X labels onto core atom indices.
    Without them the drawing is chemically fine but silently misleading: it
    shows one member of the series as though it were the whole scaffold.

    Returns {"smiles", "labels", "n_variants", "variable_atoms", "error"}.
    `variable_atoms` maps core atom index -> "X1"/"X2"/..., ready for
    render_markush_svg.
    """
    out = {"smiles": None, "labels": [], "n_variants": 0,
           "variable_atoms": {}, "error": None}
    if not rows:
        out["error"] = "no decomposed rows to take a core from"
        return out
    cores = [str(r.get("Core") or "").strip() for r in rows]
    cores = [c for c in cores if c]
    if not cores:
        out["error"] = ("the decomposed rows carry no Core column -- "
                        "markush_core() needs the output of decompose()")
        return out

    counts = Counter(cores)
    out["n_variants"] = len(counts)
    smiles = counts.most_common(1)[0][0]
    out["smiles"] = smiles

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        out["error"] = f"could not parse the core: {smiles}"
        return out
    labels = sorted(a.GetAtomMapNum() for a in mol.GetAtoms()
                    if a.GetAtomicNum() == 0 and a.GetAtomMapNum() > 0)
    out["labels"] = labels
    if not labels:
        out["error"] = ("the core has no numbered attachment points, so R "
                        "positions cannot be placed on it")

    # Transfer variable-atom labels from the candidate query onto this core.
    if query_mol is not None and variable_labels:
        try:
            probe = Chem.MolFromSmiles(smiles)
            if probe is None:
                probe = Chem.MolFromSmarts(smiles)
                if probe is not None:
                    _prepare_query_mol(probe)
            match = probe.GetSubstructMatch(query_mol) if probe is not None else ()
            if match:
                out["variable_atoms"] = {
                    match[q]: lbl for q, lbl in variable_labels.items()
                    if q < len(match)}
            else:
                out["error"] = (out["error"] or "") + (
                    " the variable-atom query did not match the decomposition's "
                    "core, so X positions could not be placed on the drawing")
        except Exception as e:                       # pragma: no cover
            out["error"] = (out["error"] or "") + f" X mapping failed: {e}"
    return out


def render_markush_svg(core_smiles: str, width: int = 460, height: int = 320,
                       highlight: bool = True,
                       variable_atoms: Optional[dict] = None) -> str:
    """Draw the scaffold with R1/R2 at the attachment points.

    `variable_atoms` maps core atom index -> "X1"/"X2"; those atoms are
    relabelled and highlighted alongside the R positions, so a benzazole
    series reads as a scaffold with an X rather than as an indole.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    if not core_smiles:
        return _EMPTY_SVG

    kekulize = True
    mol = Chem.MolFromSmiles(str(core_smiles))
    if mol is None:
        mol = Chem.MolFromSmarts(str(core_smiles))
        if mol is None:
            return _EMPTY_SVG
        _prepare_query_mol(mol)
        kekulize = False

    try:
        dummies = _label_dummies(mol)
        for idx, label in (variable_atoms or {}).items():
            if 0 <= int(idx) < mol.GetNumAtoms():
                a = mol.GetAtomWithIdx(int(idx))
                a.SetProp("atomLabel", str(label))
                # An X position has no fixed element, so its implicit-H count
                # is meaningless and RDKit would otherwise draw e.g. "X1H".
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                dummies.append(int(idx))
        rdDepictor.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = d.drawOptions()
        opts.addStereoAnnotation = False
        try:
            opts.padding = 0.08
        except Exception:
            pass
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol,
            highlightAtoms=dummies if highlight else [],
            kekulize=kekulize)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return _EMPTY_SVG


# --------------------------------------------------------------------------
# 3. R-group decomposition
# --------------------------------------------------------------------------

def decompose(
    smiles_list: Sequence[str],
    core_smarts: str,
    ids: Optional[Sequence[str]] = None,
    only_match_at_rgroups: bool = False,
    remove_hydrogen_only_rgroups: bool = False,
    exhaustive: bool = False,
    separate_geminal: bool = True,
) -> Tuple[List[dict], List[str]]:
    """R-group decomposition against a single core.

    only_match_at_rgroups defaults to False, which is the setting an
    MCS-derived core needs: the MCS SMARTS carries no R-labels, so RDKit has
    to be allowed to attach R-groups wherever a substituent hangs off the
    core. Set it True only when you hand-draw a labelled scaffold.

    remove_hydrogen_only_rgroups defaults to False so that hydrogen shows up
    explicitly as an R-group -- in a SAR table an H position is real data, not
    an absence.

    separate_geminal defaults to True: a core atom carrying two substituents
    gets an R column for each. RDKit's default packs both into one, so a
    gem-diethyl carbon reads "CCC[*:1].CC[*:1]" in a single cell -- which is
    not a substituent, it is two, and nothing downstream can treat it as
    either. It has no name, it sorts under whichever fragment came first, and
    a matched pair differing at only one of the two positions is invisible to
    the cliff analysis because the cell as a whole differs. One column per
    attachment point is what a Markush table means. Set it False to get
    RDKit's combined cells back.

    Returns (rows, unmatched_ids).
    """
    core = Chem.MolFromSmarts(core_smarts)
    if core is None:
        raise ValueError(f"core SMARTS did not parse: {core_smarts!r}")
    _prepare_query_mol(core)

    if ids is None:
        ids = [str(i + 1) for i in range(len(smiles_list))]

    mols, kept_ids, kept_smiles = [], [], []
    for _id, smi in zip(ids, smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            kept_ids.append(_id)
            kept_smiles.append(smi)

    params = rgd.RGroupDecompositionParameters()
    params.onlyMatchAtRGroups = only_match_at_rgroups
    params.removeAllHydrogenRGroups = remove_hydrogen_only_rgroups
    params.removeHydrogensPostMatch = True
    params.alignment = rgd.RGroupCoreAlignment.NoAlignment
    # Older RDKit builds have no such option; a combined cell is then the only
    # answer available, and that is better than failing to decompose at all.
    if hasattr(params, "allowMultipleRGroupsOnUnlabelled"):
        params.allowMultipleRGroupsOnUnlabelled = separate_geminal
    params.matchingStrategy = (
        rgd.RGroupMatching.Exhaustive if exhaustive else rgd.RGroupMatching.GreedyChunks
    )

    groups, unmatched_idx = rgd.RGroupDecompose(
        [core], mols, asSmiles=True, asRows=True, options=params
    )

    unmatched_set = set(unmatched_idx)
    matched_ids = [i for k, i in enumerate(kept_ids) if k not in unmatched_set]
    matched_smiles = [s for k, s in enumerate(kept_smiles) if k not in unmatched_set]

    rows = []
    for _id, smi, grp in zip(matched_ids, matched_smiles, groups):
        row = {"ID": _id, "SMILES": smi}
        row.update(grp)
        rows.append(row)

    unmatched_ids = [kept_ids[k] for k in sorted(unmatched_set)]
    return rows, unmatched_ids


def rgroup_columns(rows: List[dict]) -> List[str]:
    """R-group column names present in the decomposition, in R1..Rn order.

    Deliberately R-only, unlike `sar_report.rgroup_cols`, which also picks up
    the X columns for variable ring atoms. The two are not an oversight:

      * R cells hold substituent SMILES and must go through the naming
        cascade and the display policy.
      * X cells hold a bare element symbol, already resolved by
        `sar_variable.attach_variable_columns`. Sending them through the
        cascade would be actively wrong -- `canon_key("N")` reads "N" as
        ammonia, and the cell would come back named "azane".

    So `name_rows` uses this (R only) for naming, and anything laying out a
    table uses `sar_report.rgroup_cols` (R and X). A consequence worth
    knowing: the report `name_rows` returns is keyed on R columns only, so
    callers iterating display columns must use `report.get(c)`, not
    `report[c]`.
    """
    seen = set()
    for r in rows:
        seen.update(k for k in r if re.fullmatch(r"R\d+", k))
    return sorted(seen, key=lambda c: int(c[1:]))


# --------------------------------------------------------------------------
# 4. Naming cascade
# --------------------------------------------------------------------------

# Only applied when the attachment atom is carbon, where the transform is safe.
_SUFFIX_RULES = [
    (re.compile(r"^benzene$"), "phenyl"),
    (re.compile(r"^naphthalene$"), "naphthalenyl"),
    (re.compile(r"^(.*)ane$"), r"\1yl"),
    (re.compile(r"^(.*)ene$"), r"\1enyl"),
    (re.compile(r"^(.*)yne$"), r"\1ynyl"),
]


class NameResolver:
    """Three-tier R-group -> word resolver.

    Tier 1  curated lookup table          source = "lookup"
    Tier 2  PubChem name of the H-capped fragment, suffix-adjusted when safe
                                          source = "pubchem" | "pubchem_raw"
    Tier 3  canonical SMILES, unchanged   source = "smiles"

    Tier 2 is a convenience, not an authority. PubChem names a MOLECULE, and a
    substituent is not a molecule -- capping '[*:1]OC' with H gives methanol,
    whose correct substituent name is "methoxy", which no suffix rule can
    derive. So tier-2 names are only suffix-adjusted for carbon attachment
    points; everything else is returned with a "~" marker and source
    "pubchem_raw" so it can never be mistaken for a verified substituent name.
    Anything you care about should end up in the table.
    """

    def __init__(
        self,
        table_path: str,
        use_pubchem: bool = False,
        cache_path: Optional[str] = None,
        prefer_alias: bool = False,
        request_delay: float = 0.25,
    ):
        self.prefer_alias = prefer_alias
        self.use_pubchem = use_pubchem
        self.request_delay = request_delay
        self.cache_path = cache_path
        self._cache: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as fh:
                    self._cache = json.load(fh)
            except Exception:
                self._cache = {}

        self.table: Dict[str, dict] = {}
        self.table_errors: List[Tuple[str, str]] = []
        self._load_table(table_path)
        self.misses: Counter = Counter()

    def _load_table(self, path: str) -> None:
        """Load the lookup table.

        If the CSV carries a `canonical_smiles` column (written by
        import_legacy_table.py), it is trusted and used directly. That matters
        at scale: canonicalizing 28k+ SMILES through RDKit on every node
        execution costs seconds, and the answer never changes.
        """
        with open(path, newline="", encoding="utf-8") as fh:
            for lineno, rec in enumerate(csv.DictReader(fh), start=2):
                smi = (rec.get("smiles") or "").strip()
                name = (rec.get("name") or "").strip()
                if not smi or not name:
                    continue
                key = (rec.get("canonical_smiles") or "").strip() or canon_key(smi)
                if key is None:
                    self.table_errors.append((f"line {lineno}", f"unparseable SMILES: {smi}"))
                    continue
                if key in self.table and self.table[key]["name"] != name:
                    self.table_errors.append(
                        (f"line {lineno}",
                         f"duplicate key {key}: {self.table[key]['name']} vs {name}")
                    )
                    continue

                display = (rec.get("display") or "auto").strip().lower() or "auto"
                if display not in ("auto", "name", "picture"):
                    self.table_errors.append(
                        (f"line {lineno}",
                         f"display must be auto|name|picture, got {display!r}"
                         " -- treating as auto")
                    )
                    display = "auto"

                self.table[key] = {
                    "name": name,
                    "alias": (rec.get("alias") or "").strip(),
                    "symbol": (rec.get("symbol") or "").strip(),
                    "class": (rec.get("class") or "").strip(),
                    "display": display,
                }

    # -- tier 2 ------------------------------------------------------------

    def _pubchem_name(self, capped_smiles: str) -> Optional[str]:
        if capped_smiles in self._cache:
            return self._cache[capped_smiles] or None
        url = (
            f"{PUBCHEM_BASE}/compound/smiles/property/IUPACName/TXT"
            f"?smiles={urllib.parse.quote(capped_smiles, safe='')}"
        )
        name = ""
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                name = resp.read().decode("utf-8", "replace").strip().splitlines()[0].strip()
        except Exception:
            name = ""
        if name.lower().startswith("status:") or "not found" in name.lower():
            name = ""
        self._cache[capped_smiles] = name
        time.sleep(self.request_delay)
        return name or None

    def _attachment_is_carbon(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
        if mol is None:
            return False
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                nbrs = list(a.GetNeighbors())
                return bool(nbrs) and all(n.GetAtomicNum() == 6 for n in nbrs)
        return False

    # -- public ------------------------------------------------------------

    def resolve_full(self, rgroup_smiles: str) -> dict:
        """Everything known about one R-group.

        Keys: name, symbol, alias, source, display_pref, class,
        canonical_smiles, name_chars, heavy_atoms, n_rings, n_ring_systems,
        mw, sort.

        `name` is always the full name -- the long one. `symbol` is the short
        form (Me, CF3, Ph) when the table has one, and "" otherwise. Which of
        them a cell actually shows is name_rows' decision, not this one's; the
        caller always has both.
        """
        name, source = self.resolve(rgroup_smiles)
        key = canon_key(rgroup_smiles)
        hit = self.table.get(key) if key else None
        cx = fragment_complexity(rgroup_smiles) if rgroup_smiles else {
            "heavy_atoms": 0, "n_rings": 0, "n_ring_systems": 0, "mw": 0.0
        }
        info = {
            "name": name,
            "symbol": (hit or {}).get("symbol", ""),
            "alias": (hit or {}).get("alias", ""),
            "source": source,
            "display_pref": hit["display"] if hit else "auto",
            "class": (hit or {}).get("class", ""),
            "canonical_smiles": key or str(rgroup_smiles),
            "name_chars": len(name),
            **cx,
        }
        info["sort"] = sort_key(info)
        return info

    def resolve(self, rgroup_smiles: str) -> Tuple[str, str]:
        """Return (name, source) for one R-group SMILES."""
        if rgroup_smiles is None or str(rgroup_smiles).strip() == "":
            return "", "empty"

        key = canon_key(rgroup_smiles)
        if key is None:
            return str(rgroup_smiles), "unparseable"

        hit = self.table.get(key)
        if hit:
            if self.prefer_alias and hit["alias"]:
                return hit["alias"], "lookup"
            return hit["name"], "lookup"

        self.misses[key] += 1

        # Multi-attachment fragments (a ring bridging two core positions) have
        # no single substituent name; do not guess.
        if n_attachments(rgroup_smiles) != 1:
            return key, "smiles"

        if self.use_pubchem:
            capped = cap_with_hydrogen(rgroup_smiles)
            if capped:
                parent = self._pubchem_name(capped)
                if parent:
                    if self._attachment_is_carbon(rgroup_smiles):
                        for pat, repl in _SUFFIX_RULES:
                            if pat.match(parent):
                                return pat.sub(repl, parent), "pubchem"
                    return f"~{parent}", "pubchem_raw"

        return key, "smiles"

    def save_cache(self) -> None:
        if self.cache_path:
            try:
                with open(self.cache_path, "w") as fh:
                    json.dump(self._cache, fh, indent=1, sort_keys=True)
            except Exception:
                pass

    def miss_report(self) -> List[dict]:
        """Unnamed fragments, most frequent first -- your table to-do list."""
        return [
            {"canonical_smiles": k, "count": n, "n_attachments": n_attachments(k)}
            for k, n in self.misses.most_common()
        ]


# --------------------------------------------------------------------------
# 5. Name or picture?
# --------------------------------------------------------------------------

class DisplayPolicy:
    """Decides, per R-group, whether a cell shows a word or a drawing.

    A name stops being useful somewhere around the point where the fragment
    stops being *a substituent* and becomes *a molecule that happens to have
    substituents of its own*. "2,4-difluorophenyl" reads instantly;
    "1-(3-methylpiperazino)-2,4-difluoro-5-tert-butylphenyl" does not, and
    wants to be a picture.

    Heavy-atom count is the primary metric because it tracks that transition
    better than name length does -- "3,5-bis(trifluoromethyl)phenyl" is 30
    characters and still instantly readable. Name length is kept as a
    secondary trigger for short fragments that happen to get baroque names.

    Modes
    -----
    "column"  (default) every cell in an R-group column gets the same
              treatment; a column flips to pictures once more than
              column_picture_fraction of its cells want one.
    "cell"    each cell decides for itself.
    "name"    always words.
    "picture" always drawings.

    Why "column" is the default, and not merely a preference: **a KNIME
    column has exactly one type.** A single column physically cannot hold
    strings in some rows and SVG images in others. Cell mode therefore has to
    emit two columns per R-group (`R1` and `R1_svg`) with the unused cell in
    each row blanked, which leaves you two half-empty columns side by side.
    Column mode emits one full column. It also keeps the table scannable --
    reading down an R-group column is most of what an SAR table is for, and
    mixed text/images breaks that (row heights jump, the eye loses the line).

    Per-entry overrides from the lookup table's `display` column always win,
    in either mode. That is the point of an override -- a group you have
    decided is recognizable on sight stays a word even in a picture column.
    """

    def __init__(
        self,
        mode: str = "column",
        max_heavy_atoms: int = 12,
        max_name_chars: int = 30,
        max_ring_systems: int = 1,
        column_picture_fraction: float = 0.3,
        draw_unnamed: bool = True,
    ):
        if mode not in ("column", "cell", "name", "picture"):
            raise ValueError(f"mode must be column|cell|name|picture, got {mode!r}")
        self.mode = mode
        self.max_heavy_atoms = max_heavy_atoms
        self.max_name_chars = max_name_chars
        self.max_ring_systems = max_ring_systems
        self.column_picture_fraction = column_picture_fraction
        self.draw_unnamed = draw_unnamed

    UNNAMED_SOURCES = ("smiles", "pubchem_raw", "unparseable")

    def cell_wants_picture(self, info: dict) -> bool:
        """Would this one fragment be better as a drawing?

        The threshold predicate. `cell_decision` is what callers should use --
        this does not know about column mode or explicit overrides.
        """
        pref = info.get("display_pref", "auto")
        if pref == "name":
            return False
        if pref == "picture":
            return True

        # A fragment that fell through the naming cascade has no name to show
        # -- a drawing beats a SMILES string in a table cell every time.
        if self.draw_unnamed and info.get("source") in self.UNNAMED_SOURCES:
            return True

        if info.get("source") == "empty":
            return False

        return (
            info.get("heavy_atoms", 0) > self.max_heavy_atoms
            or info.get("name_chars", 0) > self.max_name_chars
            or info.get("n_ring_systems", 0) > self.max_ring_systems
        )

    def cell_is_column_driven(self, info: dict) -> bool:
        """False when this cell's treatment is settled regardless of its column.

        Two kinds of cell are exempt from the column vote: those with an
        explicit `display` in the lookup table, and those with no name at all.
        Both would otherwise skew the fraction that decides the column.
        """
        if info.get("display_pref", "auto") in ("name", "picture"):
            return False
        if self.draw_unnamed and info.get("source") in self.UNNAMED_SOURCES:
            return False
        return True

    def cell_decision(self, info: dict, column_mode: Optional[str] = None) -> str:
        """The final answer for one cell: "name" or "picture".

        Precedence, most specific first:
          1. an explicit `display` value in the lookup table
          2. an explicit global mode ("name"/"picture") -- you asked for it
          3. no name to show  ->  draw it, whatever the column decided
          4. "cell" mode      ->  this fragment's own size
          5. "column" mode    ->  whatever the column decided

        Rule 3 is the one that matters in practice. Without it a column of
        mostly-nameable fragments stays in "name" mode and the few unnamed
        cells print raw SMILES -- which is the exact outcome drawing them was
        meant to prevent.
        """
        pref = info.get("display_pref", "auto")
        if pref in ("name", "picture"):
            return pref
        if self.mode in ("name", "picture"):
            return self.mode
        if self.draw_unnamed and info.get("source") in self.UNNAMED_SOURCES:
            return "picture"
        if self.mode == "cell":
            return "picture" if self.cell_wants_picture(info) else "name"
        return column_mode or "name"

    def column_mode(self, infos: Sequence[dict]) -> str:
        """'name' or 'picture' for a whole R-group column."""
        if self.mode == "name":
            return "name"
        if self.mode == "picture":
            return "picture"
        voting = [i for i in infos if self.cell_is_column_driven(i)]
        if not voting:
            return "name"
        frac = sum(1 for i in voting if self.cell_wants_picture(i)) / len(voting)
        return "picture" if frac >= self.column_picture_fraction else "name"


# --------------------------------------------------------------------------
# 6. Table assembly
# --------------------------------------------------------------------------

def name_rows(
    rows: List[dict],
    resolver: NameResolver,
    keep_smiles_columns: bool = True,
    policy: Optional[DisplayPolicy] = None,
    svg_width: int = 200,
    svg_height: int = 150,
    symbols: Optional[bool] = None,
    symbols_from_cols: int = 4,
    render_all_svgs: bool = True,
    max_svg_cells: int = 4000,
) -> Tuple[List[dict], dict]:
    """Turn a decomposition table into a SAR table.

    Non-R-group columns pass through untouched, so Activity -- or whatever
    else you carried in from the input file -- survives into the SAR table.

    Returns (rows, report) where report describes what each R-group column
    was rendered as and why, so the decision is inspectable rather than
    mysterious:

        {"R1": {"mode": "name",    "picture_fraction": 0.0,  "max_heavy_atoms": 8},
         "R2": {"mode": "picture", "picture_fraction": 0.71, "max_heavy_atoms": 19}}

    `symbols` switches cells to short forms (Me, CF3, Ph). None means decide
    automatically: on once there are `symbols_from_cols` or more R-group
    columns, since that is the point where full words stop fitting. A fragment
    with no curated symbol keeps its full name either way.

    `render_all_svgs` renders a drawing for every cell, not just the drawn
    ones, so a cell showing a word still has a picture to offer on hover. It
    switches itself off above `max_svg_cells` cells to keep the HTML sane, and
    the report says so.

    Per R-group column the output carries:
        R1           the label actually shown -- symbol or name
        R1_name      the full name, always, however the cell is displayed
        R1_symbol    the short form, or "" if there isn't one
        R1_svg       the drawing
        R1_display   "name" or "picture" -- which one the cell shows
        R1_source    naming provenance
        R1_sort      chemist-order sort key
        R1_heavy     heavy-atom count (an alternative sort)
        R1_smiles    the raw fragment SMILES (if keep_smiles_columns)
    """
    if policy is None:
        policy = DisplayPolicy()

    cols = rgroup_columns(rows)
    rgset = set(cols)

    if symbols is None:
        symbols = len(cols) >= symbols_from_cols

    n_cells = len(rows) * max(1, len(cols))
    svg_everywhere = render_all_svgs and n_cells <= max_svg_cells

    # One drawing per distinct fragment, not per cell. A 26-row table has
    # about ten distinct R-groups, so this is roughly a 3x saving on both
    # render time and output size.
    _svg_cache: Dict[str, str] = {}

    def draw(smi: str) -> str:
        if not smi:
            return ""
        if smi not in _svg_cache:
            _svg_cache[smi] = render_fragment_svg(smi, svg_width, svg_height)
        return _svg_cache[smi]

    # Resolve everything once, so column decisions can see the whole column.
    infos: Dict[str, List[dict]] = {}
    for c in cols:
        infos[c] = [resolver.resolve_full(r.get(c, "")) for r in rows]

    col_modes: Dict[str, str] = {}
    report: Dict[str, dict] = {}
    for c in cols:
        col_infos = infos[c]
        voting = [i for i in col_infos if policy.cell_is_column_driven(i)]
        frac = (
            sum(1 for i in voting if policy.cell_wants_picture(i)) / len(voting)
            if voting else 0.0
        )
        col_modes[c] = policy.column_mode(col_infos)
        report[c] = {
            "mode": col_modes[c] if policy.mode != "cell" else "cell",
            "picture_fraction": round(frac, 3),
            "max_heavy_atoms": max((i.get("heavy_atoms", 0) for i in col_infos),
                                   default=0),
            "max_name_chars": max((i.get("name_chars", 0) for i in col_infos),
                                  default=0),
            "n_overrides": sum(
                1 for i in col_infos if i.get("display_pref", "auto") != "auto"
            ),
            "n_unnamed": sum(
                1 for i in col_infos
                if i.get("source") in policy.UNNAMED_SOURCES
            ),
            "n_symbols": sum(1 for i in col_infos if i.get("symbol")),
        }

    out = []
    for idx, r in enumerate(rows):
        new = {k: v for k, v in r.items() if k not in rgset}
        for c in cols:
            info = infos[c][idx]
            val = r.get(c, "")

            show = policy.cell_decision(info, column_mode=col_modes[c])
            label = (info["symbol"] if (symbols and info["symbol"])
                     else info["name"])

            if keep_smiles_columns:
                new[f"{c}_smiles"] = val
            new[c] = label
            new[f"{c}_name"] = info["name"]
            new[f"{c}_symbol"] = info["symbol"]
            new[f"{c}_display"] = show
            new[f"{c}_source"] = info["source"]
            new[f"{c}_sort"] = info["sort"]
            new[f"{c}_heavy"] = info.get("heavy_atoms", 0)
            # Canonical identity of the substituent. Matched-pair analysis
            # needs "is this the same group at this position?", and the raw
            # SMILES is not reliable for that -- two spellings of one
            # fragment would read as different substituents.
            new[f"{c}_key"] = info.get("canonical_smiles", "")
            new[f"{c}_svg"] = draw(val) if (
                svg_everywhere or show == "picture") else ""
        out.append(new)

    for c in cols:
        report[c]["symbols_used"] = bool(symbols)
        report[c]["svgs_everywhere"] = svg_everywhere
    if render_all_svgs and not svg_everywhere:
        for c in cols:
            report[c]["note"] = (
                f"hover drawings disabled: {n_cells} cells exceeds "
                f"max_svg_cells={max_svg_cells}")

    return out, report

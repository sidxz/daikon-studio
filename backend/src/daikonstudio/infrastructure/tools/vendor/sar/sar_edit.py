"""
sar_edit.py — propose a core by hand, and have the tool score it.

The MCS search proposes; this module lets a chemist dispose. Give it a core --
typed as SMARTS, drawn in the editor (`sar_draw`), or derived by editing one
the search found -- and it reports what that core would actually do to the set:
coverage, how many positions vary and which elements they hold, how many R
columns fall out, which compounds miss and *why* they miss.

Why this exists
---------------
An MCS returns the largest common substructure. A chemist frequently wants a
different one: smaller, or differently aligned, or with a position pinned for
reasons that live in the chemistry and not in the graph. Trying to make the
search guess that intent is a losing game, and every round of guessing costs a
day. A scored escape hatch ends the game -- state the core, see the numbers.

It doubles as the diagnostic. When the search misses a core the chemist knows
is there, evaluating that core by hand says whether it was reachable at all,
and `compare_to` puts it side by side with what the search did return. That
turns "the MCS won't find my core" into a number.

What comes back
---------------
`evaluate` returns a report whose "candidate" key is shaped exactly like an
entry from `sar_variable.find_core_candidates`. That is deliberate: a
hand-drawn core must flow through decomposition, naming, the Markush drawing
and the cliff analysis with no special-casing anywhere downstream. Every bug
in this project so far has been at a seam, not inside a module, and a
hand-built core that is "nearly" the right shape would be a new seam.

Layering: RDKit only, like sar_core and sar_variable. No I/O, no HTML, no
widget imports -- `sar_draw` depends on this module and never the reverse, so
the editor being missing or broken can never stop a core being scored.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

from rdkit import Chem, RDLogger, rdBase

import sar_variable as sv

RDLogger.DisableLog("rdApp.warning")


def _quietly(parse, text):
    """Run a parser whose failure is expected, without RDKit shouting.

    `parse_core` probes SMILES before SMARTS, so for every genuine SMARTS core
    the SMILES attempt fails *by design*. RDKit reports that at error level --
    four lines and a caret pointing into the string -- so a chemist who pasted
    a perfectly good SMARTS saw "SMILES Parse Error" and reasonably concluded
    their core had been rejected, when it was about to parse as SMARTS on the
    next line and be used.

    Only this one speculative call is silenced; a parse whose failure actually
    means something still reports it.
    """
    with rdBase.BlockLogs():
        return parse(text)


# --------------------------------------------------------------------------
# parsing  --  accept whatever the chemist has to hand
# --------------------------------------------------------------------------

def parse_core(core) -> tuple:
    """Turn a core given as SMARTS, SMILES or a mol into a query mol.

    Returns (query_mol, note, kind) where kind is "smarts", "smiles" or
    "mol". SMILES is tried first, because that is what a structure editor
    hands back by default and what a chemist typing a scaffold types, and
    because most SMILES parse as SMARTS as well without meaning the same
    thing. A string only SMARTS can read stays SMARTS.

    A SMILES-derived core is a stricter thing than it looks -- every atom
    fixed to its element -- so the note says so rather than letting a chemist
    wonder why their drawn benzimidazole missed the benzothiazoles.
    """
    if core is None:
        return None, "no core given", None
    if not isinstance(core, str):
        return core, "", "mol"

    text = core.strip()
    if not text:
        return None, "no core given", None

    # SMILES is tried first, because most strings that parse as SMILES parse
    # as SMARTS too -- "c1ccc2[nH]ccc2c1" is both -- and the two readings are
    # not the same query. SMARTS takes the characters literally: no aromaticity
    # perception, no implicit-H count, no ring-membership beyond what is
    # written. A scaffold is SMILES, whether a chemist typed it or a structure
    # editor handed it back, so that is the reading to take when the string
    # admits both. Anything genuinely SMARTS-only -- [#6,#7,#16], :&@, !R --
    # fails the SMILES parse and falls through untouched.
    m = _quietly(Chem.MolFromSmiles, text)
    if m is not None and m.GetNumAtoms():
        # Hand back the query re-read from its own SMARTS, so the string this
        # core prints as and the mol that gets analysed number their atoms the
        # same way; see sar_variable.canonical_query.
        q = Chem.MolFromSmarts(Chem.MolToSmarts(m))
        return (q if q is not None and q.GetNumAtoms() else m,
                ("read as SMILES, so every atom is fixed to its element. "
                 "To let a position vary, use sar_edit.free() or write it "
                 "as SMARTS, e.g. [#6,#7,#16]"), "smiles")

    q = _quietly(Chem.MolFromSmarts, text)
    if q is not None and q.GetNumAtoms():
        return q, "", "smarts"
    # Both readings failed, so let RDKit say why -- this one is a real error.
    Chem.MolFromSmarts(text)
    return None, f"could not parse as SMARTS or SMILES: {text!r}", None


# --------------------------------------------------------------------------
# why a compound missed
# --------------------------------------------------------------------------

def _skeleton(query_mol):
    """The same core with every atom's element requirement removed.

    Used only to separate the two ways a compound can miss: the shape is
    wrong, or the shape is right and an element disagrees. Those want very
    different responses from the chemist, so they should not read the same.
    """
    rw = Chem.RWMol(query_mol)
    generic = Chem.AtomFromSmarts("[*]")
    if generic is None:
        return None
    for i in range(rw.GetNumAtoms()):
        rw.ReplaceAtom(i, generic)
    return rw.GetMol()


def position_acceptors(query_mol, mol, max_matches: int = 64) -> Dict[int, set]:
    """{core position: atoms of `mol` that satisfy that position's query}.

    Each core atom is lifted into a one-atom query and matched on its own.
    That is the only way to ask "would this position accept this atom?"
    without a whole-molecule match succeeding first -- which is exactly the
    question when the whole-molecule match has failed.
    """
    out: Dict[int, set] = {}
    for q in range(query_mol.GetNumAtoms()):
        one = Chem.RWMol()
        one.AddAtom(query_mol.GetAtomWithIdx(q))
        hits = mol.GetSubstructMatches(one.GetMol(), uniquify=True,
                                       maxMatches=max_matches)
        out[q] = {h[0] for h in hits}
    return out


def explain_miss(mol, query_mol, max_matches: int = 32) -> str:
    """One sentence on why `mol` does not match `query_mol`."""
    if mol is None:
        return "structure did not parse"
    if query_mol is None:
        return "no core"
    if mol.HasSubstructMatch(query_mol):
        return "matches"

    skel = _skeleton(query_mol)
    if skel is None:
        return "does not match"
    shapes = mol.GetSubstructMatches(skel, uniquify=True,
                                     maxMatches=max_matches)
    if not shapes:
        return ("no substructure of this shape — the ring system or "
                "connectivity differs")

    # Of every placement with the right shape, report the one needing the
    # fewest changes: that is the near miss a chemist can act on.
    accept = position_acceptors(query_mol, mol)
    best, best_bad = None, None
    for shape in shapes:
        bad = [q for q, a in enumerate(shape) if a not in accept.get(q, ())]
        if best_bad is None or len(bad) < len(best_bad):
            best, best_bad = shape, bad
    if not best_bad:
        return "does not match, though every position individually would"

    detail = ", ".join(
        f"position {q} would need "
        f"{mol.GetAtomWithIdx(best[q]).GetSymbol()}"
        for q in best_bad[:4])
    more = f" (+{len(best_bad) - 4} more)" if len(best_bad) > 4 else ""
    return f"shape fits; {detail}{more}"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def evaluate(core, smiles_list: Sequence[str],
             ids: Optional[Sequence[str]] = None, *,
             optimize: bool = True,
             label: str = "hand-built") -> dict:
    """Score a proposed core against the compounds. The heart of this module.

    Nothing is filtered and nothing is rejected: a core with 20% coverage is
    reported at 20% coverage, because the chemist asked about that core. The
    search filters; this answers.
    """
    report: dict = {
        "ok": False, "error": None, "notes": [],
        "smarts": None, "query_mol": None, "n_atoms": 0,
        "coverage": 0.0, "matched": [], "unmatched": [], "misses": [],
        "n_variable": 0, "variable": {}, "n_rgroups": 0, "attachments": {},
        "ambiguous": [], "candidate": None,
    }

    qmol, note, kind = parse_core(core)
    if note:
        report["notes"].append(note)
    if qmol is None:
        report["error"] = note or "could not parse the core"
        return report

    ids = list(ids) if ids is not None else [
        str(i + 1) for i in range(len(smiles_list))]
    pairs = [(i, Chem.MolFromSmiles(s)) for i, s in zip(ids, smiles_list)]
    bad = [i for i, m in pairs if m is None]
    if bad:
        report["notes"].append(
            f"{len(bad)} structure(s) did not parse and were skipped")
    good_ids = [i for i, m in pairs if m is not None]
    good = [m for _, m in pairs if m is not None]

    report["query_mol"] = qmol
    # A SMARTS the chemist wrote is handed back verbatim. Regenerating it
    # would be a silent transformation of their input for no gain, and the
    # string is what the decomposition re-parses downstream -- so rewriting
    # it is a chance for the analysed core and the applied core to diverge.
    # A SMILES core is converted, because a SMILES string read *as SMARTS*
    # downstream is not reliably the same query.
    report["smarts"] = (core.strip() if kind == "smarts"
                        else Chem.MolToSmarts(qmol))
    report["n_atoms"] = qmol.GetNumAtoms()

    if not good:
        report["error"] = "no parseable structures to match against"
        return report

    analysis = sv.position_elements(good, qmol, good_ids, optimize=optimize)
    matched = [i for i in good_ids if i not in set(analysis["unmatched"])]
    report["coverage"] = len(matched) / len(good)
    report["matched"] = matched
    report["unmatched"] = list(analysis["unmatched"])
    report["ambiguous"] = list(analysis["ambiguous"])
    report["n_rgroups"] = analysis["n_rgroups"]
    report["attachments"] = analysis["attachments"]
    report["n_variable"] = len(analysis["variable"])

    labels = sv.label_variables(analysis["variable"])
    for q, name in sorted(labels.items()):
        per = analysis["per_position"].get(q, {})
        report["variable"][name] = {
            "position": q,
            "elements": analysis["elements"].get(q, []),
            "counts": dict(Counter(per.values())),
        }

    by_id = dict(zip(good_ids, good))
    report["misses"] = [{"id": i, "reason": explain_miss(by_id[i], qmol)}
                        for i in report["unmatched"][:20]]

    assignments, xcols = sv.variable_columns(analysis, good_ids)
    # Shaped exactly like a find_core_candidates entry, so a hand-built core
    # flows through decompose / naming / Markush / cliffs untouched.
    report["candidate"] = {
        "smarts": report["smarts"],
        "query_mol": qmol,
        "preset": label,
        "presets": {label},
        "preset_label": f"{label} core",
        "coverage": report["coverage"],
        "n_atoms": qmol.GetNumAtoms(),
        "n_bonds": qmol.GetNumBonds(),
        "n_ring_atoms": sv.n_ring_atoms(qmol),
        "thresholds": [],
        "timed_out": False,
        "analysis": analysis,
        "variable_labels": labels,
        "variable_columns": xcols,
        "assignments": assignments,
        "n_variable": len(analysis["variable"]),
        "derived": "hand",
        "rank": 0,
    }
    report["ok"] = True
    return report


def compare_to(report: dict, candidate: Optional[dict]) -> List[str]:
    """Hand-built core against what the search proposed, as readable lines.

    The point of this is not scoring but diagnosis: when a chemist knows a
    core exists and the search did not offer it, this says whether the search
    was wrong or the chemist was.
    """
    if not report.get("ok"):
        return [f"core not evaluated: {report.get('error')}"]
    lines = [
        f"                      {'yours':>12}  {'search':>12}",
        f"  coverage            {report['coverage']:>11.0%}  "
        + (f"{candidate['coverage']:>11.0%}" if candidate else f"{'—':>12}"),
        f"  core atoms          {report['n_atoms']:>12}  "
        + (f"{candidate['n_atoms']:>12}" if candidate else f"{'—':>12}"),
        f"  variable positions  {report['n_variable']:>12}  "
        + (f"{candidate['n_variable']:>12}" if candidate else f"{'—':>12}"),
        f"  R columns           {report['n_rgroups']:>12}  "
        + (f"{candidate['analysis']['n_rgroups']:>12}" if candidate
           else f"{'—':>12}"),
    ]
    if candidate:
        if (report["coverage"] >= candidate["coverage"] - 1e-9
                and report["n_variable"] < candidate["n_variable"]):
            lines.append("  → yours is better on both counts; the search "
                         "should have found it. Worth reporting.")
        elif report["coverage"] < candidate["coverage"] - 1e-9:
            lines.append(f"  → yours costs "
                         f"{(candidate['coverage'] - report['coverage']):.0%} "
                         f"coverage for the simpler core")
    return lines


def summarize(report: dict) -> List[str]:
    """Readable lines for a notebook or a terminal."""
    if not report.get("ok"):
        return [f"core rejected: {report.get('error')}"]
    lines = [
        f"core      : {report['smarts']}",
        f"coverage  : {report['coverage']:.0%}  "
        f"({len(report['matched'])}/{len(report['matched']) + len(report['unmatched'])})",
        f"variable  : {report['n_variable']} position(s)",
    ]
    for name, info in report["variable"].items():
        detail = ", ".join(f"{e}×{info['counts'].get(e, 0)}"
                           for e in info["elements"])
        lines.append(f"  {name}      = {'/'.join(info['elements'])}   ({detail})")
    lines.append(f"R columns : ~{report['n_rgroups']}")
    if report["ambiguous"]:
        lines.append(f"arbitrary : {len(report['ambiguous'])} compound(s) sit "
                     f"on this core equally well either way round")
    if report["unmatched"]:
        lines.append(f"unmatched : {len(report['unmatched'])}")
        for m in report["misses"][:8]:
            lines.append(f"  {m['id']}: {m['reason']}")
    for n in report["notes"]:
        lines.append(f"note: {n}")
    return lines


# --------------------------------------------------------------------------
# editing a core  --  chemist-level operations, no drawing required
# --------------------------------------------------------------------------
#
# Most of the time the core a chemist wants is not a new drawing but a small
# change to one the search already offered: "that one, but with position 3
# fixed as nitrogen". Making them redraw a benzimidazole to say that is busy
# work, and a redrawn core is a chance to introduce a difference nobody
# intended. These operate on the query directly.

def _core_of(core_or_candidate):
    if isinstance(core_or_candidate, dict):
        return core_or_candidate.get("query_mol")
    q, _ = parse_core(core_or_candidate)
    return q


def _replace_atom(query_mol, position: int, smarts: str):
    """A copy of the core with one atom replaced by `smarts`."""
    atom = Chem.AtomFromSmarts(smarts)
    if atom is None or query_mol is None:
        return None
    if position < 0 or position >= query_mol.GetNumAtoms():
        return None
    rw = Chem.RWMol(query_mol)
    rw.ReplaceAtom(position, atom)
    return rw.GetMol()


def fix(core_or_candidate, position: int, element: str):
    """Require `position` to be `element`, keeping every other condition.

    Narrows rather than replaces, so the aromaticity, ring membership and
    connectivity the MCS established at that atom all survive.
    """
    q = _core_of(core_or_candidate)
    if q is None:
        return None
    return sv._pin_query(q, position, element)


def allow(core_or_candidate, position: int, elements: Sequence[str],
          aromatic: bool = True, in_ring: bool = True):
    """Let `position` be any of `elements` — widening as well as narrowing.

    `fix` can only tighten a position, because ANDing a condition on is all
    RDKit's ExpandQuery does. Opening one back up means replacing the atom, so
    the aromatic and ring conditions have to be restated; they default on
    because a core position being opened up is nearly always a ring atom.
    """
    q = _core_of(core_or_candidate)
    pt = Chem.GetPeriodicTable()
    try:
        nums = ",".join(f"#{pt.GetAtomicNumber(e)}" for e in elements)
    except Exception:
        return None
    if not nums:
        return None
    extra = "".join([";a" if aromatic else "", ";r" if in_ring else ""])
    return _replace_atom(q, position, f"[{nums}{extra}]")


def free(core_or_candidate, position: int, aromatic: bool = True,
         in_ring: bool = True):
    """Let `position` be any element at all."""
    q = _core_of(core_or_candidate)
    spec = "a" if aromatic else "*"
    if in_ring:
        spec += ";r"
    return _replace_atom(q, position, f"[{spec}]")


def enumerate_fixes(candidate: dict, smiles_list: Sequence[str],
                    ids: Optional[Sequence[str]] = None, *,
                    positions: Optional[Sequence[int]] = None) -> List[dict]:
    """Every single-position fixing of a core, scored, nothing filtered out.

    `sar_variable.pin_variants` does something similar but keeps only what
    passes a coverage test, because it feeds the ranked candidate list. This
    is the browsable version: the whole menu, coverage and all, so a chemist
    can see the option they had in mind even when it costs something. Ordered
    by coverage, then by how few positions still vary.
    """
    q = candidate.get("query_mol") if isinstance(candidate, dict) else None
    if q is None:
        return []
    ids = list(ids) if ids is not None else [
        str(i + 1) for i in range(len(smiles_list))]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    good = [m for m in mols if m is not None]

    analysis = candidate.get("analysis") or {}
    targets = list(positions if positions is not None
                   else analysis.get("variable", []))
    reach = sv.reachable_elements(good, q) if good else {}

    out: List[dict] = []
    for pos in targets:
        for element in sorted(reach.get(pos, set())):
            pinned = sv._pin_query(q, pos, element)
            if pinned is None:
                continue
            rep = evaluate(pinned, smiles_list, ids,
                           label=f"position {pos} fixed to {element}")
            if not rep["ok"]:
                continue
            rep["edit"] = {"op": "fix", "position": pos, "element": element}
            out.append(rep)
    out.sort(key=lambda r: (-r["coverage"], r["n_variable"], r["n_rgroups"]))
    return out


def edits_table(reports: Sequence[dict]) -> List[str]:
    """The menu from `enumerate_fixes`, as aligned text."""
    lines = [f"{'#':>3}  {'edit':<26} {'cover':>6} {'X':>3} {'R':>3}  "
             f"variable elements"]
    for i, r in enumerate(reports, start=1):
        e = r.get("edit", {})
        what = f"fix pos {e.get('position')} = {e.get('element')}"
        els = "  ".join(f"{k}={'/'.join(v['elements'])}"
                        for k, v in r["variable"].items()) or "—"
        lines.append(f"{i:>3}  {what:<26} {r['coverage']:>5.0%} "
                     f"{r['n_variable']:>3} {r['n_rgroups']:>3}  {els}")
    return lines

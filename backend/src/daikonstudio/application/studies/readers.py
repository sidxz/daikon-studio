"""How a study reads a result, by kind: what the agent is told, may see and can check.

The agent is handed a summary of a result, never a table or a structure whole, and asks
for a bounded look when a summary is not enough. What it then claims is checked against
the result itself before it is stored. A claim that reads well and cites a residue the
pocket does not have is the failure a reader cannot catch for themselves, and the one
this exists to catch. So a kind can be talked about only where it has a checker: there is
deliberately no permissive default, and adding a kind means writing its checker, not
loosening this.

A checker returns pointers with no step. It knows what a result holds; the agent knows
which step a claim was about, and fills it in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from daikonstudio.application.studies.schemas import Claim, Look
from daikonstudio.application.tools import codecs
from daikonstudio.domain.studies.study import Pointer

# What one look may show, however many rows it asks for.
LOOK_ROWS_MAX = 50
# Columns that hold a structure or a sequence: described to the agent, never shown to it.
WITHHELD_COLUMNS = re.compile(r"^((canonical_)?smiles|sequence|inchi|mol)$", re.I)
_IDENTIFIER = re.compile(r"^(id|compound_id|identifier|name)$", re.I)


@dataclass(frozen=True)
class KindReader:
    """What a study can do with one kind of result.

    `look` and `check` are None where a kind offers none. A kind with no `check` can be
    summarized but not claimed about. `notes` are what the agent should know when it
    reads this kind, beyond what the summary says.
    """

    summarize: Callable[[Any, Mapping[str, Any]], str]
    look: Callable[[Any, Look], str] | None
    check: Callable[[Sequence[Claim], Any], tuple[list[Pointer], int]] | None
    notes: str = ""


def _lining(pocket: Mapping[str, Any]) -> set[str]:
    rows = pocket.get("residues") or []
    return {_key(f"{r.get('resname', '')}{r.get('resseq', '')}") for r in rows}


def _key(name: str) -> str:
    """A residue as written anywhere: 'A:HIS94', 'His 94' and 'HIS94' are one."""
    return name.split(":")[-1].replace(" ", "").upper()


def _supported(evidence: str, pockets: Sequence[Mapping[str, Any]]) -> bool:
    if evidence == "ligand_overlap":
        return any(not o.get("additive") for p in pockets for o in (p.get("ligand_overlap") or []))
    if evidence == "annotated_site":
        return any(p.get("annotated") for p in pockets)
    if evidence == "interface":
        return any(p.get("interface") for p in pockets)
    # lining_residues: the residues themselves are checked by the caller. Anything else
    # is a table's word, and a pocket has no such column to carry it.
    return evidence == "lining_residues"


def _check_pockets(claims: Sequence[Claim], data: Mapping[str, Any]) -> tuple[list[Pointer], int]:
    """Keep only the claims this pocket result actually supports."""
    pockets = data.get("pockets") or []
    by_rank = {int(p["rank"]): p for p in pockets if "rank" in p}
    kept: list[Pointer] = []
    dropped = 0
    for claim in claims:
        named = [by_rank[r] for r in dict.fromkeys(claim.candidates) if r in by_rank]
        if not named or len(named) != len(set(claim.candidates)):
            dropped += 1
            continue
        lining: set[str] = set()
        for pocket in named:
            lining |= _lining(pocket)
        # A claim with no residues behind it is an opinion, not a reading.
        if not claim.cites or any(_key(c) not in lining for c in claim.cites):
            dropped += 1
            continue
        if not _supported(claim.evidence, named):
            dropped += 1
            continue
        ranks = ", ".join(str(r) for r in dict.fromkeys(claim.candidates))
        kept.append(
            Pointer(
                step=None,
                at=f"rank{'s' if len(named) > 1 else ''} {ranks}",
                label=claim.label,
                evidence=claim.evidence,
                cites=tuple(claim.cites),
            )
        )
    return kept, dropped


def _describe(data: Mapping[str, Any]) -> str:
    """The result as the agent reads it: every candidate, with what lines it."""
    meta = data.get("meta") or {}
    pockets = data.get("pockets") or []
    lines = [
        f"Kept in the receptor: {', '.join(meta.get('receptor_ligands') or []) or 'nothing'}.",
        f"Numbering against UniProt: {meta.get('numbering') or 'not checked'}.",
        "",
    ]
    for pocket in pockets:
        overlap = ", ".join(
            f"{o.get('comp_id')}{' (additive)' if o.get('additive') else ''} "
            f"{o.get('shared')}/{o.get('of')}"
            for o in (pocket.get("ligand_overlap") or [])
        )
        annotated = "; ".join(
            f"{a.get('type')} {a.get('start')}" for a in (pocket.get("annotated") or [])
        )
        residues = ", ".join(
            f"{r.get('resname')}{r.get('resseq')}" for r in (pocket.get("residues") or [])
        )
        lines.append(
            f"Candidate {pocket.get('rank')}: druggability {pocket.get('druggability')}, "
            f"volume {pocket.get('volume')}, "
            f"interface {'yes' if pocket.get('interface') else 'no'}"
        )
        lines.append(f"  lines: {residues or '(none)'}")
        lines.append(f"  ligands: {overlap or '(none)'}")
        lines.append(f"  annotated: {annotated or '(none)'}")
    return "\n".join(lines)


_POCKET_NOTES = (
    "Do not read the ranking as a recommendation. Rank and score come from the tool's "
    "geometric model, which does not know what the protein does. A surface groove "
    "routinely outranks the site a drug binds. Judge each candidate on the residues that "
    "line it, the ligands it touches and the annotated sites it contains.\n"
    "Say when several candidates are one site. An extended cavity is often split across "
    "candidates sitting far apart in a ranked list, and a reader of the table alone will "
    "take them for separate sites. Put their ranks in one claim."
)


def identifier_column(columns: Sequence[str]) -> str:
    """The column that names a row: the first called an id or a name, else the first."""
    return next((c for c in columns if _IDENTIFIER.match(c)), columns[0])


def _shown(column: str, value: Any) -> str:
    """A cell as the agent may see it: a structure or a sequence is withheld."""
    return "<structure>" if WITHHELD_COLUMNS.match(column) else str(value)


def summarize_table(data: Any, summary: Mapping[str, Any]) -> str:
    """A table as the agent first meets it: its shape and what each column holds.

    Only the stored summary is read, so no row can be quoted and `data` may be None.
    """
    columns: list[str] = summary.get("columns") or []
    profile = {p["name"]: p for p in summary.get("columns_profile") or []}
    lines = [f"Rows: {summary.get('rows', 0)}. Columns: {len(columns)}."]
    if columns:
        lines.append(f"Rows are named by the values of {identifier_column(columns)}.")
    # ponytail: one line per column; cap and group them when a table is wide (descriptor
    # tables run to hundreds of columns).
    for name in columns:
        p = profile.get(name)
        if p is None:  # stored before the summary carried a profile
            lines.append(f"- {name}")
            continue
        facts = [p["dtype"], f"{p['filled']} filled", f"{p['distinct']} distinct"]
        if WITHHELD_COLUMNS.match(name):
            facts.append("values withheld")
        elif p.get("min") is not None:
            facts.append(f"from {p['min']:.4g} to {p['max']:.4g}, median {p['median']:.4g}")
        lines.append(f"- {name}: {', '.join(facts)}")
    return "\n".join(lines)


def look_table(data: Sequence[Mapping[str, Any]], look: Look) -> str:
    """The rows asked for, up to the cap, one line each: `identifier: column=value, ...`."""
    rows = data[: max(0, min(look.rows, LOOK_ROWS_MAX))]
    if not rows:
        return "(no rows)"
    present = list(rows[0])
    wanted = [c for c in look.columns if c in present] or present
    key = identifier_column(present)
    return "\n".join(
        f"{_shown(key, row[key])}: " + ", ".join(f"{c}={_shown(c, row[c])}" for c in wanted)
        for row in rows
    )


def check_table(
    claims: Sequence[Claim], data: Sequence[Mapping[str, Any]]
) -> tuple[list[Pointer], int]:
    """Keep only the claims whose citations are in this table, spelled as it spells them."""
    if not data:
        return [], len(claims)
    columns = list(data[0])
    key = identifier_column(columns)
    rows = {str(row[key]) for row in data if row[key] is not None}
    known = {*columns, *rows}
    kept: list[Pointer] = []
    dropped = 0
    for claim in claims:
        # The evidence says what is pointed at, a column or some rows, and needs a cite of
        # that kind; a cite of neither is not in this table at all.
        pool = {"column": columns, "rows": rows}.get(claim.evidence, ())
        pointed = list(dict.fromkeys(c for c in claim.cites if c in pool))
        if not pointed or any(c not in known for c in claim.cites):
            dropped += 1
            continue
        kept.append(
            Pointer(
                step=None,
                at=f"{claim.evidence} {', '.join(pointed)}",
                label=claim.label,
                evidence=claim.evidence,
                cites=tuple(claim.cites),
            )
        )
    return kept, dropped


def summarize_structure(data: Any, summary: Mapping[str, Any]) -> str:
    """What was recorded when a structure was fetched or prepared; its atoms stay out."""
    lines = []
    for key, value in summary.items():
        if value in (None, "", [], {}):
            continue
        if key == "ligands":
            value = [f"{x.get('comp_id')} {x.get('chain')}{x.get('resseq')}" for x in value]
        elif isinstance(value, list) and any(isinstance(x, Mapping) for x in value):
            value = f"{len(value)} records"
        lines.append(f"{key}: {', '.join(map(str, value)) if isinstance(value, list) else value}")
    return "\n".join(lines)


_TABLE = KindReader(summarize=summarize_table, look=look_table, check=check_table)

READERS: dict[str, KindReader] = {
    "pocket_candidates": KindReader(
        summarize=lambda data, summary: _describe(data),
        look=None,
        check=_check_pockets,
        notes=_POCKET_NOTES,
    ),
    **dict.fromkeys(sorted(codecs.TABLE_KINDS), _TABLE),
    "structure": KindReader(summarize=summarize_structure, look=None, check=None),
}

# A kind nobody wrote a reader for is summarized by its meta, and cannot be claimed about.
_UNREAD = KindReader(summarize=lambda data, summary: json.dumps(summary), look=None, check=None)


def reader(kind: str) -> KindReader:
    return READERS.get(kind, _UNREAD)


def checkable() -> list[str]:
    """The kinds a claim can be checked on, for the prompt."""
    return sorted(kind for kind, r in READERS.items() if r.check is not None)

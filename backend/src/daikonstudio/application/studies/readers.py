"""How a study reads a result, by kind: what the agent is told, may see and can check.

The agent is handed a summary of a result, never a table or a structure whole, and asks
for a bounded look when a summary is not enough. What it then claims is checked against
the result itself before it is stored. A claim that reads well and cites a residue the
pocket does not have is the failure a reader cannot catch for themselves, and the one
this exists to catch. So a kind can be claimed about only where it has a checker: there is
deliberately no permissive default, and adding a kind means writing its checker, not
loosening this.

A checker returns pointers with no step. It knows what a result holds; the agent knows
which step a claim was about, and fills it in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from daikonstudio.application.studies.schemas import LOOK_ROWS_MAX as LOOK_ROWS_MAX
from daikonstudio.application.studies.schemas import Claim, Look
from daikonstudio.application.tools import codecs
from daikonstudio.application.tools.codecs import WITHHELD_COLUMNS as WITHHELD_COLUMNS
from daikonstudio.domain.studies.study import Pointer

_IDENTIFIER = re.compile(r"^(id|compound_id|identifier|name)$", re.I)
# A cell is cut here and its whitespace collapsed, so a row stays one short line.
_CELL_MAX = 80


@dataclass(frozen=True)
class KindReader:
    """What a study can do with one kind of result.

    `look` and `check` are None where a kind offers none. A kind with no `check` can be
    summarized but not claimed about. `notes` are what the agent should know when it
    reads this kind, beyond what the summary says. Each function is also given the value's
    stored summary, because some facts, such as which columns hold structures, were decided
    when the value was encoded and cannot be read back from its rows.
    """

    summarize: Callable[[Any, Mapping[str, Any]], str]
    look: Callable[[Any, Look, Mapping[str, Any]], str] | None
    check: Callable[[Sequence[Claim], Any, Mapping[str, Any]], tuple[list[Pointer], int]] | None
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
    # Only the predictor sets ignored_ligands; it keeps nothing in a receptor, so it
    # gets its own account rather than the cavity search's.
    if "ignored_ligands" in meta:
        ligands = f"Ignored by the predictor: {', '.join(meta['ignored_ligands']) or 'nothing'}."
    else:
        kept = ", ".join(meta.get("receptor_ligands") or []) or "nothing"
        ligands = f"Kept in the receptor: {kept}."
    lines = [ligands, f"Numbering against UniProt: {meta.get('numbering') or 'not checked'}.", ""]
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
        # A predicted site carries a probability and no volume; a cavity, druggability and both.
        scored = "probability" if "probability" in pocket else "druggability"
        volume = f"volume {pocket['volume']}, " if "volume" in pocket else ""
        lines.append(
            f"Candidate {pocket.get('rank')}: {scored} {pocket.get(scored)}, {volume}"
            f"interface {'yes' if pocket.get('interface') else 'no'}"
        )
        lines.append(f"  lines: {residues or '(none)'}")
        lines.append(f"  ligands: {overlap or '(none)'}")
        lines.append(f"  annotated: {annotated or '(none)'}")
    return "\n".join(lines)


_POCKET_NOTES = (
    "Do not read the ranking as a recommendation. Rank and score come from the tool's "
    "model of the surface, which does not know what the protein does. A surface groove "
    "routinely outranks the site a drug binds. Judge each candidate on the residues that "
    "line it, the ligands it touches and the annotated sites it contains.\n"
    "Say when several candidates are one site. An extended cavity is often split across "
    "candidates sitting far apart in a ranked list, and a reader of the table alone will "
    "take them for separate sites. Put their ranks in one claim."
)


def _variable_cites(candidate: Mapping[str, Any]) -> set[str]:
    """A core's variable positions as citable: each position and each element it takes."""
    variable = candidate.get("variable") or {}
    return {*variable, *(e for v in variable.values() for e in str(v).split("/"))}


def check_cores(claims: Sequence[Claim], data: Mapping[str, Any]) -> tuple[list[Pointer], int]:
    """Keep only the claims about a core this result has, citing what it missed or let vary."""
    by_rank = {int(c["rank"]): c for c in data.get("candidates") or [] if "rank" in c}
    kept: list[Pointer] = []
    dropped = 0
    for claim in claims:
        named = [by_rank[r] for r in dict.fromkeys(claim.candidates) if r in by_rank]
        if not named or len(named) != len(set(claim.candidates)):
            dropped += 1
            continue
        known = {u for c in named for u in (*(c.get("unmatched") or []), *_variable_cites(c))}
        # A core claimed about with nothing behind it is an opinion, not a reading.
        if (
            claim.evidence not in {"coverage", "unmatched", "variable"}
            or not claim.cites
            or any(c not in known for c in claim.cites)
        ):
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


def summarize_cores(data: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    """Every candidate core, with what it covers, lets vary and misses; no SMARTS or drawing."""
    meta = data.get("meta") or {}
    lines = [
        f"{meta.get('n_candidates')} candidates for {meta.get('n_compounds')} compounds, "
        f"{meta.get('n_unparsed')} unparsed. " + " ".join(meta.get("notes") or [])
    ]
    for c in data.get("candidates") or []:
        series = f", series {c['series']}" if c.get("series") is not None else ""
        variable = ", ".join(f"{k} ({v})" for k, v in (c.get("variable") or {}).items())
        missed = c.get("unmatched") or []
        told = f"misses {len(missed)}: {', '.join(missed)}" if missed else "misses none"
        timed_out = " Search timed out." if c.get("timed_out") else ""
        lines.append(
            f"rank {c.get('rank')}{series}: coverage {c.get('coverage', 0):.0%}, "
            f"{c.get('n_atoms')} atoms, variable {variable or 'none'}, {told}."
            f"{timed_out} {' '.join(c.get('description') or [])}".rstrip()
        )
    return "\n".join(lines)


_CORE_NOTES = (
    "Do not read the ranking as a recommendation: rank one is the first option, and the "
    "largest core often reaches full coverage only by letting several positions vary. A "
    "narrower core that misses a few compounds may be the one a chemist wants. Say what each "
    "core covers, lets vary and misses, citing the compounds and positions as listed."
)


def check_fit(claims: Sequence[Claim], data: Mapping[str, Any]) -> tuple[list[Pointer], int]:
    """Keep only the claims whose citations are positions, groups or compounds this fit has."""
    meta = data.get("meta") or {}
    reference = meta.get("reference") or {}
    known = {*reference, *map(str, reference.values())}
    for t in data.get("terms") or []:
        known |= {str(t.get("position")), str(t.get("group"))}
    known |= {str(f.get("id")) for f in data.get("fitted") or []}
    for u in data.get("unmade") or []:
        known |= {*(u.get("groups") or {}), *map(str, (u.get("groups") or {}).values())}
    kept: list[Pointer] = []
    dropped = 0
    for claim in claims:
        if (
            claim.evidence not in {"term", "fit", "unmade"}
            or not claim.cites
            or any(c not in known for c in claim.cites)
        ):
            dropped += 1
            continue
        kept.append(
            Pointer(
                step=None,
                at=f"{claim.evidence} {', '.join(claim.cites)}",
                label=claim.label,
                evidence=claim.evidence,
                cites=tuple(claim.cites),
            )
        )
    return kept, dropped


def summarize_fit(data: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    """The substituent model as the agent reads it: how well it fits, then every term."""
    meta = data.get("meta") or {}
    if not meta.get("ok"):
        return f"The model was not fitted: {meta.get('reason')}"
    reference = ", ".join(f"{p} {g}" for p, g in (meta.get("reference") or {}).items())
    lines = [
        f"Activity: {meta.get('activity_column')}. {meta.get('n_compounds')} compounds fitted, "
        f"{meta.get('n_excluded_censored')} left out as censored. r2 {_r(meta.get('r2'))}, "
        f"q2 {_r(meta.get('q2'))} (leave-one-out).",
        f"Reference group at each position: {reference or 'none'}.",
        "Each term, as the change in activity from the reference, and the compounds behind it:",
    ]
    for t in data.get("terms") or []:
        lines.append(
            f"- {t.get('position')} {t.get('group')}: {round(t['coef'], 3):+} (n {t.get('n')})"
        )
    unmade = data.get("unmade") or []
    lines.append(
        "Combinations not made, with the model's prediction:" if unmade else "None unmade."
    )
    for u in unmade:
        groups = ", ".join(f"{p} {g}" for p, g in (u.get("groups") or {}).items())
        lines.append(f"- {groups}: {_r(u.get('prediction'))}")
    return "\n".join(lines)


_FIT_NOTES = (
    "A term is the change in activity from the reference group, assuming each position acts "
    "alone. Quote q2, not r2, for how far the model can be trusted: r2 only says it fits what "
    "it was shown. A prediction for a combination not made is the sum of its terms and "
    "nothing more."
)


def identifier_column(columns: Sequence[str], withheld: Collection[str] = ()) -> str:
    """The column that names a row: the first called an id or a name, else the first.

    A column whose values are withheld cannot name a row to a reader who never sees them, so
    it is passed over unless every column is withheld.
    """
    shown = [c for c in columns if c not in withheld] or list(columns)
    return next((c for c in shown if _IDENTIFIER.match(c)), shown[0])


def _withheld(columns: Sequence[str], summary: Mapping[str, Any]) -> set[str]:
    """The columns whose values the agent never sees.

    Those the encoder found by what they hold, and any named like a structure, which is all a
    summary stored before the encoder looked has to go on.
    """
    return {c for c in columns if WITHHELD_COLUMNS.search(c)} | set(summary.get("withheld") or ())


def _cell(value: Any) -> str:
    """A value on one short line, as the agent reads it and as a claim must spell it."""
    text = " ".join(str(value).split())
    return text if len(text) <= _CELL_MAX else text[: _CELL_MAX - 3] + "..."


def _shown(column: str, value: Any, withheld: Collection[str]) -> str:
    """A cell as the agent may see it: a structure or a sequence is withheld."""
    return "<structure>" if column in withheld else _cell(value)


def summarize_table(data: Any, summary: Mapping[str, Any]) -> str:
    """A table as the agent first meets it: its shape and what each column holds.

    Only the stored summary is read, so no row can be quoted and `data` may be None.
    """
    columns: list[str] = summary.get("columns") or []
    withheld = _withheld(columns, summary)
    profile = {p["name"]: p for p in summary.get("columns_profile") or []}
    lines = [f"Rows: {summary.get('rows', 0)}. Columns: {len(columns)}."]
    key = identifier_column(columns, withheld) if columns else None
    if key is not None and key not in withheld:
        lines.append(f"Rows are named by the values of {key}.")
    # ponytail: one line per column; cap and group them when a table is wide (descriptor
    # tables run to hundreds of columns).
    for name in columns:
        p = profile.get(name)
        if p is None:  # stored before the summary carried a profile
            lines.append(f"- {name}")
            continue
        facts = [p["dtype"], f"{p['filled']} filled", f"{p['distinct']} distinct"]
        if name in withheld:
            facts.append("values withheld")
        elif p.get("min") is not None:
            facts.append(f"from {p['min']:.4g} to {p['max']:.4g}, median {p['median']:.4g}")
        lines.append(f"- {name}: {', '.join(facts)}")
    return "\n".join(lines)


def look_table(data: Sequence[Mapping[str, Any]], look: Look, summary: Mapping[str, Any]) -> str:
    """The rows asked for, up to the cap, one line each: `identifier: column=value, ...`."""
    rows = data[: max(0, min(look.rows, LOOK_ROWS_MAX))]
    if not rows:
        return "(no rows)"
    present = list(rows[0])
    withheld = _withheld(present, summary)
    wanted = [c for c in look.columns if c in present] or present
    key = identifier_column(present, withheld)
    return "\n".join(
        f"{_shown(key, row[key], withheld)}: "
        + ", ".join(f"{c}={_shown(c, row[c], withheld)}" for c in wanted)
        for row in rows
    )


def check_table(
    claims: Sequence[Claim], data: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> tuple[list[Pointer], int]:
    """Keep only the claims whose citations are in this table, spelled as it spells them."""
    if not data:
        return [], len(claims)
    columns = list(data[0])
    withheld = _withheld(columns, summary)
    key = identifier_column(columns, withheld)
    # Rows are cited by the identifier as a look showed it; with every column withheld,
    # a look showed none.
    rows = set() if key in withheld else {_cell(row[key]) for row in data if row[key] is not None}
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
    """What was recorded when a structure was fetched or prepared; its atoms stay out.

    An empty list is said, not left out: a prepared structure's `kept_ligands` empty is a
    receptor stripped of every ligand, and left out, the entry's own ligands beside it would
    read as still there. Those keep their name only until the structure is prepared.
    """
    lines = []
    for key, value in summary.items():
        if value in (None, "", {}):
            continue
        if key == "ligands":
            if "kept_ligands" in summary:
                key = "ligands before preparation"
            value = [f"{x.get('comp_id')} {x.get('chain')}{x.get('resseq')}" for x in value]
        elif isinstance(value, list) and any(isinstance(x, Mapping) for x in value):
            value = f"{len(value)} records"
        told = (", ".join(map(str, value)) or "none") if isinstance(value, list) else value
        lines.append(f"{key}: {told}")
    return "\n".join(lines)


def _r(value: Any) -> str:
    """A coefficient as the card shows it, to three places."""
    return "none" if value is None else str(round(value, 3))


def summarize_correlation(data: Any, summary: Mapping[str, Any]) -> str:
    """Every property against the activity, with its r, its interval and its n, and what was
    left out: the card shows all of it, so the agent reads all of it."""
    strongest = summary.get("strongest")
    lines = [
        f"Activity: {summary.get('activity_column')}. Readings left out because they are "
        f"not numbers: {summary.get('excluded', 0)}.",
        f"Strongest: {strongest['property']}, r {_r(strongest['r'])}."
        if strongest
        else "Strongest: none, no property could be correlated.",
        "Each property, as r (95% interval) and n:",
    ]
    for p in (data or {}).get("properties") or []:
        low, high = p.get("ci_low"), p.get("ci_high")
        interval = "no interval" if low is None or high is None else f"{_r(low)} to {_r(high)}"
        lines.append(f"- {p.get('property')}: r {_r(p.get('r'))} ({interval}), n {p.get('n')}")
    return "\n".join(lines)


_TABLE = KindReader(summarize=summarize_table, look=look_table, check=check_table)

READERS: dict[str, KindReader] = {
    "pocket_candidates": KindReader(
        summarize=lambda data, summary: _describe(data),
        look=None,
        check=lambda claims, data, summary: _check_pockets(claims, data),
        notes=_POCKET_NOTES,
    ),
    "core_candidates": KindReader(
        summarize=summarize_cores,
        look=None,
        check=lambda claims, data, summary: check_cores(claims, data),
        notes=_CORE_NOTES,
    ),
    "model_fit": KindReader(
        summarize=summarize_fit,
        look=None,
        check=lambda claims, data, summary: check_fit(claims, data),
        notes=_FIT_NOTES,
    ),
    **dict.fromkeys(sorted(codecs.TABLE_KINDS), _TABLE),
    "structure": KindReader(summarize=summarize_structure, look=None, check=None),
    "correlation": KindReader(summarize=summarize_correlation, look=None, check=None),
}

# A kind nobody wrote a reader for is summarized by its meta, and cannot be claimed about.
_UNREAD = KindReader(summarize=lambda data, summary: json.dumps(summary), look=None, check=None)


def reader(kind: str) -> KindReader:
    return READERS.get(kind, _UNREAD)


def checkable() -> list[str]:
    """The kinds a claim can be checked on, for the prompt."""
    return sorted(kind for kind, r in READERS.items() if r.check is not None)

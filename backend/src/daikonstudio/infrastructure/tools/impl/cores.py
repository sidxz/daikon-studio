"""The core-finding and R-group tools, over the vendored SAR chemistry."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Any, cast

import polars as pl
from pydantic import BaseModel
from rdkit import Chem, DataStructs
from rdkit.Chem import rdAbbreviations, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from daikonstudio.infrastructure.tools.impl.chem import id_column, structure_column
from daikonstudio.infrastructure.tools.params import CoreParams
from daikonstudio.infrastructure.tools.vendor import sar

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

# The clustering constants, as the sibling's clustering.py: everything crossing
# the boundary is a Tanimoto similarity and Butina's distance conversion happens
# here. Below MIN_CLUSTER a "cluster" is an observation, not a series.
SIMILARITY = 0.6
MIN_CLUSTER = 3
MIN_SEPARATION = 0.15
# Separation alone does not justify splitting: the benzamide set splits into
# three clusters 0.27 apart and yet already gives a 14-atom core as one series.
MIN_CORE_GAIN = 1.5
RADIUS, BITS = 2, 2048
WHOLE_SET_TIMEOUT = 60
CLUSTER_TIMEOUT = 15

_POSITION = re.compile(r"[RX]\d+")
_CORE_LINE = re.compile(r"\s*core\s*:")
# The columns each tool writes, so an incoming column of the same name is moved
# aside rather than overwritten. See `_set_aside`.
_CLUSTER_WRITES = re.compile(
    r"series|scaffold_smiles|series_size|series_similarity"
    r"|whole_set_core_atoms|typical_series_core_atoms|split_recommended"
)
_DECOMPOSE_WRITES = re.compile(r"decomposed|ambiguous|[RX]\d+|R\d+_(smiles|svg)")


def _frame(inputs: Mapping[str, object], key: str = "compounds") -> pl.DataFrame:
    value = inputs[key]
    assert isinstance(value, pl.DataFrame)
    return value


def _mols(frame: pl.DataFrame) -> tuple[list[str], list[str], list[Chem.Mol | None]]:
    """Ids, structures and parsed molecules, in row order.

    A file with no identifier column still has rows, and a core found on it is
    still worth reporting, so the row number stands in for an id.
    """
    smiles = [s or "" for s in frame[structure_column(frame)].to_list()]
    try:
        ids = [str(i) for i in frame[id_column(frame)].to_list()]
    except ValueError:
        ids = [str(n) for n in range(frame.height)]
    return ids, smiles, [Chem.MolFromSmiles(s) if s else None for s in smiles]


def _refuse_repeats(column: pl.Series) -> None:
    """Refuse a repeated identifier, which every later join would multiply.

    A joined duplicate puts one compound's activity beside another's
    substituent, so the numbers downstream are wrong rather than missing. The
    scientist owns the fix, so the step stops and names the ids.
    """
    repeated = [str(v) for v in column.filter(column.is_duplicated()).unique(maintain_order=True)]
    if not repeated:
        return
    more = f" and {len(repeated) - 5} more" if len(repeated) > 5 else ""
    raise ValueError(
        f"the identifier column {column.name!r} repeats: {', '.join(repeated[:5])}{more}. "
        "Give every compound one row with its own id and run the step again."
    )


def _with_id(frame: pl.DataFrame, ids: Sequence[str]) -> pl.DataFrame:
    """The frame with its identifier column named `id` and moved first."""
    try:
        column = id_column(frame)
    except ValueError:
        return frame.insert_column(0, pl.Series("id", list(ids), dtype=pl.Utf8))
    _refuse_repeats(frame[column])
    # `Name,SMILES,ID,pIC50` picks `Name`, and renaming it onto the existing
    # `id` is a polars DuplicateError that tells a scientist nothing.
    if column != "id" and "id" in frame.columns:
        frame = frame.rename({"id": "id_input"})
    rest = [c for c in frame.columns if c != column]
    return frame.rename({column: "id"}).select(["id", *rest])


def _set_aside(frame: pl.DataFrame, written: re.Pattern[str]) -> pl.DataFrame:
    """Move any incoming column this tool is about to write to `<name>_input`.

    `with_columns` and a left join both replace a column of the same name
    silently, so an upload with its own `series` or `R1` came back holding the
    tool's answer, with the scientist's own values gone and every later tool
    computing on whichever of the two survived.
    """
    # ponytail: a frame already carrying both `series` and `series_input` keeps
    # the latter and loses the former; suffix-until-free if that ever turns up.
    moved = {
        c: f"{c}_input"
        for c in frame.columns
        if written.fullmatch(c) and f"{c}_input" not in frame.columns
    }
    return frame.rename(moved)


def _readable(lines: Sequence[str]) -> list[str]:
    """Description lines with the SMARTS-bearing ones dropped.

    `description` is what the agent reads, and a SMARTS is a structure: the
    spec keeps both it and the SVG out of the text.
    """
    return [line for line in lines if "[#" not in line and not _CORE_LINE.match(line)]


def _notes(notes: Sequence[str]) -> list[str]:
    """At most five notes, none of them the per-preset search trace."""
    return [n for n in notes if len(n) <= 300][:5]


def _positions(columns: Sequence[str]) -> list[str]:
    """The R and X columns, R first and in numeric order."""
    found = [c for c in columns if _POSITION.fullmatch(c)]
    return sorted(found, key=lambda c: (c[0] != "R", int(c[1:])))


def _variable(cand: Mapping[str, Any]) -> dict[str, str]:
    elements = cand["analysis"]["elements"]
    labels = sorted(cand["variable_labels"].items())
    return {label: "/".join(elements.get(q, [])) for q, label in labels}


def _candidate(
    cand: Mapping[str, Any], series: int | None, matched: Sequence[str]
) -> dict[str, Any]:
    """One candidate core as Task 5/6 consume it: no query mol, no RDKit objects.

    `cand["rank"]` must already be the rank the candidate is shown under: the
    description is headed with it, and approval goes by rank.
    """
    analysis = cand["analysis"]
    return {
        "rank": cand["rank"],
        "matched": list(matched),
        "smarts": cand["smarts"],
        "svg": sar.sar_core.render_core_svg(cand["smarts"], 360, 220),
        "coverage": cand["coverage"],
        "n_atoms": cand["n_atoms"],
        "n_ring_atoms": cand["n_ring_atoms"],
        "n_variable": cand["n_variable"],
        "variable": _variable(cand),
        "n_rgroups": analysis.get("n_rgroups", 0),
        "unmatched": list(analysis["unmatched"]),
        "misses": [],
        "description": _readable(sar.sar_variable.describe(cand)),
        "preset_label": cand["preset_label"],
        "derived": cand.get("derived"),
        "timed_out": bool(cand.get("timed_out")),
        "series": series,
    }


# ---------------------------------------------------------------------------
# cluster_series
# ---------------------------------------------------------------------------


def _generator() -> Any:
    return rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=BITS)


def _scaffold(mol: Chem.Mol) -> Chem.Mol:
    """What gets fingerprinted: the Murcko scaffold, or the molecule if it has none.

    On whole molecules the similarity is dominated by the substituents, which is
    the opposite of what a chemotype split wants — one scaffold wearing a
    morpholine and wearing a methyl reads as two series. A compound with no ring
    system falls back to itself rather than to an empty fingerprint.
    """
    try:
        # ignore[no-untyped-call]: installed rdkit-stubs leave these two untyped.
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call]
    except Exception:  # RDKit raises bare RuntimeErrors on odd valences
        return mol
    return scaffold if scaffold is not None and scaffold.GetNumAtoms() else mol


def _fingerprints(mols: Sequence[Chem.Mol]) -> list[Any]:
    generator = _generator()
    return [generator.GetFingerprint(_scaffold(m)) for m in mols]


def _butina(fps: Sequence[Any], similarity: float) -> list[tuple[int, ...]]:
    """Butina groups over `fps`, largest first."""
    if len(fps) < 2:
        return [(0,)] if fps else []
    flat: list[float] = []
    for i in range(1, len(fps)):
        flat.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    groups = Butina.ClusterData(  # type: ignore[no-untyped-call]
        flat, len(fps), 1.0 - similarity, isDistData=True
    )
    return sorted((tuple(g) for g in groups), key=len, reverse=True)


def _separation(fps: Sequence[Any], kept: Sequence[tuple[int, ...]]) -> float | None:
    """Mean similarity inside a cluster minus mean similarity between clusters.

    If those two are close the split is arbitrary whatever the cluster count
    says. One representative each way places two clusters without going
    quadratic over a large set.
    """
    inside: list[float] = []
    between: list[float] = []
    for n, group in enumerate(kept):
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                inside.append(DataStructs.TanimotoSimilarity(fps[group[a]], fps[group[b]]))
        for other in kept[n + 1 :]:
            between.extend(
                DataStructs.BulkTanimotoSimilarity(fps[group[0]], [fps[i] for i in other[:20]])
            )
    if not inside or not between:
        return None
    return round(sum(inside) / len(inside) - sum(between) / len(between), 3)


def _core_atoms(smiles: Sequence[str], ids: Sequence[str], timeout: int) -> int:
    """Atoms in the best core for these compounds; 0 when the search found none."""
    found = sar.sar_variable.find_core_candidates(list(smiles), list(ids), timeout=timeout)
    return int(found["candidates"][0]["n_atoms"]) if found["candidates"] else 0


def cluster_series(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    frame = _frame(inputs)
    ids, smiles, mols = _mols(frame)
    # Before the search, not after it: a repeated id fails the step, and a
    # minute of clustering first is a minute wasted.
    base = _with_id(_set_aside(frame, _CLUSTER_WRITES), ids)
    usable = [n for n, m in enumerate(mols) if m is not None]

    series: list[int | None] = [None] * len(ids)
    scaffolds: list[str | None] = [None] * len(ids)
    sizes: list[int | None] = [None] * len(ids)
    similarity: list[float | None] = [None] * len(ids)
    baseline: int | None = None
    typical: int | None = None
    recommended = False

    if len(usable) >= 2 * MIN_CLUSTER:
        fps = _fingerprints([cast(Chem.Mol, mols[n]) for n in usable])
        kept = [g for g in _butina(fps, SIMILARITY) if len(g) >= MIN_CLUSTER]
        atoms: list[int] = []
        for rank, group in enumerate(kept, start=1):
            inside = [
                DataStructs.TanimotoSimilarity(fps[group[a]], fps[group[b]])
                for a in range(len(group))
                for b in range(a + 1, len(group))
            ]
            # The biggest member's scaffold stands in for the series, so every
            # row of one series carries the same answer in this column.
            scaffold = Chem.MolToSmiles(_scaffold(cast(Chem.Mol, mols[usable[group[0]]])))
            mean = round(sum(inside) / len(inside), 3) if inside else 1.0
            for i in group:
                row = usable[i]
                series[row], scaffolds[row] = rank, scaffold
                sizes[row], similarity[row] = len(group), mean
            atoms.append(
                _core_atoms(
                    [smiles[usable[i]] for i in group],
                    [ids[usable[i]] for i in group],
                    CLUSTER_TIMEOUT,
                )
            )
        baseline = _core_atoms(
            [smiles[n] for n in usable], [ids[n] for n in usable], WHOLE_SET_TIMEOUT
        )
        # The typical cluster, not the best one: one lucky cluster does not
        # justify splitting a set into eight.
        built = sorted(n for n in atoms if n)
        typical = built[len(built) // 2] if built else None
        gap = _separation(fps, kept)
        separated = len(kept) >= 2 and gap is not None and gap >= MIN_SEPARATION
        recommended = bool(
            separated and baseline and typical and typical / baseline >= MIN_CORE_GAIN
        )

    out = base.with_columns(
        pl.Series("series", series, dtype=pl.Int64),
        pl.Series("scaffold_smiles", scaffolds, dtype=pl.Utf8),
        pl.Series("series_size", sizes, dtype=pl.Int64),
        pl.Series("series_similarity", similarity, dtype=pl.Float64),
        pl.lit(baseline, dtype=pl.Int64).alias("whole_set_core_atoms"),
        pl.lit(typical, dtype=pl.Int64).alias("typical_series_core_atoms"),
        pl.lit(recommended, dtype=pl.Boolean).alias("split_recommended"),
    )
    return {"series": out}


# ---------------------------------------------------------------------------
# find_cores and score_core
# ---------------------------------------------------------------------------


def _groups(ids: Sequence[str], series: object) -> list[tuple[int | None, list[int]]]:
    """Which rows are searched together, in the order they are reported."""
    everything: list[tuple[int | None, list[int]]] = [(None, list(range(len(ids))))]
    if not isinstance(series, pl.DataFrame) or "series" not in series.columns:
        return everything
    # The clustering's own verdict decides, not the mere presence of clusters:
    # searching per series against its advice gave every candidate a coverage
    # over its own group that the card read as the whole set. Constant per row,
    # so the first row answers for the table; no column counts as no.
    if not (
        "split_recommended" in series.columns
        and series.height
        and bool(series["split_recommended"][0])
    ):
        return everything
    assigned = dict(zip(series["id"].to_list(), series["series"].to_list(), strict=False))
    per_row = [assigned.get(i) for i in ids]
    if all(s is None for s in per_row):
        return everything
    out: list[tuple[int | None, list[int]]] = [
        (int(s), [n for n, v in enumerate(per_row) if v == s])
        for s in sorted({v for v in per_row if v is not None})
    ]
    loose = [n for n, v in enumerate(per_row) if v is None]
    if len(loose) >= 2:
        out.append((None, loose))
    return out


def find_cores(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    frame = _frame(inputs)
    ids, smiles, mols = _mols(frame)
    candidates: list[dict[str, Any]] = []
    notes: list[str] = []
    searched: list[int] = []
    found_in: list[tuple[dict[str, Any], int | None, list[str]]] = []
    for group, rows in _groups(ids, inputs.get("series")):
        group_ids = [ids[n] for n in rows]
        found = sar.sar_variable.find_core_candidates(
            [smiles[n] for n in rows], group_ids, timeout=WHOLE_SET_TIMEOUT
        )
        prefix = "" if group is None else f"series {group}: "
        notes.extend(f"{prefix}{note}" for note in found["notes"])
        found_in.extend((c, group, group_ids) for c in found["candidates"])
        if group is not None:
            searched.append(group)
    # The ranks a candidate carries are within its own search; the list the
    # chemist reads is one list, so they are reassigned across the whole of it
    # — before the description is built, which is headed with the rank.
    for rank, (cand, group, group_ids) in enumerate(found_in, start=1):
        cand["rank"] = rank
        missed = set(cand["analysis"]["unmatched"])
        candidates.append(
            _candidate(cand, group, [i for i in group_ids if i not in missed]),
        )
    return {
        "candidates": {
            "meta": {
                "n_candidates": len(candidates),
                "n_compounds": frame.height,
                "n_unparsed": sum(m is None for m in mols),
                "notes": _notes(notes),
                "series": searched,
            },
            "candidates": candidates,
        }
    }


def score_core(params: CoreParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    frame = _frame(inputs)
    ids, smiles, mols = _mols(frame)
    report = sar.sar_edit.evaluate(params.core, smiles, ids)
    if report["error"]:
        raise ValueError(report["error"])
    # Nothing is filtered here: the chemist asked about this core, so a core
    # covering a fifth of the set is reported at a fifth, with the reason each
    # compound missed. The search filters; this answers.
    report["candidate"]["rank"] = 1
    cand = _candidate(report["candidate"], None, report["matched"])
    cand.update(
        variable={name: "/".join(info["elements"]) for name, info in report["variable"].items()},
        misses=[{"id": m["id"], "reason": m["reason"]} for m in report["misses"]],
        description=_readable(sar.sar_edit.summarize(report)),
    )
    return {
        "candidates": {
            "meta": {
                "n_candidates": 1,
                "n_compounds": frame.height,
                "n_unparsed": sum(m is None for m in mols),
                "notes": _notes(report["notes"]),
                "series": [],
            },
            "candidates": [cand],
        }
    }


# ---------------------------------------------------------------------------
# decompose_rgroups
# ---------------------------------------------------------------------------


def _query(smarts: str) -> Chem.Mol:
    """A SMARTS core made usable: MolFromSmarts leaves no ring info or valences."""
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"core SMARTS did not parse: {smarts!r}")
    query.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(query)
    return query


def decompose_rgroups(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    core = inputs["core"]
    assert isinstance(core, Mapping)
    smarts = str(core["smarts"])
    frame = _frame(inputs)
    ids, smiles, mols = _mols(frame)
    query = _query(smarts)

    rows, _unmatched = sar.sar_core.decompose(
        smiles, smarts, ids, exhaustive=True, separate_geminal=True
    )
    good = [(i, m) for i, m in zip(ids, mols, strict=True) if m is not None]
    analysis = sar.sar_variable.position_elements(
        [m for _, m in good], query, [i for i, _ in good]
    )
    assignments, x_columns = sar.sar_variable.variable_columns(analysis, [i for i, _ in good])
    rows = sar.sar_variable.attach_variable_columns(rows, assignments, x_columns, id_col="ID")

    r_columns = sar.sar_core.rgroup_columns(rows)
    columns = _positions([*r_columns, *x_columns])
    # An arbitrary embedding is what manufactures a matched pair that was never
    # made, and the vendored analysis already knows which rows sit either way
    # round, so the row carries the warning instead of the wrapper dropping it.
    arbitrary = set(analysis["ambiguous"])
    decomposed = pl.DataFrame(
        {
            "id": [str(r["ID"]) for r in rows],
            "decomposed": [True] * len(rows),
            "ambiguous": [str(r["ID"]) in arbitrary for r in rows],
            # Everything an R or X cell carries beyond the fragment itself is
            # the sibling UI's business (`_name`, `_sort`, `_display`), and
            # `Core` is the core we already have.
            **{c: [str(r.get(c, "") or "") or None for r in rows] for c in columns},
        },
        schema={
            "id": pl.Utf8,
            "decomposed": pl.Boolean,
            "ambiguous": pl.Boolean,
            **{c: pl.Utf8 for c in columns},
        },
    )
    out = (
        _with_id(_set_aside(frame, _DECOMPOSE_WRITES), ids)
        .with_columns(pl.col("id").cast(pl.Utf8))
        .join(decomposed, on="id", how="left")
        .with_columns(pl.col("decomposed", "ambiguous").fill_null(False))
        # The naming step rewrites R columns in place, so the fragment SMILES is
        # kept beside each one or naming could not be undone or repeated.
        .with_columns(pl.col(c).alias(f"{c}_smiles") for c in r_columns)
    )
    return {"table": out}


# ---------------------------------------------------------------------------
# name_substituents
# ---------------------------------------------------------------------------


@cache
def _abbreviations() -> dict[str, str]:
    """RDKit's built-in abbreviations, keyed on the fragment they stand for.

    Keyed on a canonical SMILES with the hydrogen bookkeeping reset, because
    the definitions write `*C(=O)[OH]` while a decomposition writes
    `OC(=O)[*:1]` — the same fragment, and no match without this. First label
    wins, so CO2Et beats its COOEt synonym.
    """
    out: dict[str, str] = {}
    for entry in rdAbbreviations.GetDefaultAbbreviations():
        mol = Chem.Mol(entry.mol)
        try:
            for atom in mol.GetAtoms():
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
            Chem.SanitizeMol(mol)
        except Exception:  # RDKit raises bare RuntimeErrors on odd valences
            continue
        out.setdefault(Chem.MolToSmiles(mol), entry.label)
    return out


def _word(fragment: str) -> str | None:
    """The word a chemist would write for this fragment, or None for a drawing."""
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return "H"
    if len(heavy) == 1:
        atom = heavy[0]
        if atom.GetSymbol() == "C":
            return "Me"
        count = atom.GetTotalNumHs(includeNeighbors=True)
        return atom.GetSymbol() + ("" if not count else "H" if count == 1 else f"H{count}")
    return _abbreviations().get(Chem.MolToSmiles(mol))


def name_substituents(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    table = _frame(inputs, "table")
    # Word or picture per cell, not per column: one exotic group kept a whole
    # column as SMILES, which is not something a chemist reads. A group with a
    # conventional short name gets the word; anything else gets drawn.
    for column in (c for c in _positions(table.columns) if c.startswith("R")):
        source = f"{column}_smiles"
        if source not in table.columns:
            continue
        fragments = table[source].to_list()
        named = [(v, None if not v else _word(v)) for v in fragments]
        table = table.with_columns(
            pl.Series(column, [word or fragment or None for fragment, word in named], pl.Utf8),
            pl.Series(
                f"{column}_svg",
                [
                    None
                    if not fragment or word
                    else sar.sar_core.render_fragment_svg(
                        fragment, 160, 120, attachment_label=column
                    )
                    for fragment, word in named
                ],
                pl.Utf8,
            ),
        )
    return {"table": table}


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "cluster_series": cluster_series,
    "find_cores": find_cores,
    "score_core": lambda p, i: score_core(cast(CoreParams, p), i),
    "decompose_rgroups": decompose_rgroups,
    "name_substituents": name_substituents,
}

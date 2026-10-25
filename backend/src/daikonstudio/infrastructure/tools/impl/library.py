"""Six RDKit tools for working over a set of compounds rather than a series.

The SAR tools already here assume a congeneric series and a core. These assume
nothing: they search, group, rank and annotate a table of compounds, which is what a
chemist does before a series exists and whenever one arrives from somewhere else.

Three conventions hold across all six, and each exists because the obvious version
is wrong in a way nobody notices.

Nothing is dropped silently. A search annotates the table and says how many matched
rather than returning only the hits, so a step that found nothing is visible as a
count of zero instead of an empty table that reads like a failure. The one exception
is the diverse pick, whose entire purpose is to return a subset, and it reports the
denominator it picked from.

A molecule that will not parse is a row with a null, never a dropped row and never a
crash. Row order is the caller's and is never changed, because a scientist's table is
ordered the way they left it and a later tool may join on position.

Fingerprints are built once per table and reused. The Morgan generator with a count
fingerprint is the current RDKit API; the deprecated free functions produce different
numbers, so a similarity reported by two tools built at different times would not
match.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any, cast

import polars as pl
from pydantic import BaseModel
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters import rdSimDivPickers

from daikonstudio.application.studies.readers import _IDENTIFIER
from daikonstudio.infrastructure.tools.impl.chem import ToolImpl, _frame, structure_column
from daikonstudio.infrastructure.tools.params import (
    ClusterParams,
    DiverseParams,
    SimilarityParams,
    SubstructureParams,
)

# Radius 2 over 2048 bits: the default a chemist means by "Morgan" or "ECFP4", and the
# one the chemical-space map already uses, so a similarity here and a distance there
# are the same measurement.
_FINGERPRINTS = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# How many identifiers a `meta` list carries. The agent reads `meta` for free but only
# gets 50 rows from the top of a table on request, so a tool that answers "which ones"
# has to answer it here or not at all: on a 3,000-row table the matching rows are almost
# never in the first 50. Capped because `meta` is in every prompt that mentions the step.
_NAMED_MAX = 25


def _names(frame: pl.DataFrame) -> list[str]:
    """Each row's identifier, by the same convention the study readers use, else its number."""
    for name in frame.columns:
        if _IDENTIFIER.match(name):
            return [str(v) for v in frame[name].to_list()]
    return [f"row {i + 1}" for i in range(frame.height)]


def _named(frame: pl.DataFrame, rows: list[int]) -> dict[str, Any]:
    """The identifiers of `rows`, capped, with the count so a truncated list says so."""
    names = _names(frame)
    return {
        "n": len(rows),
        "ids": [names[i] for i in rows[:_NAMED_MAX]],
        "truncated": len(rows) > _NAMED_MAX,
    }


def _molecules(frame: pl.DataFrame) -> tuple[str, list[Chem.Mol | None]]:
    """The structure column's name and one molecule per row, None where it would not parse."""
    column = structure_column(frame)
    return column, [
        Chem.MolFromSmiles(smiles) if smiles is not None else None
        for smiles in frame[column].to_list()
    ]


def _fingerprints(molecules: list[Chem.Mol | None]) -> dict[int, Any]:
    """Row number to fingerprint, for the rows that parsed. Absent means unparseable."""
    return {
        row: _FINGERPRINTS.GetFingerprint(mol)
        for row, mol in enumerate(molecules)
        if mol is not None
    }


def substructure_search(
    params: SubstructureParams, inputs: Mapping[str, object]
) -> dict[str, Any]:
    """Flag the rows containing a substructure, given as SMARTS."""
    frame = _frame(inputs)
    query = Chem.MolFromSmarts(params.smarts)
    if query is None:
        # A bad SMARTS is the scientist's typo, not a server error, and the message has to
        # name what failed: `c1ccccc1(` looks like a ring to everyone but the parser.
        raise ValueError(f"Not a valid SMARTS pattern: {params.smarts!r}")
    _, molecules = _molecules(frame)
    matched = [None if mol is None else mol.HasSubstructMatch(query) for mol in molecules]
    out = frame.with_columns(pl.Series("matches_query", matched, dtype=pl.Boolean))
    return {
        "compounds": out,
        "meta": {
            "smarts": params.smarts,
            "n_matched": sum(1 for m in matched if m),
            "n_rows": frame.height,
            "n_unparsed": sum(1 for m in matched if m is None),
            # Which ones, named here because the rows that matched are rarely in the first
            # 50, and "40 matched" that cannot name one is not an answer.
            "matched": _named(frame, [i for i, m in enumerate(matched) if m]),
        },
    }


def similarity_search(params: SimilarityParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Score every compound against a query structure by Tanimoto, most similar first."""
    frame = _frame(inputs)
    query = Chem.MolFromSmiles(params.query_smiles)
    if query is None:
        raise ValueError(f"Not a valid SMILES for the query: {params.query_smiles!r}")
    reference = _FINGERPRINTS.GetFingerprint(query)
    _, molecules = _molecules(frame)
    fingerprints = _fingerprints(molecules)
    scores = [
        DataStructs.TanimotoSimilarity(reference, fingerprints[row])
        if row in fingerprints
        else None
        for row in range(frame.height)
    ]
    # Sorted, because a ranking nobody can read top-down is not a ranking. Nulls last so an
    # unparsed row never occupies the top of the list.
    out = frame.with_columns(pl.Series("tanimoto", scores, dtype=pl.Float64)).sort(
        "tanimoto", descending=True, nulls_last=True
    )
    above = [s for s in scores if s is not None and s >= params.cutoff]
    return {
        "compounds": out,
        "meta": {
            "query_smiles": params.query_smiles,
            "cutoff": params.cutoff,
            "n_at_or_above_cutoff": len(above),
            "n_rows": frame.height,
            "best": max((s for s in scores if s is not None), default=None),
            # The table is sorted, so the top of a look is already the answer. Named anyway
            # with their scores, so the neighbours survive into a later turn that only has
            # the summary.
            "nearest": [
                {"id": name, "tanimoto": round(score, 3)}
                for name, score in zip(_names(out), out["tanimoto"].to_list(), strict=True)
                if score is not None and score >= params.cutoff
            ][:_NAMED_MAX],
        },
    }


def cluster_compounds(params: ClusterParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Group compounds by Taylor-Butina at a Tanimoto distance cutoff."""
    frame = _frame(inputs)
    _, molecules = _molecules(frame)
    fingerprints = _fingerprints(molecules)
    usable = sorted(fingerprints)
    # Butina takes the lower triangle of a distance matrix as a flat list, in row order.
    distances: list[float] = []
    for row, i in enumerate(usable):
        for j in usable[:row]:
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            distances.append(1.0 - similarity)
    clusters = (
        Butina.ClusterData(  # type: ignore[no-untyped-call]
            distances, len(usable), params.cutoff, isDistData=True
        )
        if usable
        else ()
    )
    # Cluster 1 is the largest, because Butina returns them that way and a scientist reading
    # "cluster 1" means the biggest one. The first member of each is its centroid.
    cluster_of: dict[int, int] = {}
    centroids: set[int] = set()
    for number, members in enumerate(clusters, start=1):
        for position, member in enumerate(members):
            cluster_of[usable[member]] = number
            if position == 0:
                centroids.add(usable[member])
    out = frame.with_columns(
        pl.Series("cluster", [cluster_of.get(i) for i in range(frame.height)], dtype=pl.Int64),
        pl.Series(
            "is_centroid",
            [None if i not in cluster_of else i in centroids for i in range(frame.height)],
            dtype=pl.Boolean,
        ),
    )
    sizes = [len(members) for members in clusters]
    names = _names(frame)
    return {
        "compounds": out,
        "meta": {
            "cutoff": params.cutoff,
            "n_clusters": len(sizes),
            "n_singletons": sum(1 for size in sizes if size == 1),
            "largest_cluster": max(sizes, default=0),
            "n_rows": frame.height,
            "n_unparsed": frame.height - len(usable),
            # One line per cluster: its size and the compound that represents it. This is
            # the result; the column of cluster numbers is only how it is carried.
            "clusters": [
                {
                    "cluster": number,
                    "size": len(members),
                    "centroid": names[usable[members[0]]],
                }
                for number, members in enumerate(clusters[:_NAMED_MAX], start=1)
            ],
        },
    }


def pick_diverse_subset(params: DiverseParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Pick a maximally dissimilar subset, the MaxMin way."""
    frame = _frame(inputs)
    _, molecules = _molecules(frame)
    fingerprints = _fingerprints(molecules)
    usable = sorted(fingerprints)
    wanted = min(params.n, len(usable))
    if wanted <= 0:
        return {
            "compounds": frame.head(0),
            "meta": {"requested": params.n, "picked": 0, "from_rows": len(usable)},
        }
    kept = [fingerprints[i] for i in usable]

    def distance(i: int, j: int) -> float:
        return 1.0 - DataStructs.TanimotoSimilarity(kept[i], kept[j])

    # seed=0, so the same table picked twice gives the same subset. MaxMin starts from a
    # random point, and an unseeded pick that changes between runs cannot be reported.
    picks = rdSimDivPickers.MaxMinPicker().LazyPick(distance, len(kept), wanted, seed=0)
    rows = sorted(usable[p] for p in picks)
    return {
        "compounds": frame[rows],
        "meta": {
            "requested": params.n,
            "picked": len(rows),
            "from_rows": len(usable),
            "n_unparsed": frame.height - len(usable),
        },
    }


def _grouped(values: list[str | None]) -> list[int | None]:
    """Number the distinct values, commonest first, so rows can be compared without them.

    A scaffold is a structure, so the codec withholds the column from the agent -- rightly,
    since structures do not belong in its context. But the question a scaffold answers is
    "which compounds group together", and that is an identity, not a drawing. Without a
    number beside the string the agent can see that two groupings differ and cannot say
    where, which is the one thing the comparison was for.

    None for a row with no scaffold at all, and for an acyclic molecule, whose scaffold is
    the empty string: neither shares a chemotype with anything, and giving them a shared id
    would invent a group that is really a pile of leftovers.
    """
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    order = sorted(counts, key=lambda value: (-counts[value], value))
    number = {value: i for i, value in enumerate(order, start=1)}
    return [number.get(value) if value else None for value in values]


def _largest(groups: list[int | None]) -> list[tuple[int, int, list[int]]]:
    """Each group id with its size and its rows, biggest first, capped."""
    rows_by_group: dict[int, list[int]] = {}
    for row, group in enumerate(groups):
        if group is not None:
            rows_by_group.setdefault(group, []).append(row)
    ranked = sorted(rows_by_group.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    return [(group, len(rows), rows) for group, rows in ranked[:_NAMED_MAX]]


def murcko_scaffolds(_params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    """The Bemis-Murcko scaffold of each compound, and the generic ring skeleton beside it."""
    frame = _frame(inputs)
    _, molecules = _molecules(frame)
    scaffolds: list[str | None] = []
    generic: list[str | None] = []
    for mol in molecules:
        if mol is None:
            scaffolds.append(None)
            generic.append(None)
            continue
        core = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call]
        scaffolds.append(Chem.MolToSmiles(core))
        # The generic form replaces every atom with carbon and every bond with single, which
        # is what groups two chemotypes that differ only by a ring heteroatom.
        try:
            plain = MurckoScaffold.MakeScaffoldGeneric(core)  # type: ignore[no-untyped-call]
            generic.append(Chem.MolToSmiles(plain))
        # RDKit raises bare on some valences here; a null is the answer, not a failed step.
        except Exception:
            generic.append(None)
    out = frame.with_columns(
        pl.Series("scaffold", scaffolds, dtype=pl.String),
        pl.Series("scaffold_generic", generic, dtype=pl.String),
        pl.Series("scaffold_id", _grouped(scaffolds), dtype=pl.Int64),
        pl.Series("scaffold_generic_id", _grouped(generic), dtype=pl.Int64),
    )
    named = [s for s in scaffolds if s is not None]
    names_of = _names(frame)
    # An empty string is what RDKit returns for an acyclic molecule: it has no scaffold,
    # which is a finding (nothing to vary around) rather than a failure.
    return {
        "compounds": out,
        "meta": {
            "n_distinct_scaffolds": len({s for s in named if s}),
            "n_acyclic": sum(1 for s in named if s == ""),
            "n_rows": frame.height,
            "n_unparsed": frame.height - len(named),
            # The biggest chemotypes by scaffold, each with one compound that carries it.
            # The scaffold strings themselves are withheld, so the id and an example are
            # what make the grouping nameable.
            "largest_groups": [
                {"scaffold_id": group, "size": size, "example": names_of[rows[0]]}
                for group, size, rows in _largest(out["scaffold_id"].to_list())
            ],
        },
    }


def _sascorer() -> Any:
    """RDKit ships the scorer as a Contrib script, not an importable module."""
    path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if path not in sys.path:
        sys.path.append(path)
    import sascorer  # type: ignore[import-not-found]

    return sascorer


def synthetic_accessibility(
    _params: BaseModel | None, inputs: Mapping[str, object]
) -> dict[str, Any]:
    """Ertl-Schuffenhauer synthetic accessibility, 1 (easy) to 10 (hard)."""
    frame = _frame(inputs)
    scorer = _sascorer()
    _, molecules = _molecules(frame)
    scores = [None if mol is None else float(scorer.calculateScore(mol)) for mol in molecules]
    out = frame.with_columns(pl.Series("sa_score", scores, dtype=pl.Float64))
    named = [s for s in scores if s is not None]
    return {
        "properties": out,
        "meta": {
            "n_rows": frame.height,
            "n_unparsed": frame.height - len(named),
            "median": sorted(named)[len(named) // 2] if named else None,
            # The hardest few by name. A score column nobody can sort to is not an answer
            # to "does anything here look hard to make".
            "hardest": [
                {"id": name, "sa_score": round(score, 2)}
                for name, score in sorted(
                    ((n, s) for n, s in zip(_names(frame), scores, strict=True) if s is not None),
                    key=lambda pair: -pair[1],
                )[:10]
            ],
            # Said here rather than left to the reader: the score is a fragment-frequency
            # prior, so a default threshold is a convention and not a verdict on whether
            # anyone can make the molecule.
            "note": (
                "A fragment-frequency estimate, not a route. Compare within this table "
                "rather than against a fixed cutoff."
            ),
        },
    }


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "substructure_search": lambda p, i: substructure_search(cast(SubstructureParams, p), i),
    "similarity_search": lambda p, i: similarity_search(cast(SimilarityParams, p), i),
    "cluster_compounds": lambda p, i: cluster_compounds(cast(ClusterParams, p), i),
    "pick_diverse_subset": lambda p, i: pick_diverse_subset(cast(DiverseParams, p), i),
    "murcko_scaffolds": murcko_scaffolds,
    "synthetic_accessibility": synthetic_accessibility,
}

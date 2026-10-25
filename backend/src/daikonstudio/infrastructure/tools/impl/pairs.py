"""The matched-pair tools, over the R-group table.

Everything here reads one table whose pairing is already done: two rows with the
same R-vector but for one position are a matched pair, so no substructure search
is needed. The activity is read as an interval rather than a number, which is
what keeps a censored reading (">64") usable: two bounds overlap and guarantee
nothing, a bound against an exact value guarantees a floor.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import polars as pl
from pydantic import BaseModel

from daikonstudio.infrastructure.tools.params import AssayParams, CliffParams
from daikonstudio.infrastructure.tools.vendor import sar

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

# R1, R2 … are the substituents; X1, X2 … the variable ring atoms.
_POSITION = re.compile(r"^[RX]\d+$")

# A gap this small is the floor's own rounding, not a difference (sar_cliffs uses
# the same tolerance so that a stated fourfold gate accepts an exactly fourfold jump).
_TOLERANCE = 1e-9

_PAIR_SCHEMA: dict[str, Any] = {
    "id": pl.Utf8,
    "position": pl.Utf8,
    "id_better": pl.Utf8,
    "id_worse": pl.Utf8,
    "from": pl.Utf8,
    "to": pl.Utf8,
    "delta": pl.Float64,
    "fold": pl.Utf8,
    "bounded": pl.Boolean,
    "ordered": pl.Boolean,
    "activity_better": pl.Utf8,
    "activity_worse": pl.Utf8,
    "measurements_better": pl.Int64,
    "measurements_worse": pl.Int64,
}

_EFFECT_SCHEMA: dict[str, Any] = {
    "id": pl.Utf8,
    "position": pl.Utf8,
    "from": pl.Utf8,
    "to": pl.Utf8,
    "direction": pl.Utf8,
    "n_pairs": pl.Int64,
    "mean_delta": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "n_bounded": pl.Int64,
    "n_unordered": pl.Int64,
}

# A chemist reads the first handful; the product of every level at every position
# is combinatorial, so it is cut off well before it stops being a list.
_MAX_COMBINATIONS = 5000
_TOP_UNMADE = 6


def _frame(inputs: Mapping[str, object], key: str) -> pl.DataFrame:
    value = inputs[key]
    assert isinstance(value, pl.DataFrame)
    return value


def _semantics(params: AssayParams) -> dict[str, Any]:
    """What the vendored `to_interval` and `find_cliffs` need to read the column.

    Always passed in full: left to guess from the header, `find_cliffs` can read a
    percentage as a fold change or run the direction backwards.
    """
    return {
        "kind": params.kind,
        "higher_is_better": params.higher_is_better,
        "unit": params.unit or "",
    }


def _require_activity(table: pl.DataFrame, params: AssayParams) -> None:
    """Fail the step when the activity column is not there.

    Read off a column that does not exist, every row is `None`, every compound is
    skipped and the table comes back empty — with the cause stated nowhere, since
    `pairs` carries no meta to put a note in.
    """
    if params.activity_column not in table.columns:
        raise ValueError(
            f"No column {params.activity_column!r} in the table. It has: "
            f"{', '.join(table.columns)}"
        )


def _positions(table: pl.DataFrame) -> list[str]:
    return [name for name in table.columns if _POSITION.match(name)]


def _decomposed(table: pl.DataFrame) -> pl.DataFrame:
    return table.filter(pl.col("decomposed")) if "decomposed" in table.columns else table


def _rows(table: pl.DataFrame, params: AssayParams) -> list[dict[str, Any]]:
    """The decomposed rows as `find_cliffs` wants them.

    Per position it groups on `{col}_smiles` (the fragment, or the element at an X
    position) and labels with `{col}` — the substituent's word once the table has
    been named, and the fragment itself before that.
    """
    identifier = table.columns[0]  # the rgroup table's contract: the id comes first
    positions = _positions(table)
    out: list[dict[str, Any]] = []
    for row in _decomposed(table).iter_rows(named=True):
        built: dict[str, Any] = {
            "id": str(row[identifier]),
            params.activity_column: row.get(params.activity_column),
        }
        for column in positions:
            cell = row[column]
            smiles = row.get(f"{column}_smiles", cell)
            built[column] = cell
            built[f"{column}_smiles"] = smiles
            if cell != smiles:
                built[f"{column}_name"] = cell
        if "duplicates" in row:
            built["duplicates"] = row["duplicates"]
        out.append(built)
    return out


def matched_pairs(params: AssayParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    table = _frame(inputs, "table")
    _require_activity(table, params)
    semantics = _semantics(params)
    rows = _rows(table, params)
    measurements = {
        row["id"]: row["duplicates"] for row in rows if row.get("duplicates") is not None
    }
    found = sar.sar_cliffs.find_cliffs(
        rows,
        _positions(table),
        params.activity_column,
        min_delta=0.0,
        id_col="id",
        **semantics,
    )
    out = [_pair(pair, semantics, measurements) for pair in found["cliffs"]]
    return {"pairs": pl.DataFrame(out, schema=_PAIR_SCHEMA)}


def _pair(
    pair: Mapping[str, Any], semantics: Mapping[str, Any], measurements: Mapping[str, int]
) -> dict[str, Any]:
    better = sar.sar_cliffs.to_interval(pair["activity_better"], semantics)
    worse = sar.sar_cliffs.to_interval(pair["activity_worse"], semantics)
    gap = float(sar.sar_cliffs.interval_gap(better, worse))
    censored = sar.sar_cliffs.is_bounded(better) or sar.sar_cliffs.is_bounded(worse)
    # Nothing is guaranteed when the intervals overlap and one of them is open: two
    # readings of ">64" are not equally active, nobody knows which is better. Two
    # exact equal readings are a measured zero, which is a finding and stays 0.0.
    # The gap, recomputed rather than read off the finding: `find_cliffs` rounds it
    # to three places for display, and a stated fourfold gate has to hold exactly.
    delta = None if (censored and gap <= _TOLERANCE) else gap
    # A pair that guarantees nothing is unordered, and nothing in its row may say
    # otherwise: the two compounds go in by id ascending rather than by activity,
    # the id joins them with "|", and the vendored `fold` goes — ">=1.0x" reads as
    # an improvement when the open bound could run either way.
    ordered = delta is not None
    low, high = "worse", "better"  # the from-side and the to-side
    if not ordered and pair["id_worse"] > pair["id_better"]:
        low, high = "better", "worse"
    id_low, id_high = pair[f"id_{low}"], pair[f"id_{high}"]
    return {
        "id": f"{id_low}>{id_high}" if ordered else f"{id_low}|{id_high}",
        "position": pair["position"],
        "id_better": id_high,
        "id_worse": id_low,
        "from": pair[f"label_{low}"],
        "to": pair[f"label_{high}"],
        "delta": delta,
        "fold": pair["fold"] if ordered else None,
        "bounded": bool(pair["bounded"]),
        "ordered": ordered,
        "activity_better": _text(pair[f"activity_{high}"]),
        "activity_worse": _text(pair[f"activity_{low}"]),
        "measurements_better": measurements.get(id_high),
        "measurements_worse": measurements.get(id_low),
    }


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def activity_cliffs(params: CliffParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Every pair, with the ones over the floor flagged and ranked.

    The whole table comes back rather than the cliffs alone, so a step reading the
    pairs afterwards still sees the series it came from.
    """
    pairs = _frame(inputs, "pairs")
    flagged = pairs.with_columns(
        (pl.col("delta").fill_null(-math.inf) >= params.min_delta - _TOLERANCE).alias("cliff")
    ).sort(["cliff", "delta"], descending=True, nulls_last=True)
    return {
        "cliffs": flagged.with_columns(
            pl.when(pl.col("cliff"))
            .then(pl.col("cliff").cum_sum().cast(pl.Int64))
            .alias("cliff_rank")
        )
    }


def effect_sizes(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    """What each substituent change is worth on average, with a bootstrap interval.

    A change and its reverse are the same change: both are read in one direction
    (alphabetically), negating the difference on the ones that were the other way
    round, so Me->Cl and Cl->Me are not reported as two unrelated findings.
    """
    pairs = _frame(inputs, "pairs")
    flip = pl.col("from") > pl.col("to")
    oriented = pairs.select(
        "position",
        pl.when(flip).then(pl.col("to")).otherwise(pl.col("from")).alias("from"),
        pl.when(flip).then(pl.col("from")).otherwise(pl.col("to")).alias("to"),
        pl.when(flip).then(-pl.col("delta")).otherwise(pl.col("delta")).alias("delta"),
        "bounded",
    )
    rng = np.random.default_rng(0)
    out: list[dict[str, Any]] = []
    # An unordered pair has no delta — it guarantees nothing about the direction —
    # but it is still a pair of this change, and hiding it made `n_pairs` disagree
    # with the pairs table the chemist is reading beside it. The three counts
    # (exact, bounded, unordered) add up to `n_pairs`.
    is_exact = ~pl.col("bounded") & pl.col("delta").is_not_null()
    grouped = oriented.group_by("position", "from", "to").agg(
        pl.len().alias("n_pairs"),
        pl.col("delta").filter(is_exact).alias("exact"),
        (pl.col("bounded") & pl.col("delta").is_not_null()).sum().alias("n_bounded"),
        pl.col("delta").is_null().sum().alias("n_unordered"),
    )
    for group in grouped.sort("position", "from", "to").iter_rows(named=True):
        # A bounded pair's delta is a *lower* bound on the improvement, and the
        # alphabetical orientation negates half of them into upper bounds, so the
        # mean rests on the exact pairs alone. The counts say what it rests on.
        exact = np.array(group["exact"], dtype=float)
        mean_delta = float(exact.mean()) if len(exact) else None
        low, high = _interval(exact, rng)
        out.append(
            {
                "id": f"{group['position']} {group['from']}>{group['to']}",
                "position": group["position"],
                "from": group["from"],
                "to": group["to"],
                "direction": _direction(mean_delta),
                "n_pairs": int(group["n_pairs"]),
                "mean_delta": mean_delta,
                "ci_low": low,
                "ci_high": high,
                "n_bounded": int(group["n_bounded"]),
                "n_unordered": int(group["n_unordered"]),
            }
        )
    # The changes the data can carry first, and among those the ones that matter most.
    effects = pl.DataFrame(out, schema=_EFFECT_SCHEMA).sort(
        ["n_pairs", pl.col("mean_delta").abs()],
        descending=True,
        nulls_last=True,
        maintain_order=True,
    )
    return {"effects": effects}


def _direction(mean_delta: float | None) -> str | None:
    """The finding in words — on a table oriented alphabetically, a sign is not one."""
    if not mean_delta:  # no exact pair behind it, or a measured zero
        return None
    return "to is better" if mean_delta > 0 else "from is better"


def _interval(deltas: np.ndarray, rng: np.random.Generator) -> tuple[float | None, float | None]:
    """A percentile bootstrap, or nothing: two pairs cannot carry an interval."""
    if len(deltas) < 3:
        return None, None
    picks = rng.integers(0, len(deltas), size=(1000, len(deltas)))
    means = deltas[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _records(
    table: pl.DataFrame, params: AssayParams
) -> tuple[list[tuple[str, tuple[str, ...], float]], int, int]:
    """The compounds the fit can use, and how many were dropped, and why.

    A bound is not a measurement: a least-squares fit cannot carry ">64" without
    turning it into 64, so those rows are left out and counted rather than invented.
    A null at a position is not a substituent either: `str(None)` fits a term for a
    group called "None", so the compound is left out and counted the same way.
    """
    semantics = _semantics(params)
    positions = _positions(table)
    records: list[tuple[str, tuple[str, ...], float]] = []
    censored = 0
    incomplete = 0
    for row in _rows(table, params):
        interval = sar.sar_cliffs.to_interval(row.get(params.activity_column), semantics)
        if interval is None:
            continue
        if sar.sar_cliffs.is_bounded(interval):
            censored += 1
            continue
        cells = [row[column] for column in positions]
        if any(cell is None for cell in cells):
            incomplete += 1
            continue
        vector = tuple(str(cell) for cell in cells)
        records.append((row["id"], vector, float(sar.sar_cliffs.point_of(interval))))
    return records, censored, incomplete


def free_wilson(params: AssayParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    """Activity as a sum of per-substituent contributions.

    Ported from sar-workbench's `extras/freewilson.py`. How well it fits is the
    finding: where it holds the positions are independent and can be optimized
    separately, where it fails they interact.
    """
    table = _frame(inputs, "table")
    _require_activity(table, params)
    columns = _positions(table)
    records, censored, incomplete = _records(table, params)
    meta: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "activity_column": params.activity_column,
        "n_compounds": len(records),
        "n_excluded_censored": censored,
        "n_excluded_incomplete": incomplete,
        "n_terms": 0,
        "r2": None,
        "q2": None,
        "reference": {},
        "varying_positions": [],
    }
    if not records or not columns:
        meta["reason"] = (
            f"No compound has an exact value in {params.activity_column!r} at a "
            f"decomposed position"
        )
        return {"fit": {"meta": meta, "terms": [], "fitted": [], "unmade": []}}
    if len(records) < 3:
        meta["reason"] = (
            "Model fitting requires at least three compounds with exact activity values."
        )
        return {"fit": {"meta": meta, "terms": [], "fitted": [], "unmade": []}}

    counts = {col: Counter(vec[i] for _, vec, _ in records) for i, col in enumerate(columns)}
    # The most common group at each position is the baseline; ties go alphabetically.
    reference = {col: max(sorted(counts[col]), key=lambda k: counts[col][k]) for col in columns}
    terms = [(col, key) for col in columns for key in sorted(counts[col]) if key != reference[col]]
    meta["n_terms"] = len(terms)
    meta["reference"] = reference
    if len(terms) + 1 >= len(records):
        meta["reason"] = (
            f"Model fitting was not performed: {len(records)} compounds are insufficient to "
            f"estimate {len(terms)} substituent coefficients plus an intercept with "
            "residual degrees of freedom."
        )
        return {"fit": {"meta": meta, "terms": [], "fitted": [], "unmade": []}}

    index = {term: n + 1 for n, term in enumerate(terms)}

    def design(vectors: Sequence[tuple[str, ...]]) -> np.ndarray:
        matrix = np.zeros((len(vectors), len(terms) + 1))
        matrix[:, 0] = 1.0
        for row, vector in enumerate(vectors):
            for i, col in enumerate(columns):
                position = index.get((col, vector[i]))
                if position:
                    matrix[row, position] = 1.0
        return matrix

    design_matrix = design([vec for _, vec, _ in records])
    y = np.array([value for _, _, value in records])
    beta = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    fitted = design_matrix @ beta
    ss_total = float(((y - y.mean()) ** 2).sum())
    press = 0.0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        left_out = np.linalg.lstsq(design_matrix[keep], y[keep], rcond=None)[0]
        press += float((y[i] - design_matrix[i] @ left_out) ** 2)
    if ss_total > 0:
        meta["r2"] = round(1 - float(((y - fitted) ** 2).sum()) / ss_total, 3)
        meta["q2"] = round(1 - press / ss_total, 3)

    made = {vec for _, vec, _ in records}
    levels = [[key for key, n in counts[col].items() if n >= 2] for col in columns]
    meta["varying_positions"] = [columns[i] for i, keys in enumerate(levels) if len(keys) > 1]
    combinations = (v for v in itertools.product(*levels) if v not in made)
    unmade = sorted(
        (
            (float(design([v])[0] @ beta), v)
            for v in itertools.islice(combinations, _MAX_COMBINATIONS)
        ),
        key=lambda pair: -pair[0],
    )
    meta["ok"] = True
    return {
        "fit": {
            "meta": meta,
            "terms": [
                {
                    "position": col,
                    "group": key,
                    "n": n,
                    "coef": 0.0
                    if key == reference[col]
                    else round(float(beta[index[(col, key)]]), 3),
                    "reference": key == reference[col],
                }
                for col in columns
                for key, n in counts[col].most_common()
            ],
            "fitted": [
                {"id": record[0], "fitted": round(float(p), 3)}
                for record, p in zip(records, fitted, strict=True)
            ],
            "unmade": [
                {"groups": dict(zip(columns, v, strict=True)), "prediction": round(p, 3)}
                for p, v in unmade[:_TOP_UNMADE]
            ],
        }
    }


def coverage_gaps(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    """The combinations of substituents the series never made.

    Most surrounded first: a gap whose every neighbour exists is one compound away
    from being decided, and is the one worth making.
    """
    table = _frame(inputs, "table")
    rows = _decomposed(table)
    columns = _positions(table)
    # A null is not a substituent: grouped by `str`, an unassigned position becomes
    # the level "None" and gets offered as a compound to make.
    groups = {
        col: Counter(str(v) for v in rows[col].to_list() if v is not None) for col in columns
    }
    varying = [col for col in columns if len(groups[col]) > 1]
    schema: dict[str, Any] = {
        "id": pl.Utf8,
        **dict.fromkeys(varying, pl.Utf8),
        "nearest_made": pl.Utf8,
        "n_neighbours": pl.Int64,
    }
    if not varying:
        return {"gaps": pl.DataFrame([], schema=schema)}

    made: dict[tuple[str, ...], str] = {}
    for row in rows.select(table.columns[0], *varying).iter_rows():
        if any(v is None for v in row[1:]):
            continue  # no level of its own, so it is neither a gap filled nor a neighbour
        made.setdefault(tuple(str(v) for v in row[1:]), str(row[0]))
    # Only groups the series made twice: a one-off substituent says nothing about
    # what combining it would do, and squaring the levels on it buries the real gaps.
    levels = [[key for key, n in groups[col].items() if n >= 2] for col in varying]
    out: list[dict[str, Any]] = []
    for vector in itertools.islice(itertools.product(*levels), _MAX_COMBINATIONS):
        if vector in made:
            continue
        neighbours = [
            identifier
            for other, identifier in made.items()
            if sum(a != b for a, b in zip(vector, other, strict=True)) == 1
        ]
        out.append(
            {
                "id": "|".join(vector),
                **dict(zip(varying, vector, strict=True)),
                # Nearest by how many positions differ, so a gap with no neighbour
                # still says what it is closest to; the first such row wins the tie.
                "nearest_made": min(neighbours) if neighbours else _nearest(vector, made),
                "n_neighbours": len(neighbours),
            }
        )
    return {
        "gaps": pl.DataFrame(out, schema=schema).sort(
            "n_neighbours", descending=True, maintain_order=True
        )
    }


def _nearest(vector: tuple[str, ...], made: Mapping[tuple[str, ...], str]) -> str | None:
    def distance(other: tuple[str, ...]) -> int:
        return sum(a != b for a, b in zip(vector, other, strict=True))

    closest = min(made, key=distance, default=None)
    return None if closest is None else made[closest]


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "matched_pairs": lambda p, i: matched_pairs(cast(AssayParams, p), i),
    "effect_sizes": effect_sizes,
    "activity_cliffs": lambda p, i: activity_cliffs(cast(CliffParams, p), i),
    "free_wilson": lambda p, i: free_wilson(cast(AssayParams, p), i),
    "coverage_gaps": coverage_gaps,
}

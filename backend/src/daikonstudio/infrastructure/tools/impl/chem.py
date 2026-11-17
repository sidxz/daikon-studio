"""The four RDKit tools: pure functions over polars frames."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, cast

import numpy as np
import polars as pl
from pydantic import BaseModel
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize

from daikonstudio.application.studies.readers import _IDENTIFIER
from daikonstudio.infrastructure.tools.measurements import read_measurements
from daikonstudio.infrastructure.tools.params import ActivityParams, ConfoundParams

ToolImpl = Callable[[BaseModel | None, Mapping[str, object]], Mapping[str, object]]

# Unparseable input is reported in a column, not on stderr once per row.
# ignore[attr-defined]: installed rdkit-stubs omit DisableLog; RDKit itself has it.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

PROPERTY_COLUMNS = (
    "mol_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "fsp3",
    "rotatable_bonds",
    "heavy_atoms",
)
# ignore[attr-defined]: installed rdkit-stubs omit these generated descriptor functions.
_DESCRIPTORS = (
    Descriptors.MolWt,  # type: ignore[attr-defined]
    Descriptors.MolLogP,  # type: ignore[attr-defined]
    Descriptors.TPSA,  # type: ignore[attr-defined]
    Descriptors.NumHDonors,  # type: ignore[attr-defined]
    Descriptors.NumHAcceptors,  # type: ignore[attr-defined]
    Descriptors.FractionCSP3,  # type: ignore[attr-defined]
    Descriptors.NumRotatableBonds,  # type: ignore[attr-defined]
    Descriptors.HeavyAtomCount,
)


def _frame(inputs: Mapping[str, object], key: str = "compounds") -> pl.DataFrame:
    value = inputs[key]
    assert isinstance(value, pl.DataFrame)
    return value


def structure_column(frame: pl.DataFrame) -> str:
    """The first text column at least half of whose first 100 values parse as SMILES."""
    # standardize writes canonical_smiles; the raw column stays beside it.
    if "canonical_smiles" in frame.columns:
        return "canonical_smiles"
    tried = []
    for name, dtype in frame.schema.items():
        if dtype != pl.Utf8:
            continue
        tried.append(name)
        values = frame[name].drop_nulls().head(100).to_list()
        if values and sum(Chem.MolFromSmiles(v) is not None for v in values) >= len(values) / 2:
            return name
    raise ValueError(f"No column parses as SMILES; tried: {', '.join(tried)}")


def id_column(frame: pl.DataFrame) -> str:
    """The first column named like an identifier; callers wanting a fallback add a row number."""
    for name in frame.columns:
        if _IDENTIFIER.match(name):
            return name
    raise ValueError(f"No column is named like an identifier; tried: {', '.join(frame.columns)}")


def standardize(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    frame = _frame(inputs)
    column = structure_column(frame)
    uncharger = rdMolStandardize.Uncharger()
    tautomers = rdMolStandardize.TautomerEnumerator()
    canonical: list[str | None] = []
    changed: list[bool] = []
    errors: list[str | None] = []
    for smiles in frame[column].to_list():
        mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
        if mol is None:
            canonical.append(None)
            changed.append(False)
            errors.append("Empty structure" if smiles is None else f"Cannot parse {smiles!r}")
            continue
        try:
            std = tautomers.Canonicalize(
                uncharger.uncharge(rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol)))
            )
            out = Chem.MolToSmiles(std)
            errors.append(None)
        except Exception as error:  # RDKit raises bare RuntimeErrors on odd valences
            canonical.append(None)
            changed.append(False)
            errors.append(str(error))
            continue
        canonical.append(out)
        changed.append(out != Chem.MolToSmiles(mol))
    out_frame = frame.with_columns(
        pl.Series("canonical_smiles", canonical, dtype=pl.Utf8),
        pl.Series("changed", changed, dtype=pl.Boolean),
        pl.Series("parse_error", errors, dtype=pl.Utf8),
    )
    # Rows sharing a canonical structure collapse to one; unparseable rows stay.
    counts = Counter(s for s in canonical if s is not None)
    seen: set[str] = set()
    keep: list[bool] = []
    for s in canonical:
        keep.append(s is None or s not in seen)
        if s is not None:
            seen.add(s)
    groups: dict[str, list[int]] = {}
    for row, s in enumerate(canonical):
        if s is not None and counts[s] > 1:
            groups.setdefault(s, []).append(row)
    out_frame = out_frame.with_columns(
        pl.Series("duplicates", [counts.get(s, 1) if s else 1 for s in canonical], dtype=pl.Int64)
    )
    merged, spreads = _merge_replicates(out_frame, groups)
    out_frame = merged.filter(pl.Series(keep))
    return {
        "compounds": out_frame,
        "meta": {
            "n_rows_in": frame.height,
            "n_structures_out": out_frame.height,
            "n_structures_with_replicates": len(groups),
            "n_rows_merged": sum(len(rows) - 1 for rows in groups.values()),
            "replicate_rule": (
                "Rows that standardize to the same structure are one compound. Their "
                "numeric readings are replaced by the median across the copies, which is "
                "the same compound whether the column is a concentration or a log "
                "potency; text columns keep the first copy's value."
            ),
            # The spread between copies of one compound is the assay's own reproducibility,
            # measured on this very plate. It bounds every difference the study goes on to
            # call real, so it is reported here rather than left for anyone to notice.
            "replicate_spread": spreads[:_SPREAD_SHOWN],
            "widest_spread": spreads[0]["spread"] if spreads else None,
        },
    }


#: How many replicate groups are named in `meta`, worst first.
_SPREAD_SHOWN = 25


def _spread(values: list[float]) -> tuple[float, str]:
    """How far apart the copies of one reading are, and how that distance is written.

    A fold change for readings that are all positive, which is how an assay's
    reproducibility is quoted; a plain difference otherwise, since a fold change across
    zero is meaningless.
    """
    low, high = min(values), max(values)
    if low > 0:
        return high / low, "fold"
    return high - low, "difference"


def _middle(values: list[float]) -> float:
    """The middle reading, geometric where that is the one that can be wrong by a factor."""
    # Cut to six figures on the way out. Combining 6.4 and 0.1 through logs lands on
    # 0.8000000000000002, and a reading that arrives in a table like that is one a chemist
    # stops trusting, however right it is.
    if min(values) > 0:
        middle = math.exp(statistics.median([math.log(v) for v in values]))
    else:
        middle = float(statistics.median(values))
    return float(f"{middle:.6g}")


def _merge_replicates(
    frame: pl.DataFrame, groups: Mapping[str, list[int]]
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """One row per structure, with numeric readings taken as the median of the copies.

    The median, not the first copy and not the arithmetic mean. Taking the first silently
    keeps one arbitrary measurement and throws the rest away, which can move the most
    potent compound in a set.

    The median is taken in log space whenever every copy is positive, which makes it the
    geometric mean of the middle two when there is an even number of them. This step is
    not told whether it is holding a concentration or a log potency, and the two mistakes
    are not the same size: two IC50 readings of 2.31 and 0.1 uM combine to 0.48
    geometrically and to 1.205 arithmetically, a factor of 2.5 and the difference between
    a hit and a miss, while two pIC50 readings of 7.0 and 8.0 combine to 7.483 and 7.5, a
    difference of no consequence. So the rule that is catastrophic for one scale and
    harmless for the other is the one that is not used. With an odd number of copies both
    rules return the same observed reading.
    """
    if not groups:
        return frame, []
    numeric = [name for name, dtype in frame.schema.items() if dtype in (pl.Float64, pl.Int64)]
    numeric = [name for name in numeric if name != "duplicates"]
    identifier = next((c for c in frame.columns if _IDENTIFIER.match(c)), None)
    updates: dict[str, dict[int, float]] = {name: {} for name in numeric}
    spreads: list[dict[str, Any]] = []
    for rows in groups.values():
        first = rows[0]
        for name in numeric:
            values = [v for v in (frame[name][row] for row in rows) if v is not None]
            if not values:
                continue
            floats = [float(v) for v in values]
            updates[name][first] = _middle(floats)
            if len(set(floats)) > 1:
                size, how = _spread(floats)
                spreads.append(
                    {
                        "id": str(frame[identifier][first]) if identifier else f"row {first + 1}",
                        "merged": [
                            str(frame[identifier][row]) if identifier else f"row {row + 1}"
                            for row in rows
                        ],
                        "column": name,
                        "values": floats,
                        "kept": updates[name][first],
                        "spread": round(size, 3),
                        "spread_is": how,
                    }
                )
    for name in numeric:
        if not updates[name]:
            continue
        column = frame[name]
        values = [updates[name].get(row, column[row]) for row in range(frame.height)]
        # Back to the column's own type: a median of whole numbers may land between two
        # of them, and a reading column that arrived as integers stays integers.
        series = pl.Series(name, values, dtype=pl.Float64)
        frame = frame.with_columns(
            series.round(0).cast(pl.Int64) if column.dtype == pl.Int64 else series
        )
    spreads.sort(key=lambda s: s["spread"], reverse=True)
    return frame, spreads


def property_profile(params: ActivityParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    frame = _frame(inputs)
    column = structure_column(frame)
    rows: list[list[float | None]] = []
    for smiles in frame[column].to_list():
        mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
        rows.append([None if mol is None else float(d(mol)) for d in _DESCRIPTORS])
    out = frame.with_columns(
        pl.Series(name, [r[i] for r in rows], dtype=pl.Float64)
        for i, name in enumerate(PROPERTY_COLUMNS)
    ).with_columns(pl.col("heavy_atoms").cast(pl.Int64))
    meta: dict[str, Any] = {"n_rows": frame.height}
    if params.activity_column:
        # A censored reading ('>64') keeps its bound instead of becoming a null. Both
        # efficiencies are monotone in the log potency -- one multiplies by a positive
        # constant, the other subtracts logp -- so a bound on it is a bound on them in
        # the same direction, and the relation column describes all three.
        readings = read_measurements(frame[params.activity_column])
        potency, relations, note = _log_potency(params, readings)
        meta["activity_column"] = params.activity_column
        meta["activity_read_as"] = note
        if potency is None:
            # Refused rather than computed on the wrong scale. An efficiency is a number
            # a chemist compares against remembered ones -- 0.3 is good, 0.5 is unusual --
            # so one computed from micromolar is not approximate, it is meaningless, and
            # it looks like a result.
            meta["efficiencies"] = (
                "Not computed. A concentration column must say its unit before it can be "
                "turned into a log potency, and an efficiency on the raw concentration "
                "would be a number of the right shape and no meaning."
            )
        else:
            out = out.with_columns(relations).with_columns(
                (1.37 * pl.lit(potency) / pl.col("heavy_atoms")).alias("ligand_efficiency"),
                (pl.lit(potency) - pl.col("logp")).alias("lipophilic_efficiency"),
            )
            meta["efficiencies"] = (
                "Ligand efficiency is 1.37 x log potency / heavy atoms, and lipophilic "
                "efficiency is log potency - logP. Both are read against remembered "
                "values: around 0.3 is a usual ligand efficiency."
            )
    return {"properties": out, "meta": meta}


#: How many of a unit there are in one molar. A log potency is -log10 of the
#: concentration in molar, so the unit is not decoration: getting it wrong moves every
#: efficiency by a whole number of log units.
_PER_MOLAR = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12}


def _log_potency(params: ActivityParams, readings: Any) -> tuple[pl.Series | None, pl.Series, str]:
    """The activity as a log potency, with the relations that still hold on it.

    A concentration is converted; a column that is already a log potency is used as it
    stands. The conversion turns the scale upside down, so a reading recorded as greater
    than 64 micromolar is a log potency *less* than the matching value, and the relation
    has to turn over with it. Left alone it would say a weak compound is strong.
    """
    if params.kind == "log":
        return readings.values, readings.relations, "a log potency, as given"
    factor = _PER_MOLAR.get((params.unit or "").strip().lower())
    if factor is None:
        return None, readings.relations, f"a concentration in an unstated unit ({params.unit})"
    potency = (readings.values.cast(pl.Float64) * factor).log10() * -1.0
    flipped = {">": "<", "<": ">", ">=": "<=", "<=": ">="}
    relations = pl.Series(
        readings.relations.name,
        [flipped.get(r, r) for r in readings.relations.to_list()],
        dtype=pl.String,
    )
    return potency, relations, f"a concentration in {params.unit}, converted to a log potency"


def property_confound(params: ConfoundParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    if "properties" in inputs:
        frame = _frame(inputs, "properties")
    else:
        frame = property_profile(ActivityParams(), inputs)["properties"]
    rng = np.random.default_rng(0)
    # A bound ('>64') cannot enter a correlation: the true value is somewhere above it, so
    # a Pearson r computed on 64 is a number about data nobody has. Bounds are left out and
    # counted -- separately from cells that were never readings ('NT'), because "9 left out
    # as censored" and "9 unreadable" send a scientist to different places.
    readings = read_measurements(frame[params.activity_column])
    activity = readings.exact_only
    excluded = readings.n_censored
    results: list[dict[str, Any]] = []
    for name in PROPERTY_COLUMNS:
        pair = pl.DataFrame({"x": frame[name].cast(pl.Float64), "y": activity}).drop_nulls()
        x, y = pair["x"].to_numpy(), pair["y"].to_numpy()
        n = len(x)
        r = _pearson(x, y)
        low = high = None
        if n >= 3 and r is not None:
            picks = rng.integers(0, n, size=(1000, n))
            boots = [_pearson(x[p], y[p]) for p in picks]
            valid = [b for b in boots if b is not None]
            if valid:
                low, high = (float(v) for v in np.percentile(valid, [2.5, 97.5]))
        results.append({"property": name, "r": r, "ci_low": low, "ci_high": high, "n": n})
    scored = [p for p in results if p["r"] is not None]
    strongest = max(scored, key=lambda p: abs(p["r"]), default=None)
    n_rows = max((p["n"] for p in results), default=0)
    return {
        "correlation": {
            "meta": {
                "activity_column": params.activity_column,
                "n": n_rows,
                "excluded": excluded,
                "excluded_unreadable": readings.n_unreadable,
                "strongest": (
                    {"property": strongest["property"], "r": strongest["r"]} if strongest else None
                ),
            },
            "properties": results,
        }
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


@cache
def _catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    for name in ("PAINS", "BRENK", "NIH"):
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


def structural_alerts(params: BaseModel | None, inputs: Mapping[str, object]) -> dict[str, Any]:
    frame = _frame(inputs)
    column = structure_column(frame)
    catalog = _catalog()
    flagged: list[dict[str, Any]] = []
    for row, smiles in enumerate(frame[column].to_list()):
        mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
        if mol is None:
            continue
        hits = [
            {
                "catalog": entry.GetProp("FilterSet").split("_")[0].upper(),
                "description": entry.GetDescription(),
            }
            for entry in catalog.GetMatches(mol)
        ]
        if hits:
            flagged.append({"row": row, "smiles": smiles, "alerts": hits})
    return {"alerts": flagged}


IMPLEMENTATIONS: dict[str, ToolImpl] = {
    "standardize": standardize,
    "property_profile": lambda p, i: property_profile(cast(ActivityParams, p), i),
    "property_confound": lambda p, i: property_confound(cast(ConfoundParams, p), i),
    "structural_alerts": structural_alerts,
}

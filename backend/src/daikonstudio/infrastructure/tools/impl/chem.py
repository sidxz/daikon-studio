"""The four RDKit tools: pure functions over polars frames."""

from __future__ import annotations

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
    # Rows sharing a canonical structure collapse to the first; unparseable rows stay.
    counts = Counter(s for s in canonical if s is not None)
    seen: set[str] = set()
    keep: list[bool] = []
    for s in canonical:
        keep.append(s is None or s not in seen)
        if s is not None:
            seen.add(s)
    out_frame = out_frame.with_columns(
        pl.Series("duplicates", [counts.get(s, 1) if s else 1 for s in canonical], dtype=pl.Int64)
    ).filter(pl.Series(keep))
    return {"compounds": out_frame}


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
    if params.activity_column:
        # strict=False: a censored reading ('>64') is null here, not a crash.
        activity = pl.col(params.activity_column).cast(pl.Float64, strict=False)
        out = out.with_columns(
            (1.37 * activity / pl.col("heavy_atoms")).alias("ligand_efficiency"),
            (activity - pl.col("logp")).alias("lipophilic_efficiency"),
        )
    return {"properties": out}


def property_confound(params: ConfoundParams, inputs: Mapping[str, object]) -> dict[str, Any]:
    if "properties" in inputs:
        frame = _frame(inputs, "properties")
    else:
        frame = property_profile(ActivityParams(), inputs)["properties"]
    rng = np.random.default_rng(0)
    raw = frame[params.activity_column]
    # Censored readings ('>64', '<0.01') are not numbers. They are left out and
    # counted, so the exclusion is stated rather than silently shrinking n.
    activity = raw.cast(pl.Float64, strict=False)
    excluded = int(activity.null_count() - raw.null_count())
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

"""
sar_cliffs.py — find the activity jumps in an SAR table.

An *activity cliff* here is a matched molecular pair restricted to the R-group
table: two compounds identical at every R position except one, with a large
activity difference. Restricting it to the decomposition makes this much
cheaper than general MMP analysis -- the pairing is already done, we just read
the R-vectors.

No RDKit and no HTML. It takes the rows `sar_core.name_rows` produced and
returns findings, which means the whole thing is testable without a chemistry
stack.

What counts as "large"
----------------------
`min_delta`, an absolute floor in log units, is the gate. 1.0 = ten-fold,
2.0 = hundred-fold; the MMP activity-cliff literature usually sits at 1.5-2.
This is what makes a flat SAR return nothing: a series where everything falls
within half a log never reaches the floor, however the numbers are
distributed.

`min_z` -- standard deviations above the mean pair difference -- is available
but **off by default**, and that is a deliberate reversal. It was originally
the co-gate, ANDed with the floor, and two test cases killed it:

  * A series where one position is uniformly steep. Pooling all pair
    differences inflates the SD until mean + 2*SD lands *above the largest
    observed difference*, so a genuine 100-fold cliff was silently rejected.
    Mean/SD outlier detection assumes a unimodal distribution, and "a few big
    jumps among many small ones" is precisely bimodal.

  * A single matched pair. SD of one value is zero, so no z-score exists and
    the AND gate rejected the only pair there was.

So: the floor decides what counts as a jump, and z is an opt-in extra filter
for a series where ten-fold swings are routine and you only want the extremes.
When it is on, z is computed against that R position's own distribution rather
than the pooled one -- R1 and R2 generally have nothing to do with each other.
Either way the full distribution is reported so you can pick a floor rather
than trust a default.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# what kind of number is in the activity column?
# --------------------------------------------------------------------------
#
# An earlier version had one flag, `scale`, set to "log" or "linear". That
# fused three independent questions and got two of them wrong on anything but
# an IC50 sheet:
#
#   kind             is the number already log-transformed (pIC50), a raw
#                    concentration (IC50, MIC, MBC), or an interval quantity
#                    where ratios are meaningless (% inhibition, a Tm shift)?
#   higher_is_better does a bigger number mean a better compound? pIC50 yes,
#                    MIC no. Guess this wrong and every cliff reads backwards.
#   unit             nM, ug/mL, %. Carried so labels can say what they mean.
#
# The failures were silent. "MBC" matched no hint, so it was assumed already
# logged and lower-is-better was lost. "% inhibition at 10uM" was assumed
# logged too, so a one-point difference in percent was reported as "10x".
#
# Each entry below is (kind, higher_is_better). Longest match wins, so
# "pic50" beats "ic50" and "logmic" beats "mic".
ASSAY_HINTS: Dict[str, Tuple[str, bool]] = {
    # already log-transformed, bigger = better
    "pic50": ("log", True), "pec50": ("log", True), "pki": ("log", True),
    "pkd": ("log", True), "pgi50": ("log", True), "pa2": ("log", True),
    "pkb": ("log", True), "pmic": ("log", True), "pxc50": ("log", True),
    # already log-transformed, bigger = worse
    "logic50": ("log", False), "logmic": ("log", False),
    "logec50": ("log", False), "logki": ("log", False),
    # concentrations: bigger = worse
    "ic50": ("concentration", False), "ic90": ("concentration", False),
    "ec50": ("concentration", False), "ec90": ("concentration", False),
    "gi50": ("concentration", False), "cc50": ("concentration", False),
    "td50": ("concentration", False), "ld50": ("concentration", False),
    "xc50": ("concentration", False), "km": ("concentration", False),
    "ki": ("concentration", False), "kd": ("concentration", False),
    "mic": ("concentration", False), "mic50": ("concentration", False),
    "mic90": ("concentration", False), "mbc": ("concentration", False),
    "mfc": ("concentration", False), "mbec": ("concentration", False),
    "motc": ("concentration", False), "conc": ("concentration", False),
    "potency": ("concentration", False), "dose": ("concentration", False),
    # ratios: bigger = better
    "selectivity": ("concentration", True), "si": ("concentration", True),
    "therapeuticindex": ("concentration", True),
    "foldselectivity": ("concentration", True),
    # interval quantities: differences mean something, ratios do not
    "inhibition": ("linear", True), "inhib": ("linear", True),
    "percentinhibition": ("linear", True), "effect": ("linear", True),
    "occupancy": ("linear", True), "response": ("linear", True),
    "tmshift": ("linear", True), "dtm": ("linear", True),
    "deltatm": ("linear", True), "tm": ("linear", True),
    "zscore": ("linear", True), "score": ("linear", True),
    "viability": ("linear", False), "survival": ("linear", False),
}

_UNIT_RE = re.compile(
    r"[\(\[]\s*(n|u|\u00b5|m|p)?\s*(m|molar|g\s*/\s*m\s*l|g/l|%)\s*[\)\]]",
    re.I)


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def detect_activity(activity_col: Optional[str],
                    values: Optional[Sequence] = None) -> dict:
    """Work out kind, direction and unit for an activity column.

    Returns {"kind", "higher_is_better", "unit", "why", "confident"}. The
    guess is always reported and never silent, because getting `kind` wrong
    turns percentages into fold-changes and getting `higher_is_better` wrong
    reverses every conclusion in the table.
    """
    out = {"kind": "log", "higher_is_better": True, "unit": "",
           "why": "", "confident": False}
    name = _norm(activity_col)

    if activity_col:
        m = _UNIT_RE.search(str(activity_col))
        if m:
            out["unit"] = re.sub(r"\s+", "", m.group(0).strip("()[]"))
        elif "%" in str(activity_col):
            out["unit"] = "%"

    hit = max((h for h in ASSAY_HINTS if h in name), key=len, default=None)
    if hit:
        kind, higher = ASSAY_HINTS[hit]
        out.update(kind=kind, higher_is_better=higher, confident=True,
                   why=(f"column {activity_col!r} matched {hit!r}: "
                        f"{kind}, higher is "
                        f"{'better' if higher else 'worse'}"))
    elif name.startswith("log"):
        out.update(kind="log", higher_is_better=True, confident=False,
                   why=f"column {activity_col!r} starts with 'log'; assumed "
                       f"already log-scaled and higher-is-better")
    elif not activity_col:
        out["why"] = "no activity column named"
        return out

    # The data gets a say where the name was not decisive, and can veto a
    # nonsense reading either way.
    nums = [p[1] for p in (parse_value(v) for v in (values or [])) if p]
    if nums:
        lo, hi = min(nums), max(nums)
        spread = (hi / lo) if lo > 0 else float("inf")
        if not out["confident"]:
            if out["unit"] == "%" or (0 <= lo and hi <= 100 and spread < 100
                                      and "inhib" in name):
                out.update(kind="linear", higher_is_better=True,
                           why=f"values 0-100 in {activity_col!r}; treated as "
                               f"an interval quantity, not a ratio")
            elif lo > 0 and spread >= 1000:
                out.update(kind="concentration", higher_is_better=False,
                           why=f"values in {activity_col!r} span {spread:.0f}x; "
                               f"treated as a concentration, lower is better")
            elif 0 < lo and hi <= 20:
                out.update(kind="log", higher_is_better=True,
                           why=f"values in {activity_col!r} sit in {lo:g}-{hi:g}; "
                               f"treated as already log-scaled")
        if (out["kind"] == "concentration" and lo <= 0
                and not out["confident"]):
            # Only overrides a guess. A column plainly named IC50 with a
            # stray zero in it is a column with one bad cell, not a column
            # that has stopped being a concentration -- and reclassifying
            # the lot would quietly keep the bad cell instead of skipping it.
            out.update(kind="linear",
                       why=out["why"] + "; but values <= 0 are present, so "
                                        "it cannot be a concentration")
    if not out["why"]:
        out["why"] = (f"could not tell what {activity_col!r} is; assumed "
                      f"already log-scaled and higher-is-better. Pass "
                      f"kind= and higher_is_better= if that is wrong")
    return out


# Kept so existing callers and saved commands keep working.
def detect_scale(activity_col: Optional[str]) -> Tuple[str, str]:
    """Legacy two-value form of `detect_activity`."""
    a = detect_activity(activity_col)
    return ("log" if a["kind"] == "log" else "linear"), a["why"]


# --------------------------------------------------------------------------
# censored values  --  ">64" is data, not a blank
# --------------------------------------------------------------------------
#
# Real assay sheets are full of qualifiers, and MIC sheets are mostly
# qualifiers. The old parser returned None for anything that was not a bare
# number, so every ">64" was dropped as unusable -- quietly, and in a whole
# dose-limited series that is most of the data.
#
# A qualifier is a bound, so each value becomes an interval in comparable
# units and the distance between two intervals is the difference we can
# *guarantee*. Two compounds both at ">64" overlap, so their guaranteed
# difference is zero and they can never form a cliff, which is right: nobody
# knows which is better. ">64" against an exact 2 are disjoint, so the
# difference is at least 32-fold, and a cliff called on that is sound.

_QUAL_RE = re.compile(r"^\s*(>=|<=|=>|=<|\u2265|\u2264|>|<|~|=)?\s*"
                      r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$")

# Identity entries are not padding: without them `.get(">", "=")` quietly
# turned ">64" into an exact 64, which is worse than dropping it -- a bound
# reported as a measurement invents precision nobody has.
_OP_CANON = {">": ">", "<": "<", ">=": ">=", "<=": "<=",
             "\u2265": ">=", "\u2264": "<=", "=>": ">=", "=<": "<=",
             "~": "=", "=": "=", None: "=", "": "="}
_OP_FLIP = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "=": "="}


def parse_value(v) -> Optional[Tuple[str, float]]:
    """('>', 64.0) for '>64'; ('=', 12.5) for '12.5'; None for 'NT'."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return ("=", float(v)) if math.isfinite(float(v)) else None
    m = _QUAL_RE.match(str(v).replace(",", ""))
    if not m:
        return None
    try:
        x = float(m.group(2))
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return _OP_CANON.get(m.group(1), "="), x


def to_interval(raw, scale: dict) -> Optional[Tuple[float, float]]:
    """A raw cell as an interval in comparable units where higher = better.

    Everything is converted to one scale on which bigger means better, so
    nothing downstream has to remember which way the assay runs.
    """
    parsed = parse_value(raw)
    if parsed is None:
        return None
    op, x = parsed
    kind = scale.get("kind", "log")
    higher = bool(scale.get("higher_is_better", True))

    if kind == "concentration":
        if x <= 0:
            return None
        t = math.log10(x)
    else:
        t = x
    if not higher:
        t, op = -t, _OP_FLIP[op]      # negating reverses the inequality

    if op in (">", ">="):
        return (t, math.inf)
    if op in ("<", "<="):
        return (-math.inf, t)
    return (t, t)


def interval_gap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """The difference two intervals *guarantee*; 0 if they overlap."""
    (alo, ahi), (blo, bhi) = a, b
    if ahi < blo:
        return blo - ahi
    if bhi < alo:
        return alo - bhi
    return 0.0


def is_bounded(iv: Tuple[float, float]) -> bool:
    return not (math.isfinite(iv[0]) and math.isfinite(iv[1]))


def point_of(iv: Tuple[float, float]) -> float:
    """A single number for sorting and colouring: the finite end."""
    lo, hi = iv
    if math.isfinite(lo) and math.isfinite(hi):
        return (lo + hi) / 2.0
    return hi if math.isfinite(hi) else lo


def _to_float(v) -> Optional[float]:
    """Legacy helper: the bare number, ignoring any qualifier."""
    p = parse_value(v)
    return p[1] if p else None


def fold_label(delta: float, kind: str = "log", unit: str = "",
               bounded: bool = False) -> str:
    """A difference in the words the assay deserves.

    For a ratio scale that is a fold change. For an interval quantity it is
    not: reporting "10x" for a ten-point difference in percent inhibition is
    how the old single-flag model embarrassed itself.
    """
    if kind == "linear":
        u = f" {unit}" if unit else ""
        return f"{'>=' if bounded else ''}{abs(delta):.3g}{u}"
    fold = 10.0 ** abs(delta)
    pre = ">=" if bounded else ""
    if fold >= 10000:
        return f"{pre}>10,000x"
    if fold >= 100:
        return f"{pre}{fold:,.0f}x"
    if fold >= 10:
        return f"{pre}{fold:.0f}x"
    return f"{pre}{fold:.1f}x"


def dilutions(delta: float) -> float:
    """A log difference as doubling dilutions, which is how MIC is read."""
    return abs(delta) / math.log10(2.0)


def find_cliffs(
    rows: Sequence[dict],
    rgroup_cols: Sequence[str],
    activity_col: Optional[str],
    *,
    scale: str = "auto",
    kind: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
    unit: Optional[str] = None,
    min_delta: Optional[float] = None,
    min_fold: Optional[float] = None,
    min_z: Optional[float] = None,
    id_col: str = "ID",
    key_suffix: str = "_key",
    label_suffix: str = "",
    max_cliffs: Optional[int] = None,
) -> dict:
    """Find single-point R-group changes with a large activity swing.

    Returns a dict with:
        cliffs      ranked findings, biggest delta first
        scale       "log" or "linear", and why in `notes`
        n_pairs     single-point pairs examined
        mean_delta, sd_delta, threshold
        per_position    stats for each R position
        notes       anything the caller should know
        skipped     compounds dropped for want of a usable activity value
    """
    out = {
        "cliffs": [], "scale": scale, "n_pairs": 0, "mean_delta": 0.0,
        "sd_delta": 0.0, "threshold": min_delta, "max_delta": 0.0,
        "per_position": {}, "notes": [], "skipped": [],
        "params": {"min_delta": min_delta, "min_z": min_z},
    }
    cols = list(rgroup_cols)

    if not activity_col:
        out["notes"].append("no activity column, so there is nothing to "
                            "compare; cliff detection needs one")
        return out
    if not cols:
        out["notes"].append("no R-group columns")
        return out

    # ---- what kind of number are we looking at? --------------------------
    guess = detect_activity(activity_col, [r.get(activity_col) for r in rows])
    if scale in ("log", "linear") and kind is None:
        # Legacy two-value form: "linear" used to mean "a concentration".
        kind = "log" if scale == "log" else "concentration"
        out["notes"].append(f"scale={scale!r} is the old spelling; "
                            f"kind={kind!r} is the new one")
    act = {
        "kind": kind or guess["kind"],
        "higher_is_better": (guess["higher_is_better"]
                             if higher_is_better is None else
                             bool(higher_is_better)),
        "unit": guess["unit"] if unit is None else unit,
    }
    out["activity"] = act
    out["scale"] = "log" if act["kind"] == "log" else "linear"
    if kind is None and higher_is_better is None:
        out["notes"].append(guess["why"])
        if not guess["confident"]:
            out["notes"].append(
                "that was a guess. Pass kind= (log | concentration | linear) "
                "and higher_is_better= to be certain — getting the direction "
                "wrong reverses every cliff in the table")
    else:
        out["notes"].append(
            f"activity read as {act['kind']}, higher is "
            f"{'better' if act['higher_is_better'] else 'worse'}"
            + (f", in {act['unit']}" if act["unit"] else ""))

    # ---- usable compounds -------------------------------------------------
    # Values become intervals in comparable units where bigger always means
    # better, so nothing below has to remember which way the assay runs, and
    # a qualifier like ">64" stays usable instead of being dropped.
    recs, n_censored = [], 0
    for r in rows:
        rid = str(r.get(id_col, ""))
        cell = r.get(activity_col)
        iv = to_interval(cell, act)
        if iv is None:
            out["skipped"].append(
                (rid, f"no usable activity value ({cell!r})"))
            continue
        if is_bounded(iv):
            n_censored += 1
        vec = tuple(str(r.get(f"{c}{key_suffix}",
                             r.get(f"{c}_smiles", ""))) for c in cols)
        recs.append({"id": rid, "raw": cell, "iv": iv,
                     "value": point_of(iv), "vec": vec, "row": r})
    if n_censored:
        out["notes"].append(
            f"{n_censored} value(s) carry a qualifier such as '>' or '<'. "
            f"Those are treated as bounds: a pair is only called a cliff "
            f"when the difference is certain, so two compounds both at "
            f"'>64' can never be one")

    if len(recs) < 2:
        out["notes"].append(f"only {len(recs)} compound(s) with a usable "
                            f"activity value")
        return out

    # ---- what counts as a jump, in units that suit the assay -------------
    # A ratio scale wants a fold change; an interval quantity cannot have
    # one. The old single default of "1.0 log units" was quietly wrong for
    # both MIC (where the conventional floor is two doubling dilutions, not
    # ten-fold) and percent inhibition (where log units mean nothing).
    if min_fold is not None:
        if act["kind"] == "linear":
            out["notes"].append(
                "min_fold ignored: a fold change is meaningless on an "
                "interval quantity. Use min_delta, in the column's own units")
        elif min_fold <= 1:
            out["notes"].append(f"min_fold={min_fold} is not a jump; ignored")
        else:
            min_delta = math.log10(min_fold)
    if min_delta is None:
        if act["kind"] == "linear":
            vals = [r["value"] for r in recs]
            rng = max(vals) - min(vals)
            min_delta = 0.2 * rng if rng > 0 else 0.0
            u = f" {act['unit']}" if act["unit"] else ""
            out["notes"].append(
                f"no threshold given, and no conventional one exists for an "
                f"interval quantity, so 20% of the observed range was used: "
                f"{min_delta:.3g}{u}. Set min_delta to something the assay "
                f"justifies rather than trusting this")
        else:
            min_delta = 1.0
            out["notes"].append(
                "threshold: 1.0 log unit = 10-fold. For a doubling-dilution "
                "assay such as MIC or MBC, min_fold=4 (two dilutions) is the "
                "usual floor, since one dilution is assay noise")
    out["params"] = {"min_delta": min_delta, "min_fold": min_fold,
                     "min_z": min_z, "kind": act["kind"],
                     "higher_is_better": act["higher_is_better"],
                     "unit": act["unit"]}
    if act["kind"] != "linear":
        out["params"]["threshold_fold"] = round(10.0 ** min_delta, 3)
        out["params"]["threshold_dilutions"] = round(dilutions(min_delta), 2)

    # ---- single-point pairs ----------------------------------------------
    # Group by "the vector with position i blanked out". Everything inside a
    # group is identical everywhere except position i, so every pair within
    # a group differs at exactly that one point. O(n*k) instead of O(n^2*k).
    pairs = []
    for i, col in enumerate(cols):
        buckets: Dict[tuple, List[dict]] = defaultdict(list)
        for rec in recs:
            buckets[rec["vec"][:i] + ("*",) + rec["vec"][i + 1:]].append(rec)
        for group in buckets.values():
            if len(group) < 2:
                continue
            for a_i in range(len(group)):
                for b_i in range(a_i + 1, len(group)):
                    a, b = group[a_i], group[b_i]
                    if a["vec"][i] == b["vec"][i]:
                        continue          # same substituent: not a pair
                    # The difference we can *guarantee*, not the difference
                    # between two convenient point estimates.
                    pairs.append((col, i, a, b,
                                  interval_gap(a["iv"], b["iv"])))

    out["n_pairs"] = len(pairs)
    if not pairs:
        out["notes"].append(
            "no pairs differ at exactly one R position — every compound "
            "differs from every other at two or more points, so there are no "
            "matched pairs to compare")
        return out

    deltas = [p[4] for p in pairs]
    mu = mean(deltas)
    sd = pstdev(deltas) if len(deltas) > 1 else 0.0
    out["mean_delta"] = round(mu, 4)
    out["sd_delta"] = round(sd, 4)
    out["max_delta"] = round(max(deltas), 4)
    out["threshold"] = round(min_delta, 4)

    # ---- per-position stats, which is also where z is measured ----------
    by_pos = defaultdict(list)
    for col, _i, _a, _b, d in pairs:
        by_pos[col].append(d)
    pos_stats = {}
    for col, ds in by_pos.items():
        p_mu = mean(ds)
        p_sd = pstdev(ds) if len(ds) > 1 else 0.0
        pos_stats[col] = (p_mu, p_sd)
        out["per_position"][col] = {
            "n_pairs": len(ds),
            "mean_delta": round(p_mu, 3),
            "max_delta": round(max(ds), 3),
            "sd_delta": round(p_sd, 3),
            "z_cut": (round(p_mu + min_z * p_sd, 3)
                      if (min_z is not None and p_sd > 0) else None),
        }

    if min_z is not None:
        flat_positions = [c for c, (_m, s) in pos_stats.items() if s <= 0]
        if flat_positions:
            out["notes"].append(
                f"z filter inactive at {', '.join(flat_positions)}: too few "
                f"pairs there to have a spread")

    # ---- select and describe --------------------------------------------
    def label_of(rec, col):
        r = rec["row"]
        for k in (f"{col}{label_suffix}" if label_suffix else col,
                  f"{col}_name", f"{col}_smiles"):
            v = r.get(k)
            if v:
                return str(v)
        return "?"

    found = []
    for col, _i, a, b, d in pairs:
        p_mu, p_sd = pos_stats[col]
        z = (d - p_mu) / p_sd if p_sd > 0 else 0.0
        # Tolerance, not sloppiness: log10(8) - log10(2) lands one ULP
        # below log10(4), so an exactly-fourfold MIC jump failed a fourfold
        # gate. A threshold a chemist states in round numbers has to behave
        # like a round number.
        passes_delta = d >= min_delta - 1e-9
        # z, when enabled, only narrows. A position with no spread can't
        # produce a z-score, and must not therefore lose all its pairs.
        passes_z = True if (min_z is None or p_sd <= 0) else (z >= min_z)
        if not (passes_delta and passes_z):
            continue
        # Order the pair so the more active compound comes first. Both are
        # in comparable units where bigger is better, whichever way the
        # assay itself runs.
        hi, lo = (a, b) if a["value"] >= b["value"] else (b, a)
        bounded = is_bounded(a["iv"]) or is_bounded(b["iv"])
        found.append({
            "position": col,
            "id_better": hi["id"], "id_worse": lo["id"],
            "label_better": label_of(hi, col), "label_worse": label_of(lo, col),
            "smiles_better": hi["row"].get(f"{col}_smiles", ""),
            "smiles_worse": lo["row"].get(f"{col}_smiles", ""),
            "activity_better": hi["raw"], "activity_worse": lo["raw"],
            "delta": round(d, 3),
            "fold": fold_label(d, act["kind"], act["unit"], bounded),
            "bounded": bounded,
            # Dilutions only mean something for a doubling-dilution assay.
            # Quoting "10.96 dilutions" for a pIC50 pair is noise.
            "dilutions": (round(dilutions(d), 2)
                          if act["kind"] == "concentration" else None),
            "z": round(z, 2),
            "passes_delta": passes_delta, "passes_z": passes_z,
        })

    found.sort(key=lambda c: -c["delta"])
    if max_cliffs is not None:
        found = found[:max_cliffs]
    for n, c in enumerate(found, 1):
        c["rank"] = n
        c["cliff_id"] = n - 1
    out["cliffs"] = found

    if not found:
        out["notes"].append(
            f"no cliffs: {len(pairs)} matched pair(s) examined, largest "
            f"difference {max(deltas):.2f} log units, floor {min_delta:.2f}"
            + (f", z >= {min_z}" if min_z is not None else "")
            + ". A flat SAR is supposed to look like this")
    return out


def cell_marks(result: dict, id_col: str = "ID") -> Dict[Tuple[str, str], List[int]]:
    """Which (compound id, column) cells belong to which cliffs.

    Returns {(id, "R2"): [cliff_id, ...]}. A compound can sit in several
    cliffs, and the activity cell is marked too, under the key
    (id, "__activity__").
    """
    marks: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for c in result.get("cliffs", []):
        cid = c["cliff_id"]
        for who in ("id_better", "id_worse"):
            marks[(c[who], c["position"])].append(cid)
            marks[(c[who], "__activity__")].append(cid)
    return dict(marks)


def summarize(result: dict) -> List[str]:
    """Plain-text summary lines, for a terminal or a notebook cell.

    Everything is phrased in the assay's own terms: fold changes and doubling
    dilutions for a ratio scale, the column's own units for an interval one.
    Reporting "log units" for a percentage was how the old single-flag model
    hid the fact that it had misread the column.
    """
    lines = []
    act = result.get("activity") or {}
    kind = act.get("kind", result.get("scale", "log"))
    unit = act.get("unit", "")
    linear = kind == "linear"
    u = f" {unit}" if unit else ""
    units = u if linear else " log units"

    lines.append(
        f"activity         {kind}"
        + (f" in {unit}" if unit else "")
        + f", higher is {'better' if act.get('higher_is_better', True) else 'worse'}")
    lines.append(f"matched pairs    {result.get('n_pairs', 0)}  "
                 f"(differ at exactly one R position)")
    lines.append(f"mean difference  {result.get('mean_delta', 0):.3g}{units}")
    lines.append(f"sd               {result.get('sd_delta', 0):.3g}")
    lines.append(f"largest          {result.get('max_delta', 0):.3g}{units}")

    p = result.get("params", {})
    md = p.get("min_delta")
    if md is None:
        gate = "none"
    elif linear:
        gate = f"difference >= {md:.3g}{u}"
    else:
        gate = (f"difference >= {md:.3g} log units "
                f"= {p.get('threshold_fold', 10 ** md):.3g}-fold "
                f"= {p.get('threshold_dilutions', 0):.3g} dilutions")
    if p.get("min_z") is not None:
        gate += f"  AND  z >= {p['min_z']} (per position)"
    lines.append(f"gate             {gate}")

    cliffs = result.get("cliffs", [])
    lines.append(f"cliffs found     {len(cliffs)}")
    for c in cliffs:
        lines.append(
            f"  #{c['rank']}  {c['position']}: {c['label_worse']} -> "
            f"{c['label_better']}   {c['fold']}"
            + ("" if linear or c.get("dilutions") is None
               else f" ({c['dilutions']:g} dilutions)")
            + f"   {c['id_worse']} ({c['activity_worse']}) -> "
              f"{c['id_better']} ({c['activity_better']})   z={c['z']}")
    for pos, st in sorted(result.get("per_position", {}).items()):
        lines.append(f"  {pos}: {st['n_pairs']} pairs, mean "
                     f"{st['mean_delta']:.3g}, max {st['max_delta']:.3g}")
    for n in result.get("notes", []):
        lines.append(f"note: {n}")
    if result.get("skipped"):
        lines.append(f"skipped {len(result['skipped'])} compound(s) without a "
                     f"usable activity value: "
                     + ", ".join(i for i, _ in result["skipped"][:6]))
    return lines


def cliffs_csv_rows(result: dict) -> List[dict]:
    """Flat rows for export."""
    return [{
        "rank": c["rank"], "position": c["position"],
        "from_id": c["id_worse"], "from_group": c["label_worse"],
        "from_activity": c["activity_worse"],
        "to_id": c["id_better"], "to_group": c["label_better"],
        "to_activity": c["activity_better"],
        "delta_log": c["delta"], "fold": c["fold"], "z": c["z"],
    } for c in result.get("cliffs", [])]

"""A measured value and the relation it was reported with.

`>64` is data, not a blank. A dose-limited series is mostly qualifiers, and the two
things a reader can do with one are both wrong: drop it, which quietly shrinks n and
removes exactly the compounds that failed, or strip the qualifier, which reports a
bound as a measurement and invents precision nobody has.

So a reading parses into a relation and a number, and the relation travels with the
number from then on. What a tool then does with a bound is the tool's decision --
`matched_pairs` already reasons over intervals, a correlation leaves bounds out and
says how many -- but no tool has to re-derive the bound from a string, and none can
lose it by accident.

Pure stdlib on purpose: the domain may not import polars, so the polars-facing
helper that applies this to a column lives beside the tools that need it. The regex
and the canonical operator map are the ones proven in the vendored SAR code, kept
here so application and infrastructure share one parser instead of two.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

# The suffix a relation column takes from the column it describes, so a reader meeting
# `IC50 (nM)` and `IC50 (nM)_relation` can pair them without being told. Named in the
# domain because both the tool that writes the column and the codec that decides what an
# agent may see have to agree on it, and they sit in different layers.
RELATION_SUFFIX = "_relation"


class Relation(StrEnum):
    """How a reported value stands to the true one."""

    EXACT = "="
    GREATER = ">"
    LESS = "<"
    AT_LEAST = ">="
    AT_MOST = "<="


# A bound, not an equality. Every qualifier that means "exact" maps to EXACT, and
# every one that means a bound keeps its direction. The identity entries are not
# padding: a lookup that defaulted to "=" would turn `>64` into an exact 64, which is
# worse than dropping it.
_OPERATORS: dict[str, Relation] = {
    "": Relation.EXACT,
    "=": Relation.EXACT,
    "~": Relation.EXACT,  # approximate is still a point estimate, not a bound
    ">": Relation.GREATER,
    "<": Relation.LESS,
    ">=": Relation.AT_LEAST,
    "<=": Relation.AT_MOST,
    "=>": Relation.AT_LEAST,
    "=<": Relation.AT_MOST,
    "≥": Relation.AT_LEAST,
    "≤": Relation.AT_MOST,
}

_READING = re.compile(
    r"^\s*(>=|<=|=>|=<|≥|≤|>|<|~|=)?\s*"
    r"([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$"
)


@dataclass(frozen=True, kw_only=True, slots=True)
class Measurement:
    relation: Relation
    value: float

    @property
    def is_censored(self) -> bool:
        return self.relation is not Relation.EXACT


def parse_measurement(raw: object) -> Measurement | None:
    """`>64` to a bound, `12.5` to an exact value, `NT` and a blank to None.

    None means "not a reading", which is a third thing from a bound and from a
    number: a caller that wants to count what it left out needs them apart.
    A thousands separator is tolerated because sheets carry them; anything else
    unparseable stays None rather than guessing.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):  # a flag is not a reading, and bool is an int in Python
        return None
    if isinstance(raw, int | float):
        number = float(raw)
        if not math.isfinite(number):
            return None
        return Measurement(relation=Relation.EXACT, value=number)
    if not isinstance(raw, str):
        return None
    found = _READING.match(raw.replace(",", ""))
    if found is None:
        return None
    try:
        number = float(found.group(2))
    except ValueError:  # pragma: no cover -- the pattern already guarantees a float
        return None
    if not math.isfinite(number):  # "inf" never matches the pattern, but nan/inf must not pass
        return None
    return Measurement(relation=_OPERATORS[found.group(1) or ""], value=number)

"""Reading a table's shape, with no opinion about what it holds.

Parsers, not rules. Each column is offered to every parser this server owns and
the profile records how many values each one accepted. Teaching the profile about
a new kind of data is a parser in `_PARSERS`; nothing else changes, and no column
needs a rule written for it.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable

import polars as pl
from rdkit import Chem

from daikonstudio.application.ports.table_profiler import ColumnProfile, TableProfile
from daikonstudio.domain.shared.errors import ValidationError

# Importing the chem package silences RDKit's parse-failure logging (its
# __init__ calls RDLogger.DisableLog), which matters here because profiling a
# text column deliberately offers it a few hundred things that are not molecules.
import daikonstudio.infrastructure.chem  # noqa: F401  isort:skip

# Enough rows to see the shape, few enough that a large upload is still quick.
# The profile says when it hit this, because a sampled count presented as a whole
# count is a lie the model has no way to detect.
_MAX_ROWS = 5_000

# Values shown per column. Four is enough to see a format and short enough that
# ten columns still cost a couple of hundred tokens.
_EXAMPLES = 4

# How much of a column a parser must accept before the values it rejects are
# worth calling exceptions. Deliberately low: a column where half the entries are
# numbers and half read ">50" is a heavily censored measurement, which is the
# single most important thing to surface, and a stricter threshold would dismiss
# it as "not a number column" and say nothing.
_ENOUGH_OF_A_PATTERN = 1 / 3

_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_molecule(value: str) -> bool:
    # A bare element symbol is a valid molecule and also a perfectly ordinary
    # categorical value ("N", "C"), so a one-character cell is not evidence.
    if len(value) < 2:
        return False
    return Chem.MolFromSmiles(value) is not None


def _is_sequence(value: str) -> bool:
    stripped = value.strip().upper()
    if len(stripped) < 20:
        return False
    return all(character in _AMINO_ACIDS for character in stripped)


def _is_date(value: str) -> bool:
    return bool(_DATE.match(value.strip()))


_PARSERS: dict[str, Callable[[str], bool]] = {
    "number": _is_number,
    "molecule": _is_molecule,
    "sequence": _is_sequence,
    "date": _is_date,
}

# A column the parser below accepts is the thing worth protecting, so its values
# are described rather than quoted. The shape still reaches the model; the
# structures do not. Everything else -- numbers, dates, short labels -- is shown,
# because a potency of 7.2 is not what anyone means by proprietary chemistry.
_WITHHOLD = {"molecule", "sequence"}


class PolarsTableProfiler:
    def profile(self, content: bytes, *, filename: str) -> TableProfile:
        separator = "\t" if filename.lower().endswith((".tsv", ".tab")) else ","
        try:
            frame = pl.read_csv(
                io.BytesIO(content),
                separator=separator,
                n_rows=_MAX_ROWS,
                infer_schema_length=0,  # everything as text; the parsers decide
                truncate_ragged_lines=True,
            )
        except Exception as exc:
            raise ValidationError(
                "That file could not be read as a table.",
                detail="Upload a CSV or TSV with a header row.",
            ) from exc

        if frame.height == 0 or frame.width == 0:
            raise ValidationError("That file has no rows to describe.")

        return TableProfile(
            rows=frame.height,
            truncated=frame.height == _MAX_ROWS,
            columns=tuple(_column(frame, name) for name in frame.columns),
        )


def _column(frame: pl.DataFrame, name: str) -> ColumnProfile:
    values = [v for v in frame[name].to_list() if v is not None and str(v).strip() != ""]
    text = [str(v) for v in values]
    filled = len(text)

    parses: dict[str, int] = {}
    for parser_name, accepts in _PARSERS.items():
        # Cheap parsers run over everything; molecule parsing does not, so it
        # sees a sample and the count is scaled. A column is or is not structures;
        # two hundred rows settle that as well as fifty thousand.
        sample = text if parser_name != "molecule" else text[:200]
        if not sample:
            parses[parser_name] = 0
            continue
        accepted = sum(1 for v in sample if accepts(v))
        parses[parser_name] = (
            accepted if sample is text else round(accepted * filled / len(sample))
        )

    best = max(parses, key=lambda k: parses[k]) if parses else ""
    mostly = best if filled and parses.get(best, 0) >= filled * _ENOUGH_OF_A_PATTERN else None

    # Withholding is a stronger claim than having a pattern: a column is only
    # treated as the asset when most of it really does parse that way.
    withheld = mostly in _WITHHOLD and parses.get(mostly or "", 0) >= filled * 0.6

    odd: tuple[str, ...] = ()
    odd_count = 0
    if mostly is not None:
        accepts = _PARSERS[mostly]
        rejected = [v for v in text if not accepts(v)]
        odd_count = len(rejected)
        # De-duplicated: forty rows reading ">50" is one fact, not forty. Quoted
        # only where the column is not withheld -- a structure that failed to
        # parse is still the scientist's compound, and showing the failures of a
        # column whose values are deliberately not shown would leak exactly what
        # withholding them was for.
        odd = () if withheld else tuple(dict.fromkeys(rejected))[:_EXAMPLES]
    # Computed whenever there are numbers at all, not only when the column is
    # mostly numeric: the range of the numeric half of a censored column is
    # exactly what says whether the assay had room to measure a difference.
    numbers = sorted(float(v) for v in text if _is_number(v))

    counts = {v: text.count(v) for v in set(text)} if filled else {}
    repeated = max(counts.items(), key=lambda kv: kv[1]) if counts else None

    return ColumnProfile(
        name=name,
        filled=filled,
        distinct=len(set(text)),
        parses_as={k: v for k, v in parses.items() if v},
        examples=() if withheld else tuple(dict.fromkeys(text))[:_EXAMPLES],
        withheld=withheld,
        odd_values=odd,
        odd_count=odd_count,
        minimum=numbers[0] if numbers else None,
        maximum=numbers[-1] if numbers else None,
        median=numbers[len(numbers) // 2] if numbers else None,
        most_repeated=repeated if repeated and repeated[1] > 1 else None,
    )

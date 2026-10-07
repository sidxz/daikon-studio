"""Which kind of structure a column holds, and the refusals that depend on knowing.

The whole risk here is a column read as the wrong kind. Nothing downstream crashes when
that happens -- a featurizer handed the wrong input still returns a matrix of the right
shape, the fit converges, and a Scorecard renders -- so these tests are the only thing
standing between a mis-read column and a confident, meaningless result.
"""

import polars as pl
import pytest

from daikonstudio.application.data.create_dataset import _split_kind_error
from daikonstudio.application.data.prepare_frame import prepare_frame
from daikonstudio.domain.data.split import SplitStrategy
from daikonstudio.domain.data.structure_kind import (
    StructureKind,
    looks_like_sequence,
    normalize_sequence,
)
from daikonstudio.domain.data.target import TargetKind, TargetSpec
from daikonstudio.domain.data.validation import report_from_dict, report_to_dict
from daikonstudio.infrastructure.chem.normalizer import RdkitStructureNormalizer

NUMERIC = TargetSpec(column="y", kind=TargetKind.NUMERIC, unit=None)
NORMALIZER = RdkitStructureNormalizer()

# TEM-1 beta-lactamase, the first 81 residues of the real protein.
_PROTEIN = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSR"


def _sequences(*rows: str) -> pl.DataFrame:
    return pl.DataFrame({"structure": list(rows), "y": [1.0 + i for i in range(len(rows))]})


def test_a_protein_column_is_read_as_sequences():
    frame = _sequences(_PROTEIN, _PROTEIN[:-1] + "A", _PROTEIN[:-2] + "AA")
    prepared, report = prepare_frame(frame, "structure", (NUMERIC,), NORMALIZER)

    assert report.structure_kind is StructureKind.SEQUENCE
    assert report.valid_rows == 3
    assert report.invalid == []
    # Stored as uploaded, not rewritten: a sequence has no second spelling to resolve,
    # and `embed.py` has to see the residues the scientist actually submitted.
    assert prepared["structure"].to_list()[0] == _PROTEIN


def test_a_smiles_column_is_unaffected():
    """The backward-compatibility guarantee: a molecule column takes the path it always
    took, and the stored structures are still canonical SMILES."""
    frame = pl.DataFrame({"structure": ["CCO", "c1ccccc1"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "structure", (NUMERIC,), NORMALIZER)

    assert report.structure_kind is StructureKind.MOLECULE
    assert sorted(prepared["structure"].to_list()) == ["CCO", "c1ccccc1"]


def test_an_ambiguous_short_string_is_a_molecule():
    """`CCN` is ethylamine *and* Cys-Cys-Asn. SMILES is tried first precisely so that a
    chemistry app answers ethylamine."""
    frame = pl.DataFrame({"structure": ["CCN", "CCO"], "y": [1.0, 2.0]})
    _, report = prepare_frame(frame, "structure", (NUMERIC,), NORMALIZER)

    assert report.structure_kind is StructureKind.MOLECULE


def test_unreadable_rows_stay_unreadable_rather_than_becoming_a_sequence():
    """The regression this nearly shipped. `NOTAMOLECULE` and `ALSOGARBAGE` are spelled
    entirely from residue letters, so an alphabet test alone calls them peptides and turns
    a file with nothing usable in it into a protein dataset that trains happily."""
    frame = pl.DataFrame({"structure": ["NOTAMOLECULE", "ALSOGARBAGE"], "y": [1.0, 2.0]})
    prepared, report = prepare_frame(frame, "structure", (NUMERIC,), NORMALIZER)

    assert report.structure_kind is StructureKind.MOLECULE
    assert prepared.height == 0
    assert len(report.invalid) == 2


@pytest.mark.parametrize("text", ["NOTAMOLECULE", "ALSOGARBAGE", "ACDEFGHIKL", "CCN", ""])
def test_short_strings_are_never_sequences(text: str):
    assert looks_like_sequence(text) is False


def test_whitespace_gaps_and_case_are_normalized():
    messy = f"  {_PROTEIN[:40].lower()}-{_PROTEIN[40:].lower()}  "
    assert normalize_sequence(messy) == _PROTEIN
    assert looks_like_sequence(messy) is True


def test_a_sequence_column_is_not_scanned_for_salts():
    """`has_multiple_components` parses its argument as SMILES, so on a sequence column it
    would be answering a question about protein with a chemistry parser."""
    _, report = prepare_frame(
        _sequences(_PROTEIN, _PROTEIN[:-1] + "A"), "structure", (NUMERIC,), NORMALIZER
    )
    assert report.salts_flagged == 0


def test_a_report_written_before_this_existed_reads_as_molecules():
    """Every dataset already frozen has no `structure_kind` key at all."""
    assert report_from_dict({"total_rows": 1, "valid_rows": 1}).structure_kind is (
        StructureKind.MOLECULE
    )


def test_the_kind_survives_the_jsonb_round_trip():
    _, report = prepare_frame(
        _sequences(_PROTEIN, _PROTEIN[:-1] + "A"), "structure", (NUMERIC,), NORMALIZER
    )
    stored = report_to_dict(report)

    assert stored["structure_kind"] == "sequence"
    assert report_from_dict(stored).structure_kind is StructureKind.SEQUENCE


@pytest.mark.parametrize("strategy", [SplitStrategy.IDENTITY, SplitStrategy.POSITION])
def test_a_sequence_split_is_refused_on_molecules(strategy: SplitStrategy):
    error = _split_kind_error(strategy, StructureKind.MOLECULE)
    assert error is not None
    assert "small molecules" in str(error)


def test_a_scaffold_split_is_refused_on_sequences():
    error = _split_kind_error(SplitStrategy.SCAFFOLD, StructureKind.SEQUENCE)
    assert error is not None
    assert "ring system" in str(error)
    # The message has to send the reader somewhere useful, not just say no.
    assert "identity split" in str(error) and "position split" in str(error)


@pytest.mark.parametrize(
    ("strategy", "kind"),
    [
        (SplitStrategy.RANDOM, StructureKind.SEQUENCE),
        (SplitStrategy.RANDOM, StructureKind.MOLECULE),
        (SplitStrategy.SCAFFOLD, StructureKind.MOLECULE),
        (SplitStrategy.IDENTITY, StructureKind.SEQUENCE),
        (SplitStrategy.POSITION, StructureKind.SEQUENCE),
    ],
)
def test_the_sensible_combinations_are_allowed(strategy: SplitStrategy, kind: StructureKind):
    assert _split_kind_error(strategy, kind) is None

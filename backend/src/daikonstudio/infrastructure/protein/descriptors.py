"""Physicochemical protein descriptors -- the sequence side of `chem/featurize.py`.

The baseline featurizer, and deliberately not a learned one. A Scorecard's baseline
exists to be the floor a model must clear, so it has to be cheap, interpretable and
independent of whatever the model does -- composition and bulk physicochemistry is the
classical answer for sequences, the way a fingerprint is for molecules.

Biopython's `ProteinAnalysis` rather than our own arithmetic, because the parts worth
having are not the arithmetic: Kyte-Doolittle hydropathy, the Guruprasad instability
weights, and the pKa tables behind the isoelectric point are published constants that are
easy to transcribe subtly wrong and impossible to notice afterwards. `Bio/SeqUtils/
ProtParam.py` is dual licensed under the Biopython License Agreement or BSD-3-Clause.

Rejected on licence, for the record, so this is not revisited: `peptides` (0.5.0) ships
the full GPLv3 text while its PyPI classifier claims MIT -- the shipped licence governs --
and `propy3` is GPL-2.0-only. Both are copyleft and neither can be linked here.
"""

from __future__ import annotations

import numpy as np

#: The 20 standard residues, in a fixed order. The order is part of the feature
#: contract: a stored artifact names this featurizer and rebuilds it from that name
#: alone, so a reordering here would silently re-map every column of every model
#: already fitted.
_STANDARD = "ACDEFGHIKLMNPQRSTVWY"

#: Names in feature order, for the Scorecard's descriptor attribution.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    *(f"fraction {residue}" for residue in _STANDARD),
    "fraction non-standard",
    "length",
    "molecular weight",
    "aromaticity",
    "instability index",
    "hydropathy (GRAVY)",
    "isoelectric point",
    "charge at pH 7",
    "helix fraction",
    "turn fraction",
    "sheet fraction",
)

DIM = len(DESCRIPTOR_NAMES)


def protein_descriptors(sequences: list[str]) -> np.ndarray:
    """Composition and bulk physicochemistry, one row per sequence, aligned with input.

    Shape `(len(sequences), DIM)`, float32, and a pure function of the sequence text --
    no fitted state, nothing length-dependent. That is what lets the engine contract
    rebuild it from a stored artifact knowing only its name, and it is why this is
    composition-based rather than the per-position one-hot the variant-effect literature
    uses as its baseline: one-hot needs a fixed alignment length, which is a parameter,
    and a featurizer with a parameter cannot be rebuilt from its name.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    out = np.zeros((len(sequences), DIM), dtype=np.float32)
    for row, raw in enumerate(sequences):
        sequence = "".join(str(raw).split()).upper()
        # Ambiguity and rare codes (X, B, Z, U, O) are in the alphabet this app accepts,
        # and `ProteinAnalysis` raises `KeyError` on every one of them. They are dropped
        # for the physicochemical half -- hydropathy of an unknown residue is not a
        # number -- and their share is kept as a feature of its own, so a sequence that
        # is mostly unknown is visible to the model rather than silently shortened.
        standard = "".join(residue for residue in sequence if residue in _STANDARD)
        total = len(sequence)
        if total:
            for index, residue in enumerate(_STANDARD):
                out[row, index] = standard.count(residue) / total
            out[row, 20] = (total - len(standard)) / total
        out[row, 21] = float(total)
        if not standard:
            continue
        # Biopython ships no type information, so every call into it needs the same
        # inline ignore `chem/featurize.py` already uses for RDKit.
        analysis = ProteinAnalysis(standard)  # type: ignore[no-untyped-call]
        helix, turn, sheet = analysis.secondary_structure_fraction()  # type: ignore[no-untyped-call]
        out[row, 22:] = (
            analysis.molecular_weight(),  # type: ignore[no-untyped-call]
            analysis.aromaticity(),  # type: ignore[no-untyped-call]
            analysis.instability_index(),  # type: ignore[no-untyped-call]
            analysis.gravy(),  # type: ignore[no-untyped-call]
            analysis.isoelectric_point(),  # type: ignore[no-untyped-call]
            analysis.charge_at_pH(7.0),  # type: ignore[no-untyped-call]
            helix,
            turn,
            sheet,
        )
    return out

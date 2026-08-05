"""The small set of physicochemical descriptors a chemist reads to place a dataset.

Nine, not ninety. RDKit exposes some two hundred descriptors and a page plotting
all of them tells a scientist nothing; these are the ones that answer "what kind
of chemistry is this" -- size, lipophilicity, polarity, flexibility, saturation --
plus the Lipinski counts, because a distribution centred well outside Ro5 space is
a fact about where the data came from, not a fault to be corrected.

Column-oriented (name -> one value per input structure), because every consumer
histograms or correlates a whole column at a time; returning a dict per molecule
would be transposed again at every call site.

A structure RDKit cannot parse yields `None`, never a substituted zero. A zero
molecular weight would land in a histogram bin and be read as a measurement.
"""

from collections.abc import Callable
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

# Ordered: the profile renders them in this order, and "size, then greasiness,
# then polarity" is the order a chemist scans them in.
#
# ignore[attr-defined]: RDKit builds most of `Descriptors`/`Crippen`/`Lipinski`
# at import time, so the installed rdkit-stubs cannot see names that genuinely
# exist at runtime -- the same stub gap that already forces an ignore on
# `GetScaffoldForMol` in `scaffold.py`. The smoke test below is what actually
# holds these names honest.
_DESCRIPTORS: dict[str, Callable[[Any], float]] = {
    "molecular_weight": Descriptors.MolWt,  # type: ignore[attr-defined]
    "clogp": Crippen.MolLogP,  # type: ignore[attr-defined]
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": Lipinski.NumHDonors,  # type: ignore[attr-defined]
    "hba": Lipinski.NumHAcceptors,  # type: ignore[attr-defined]
    "rotatable_bonds": Lipinski.NumRotatableBonds,  # type: ignore[attr-defined]
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "heavy_atoms": Descriptors.HeavyAtomCount,
}

DESCRIPTOR_NAMES: tuple[str, ...] = tuple(_DESCRIPTORS)


def descriptors(smiles_list: list[str]) -> dict[str, list[float | None]]:
    """One column per descriptor, aligned by position with `smiles_list`."""
    columns: dict[str, list[float | None]] = {name: [] for name in _DESCRIPTORS}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        for name, compute in _DESCRIPTORS.items():
            # A descriptor that raises on an otherwise-parseable molecule (a
            # valence RDKit accepts but a descriptor cannot handle) is one
            # unmeasurable value, not a failed profile: record the gap and
            # carry on, the same way an unparseable structure is recorded.
            if mol is None:
                columns[name].append(None)
                continue
            try:
                columns[name].append(float(compute(mol)))
            except (ValueError, RuntimeError):
                columns[name].append(None)
    return columns

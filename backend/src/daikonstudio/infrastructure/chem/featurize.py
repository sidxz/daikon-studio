"""Model inputs. Not to be confused with `chem/descriptors.py`, which computes the nine
physicochemical properties a chemist *reads* on the dataset profile page -- that module
is deliberately "nine, not ninety" for human consumption, this one feeds estimators."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

#: Column order for `rdkit_descriptors`, captured once at import.
#:
#: RDKit adds descriptors between releases, so this list's length and order are a
#: property of the installed rdkit, not a constant. A model fitted under one version and
#: used to predict under another would silently read every column as a different
#: descriptor -- garbage predictions, no exception. That is why the descriptor engine
#: persists these names in its artifact and refuses a mismatch on load rather than
#: trusting the count to have stayed put.
DESCRIPTOR_NAMES: tuple[str, ...] = tuple(name for name, _ in Descriptors.descList)


def ecfp4(smiles_list: list[str]) -> np.ndarray:
    """ECFP4 (Morgan radius 2, 2048 bits). Invalid SMILES yield an all-zero row."""
    rows = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # GetFingerprintAsNumPy already returns a uint8 array of shape (2048,);
            # assign directly rather than round-tripping through tobytes()/frombuffer.
            rows[index] = _GENERATOR.GetFingerprintAsNumPy(mol)
    return rows


def rdkit_descriptors(smiles_list: list[str]) -> np.ndarray:
    """All RDKit 2D descriptors, one row per structure. Anything unmeasurable is NaN.

    NaN rather than 0.0, which is the opposite of `ecfp4`'s convention above and is the
    right one here: a zero bit in a fingerprint means "this substructure is absent", a
    true statement, while a zero molecular weight is a *measurement* that never
    happened. `chem/descriptors.py` makes the same call for the same reason. Three
    things collapse to NaN -- a SMILES RDKit cannot parse (whole row), a descriptor that
    raises on an otherwise fine molecule (`missingVal`, e.g. the Gasteiger-charge
    descriptors on exotic elements), and a descriptor that overflows to +/-inf (`Ipc`
    does this on larger molecules).

    The inf sweep is not optional: XGBoost accepts NaN natively and learns a split
    direction for it, but rejects inf outright with "Input data contains `inf`". So the
    consumer this featurizer exists for fails loudly on the one value we would otherwise
    pass through.

    No scaling, deliberately. Gradient boosting is scale-invariant, so the raw values
    are the honest input; a featurizer that normalised here would be silently wrong for
    any future consumer that is not a tree.
    """
    rows = np.full((len(smiles_list), len(DESCRIPTOR_NAMES)), np.nan, dtype=np.float64)
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # ignore[no-untyped-call]: the installed rdkit-stubs carry no signature for
        # CalcMolDescriptors. Same stub gap that already forces ignores in
        # `chem/descriptors.py` and `scaffold.py`; the tests are what hold it honest.
        values = Descriptors.CalcMolDescriptors(  # type: ignore[no-untyped-call]
            mol, missingVal=float("nan"), silent=True
        )
        rows[index] = list(values.values())
    rows[~np.isfinite(rows)] = np.nan
    return rows

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


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

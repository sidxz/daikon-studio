from rdkit import Chem

# RDLogger.DisableLog("rdApp.*") lives in the chem package's __init__.py, not here, so
# it applies to every submodule (featurize, scaffold, similarity) regardless of which
# one is imported first.


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def has_multiple_components(smiles: str) -> bool:
    return "." in (canonicalize(smiles) or "")

from rdkit import Chem, RDLogger

# Invalid input is expected and reported via the return value, not logged.
# ignore[attr-defined]: installed rdkit-stubs omit DisableLog; RDKit itself has it.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def has_multiple_components(smiles: str) -> bool:
    return "." in (canonicalize(smiles) or "")

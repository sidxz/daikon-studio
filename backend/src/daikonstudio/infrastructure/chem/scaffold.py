from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold. Returns "" for invalid input or acyclic molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    # ignore[no-untyped-call]: installed rdkit-stubs leave GetScaffoldForMol unannotated.
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call]
    return Chem.MolToSmiles(scaffold_mol)

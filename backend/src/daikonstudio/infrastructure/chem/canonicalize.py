from rdkit import Chem

from daikonstudio.infrastructure.chem.helm import helm_to_smiles, looks_like_helm

# RDLogger.DisableLog("rdApp.*") lives in the chem package's __init__.py, not here, so
# it applies to every submodule (featurize, scaffold, similarity) regardless of which
# one is imported first.


def canonicalize(smiles: str) -> str | None:
    """A canonical SMILES for one uploaded structure, or None if it cannot be read.

    Peptides are uploaded as HELM, which is converted here and nowhere else: a dataset
    stores the canonical SMILES, so nothing after ingestion -- fingerprints, the scaffold
    split, similarity, descriptors, the chemical-space map -- has to know the upload was
    HELM. The converted peptide then takes the ordinary SMILES path below, which is what
    guarantees the stored structure is one RDKit can read back.
    """
    if looks_like_helm(smiles):
        converted = helm_to_smiles(smiles)
        if converted is None:
            return None
        smiles = converted
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def has_multiple_components(smiles: str) -> bool:
    return "." in (canonicalize(smiles) or "")

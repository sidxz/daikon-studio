"""The concrete `StructureNormalizer` (application/ports) adapter, backed by RDKit.

A thin wrapper, not a reimplementation: every method delegates straight to Task 6's
module-level functions. Importing this module still imports the `chem` package, so
the RDKit log suppression in `chem/__init__.py` still applies.
"""

from daikonstudio.infrastructure.chem.canonicalize import canonicalize, has_multiple_components
from daikonstudio.infrastructure.chem.scaffold import murcko_scaffold


class RdkitStructureNormalizer:
    def canonicalize(self, smiles: str) -> str | None:
        return canonicalize(smiles)

    def has_multiple_components(self, smiles: str) -> bool:
        return has_multiple_components(smiles)

    def murcko_scaffold(self, smiles: str) -> str:
        return murcko_scaffold(smiles)

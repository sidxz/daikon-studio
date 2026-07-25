from rdkit import RDLogger

# Invalid structures are expected input reported through return values (None / "" /
# all-zero rows), not logged. This must live at package level (not in canonicalize.py)
# so it fires for every submodule of daikonstudio.infrastructure.chem, including ones
# that import rdkit directly without importing canonicalize.
# ignore[attr-defined]: installed rdkit-stubs omit DisableLog; RDKit itself has it.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

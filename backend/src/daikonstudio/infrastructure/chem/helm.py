"""HELM peptide notation to SMILES, read at the one place structures are canonicalized.

Peptide assays arrive as HELM -- `PEPTIDE1{A.[dA].[Aib]}$$$$` -- and that was the whole
gap for peptides here. Fingerprints already lead the non-canonical peptide benchmarks,
so nothing downstream needed a peptide-specific path; the structures simply could not be
read. Converting HELM in `canonicalize` means a peptide dataset becomes peptide SMILES
once, at ingestion, and ECFP, the Bemis-Murcko scaffold split, Tanimoto similarity, the
descriptors, the dataset profile, the Scorecard's applicability domain and the
chemical-space map all work on it unchanged.

`helmkit` (MIT) does the building. It covers what the chemistry needs: D- against
L-residues as distinct stereocenters, non-canonical monomers such as Aib, unknown
monomers given as inline SMILES, head-to-tail cyclics and disulfide bridges.

BILN is deliberately not read here. It is a second peptide notation, it would need a
second library, and no dataset in hand uses it.
"""

from __future__ import annotations

import logging

from rdkit import Chem

logger = logging.getLogger(__name__)


def looks_like_helm(text: str) -> bool:
    """Whether `text` is HELM rather than SMILES.

    HELM's grammar requires both characters -- `CHAIN{sequence}` for every polymer, and
    the three `$` separators that close the string -- and requiring *both* is what makes
    the test exact. `$` on its own would not separate the notations: it is SMILES'
    quadruple-bond symbol (RDKit reads `O->[Os]$[Os]<-O`) and it delimits atom labels in
    CXSMILES (`|$R1;;$|`). `{` is the character no SMILES or CXSMILES form produces
    unescaped, so the conjunction has no false positives and nothing in it to tune.
    """
    return "{" in text and "$" in text


def helm_to_smiles(text: str) -> str | None:
    """The SMILES for a HELM string, or None if it cannot be read.

    None rather than an exception because an unreadable structure is ordinary input at
    this boundary: `prepare_frame` reports the row as invalid and keeps the rest of the
    upload, exactly as it does for an unparseable SMILES. helmkit's messages are specific
    enough to act on ("Monomer Zq not in monomer library and is not a valid SMILES
    string"), so they are logged rather than discarded.

    RDKit's own console chatter -- it prints a parse error for a bad inline-SMILES
    monomer -- is already silenced: `chem/__init__.py` disables `rdApp.*` for every
    submodule of this package, which includes this one.
    """
    # Imported inside the function, the house rule for this repo's chemistry and ML
    # dependencies (`engines/molformer_xl.py`, `protein/embed.py`): the `chem` package is
    # imported on every API boot and by every worker, and only a HELM upload reaches
    # this line.
    import helmkit

    try:
        return Chem.MolToSmiles(helmkit.Molecule(text).mol)
    # Every malformed string tried raises ValueError, but the no-raise guarantee is the
    # point of this function -- one unreadable row must not end a scientist's upload.
    except Exception as exc:
        # debug, not warning: the scientist already gets the row back as an invalid row
        # in the validation report, and a column of bad HELM would otherwise write one
        # warning per row into the worker's log.
        logger.debug("Could not read '%s' as HELM: %s", text[:120], exc)
        return None

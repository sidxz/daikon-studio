from __future__ import annotations

from enum import StrEnum

#: Residues a sequence may contain after `normalize_sequence` has done its work: the 20
#: standard amino acids, the ambiguity codes B (Asx), X (any) and Z (Glx), the two
#: genetically encoded extras U (selenocysteine) and O (pyrrolysine), and `*` for a stop.
#: A gap is not here because it is stripped, not accepted.
_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO*")

#: A sequence has to be at least this long before this module will call it one.
#:
#: The floor is high because the residue alphabet is close to useless as a test on short
#: strings -- `NOTAMOLECULE` and `ALSOGARBAGE` are both spelled entirely from residue
#: letters, and so is most of any capitalized English. Without a floor, a column of gene
#: names or in-house compound codes would be read as protein, which is the worst failure
#: this codebase has: a model that trains, scores and reports on nonsense rather than
#: refusing. A dataset that is genuinely unreadable must stay unreadable.
#:
#: 25 is principled rather than tuned. Short peptides do not need this path at all: they
#: arrive as HELM and `chem/helm.py` converts them to SMILES, so they are molecules by the
#: time anything asks. What is left for the sequence path is canonical proteins, and those
#: are hundreds of residues -- the smallest real one here, TEM-1, is 286.
#:
#: If a short canonical-peptide dataset ever needs this, the answer is not a lower floor
#: but an explicit declaration from the uploader. Inference cannot be made safe at that
#: length, and guessing is what this constant exists to prevent.
_MINIMUM_RESIDUES = 25


class StructureKind(StrEnum):
    """What the structure column holds -- small molecules, or amino-acid sequences.

    Recorded on the Dataset rather than re-derived wherever it is needed, because the two
    are genuinely ambiguous and only ingestion is in a position to decide. `CCN` is valid
    SMILES (ethylamine) *and* a valid peptide (Cys-Cys-Asn); so is any sequence written
    only from the residues whose letters are also organic-subset atoms. Ingestion settles
    it with RDKit; a cheaper test used later could disagree, and a disagreement here is
    not a crash but a silently wrong experiment.
    """

    MOLECULE = "molecule"
    SEQUENCE = "sequence"


def normalize_sequence(text: str) -> str:
    """Upper-case, with whitespace and alignment gaps removed.

    All the canonicalization a sequence gets, and all it needs: unlike SMILES there is no
    choice of spelling to resolve, so the stored value stays the one the scientist
    uploaded apart from presentation. `embed.py:_clean` applies the same rules before
    embedding, so what is stored is what is featurized.
    """
    return "".join(text.split()).replace("-", "").replace(".", "").upper()


def looks_like_sequence(text: str) -> bool:
    """True when `text` reads as an amino-acid sequence.

    Only ever asked of a value RDKit has already refused, so this never has to out-argue
    SMILES -- which is what keeps every existing molecule dataset behaving exactly as it
    did.
    """
    residues = normalize_sequence(text)
    return len(residues) >= _MINIMUM_RESIDUES and all(residue in _RESIDUES for residue in residues)

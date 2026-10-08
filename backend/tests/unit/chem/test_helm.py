"""HELM peptide ingestion: the notation becomes SMILES inside `canonicalize`.

The design claim these pin is that nothing downstream needs a peptide path -- so the
tests go through `canonicalize`, the function the dataset pipeline actually calls, and
one of them fingerprints the result.
"""

import pytest
from rdkit import Chem

from daikonstudio.infrastructure.chem.canonicalize import canonicalize
from daikonstudio.infrastructure.chem.featurize import ecfp4
from daikonstudio.infrastructure.chem.helm import helm_to_smiles, looks_like_helm

_PENTAPEPTIDE = "PEPTIDE1{A.G.C.K.Y}$$$$"
_L_ALANINES = "PEPTIDE1{A.A.A}$$$$"
_D_ALANINES = "PEPTIDE1{[dA].[dA].[dA]}$$$$"
#: The two cysteine side chains (R3) bonded to each other, which closes the backbone
#: into a macrocycle.
_DISULFIDE = "PEPTIDE1{C.A.A.A.C}$PEPTIDE1,PEPTIDE1,1:R3-5:R3$$$"


def test_looks_like_helm_accepts_helm_and_rejects_smiles():
    """`{` and `$` are both required by HELM, and `{` is what no SMILES or CXSMILES form
    produces unescaped, so the SMILES features that come closest -- square brackets,
    ring-closure digits, a multi-component salt -- still cannot be mistaken for it."""
    assert looks_like_helm(_PENTAPEPTIDE)
    assert looks_like_helm(_DISULFIDE)
    for smiles in ("CCO", "C[C@H](N)C(=O)O", "c1ccc2ccccc2c1", "C1CC1", "[Na+].CC(=O)[O-]"):
        assert not looks_like_helm(smiles), smiles


def test_a_dollar_sign_alone_does_not_make_a_string_helm():
    """`$` is legal SMILES -- it is the quadruple bond -- so `{` is what actually
    separates the two notations. Pinned because the opposite claim is the tempting one
    and it would license dropping the `{` test."""
    quadruple_bond = "O->[Os]$[Os]<-O"

    assert Chem.MolFromSmiles(quadruple_bond) is not None, "premise: RDKit reads this"
    assert looks_like_helm(quadruple_bond) is False


def test_a_peptide_converts_and_round_trips_through_rdkit():
    smiles = helm_to_smiles(_PENTAPEPTIDE)

    assert smiles is not None
    assert Chem.MolFromSmiles(smiles) is not None
    # Already canonical, so ingestion stores exactly what the conversion produced.
    assert canonicalize(_PENTAPEPTIDE) == smiles


def test_d_and_l_residues_are_different_structures():
    """The load-bearing property of the whole build: a D-residue and an L-residue are
    the point of most peptide assays, so ingestion must not flatten them into one
    structure. Both must survive HELM -> SMILES, or every downstream fingerprint,
    scaffold and similarity reads two different molecules as the same one."""
    d_form = canonicalize(_D_ALANINES)
    l_form = canonicalize(_L_ALANINES)

    assert d_form is not None and l_form is not None
    assert d_form != l_form


def test_a_disulfide_bridge_closes_a_ring():
    smiles = canonicalize(_DISULFIDE)

    assert smiles is not None
    mol = Chem.MolFromSmiles(smiles)
    assert mol.GetRingInfo().NumRings() == 1
    assert any(
        bond.GetBeginAtom().GetSymbol() == "S" and bond.GetEndAtom().GetSymbol() == "S"
        for bond in mol.GetBonds()
    )


def test_a_non_canonical_residue_converts():
    """Aib is not one of the twenty, and it is the reason a peptide dataset exists at
    all -- a notation that only covered the canonical residues would be pointless."""
    smiles = canonicalize("PEPTIDE1{A.[Aib].G}$$$$")

    assert smiles is not None
    assert smiles != canonicalize("PEPTIDE1{A.A.G}$$$$")


@pytest.mark.parametrize(
    "text",
    [
        "PEPTIDE1{A.Zq.G}$$$$",  # a monomer that is neither in the library nor a SMILES
        "PEPTIDE1{A.[Zq].G}$$$$",  # the same, offered as inline SMILES
        "PEPTIDE1{}$$$$",  # an empty sequence
        "{$",  # passes the detection test and means nothing
        "PEPTIDE1{A.G}$PEPTIDE1,PEPTIDE1,9:R2-1:R1$$$",  # a bond to a residue not there
    ],
)
def test_unreadable_helm_returns_none_rather_than_raising(text):
    """One bad row is reported as an invalid row; it must never end the upload."""
    assert helm_to_smiles(text) is None
    assert canonicalize(text) is None


def test_the_converted_peptide_fingerprints_like_any_other_structure():
    """The reason the conversion lives in `canonicalize`: ECFP, the scaffold split,
    similarity and the descriptors all read the stored column and need no peptide
    branch of their own."""
    assert ecfp4([canonicalize(_PENTAPEPTIDE)]).sum() > 0

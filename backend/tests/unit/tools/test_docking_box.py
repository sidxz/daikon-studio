"""The box a pocket implies. Pure geometry: no binary, no lane.

A pocket carries residues and no coordinates, so this is the step between an approved
site and anything that docks into it, and the place where the two can silently disagree.
"""

from __future__ import annotations

import pytest

from daikonstudio.infrastructure.tools.impl.structure import BOX_MINIMUM, box_for_pocket


def _atom(serial: int, resseq: int, x: float, y: float, z: float, chain: str = "A") -> str:
    """One ATOM line in the fixed columns a PDB uses."""
    return (
        f"ATOM  {serial:>5} CA   ALA {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C"
    )


def test_the_box_is_centred_on_the_lining_and_padded_around_it() -> None:
    text = "\n".join(
        [
            _atom(1, 10, 0.0, 0.0, 0.0),
            _atom(2, 11, 20.0, 10.0, 0.0),
            _atom(3, 99, 500.0, 500.0, 500.0),  # a residue outside the pocket
        ]
    )
    pocket = [{"chain": "A", "resname": "ALA", "resseq": 10}, {"chain": "A", "resseq": 11}]

    box = box_for_pocket(text, pocket, padding=4.0)

    # Centred between the two lining atoms, and the far residue is ignored entirely.
    assert (box["center_x"], box["center_y"], box["center_z"]) == (10.0, 5.0, 0.0)
    # Each axis is its extent plus padding on both sides: 20 + 8 across, 10 + 8 up. The
    # third has no extent at all, and 0 + 8 is below the floor, so it takes the floor.
    assert box["size_x"] == 28.0
    assert box["size_y"] == 18.0
    assert box["size_z"] == BOX_MINIMUM
    assert (box["n_residues"], box["n_atoms"]) == (2, 2)


def test_a_pocket_from_another_structure_refuses_rather_than_guesses() -> None:
    text = _atom(1, 10, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="not the same molecule"):
        box_for_pocket(text, [{"chain": "A", "resseq": 777}])


def test_only_atom_records_count_so_a_cofactor_cannot_move_the_box() -> None:
    """Pocket residues are read from ATOM lines, so the box must be too: a HETATM ligand
    sitting in the site would otherwise drag the centre onto itself."""
    text = "\n".join(
        [
            _atom(1, 10, 0.0, 0.0, 0.0),
            "HETATM    2 ZN    ZN A  10     100.000 100.000 100.000  1.00  0.00          ZN",
        ]
    )
    box = box_for_pocket(text, [{"chain": "A", "resseq": 10}])
    assert (box["center_x"], box["n_atoms"]) == (0.0, 1)


def test_pose_geometry_is_withheld_and_the_kind_reads_as_a_table() -> None:
    """A pose is a structure. It must never reach the agent's context, and the poses kind
    has to be a table so it gets the generic reader rather than needing one of its own."""
    import polars as pl

    from daikonstudio.application.tools import codecs

    poses = pl.DataFrame(
        {
            "compound": ["a"],
            "smiles": ["CCO"],
            "pose_rank": [1],
            "affinity_kcal_mol": [-7.1],
            "pose_pdbqt": ["ATOM      1  C   LIG     1       0.000   0.000   0.000"],
        }
    )
    assert "poses" in codecs.TABLE_KINDS
    _, summary = codecs.encode("poses", poses)
    assert "pose_pdbqt" in summary["withheld"]
    assert "affinity_kcal_mol" not in summary["withheld"]


def test_docking_is_declared_and_implemented() -> None:
    from daikonstudio.infrastructure.tools.catalog import default_registry
    from daikonstudio.infrastructure.tools.runtime import IMPLEMENTATIONS

    tool = default_registry().get("dock_compounds")
    assert tool.lane == "chem-tools"  # it needs the vina binary
    # The receptor comes back too, so a pose can be drawn in what it was docked into.
    assert [slot.kind for slot in tool.outputs] == ["poses", "structure"]
    assert "dock_compounds" in IMPLEMENTATIONS


def test_a_receptor_without_hydrogens_is_refused_not_analysed() -> None:
    """ProLIF converts the receptor through MDAnalysis, which requires explicit hydrogens.
    Given none it does not raise: it segfaults, taking the runner with it. So the guard
    that refuses first is the whole safety of this tool, not a nicety."""
    import polars as pl
    import pytest

    from daikonstudio.infrastructure.tools.impl.structure import pose_interactions

    bare = _atom(1, 10, 0.0, 0.0, 0.0)  # a carbon alpha and nothing else
    poses = pl.DataFrame(
        {
            "compound": ["a"],
            "pose_rank": [1],
            "affinity_kcal_mol": [-7.0],
            "pose_pdbqt": ["ATOM      1  C   UNL     1       0.000   0.000   0.000"],
        }
    )
    with pytest.raises(ValueError, match="no hydrogens"):
        pose_interactions(None, {"poses": poses, "receptor": {"text": bare}})


def test_hydrogens_are_counted_by_element_and_by_name() -> None:
    from daikonstudio.infrastructure.tools.impl.structure import _has_hydrogens

    heavy = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C"
    # Element column present.
    light = "ATOM      2  HA  ALA A   1       1.000   0.000   0.000  1.00  0.00           H"
    # Element column missing: the atom name has to carry it.
    unlabelled = "ATOM      3  H   ALA A   1       2.000   0.000   0.000  1.00  0.00"
    assert not _has_hydrogens(heavy)
    assert _has_hydrogens("\n".join([heavy, light]))
    assert _has_hydrogens("\n".join([heavy, unlabelled]))
    assert not _has_hydrogens("")


def test_interactions_are_declared_and_implemented() -> None:
    from daikonstudio.infrastructure.tools.catalog import default_registry
    from daikonstudio.infrastructure.tools.runtime import IMPLEMENTATIONS

    tool = default_registry().get("pose_interactions")
    assert tool.lane == "chem-tools"
    assert [slot.kind for slot in tool.inputs] == ["poses", "structure"]
    assert [slot.kind for slot in tool.outputs] == ["interactions"]
    assert "pose_interactions" in IMPLEMENTATIONS


def test_a_pose_fails_on_anything_that_is_not_true() -> None:
    """A check that could not run reads as a failure: a pose that cannot be judged is
    not a pose to trust. The names come back so the reader knows what to disbelieve."""
    import pandas as pd

    from daikonstudio.infrastructure.tools.impl.structure import _verdict

    passing = pd.Series({"sanitization": True, "minimum_distance_to_protein": True})
    assert _verdict(passing) == (True, "")
    failing = pd.Series(
        {"sanitization": True, "minimum_distance_to_protein": False, "internal_energy": None}
    )
    assert _verdict(failing) == (False, "minimum_distance_to_protein, internal_energy")


def test_validity_is_declared_and_implemented() -> None:
    from daikonstudio.infrastructure.tools.catalog import default_registry
    from daikonstudio.infrastructure.tools.runtime import IMPLEMENTATIONS

    tool = default_registry().get("check_poses")
    assert tool.lane == "chem-tools"
    assert [slot.kind for slot in tool.inputs] == ["poses", "structure"]
    # Poses in, poses out, with the receptor carried along so the survivors can still be
    # drawn in what they were docked into.
    assert [slot.kind for slot in tool.outputs] == ["poses", "structure"]
    assert "check_poses" in IMPLEMENTATIONS

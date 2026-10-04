"""A binary dataset large enough to tune a decision cutoff on."""

from __future__ import annotations

_RINGS = (
    "c1ccccc1",
    "c1ccsc1",
    "C1CCCCC1",
    "C1CCCC1",
    "C1CCC1",
    "C1CC1",
    "C1CCCCCC1",
    "c1ccc2ccccc2c1",
    "c1ccc(F)cc1",
    "c1ccc(Cl)cc1",
    "c1ccc(C)cc1",
    "C1CCC2CCCCC2C1",
)


def tunable_csv() -> bytes:
    """504 distinct compounds with a binary `y`: every alcohol inactive, two thirds of
    the amines active (about 40 % active overall).

    An 80/10/10 random split leaves about 50 validation rows, which holds at least
    `MIN_CUTOFF_CLASS_COUNT` of each class (24 and 20 actives under seeds 7 and 1). The
    20-compound fixtures leave 2. Amines and alcohols are told apart by one atom, so a
    fingerprint model separates them cleanly: a tuned cutoff sits on the model's lowest
    validation active, not at 0.5.
    """
    rows = []
    for ring in _RINGS:
        for length in range(1, 26):
            rows.append(f"{ring}{'C' * length}O,0")
            if length % 3:
                rows.append(f"{ring}{'C' * length}N,1")
    return ("smiles,y\n" + "\n".join(rows) + "\n").encode()

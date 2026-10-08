"""`_cluster_ids` parsing, without the `mmseqs` binary.

The end-to-end test needs MMseqs2 installed, so CI skips it and these are the only
coverage the parser gets there. What they pin is the determinism `assign_split` promises:
identical `(frame, seed)` yields identical partitions, which holds only if the ids depend
on the input order and not on the order MMseqs2 happened to write its rows.
"""

from pathlib import Path

import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.infrastructure.protein.cluster import _cluster_ids


def _tsv(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "clustered_cluster.tsv"
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


#: Members 0 and 2 under one representative, 1 and 3 under another.
_ROWS = ["0\t0", "0\t2", "1\t1", "1\t3"]


def test_ids_are_dense_from_zero(tmp_path):
    assert _cluster_ids(_tsv(tmp_path, _ROWS), 4) == [0, 1, 0, 1]


def test_ids_follow_input_order_not_file_order(tmp_path):
    """The load-bearing property: reordering MMseqs2's rows must not renumber anything.
    Here the file leads with member 1, so numbering by file order would give it id 0."""
    shuffled = ["1\t3", "0\t2", "1\t1", "0\t0"]

    assert _cluster_ids(_tsv(tmp_path, shuffled), 4) == _cluster_ids(_tsv(tmp_path, _ROWS), 4)


def test_a_member_missing_from_the_file_is_rejected(tmp_path):
    """Never silently co-clustered: the message names how many of how many are missing,
    because that is the number a scientist can act on."""
    with pytest.raises(ValidationError, match="2 of 4"):
        _cluster_ids(_tsv(tmp_path, ["0\t0", "1\t1"]), 4)


def test_blank_lines_are_skipped(tmp_path):
    assert _cluster_ids(_tsv(tmp_path, ["0\t0", "", "0\t1", "  "]), 2) == [0, 0]

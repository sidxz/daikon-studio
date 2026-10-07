"""MMseqs2 homology clustering -- the sequence counterpart of a Bemis-Murcko scaffold.

Why an external binary rather than a few lines of Python: leakage between homologous
sequences is the single largest source of overstated protein-model performance, it
persists at sequence-identity thresholds as low as 0.2, and homology partitioning costs
up to 50% of apparent performance. Approximating the clustering would be approximating
the one thing the split exists to guarantee. MMseqs2 is MIT, and measured here at 5 s
single-threaded for 33,069 sequences, so there is nothing to trade away.

Measured on real ProteinGym assays, both behaviours this module is relied on for:

* 33,069 sequences drawn from 9 unrelated proteins cluster into **exactly** 9 groups,
  each one precisely a protein. Cross-family separation works.
* 40 single-substitution variants of one parent collapse into **one** cluster -- which
  is correct, and is why this strategy is wrong for single-parent variant data.
  `_position_labels` serves that regime; see `assign_split.py`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from daikonstudio.domain.shared.errors import ValidationError

logger = logging.getLogger(__name__)

#: Single-threaded on purpose. `assign_split`'s module docstring promises that identical
#: `(frame, seed)` yields identical partitions across processes, and a split that
#: reshuffled with the host's core count would quietly break that. Verified identical
#: across runs and across 1-vs-8 threads at 33k sequences, so this pins a guarantee that
#: already held rather than buying one -- and it costs 5 s at 33k, against ~2 s at 8.
_THREADS = "1"


class Mmseqs2Clusterer:
    """`SequenceClusterer` backed by `mmseqs easy-cluster`."""

    def cluster(self, sequences: list[str], *, min_identity: float, coverage: float) -> list[int]:
        if not sequences:
            return []

        binary = shutil.which("mmseqs")
        if binary is None:
            raise ValidationError(
                "An identity-clustered split needs the MMseqs2 sequence-clustering "
                "tool, which is not installed on this machine. An administrator can "
                "install it, or you can choose a different split strategy."
            )

        with tempfile.TemporaryDirectory(prefix="daikon-mmseqs-") as workspace:
            root = Path(workspace)
            fasta = root / "input.fasta"
            # Indices as identifiers: the sequence text itself cannot be an identifier
            # (duplicates collide, and FASTA headers cannot hold arbitrary text), and an
            # index is what the caller needs back anyway.
            fasta.write_text(
                "".join(f">{index}\n{sequence}\n" for index, sequence in enumerate(sequences))
            )
            prefix = root / "clustered"
            result = subprocess.run(
                [
                    binary,
                    "easy-cluster",
                    str(fasta),
                    str(prefix),
                    str(root / "tmp"),
                    "--min-seq-id",
                    str(min_identity),
                    "-c",
                    str(coverage),
                    "--threads",
                    _THREADS,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                # MMseqs2 is chatty on success and the failure reason is at the end.
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
                logger.error("mmseqs easy-cluster failed: %s", " | ".join(tail))
                raise ValidationError(
                    "Clustering these sequences by identity failed. Check that the "
                    "structure column holds amino-acid sequences."
                )
            return _cluster_ids(prefix.with_name(prefix.name + "_cluster.tsv"), len(sequences))


def _cluster_ids(tsv: Path, count: int) -> list[int]:
    """Dense cluster ids per input index, read from MMseqs2's `representative<TAB>member`.

    Ids are numbered by first appearance **in input order**, not in the order MMseqs2
    happened to write its rows. That keeps the output a pure function of the input even
    if a future MMseqs2 reorders the file, which is the determinism the split promises.
    """
    representative_of: dict[int, str] = {}
    for line in tsv.read_text().splitlines():
        if not line.strip():
            continue
        representative, member = line.split("\t")
        representative_of[int(member)] = representative

    missing = count - len(representative_of)
    if missing:
        # Never silently drop a row into a wrong cluster: a sequence with no cluster
        # line would otherwise share an id with whatever happened to be first.
        raise ValidationError(
            f"Identity clustering returned no cluster for {missing} of {count} "
            f"sequences. Check that the structure column holds amino-acid sequences."
        )

    ids: list[int] = []
    numbering: dict[str, int] = {}
    for index in range(count):
        representative = representative_of[index]
        if representative not in numbering:
            numbering[representative] = len(numbering)
        ids.append(numbering[representative])
    logger.info("MMseqs2 grouped %d sequences into %d clusters.", count, len(numbering))
    return ids

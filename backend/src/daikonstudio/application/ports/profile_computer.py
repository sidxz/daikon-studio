from typing import Any, Protocol

from daikonstudio.domain.data.structure_kind import StructureKind
from daikonstudio.domain.data.target import TargetSpec


class ProfileComputer(Protocol):
    async def compute(
        self,
        *,
        raw: bytes,
        structure_column: str,
        target: TargetSpec,
        structure_kind: StructureKind,
    ) -> dict[str, Any]:
        """`build_profile` over one frozen snapshot, as `profile_to_dict` stores it.

        Bytes in and the storable dict out -- deliberately not a frame in and a
        `DatasetProfile` out -- because an implementation runs the computation in
        another process, and only plain data can cross that boundary. The dict is
        the small end of the computation: the ~200 descriptor columns of a
        400k-compound dataset are gigabytes of Python floats, and they stay in the
        process that built them while twenty-odd histograms come back.

        Reading the snapshot and saving the result stay with the caller
        (`GetDatasetProfile`), which owns the cache; this is the computation alone.
        """
        ...

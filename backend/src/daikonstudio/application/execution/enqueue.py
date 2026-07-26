"""Port for handing a Run off to background execution.

Lives beside the execution use cases rather than under `application/ports/`,
mirroring how the sibling prot-cellar project places its `JobEnqueuer` in
`application/imports/job_enqueuer.py`: this port only ever has one bounded
context as a caller, so there's no shared-abstraction reason to park it in the
generic ports package alongside `BlobStore` and `StructureNormalizer`, which
several contexts use. Either location satisfies the layering contract --
`application` still never imports `infrastructure` -- so this is a naming/
grouping choice, not an architectural one.

Two implementations sit behind this Protocol (`infrastructure/worker.py`):
`ArqEnqueuer` pushes to Redis for a separate worker process to pick up;
`InlineEnqueuer` runs the job in the caller's own process, selected by
`STUDIO_INLINE_JOBS=1` so tests and local dev need no Valkey at all -- the
same dual-implementation trick as chem-cellar's `NullJobOrchestrator`.
"""

import uuid
from typing import Protocol


class JobEnqueuer(Protocol):
    async def enqueue(self, run_id: uuid.UUID) -> None: ...

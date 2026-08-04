"""Fetch, verify and cache pretrained foundation-model weights.

Its own module rather than more lines in `chemprop_dmpnn.py`: downloading and
checksumming a file has nothing to do with message passing, and a second engine
wanting a foundation model should not import a D-MPNN to get it.

No torch import here. This module deals in bytes on disk, which is what lets it
be tested on a worker that has no gpu extra installed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from daikonstudio.domain.shared.errors import ValidationError

_CHUNK = 1024 * 1024


@dataclass(frozen=True, kw_only=True)
class WeightSet:
    url: str
    md5: str
    filename: str


# The published artifacts, keyed by the string an engine's `pretrained` condition
# offers. Adding a set is one entry -- but note a chemprop v1-format checkpoint
# would also need `chemprop convert` and the V1 atom featurizer, which this
# structure cannot express. That is a reason to extend it then, not now.
WEIGHT_SETS: dict[str, WeightSet] = {
    "CheMeleon": WeightSet(
        url="https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        md5="6a80b54fdb7de37ef0374d302f01e8ce",
        filename="chemeleon_mp.pt",
    ),
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def weights_path(name: str, directory: str) -> Path:
    """The cached weights for `name`, downloading them once if absent.

    Bytes from the internet become model weights, so the checksum is a trust
    boundary and not ceremony: a truncated download otherwise surfaces as an
    unreadable-tensor error deep inside torch with no hint that the network was
    the cause.

    Download-to-temp-then-rename because two workers can share one machine, and
    `os.replace` is atomic within a filesystem -- so a second worker either sees
    no file or sees a whole one, never a half-written one.
    """
    try:
        weight_set = WEIGHT_SETS[name]
    except KeyError:
        raise ValidationError(
            f"Unknown pretrained weight set '{name}'. Available: {', '.join(sorted(WEIGHT_SETS))}."
        ) from None

    cache = Path(directory).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / weight_set.filename
    if target.exists() and _md5(target) == weight_set.md5:
        return target

    handle, staging = tempfile.mkstemp(dir=cache, suffix=".partial")
    os.close(handle)
    staged = Path(staging)
    try:
        urlretrieve(weight_set.url, staged)
        actual = _md5(staged)
        if actual != weight_set.md5:
            raise ValidationError(
                f"Downloaded weights for '{name}' failed their checksum "
                f"(expected {weight_set.md5}, got {actual}). The download from "
                f"{weight_set.url} was corrupt or truncated; nothing was cached."
            )
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    return target

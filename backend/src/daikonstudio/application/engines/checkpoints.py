"""A training run's saved progress: what a stopped run resumes from.

Spec: `docs/superpowers/specs/2026-10-03-resumable-training-design.md`. Each completed
fit is saved as a packed `TrainResult`; chemprop and MoLFormer also save their Lightning
training state while a fit runs. All of it lives under one root per run, so a successful
run, a Start over and a deleted dataset can each remove it in one call.

Saving is best effort: every failure here is logged and swallowed. A save that fails
costs a resume some progress; a save that raised would cost the run itself.

Integrity: each name has two slots, `{name}.a` and `{name}.b`. A save writes the slot
the marker does NOT point at, then rewrites the small JSON marker -- slot, length,
sha256 and the fingerprint it was saved under -- to point at it. The marker is written
last, so a save interrupted before it leaves the previous save loadable; a marker whose
fingerprint or checksum disagrees reads as nothing saved. Two fixed slots rather than
content-addressed blobs because a runner cannot delete a single blob: storage stays at
twice one save however many times a long fit saves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import lzma
import re
import struct
import uuid
from typing import Any

from daikonstudio.application.engines.context import TrainResult
from daikonstudio.application.ports.blob_store import BlobStore

logger = logging.getLogger(__name__)

#: How often an in-progress neural fit saves its training state. A runner overrides it
#: with STUDIO_CHECKPOINT_INTERVAL_SECONDS.
DEFAULT_INTERVAL_SECONDS = 600.0

#: The packed-`TrainResult` layout. Part of every fit's fingerprint: bump it when
#: `pack_result` changes and older saves read as absent instead of misreading.
RESULT_FORMAT = "2"

#: Where a neural engine keeps its Lightning training state inside a fit's scope. Named
#: here so `discard` can free it without importing the engines.
TRAINING_STATE_SCOPE = "lightning"
TRAINING_STATE = "training-state"

_SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def checkpoint_root(workspace_id: uuid.UUID, dataset_id: uuid.UUID, run_id: uuid.UUID) -> str:
    """Under the dataset's folder on purpose: `DeleteDataset` removes that folder with
    `delete_prefix`, which takes its runs' saved progress with it."""
    return f"{workspace_id}/datasets/{dataset_id}/runs/{run_id}/checkpoints/"


def _require_safe(name: str) -> None:
    # Names become blob key segments; the runner API refuses `..` and the like, and a
    # user-chosen target column could contain anything. Scopes are code-chosen names.
    if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"Not a safe checkpoint name: {name!r}")


class Checkpoints:
    def __init__(
        self,
        store: BlobStore,
        root: str,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        fingerprint: dict[str, str] | None = None,
    ) -> None:
        if not root.endswith("/"):
            raise ValueError(f"A checkpoint root is a folder key ending in '/', got {root!r}")
        self._store = store
        self.root = root
        self.interval_seconds = interval_seconds
        self._fingerprint = dict(fingerprint or {})

    def scoped(self, name: str, **fingerprint: str) -> Checkpoints:
        """A sub-store under `{root}{name}/` whose saves also record `fingerprint`."""
        _require_safe(name)
        return Checkpoints(
            self._store,
            f"{self.root}{name}/",
            interval_seconds=self.interval_seconds,
            fingerprint={**self._fingerprint, **fingerprint},
        )

    def save(self, name: str, data: bytes) -> bool:
        _require_safe(name)
        digest = hashlib.sha256(data).hexdigest()
        marker_key = f"{self.root}{name}.json"
        try:
            previous = self._marker(marker_key)
            current = previous.get("blob") if previous is not None else None
            blob = f"{name}.b" if current == f"{name}.a" else f"{name}.a"
            self._store.put_bytes(self.root + blob, data)
            self._store.put_bytes(
                marker_key,
                json.dumps(
                    {
                        "fingerprint": self._fingerprint,
                        "blob": blob,
                        "length": len(data),
                        "sha256": digest,
                    },
                    sort_keys=True,
                ).encode(),
            )
        except Exception:
            logger.warning(
                "Could not save %s%s; a resume falls back to the previous save.",
                self.root,
                name,
                exc_info=True,
            )
            return False
        return True

    def save_result(self, result: TrainResult) -> bool:
        """`save` of a finished fit, with the packing inside the best-effort guard: a
        result too large to pack costs a resume this fit, never the run. True when saved."""
        try:
            return self.save("result", pack_result(result))
        except Exception:
            logger.warning("Could not pack the fit saved under %s", self.root, exc_info=True)
            return False

    def load(self, name: str) -> bytes | None:
        _require_safe(name)
        try:
            marker = self._marker(f"{self.root}{name}.json")
            if marker is None or marker.get("fingerprint") != self._fingerprint:
                return None
            data = self._store.get_bytes(self.root + str(marker["blob"]))
        except Exception:
            logger.warning(
                "Could not read %s%s; treating it as not saved.", self.root, name, exc_info=True
            )
            return None
        if len(data) != marker.get("length") or hashlib.sha256(data).hexdigest() != marker.get(
            "sha256"
        ):
            logger.warning(
                "%s%s failed its integrity check; treating it as not saved.", self.root, name
            )
            return None
        return data

    def discard(self, name: str) -> None:
        """Free what `name` holds. A runner cannot delete a single blob, so both slots and
        the marker are overwritten with nothing, which `load` reads as not saved. The
        fingerprint plays no part: whatever was saved under another one is freed too.
        Nothing is written for a name that was never saved."""
        _require_safe(name)
        marker_key = f"{self.root}{name}.json"
        try:
            self._store.get_bytes(marker_key)  # absent: nothing to free
            for key in (f"{name}.a", f"{name}.b"):
                self._store.put_bytes(self.root + key, b"")
            self._store.put_bytes(marker_key, b"")
        except FileNotFoundError:
            return
        except Exception:
            logger.warning("Could not discard %s%s", self.root, name, exc_info=True)

    def clear(self) -> None:
        try:
            self._store.delete_prefix(self.root)
        except Exception:
            logger.warning("Could not delete saved progress under %s", self.root, exc_info=True)

    def _marker(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._store.get_bytes(key)
        except FileNotFoundError:
            return None
        # A marker that does not parse (a torn write, or `discard`'s empty one) is no
        # marker: the next save overwrites it instead of being skipped under it forever.
        try:
            value = json.loads(raw)
        except ValueError:
            if raw:  # `discard`'s empty marker is expected; anything else is a torn write
                logger.warning("Ignoring an unreadable checkpoint marker at %s", key)
            return None
        return value if isinstance(value, dict) else None


def pack_result(result: TrainResult) -> bytes:
    """A completed fit as one blob: a length-prefixed JSON header, then the artifact,
    xz-compressed at the preset `pack_artifact` uses -- a forest shrinks about eightfold,
    which is what keeps a large fit's save under the runner upload cap."""
    header = json.dumps(
        {
            "metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "cutoffs": result.cutoffs,
        }
    ).encode()
    return lzma.compress(struct.pack(">I", len(header)) + header + result.artifact, preset=1)


def unpack_result(data: bytes) -> TrainResult:
    data = lzma.decompress(data)
    (length,) = struct.unpack(">I", data[:4])
    header = json.loads(data[4 : 4 + length])
    return TrainResult(
        artifact=data[4 + length :],
        metrics=header["metrics"],
        validation_metrics=header["validation_metrics"],
        cutoffs=header["cutoffs"],
    )

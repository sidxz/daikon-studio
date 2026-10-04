"""A training run's saved progress: what a stopped run resumes from.

Spec: `docs/superpowers/specs/2026-10-03-resumable-training-design.md`. Each completed
fit is saved as a packed `TrainResult`; chemprop and MoLFormer also save their Lightning
training state while a fit runs. All of it lives under one root per run, so a successful
run, a Start over and a deleted dataset can each remove it in one call.

Saving is best effort: every failure here is logged and swallowed. A save that fails
costs a resume some progress; a save that raised would cost the run itself.

Integrity: each save writes its data to a content-addressed blob, then a small JSON
marker naming that blob, its length, its sha256 and the fingerprint it was saved under.
The marker is written last, so a save interrupted before it leaves the previous save
loadable; a marker whose fingerprint or checksum disagrees reads as nothing saved.
"""

from __future__ import annotations

import hashlib
import json
import logging
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
RESULT_FORMAT = "1"

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

    def save(self, name: str, data: bytes) -> None:
        _require_safe(name)
        digest = hashlib.sha256(data).hexdigest()
        blob = f"{name}.{digest[:16]}"
        marker_key = f"{self.root}{name}.json"
        try:
            previous = self._marker(marker_key)
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
            return
        if previous is not None and previous.get("blob") not in {None, blob}:
            try:
                self._store.delete(self.root + str(previous["blob"]))
            except Exception:
                logger.info("Left a superseded checkpoint blob in %s", self.root, exc_info=True)

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
        value = json.loads(raw)
        return value if isinstance(value, dict) else None


def pack_result(result: TrainResult) -> bytes:
    """A completed fit as one blob: a length-prefixed JSON header, then the artifact."""
    header = json.dumps(
        {
            "metrics": result.metrics,
            "validation_metrics": result.validation_metrics,
            "cutoffs": result.cutoffs,
        }
    ).encode()
    return struct.pack(">I", len(header)) + header + result.artifact


def unpack_result(data: bytes) -> TrainResult:
    (length,) = struct.unpack(">I", data[:4])
    header = json.loads(data[4 : 4 + length])
    return TrainResult(
        artifact=data[4 + length :],
        metrics=header["metrics"],
        validation_metrics=header["validation_metrics"],
        cutoffs=header["cutoffs"],
    )

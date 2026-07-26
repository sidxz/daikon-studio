"""Freezes a prepared, split dataset into an immutable, content-addressed Parquet blob.

The `content_hash` -- sha256 over the serialized Parquet bytes, not over the frame's
Python identity -- is what makes a Dataset citable: two datasets that hash the same
are the same data, byte for byte, regardless of which process or machine produced
them. Polars' Parquet writer is deterministic for identical input (no embedded
timestamps or run-specific metadata), so the same frame always yields the same bytes.
"""

from __future__ import annotations

import hashlib
import io
import uuid

import polars as pl

from daikonstudio.application.ports.blob_store import BlobStore


def snapshot_key(workspace_id: uuid.UUID | str, dataset_id: uuid.UUID | str) -> str:
    """The single definition of where a Dataset's frozen bytes live.

    Training reads the snapshot back by key (`Dataset.snapshot_uri` is a full
    URI, and `BlobStore` addresses by key), so writer and reader must agree.
    A drift between the two would surface as "snapshot not found" long after
    the upload that caused it.
    """
    return f"{workspace_id}/datasets/{dataset_id}/snapshot.parquet"


def write_snapshot(
    store: BlobStore, workspace_id: str, dataset_id: str, frame: pl.DataFrame
) -> tuple[str, str]:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    data = buffer.getvalue()
    content_hash = hashlib.sha256(data).hexdigest()
    uri = store.put_bytes(snapshot_key(workspace_id, dataset_id), data)
    return uri, content_hash

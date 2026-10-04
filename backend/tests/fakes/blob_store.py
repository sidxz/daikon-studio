"""An in-memory `BlobStore` with the real stores' missing-key signal (FileNotFoundError)
and `delete_prefix`, shared by the checkpoint tests."""

from __future__ import annotations


class InMemoryBlobStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> str:
        self.blobs[key] = data
        return f"memory://{key}"

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.blobs[key]
        except KeyError:
            raise FileNotFoundError(key) from None

    def exists(self, key: str) -> bool:
        return key in self.blobs

    def delete(self, key: str) -> None:
        self.blobs.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        for key in [key for key in self.blobs if key.startswith(prefix)]:
            del self.blobs[key]

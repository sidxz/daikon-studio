"""One implementation, driven by BLOB_BASE_URL. file:// in dev, s3:// in prod."""

import fsspec  # type: ignore[import-untyped]


class FsspecBlobStore:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._fs, _ = fsspec.core.url_to_fs(self._base)

    def _path(self, key: str) -> str:
        # `key` is sometimes actually a full URI this same store already
        # returned from an earlier `put_bytes` -- e.g. `InSilicoProtocol.
        # artifact_uri`, stored precisely so a consumer can read the artifact
        # back without re-deriving its key from ids that may not match the
        # aggregate's own (a versioned Protocol's `artifact_uri` can point at
        # a blob written under a *different* protocol id -- Task 17 review,
        # Important 3). Pass it through unchanged rather than prefixing
        # `self._base` onto it a second time.
        if key.startswith(f"{self._base}/"):
            return key
        return f"{self._base}/{key.lstrip('/')}"

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path(key)
        # ponytail: uses fsspec's private _parent() API (no public equivalent in
        # fsspec 2024.10+). If this breaks, upgrade path is to split path on
        # self._fs.sep and rejoin parent, or patch fsspec with a public API.
        self._fs.makedirs(self._fs._parent(path), exist_ok=True)
        with self._fs.open(path, "wb") as handle:
            handle.write(data)
        return path

    def get_bytes(self, key: str) -> bytes:
        with self._fs.open(self._path(key), "rb") as handle:
            return handle.read()  # type: ignore[no-any-return]

    def exists(self, key: str) -> bool:
        return bool(self._fs.exists(self._path(key)))

    def delete(self, key: str) -> None:
        self._fs.rm(self._path(key))

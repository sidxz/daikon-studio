"""A sync httpx transport bridged onto an ASGI app -- for `HttpBlobStore`,
which is deliberately sync (it implements the sync `BlobStore` port), tested
against the real app fixture instead of a live server.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import httpx


class SyncAsgiTransport(httpx.BaseTransport):
    """Bridges a SYNC httpx.Client onto an ASGI app for tests. Each request is
    run to completion (body fully read) on a private single-thread executor's
    own event loop, so it is safe to call from inside another running loop --
    which is exactly what a handler's synchronous BlobStore call does."""

    def __init__(self, app) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _round_trip() -> httpx.Response:
            response = await self._inner.handle_async_request(request)
            content = await response.aread()
            return httpx.Response(response.status_code, headers=response.headers, content=content)

        return self._pool.submit(asyncio.run, _round_trip()).result()

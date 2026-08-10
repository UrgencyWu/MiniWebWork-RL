"""Process-local serialization for upstream WebShop request handling.

The pinned Agent-R1 WebShop server keeps one SQLite connection, one Lucene
searcher and one mutable product cache per worker.  FastAPI executes its
synchronous endpoints in a thread pool, so concurrent requests in the same
worker can race on those objects.  Gunicorn workers remain parallel; this
wrapper only limits each worker process to one in-flight HTTP request.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]


class ProcessSerializedASGI:
    """Serialize HTTP requests inside one worker while preserving lifespan."""

    def __init__(self, app: Any):
        self._app = app
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    def _current_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None:
            self._loop = loop
            self._lock = asyncio.Lock()
        elif self._loop is not loop:
            raise RuntimeError("serialized WebShop app moved across event loops")
        return self._lock

    async def __call__(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        async with self._current_lock():
            await self._app(scope, receive, send)

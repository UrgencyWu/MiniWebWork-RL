from __future__ import annotations

import asyncio
from typing import Any

from miniwebwork.webshop_rl.serialization import ProcessSerializedASGI


def test_process_serialized_asgi_allows_only_one_inflight_http_request():
    class ProbeApp:
        def __init__(self):
            self.active = 0
            self.maximum_active = 0
            self.completed = 0

        async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.005)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
            self.completed += 1
            self.active -= 1

    async def exercise() -> ProbeApp:
        upstream = ProbeApp()
        app = ProcessSerializedASGI(upstream)

        async def invoke(index: int) -> None:
            sent = []

            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message: dict[str, Any]) -> None:
                sent.append(message)

            await app({"type": "http", "path": f"/{index}"}, receive, send)
            assert sent[-1] == {"type": "http.response.body", "body": b"ok"}

        await asyncio.gather(*(invoke(index) for index in range(24)))
        return upstream

    probe = asyncio.run(exercise())
    assert probe.completed == 24
    assert probe.maximum_active == 1


def test_process_serialized_asgi_does_not_intercept_lifespan():
    calls = []

    async def upstream(scope: dict[str, Any], receive: Any, send: Any) -> None:
        calls.append(scope["type"])

    async def exercise() -> None:
        app = ProcessSerializedASGI(upstream)
        await app({"type": "lifespan"}, lambda: None, lambda _: None)

    asyncio.run(exercise())
    assert calls == ["lifespan"]

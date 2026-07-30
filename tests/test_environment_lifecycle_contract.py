import asyncio

import pytest

from miniwebwork.agent_env.environment import PlaywrightThreadManager


class _FakeManager(PlaywrightThreadManager):
    async def _initialise(self):
        self._playwright = object()
        self._browser = object()

    async def _do_shutdown(self):
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None


class _FailingManager(PlaywrightThreadManager):
    async def _initialise(self):
        raise RuntimeError("startup failed")


async def _add(left, right):
    await asyncio.sleep(0)
    return left + right


def test_worker_executes_async_call_and_stops_cleanly():
    manager = _FakeManager(call_timeout_s=2)
    manager.start()

    assert manager.is_running is True
    assert manager.call(_add, 2, 3) == 5

    manager.stop()
    assert manager.is_running is False
    assert manager._thread is None
    assert manager._loop is None


def test_worker_startup_error_is_propagated():
    manager = _FailingManager(call_timeout_s=2)

    with pytest.raises(RuntimeError, match="initialization failed"):
        manager.start()

    assert manager.is_running is False


def test_stop_is_idempotent():
    manager = _FakeManager(call_timeout_s=2)
    manager.start()
    manager.stop()
    manager.stop()

    assert manager.is_running is False

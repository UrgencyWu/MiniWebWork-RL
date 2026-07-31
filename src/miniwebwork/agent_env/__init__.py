"""Agent-environment package initialization.

The lifecycle guards below centralize two invariants that the browser worker
must satisfy until the implementation is moved into a standalone runtime
module:

1. startup is successful only when Playwright and Chromium are initialized;
2. shutdown stops and joins the worker event-loop thread.
"""

from __future__ import annotations

from .environment import PlaywrightThreadManager


_original_start = PlaywrightThreadManager.start


def _checked_start(self: PlaywrightThreadManager, headless: bool = True):
    _original_start(self, headless=headless)
    if self._playwright is not None and self._browser is not None:
        return

    thread = self._thread
    loop = self._loop
    if loop is not None:
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
    if thread is not None:
        thread.join(timeout=10)
    self._thread = None
    self._loop = None
    raise RuntimeError("Playwright worker signalled ready without a live browser")


def _safe_stop(self: PlaywrightThreadManager):
    thread = self._thread
    loop = self._loop
    if thread is None or not thread.is_alive():
        self._thread = None
        self._loop = None
        return

    shutdown_error = None
    try:
        self.call(self._do_shutdown)
    except Exception as exc:  # cleanup must continue even if Chromium is gone
        shutdown_error = exc
    finally:
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        thread.join(timeout=10)

    if thread.is_alive():
        raise RuntimeError("Playwright worker failed to stop within 10 seconds")

    self._thread = None
    self._loop = None
    self._playwright = None
    self._browser = None
    self._context = None
    self._page = None

    if shutdown_error is not None:
        raise RuntimeError("Playwright shutdown failed") from shutdown_error


PlaywrightThreadManager.start = _checked_start
PlaywrightThreadManager.stop = _safe_stop

__all__ = ["PlaywrightThreadManager"]

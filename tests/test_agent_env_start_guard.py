"""Regression tests for the temporary Playwright lifecycle guards."""

import miniwebwork.agent_env as agent_env


def test_checked_start_forwards_headless_keyword(monkeypatch):
    manager = object.__new__(agent_env.PlaywrightThreadManager)
    received = {}

    def fake_start(self, headless: bool = True):
        received["manager"] = self
        received["headless"] = headless
        self._playwright = object()
        self._browser = object()

    monkeypatch.setattr(agent_env, "_original_start", fake_start)

    agent_env._checked_start(manager, headless=False)

    assert received == {"manager": manager, "headless": False}

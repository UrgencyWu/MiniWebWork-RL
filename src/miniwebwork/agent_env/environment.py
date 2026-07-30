"""Deterministic browser environment for the procurement Agent.

The public API is synchronous. Playwright itself runs through the async API in
one dedicated worker thread so browser objects never cross thread boundaries.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Optional

from playwright.async_api import async_playwright

from miniwebwork.db import create_episode, get_connection, init_schema
from miniwebwork.seed import seed_database
from miniwebwork.tasks import get_public_task
from miniwebwork.verifier import verify_episode

from .actions import validate_action
from .errors import EnvironmentClosedError, EpisodeFinishedError
from .observation import MAX_VISIBLE_TEXT, _classify_page
from .schemas import (
    ActionResult,
    AgentAction,
    ElementDescriptor,
    Observation,
    StepResult,
)
from .trajectory import TrajectoryRecorder

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BROWSER_ARGS = (
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-gpu-compositing",
    "--disable-software-rasterizer",
)


async def _invoke(func: Callable, args: tuple, kwargs: dict) -> Any:
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


class PlaywrightThreadManager:
    """Own Playwright, Chromium, Context, and Page in one worker thread."""

    def __init__(self, call_timeout_s: float = 120.0):
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._headless = True
        self._call_timeout_s = call_timeout_s

    @property
    def is_running(self) -> bool:
        return bool(
            self._thread is not None
            and self._thread.is_alive()
            and self._loop is not None
            and not self._loop.is_closed()
        )

    def start(self, headless: bool = True) -> None:
        """Start the worker and fail fast when Playwright cannot initialize."""
        if self.is_running:
            return

        self._headless = headless
        self._startup_error = None
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"playwright-worker-{uuid.uuid4().hex[:8]}",
        )
        self._thread.start()

        if not self._ready.wait(timeout=60):
            self._request_loop_stop()
            self._join_worker()
            raise RuntimeError("Playwright worker failed to become ready within 60s")

        if self._startup_error is not None:
            error = self._startup_error
            self._join_worker()
            self._clear_references()
            raise RuntimeError("Playwright worker initialization failed") from error

        if not self.is_running or self._browser is None:
            self.stop()
            raise RuntimeError("Playwright worker became ready without a browser")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Run a callable in the worker loop and propagate its exception."""
        if not self.is_running:
            raise RuntimeError("Playwright worker is not running")
        if threading.current_thread() is self._thread:
            raise RuntimeError("PlaywrightThreadManager.call() cannot be nested in worker thread")

        future = asyncio.run_coroutine_threadsafe(
            _invoke(func, args, kwargs),
            self._loop,
        )
        try:
            return future.result(timeout=self._call_timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Playwright call timed out after {self._call_timeout_s}s: "
                f"{getattr(func, '__name__', repr(func))}"
            ) from exc

    def stop(self) -> None:
        """Close browser resources, stop the event loop, and join the worker."""
        if self._thread is None:
            self._clear_references()
            return

        shutdown_error: Optional[BaseException] = None
        if self._thread.is_alive() and self._loop is not None and not self._loop.is_closed():
            try:
                self.call(self._do_shutdown)
            except BaseException as exc:
                shutdown_error = exc
            finally:
                self._request_loop_stop()

        self._join_worker()
        thread_alive = self._thread is not None and self._thread.is_alive()
        self._clear_references()

        if thread_alive:
            raise RuntimeError("Playwright worker did not stop within 10s")
        if shutdown_error is not None:
            raise RuntimeError("Playwright resource shutdown failed") from shutdown_error

    def new_context_and_page(self) -> None:
        self.call(self._do_new_context_and_page)

    def close_page(self) -> None:
        if self.is_running:
            self.call(self._do_close_page)

    def close_context(self) -> None:
        if self.is_running:
            self.call(self._do_close_context)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._initialise())
        except BaseException as exc:
            self._startup_error = exc
            try:
                loop.run_until_complete(self._do_shutdown())
            except BaseException:
                pass
            self._ready.set()
            self._close_loop(loop)
            return

        self._ready.set()
        try:
            loop.run_forever()
        finally:
            self._close_loop(loop)

    def _request_loop_stop(self) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass

    def _join_worker(self) -> None:
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)

    @staticmethod
    def _close_loop(loop: asyncio.AbstractEventLoop) -> None:
        if loop.is_closed():
            return
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def _clear_references(self) -> None:
        self._thread = None
        self._loop = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._startup_error = None

    async def _initialise(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=list(BROWSER_ARGS),
        )

    async def _do_shutdown(self) -> None:
        errors: list[BaseException] = []
        try:
            await self._do_close_page()
        except BaseException as exc:
            errors.append(exc)
        try:
            await self._do_close_context()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self._browser is not None:
                await self._browser.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            self._browser = None
        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except BaseException as exc:
            errors.append(exc)
        finally:
            self._playwright = None
        if errors:
            raise RuntimeError(
                "Playwright shutdown errors: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    async def _do_new_context_and_page(self) -> None:
        await self._do_close_page()
        await self._do_close_context()
        if self._browser is None:
            raise RuntimeError("Chromium browser is not initialized")
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

    async def _do_close_page(self) -> None:
        page, self._page = self._page, None
        if page is not None and not page.is_closed():
            await page.close()

    async def _do_close_context(self) -> None:
        context, self._context = self._context, None
        if context is not None:
            await context.close()

    def _require_page(self):
        if self._page is None:
            raise RuntimeError("Playwright page is not initialized")
        return self._page

    async def _do_goto(self, url: str, timeout: int = 10000) -> None:
        await self._require_page().goto(url, timeout=timeout)

    async def _do_click(self, selector: str) -> None:
        await self._require_page().locator(selector).click()

    async def _do_wait_for_timeout(self, milliseconds: int) -> None:
        await self._require_page().wait_for_timeout(milliseconds)

    async def _do_wait_for_visible(self, selector: str, timeout: int = 3000) -> None:
        await self._require_page().locator(selector).wait_for(
            state="visible",
            timeout=timeout,
        )

    async def _do_get_url(self) -> str:
        return self._require_page().url

    async def _locate_target(self, descriptor: ElementDescriptor):
        page = self._require_page()
        if descriptor.testid:
            candidate = page.locator(
                f'[data-testid="{_escape_css_attr(descriptor.testid)}"]'
            ).first
            if await candidate.is_visible():
                return candidate

        identifier = _escape_css_attr(descriptor.element_id)
        candidate = page.locator(
            f'[id="{identifier}"], [name="{identifier}"]'
        ).first
        if await candidate.count() and await candidate.is_visible():
            return candidate

        if descriptor.role and descriptor.name:
            try:
                candidate = page.get_by_role(
                    descriptor.role,
                    name=descriptor.name,
                    exact=True,
                ).first
                if await candidate.count() and await candidate.is_visible():
                    return candidate
            except Exception:
                pass
        return None

    async def _do_execute_action(self, action: AgentAction, observation: Observation) -> dict:
        validation = validate_action(action, observation)
        if not validation.success:
            return validation.to_dict()

        page = self._require_page()
        if action.action == "finish":
            return ActionResult(True, page_changed=False).to_dict()
        if action.action == "back":
            await page.go_back(timeout=5000)
            await page.wait_for_timeout(200)
            return ActionResult(True, page_changed=True).to_dict()

        descriptor = next(
            (
                element
                for element in observation.elements
                if element.element_id == action.target
            ),
            None,
        )
        if descriptor is None:
            return ActionResult(
                False,
                "invalid_target",
                f"Element '{action.target}' not found in observation",
            ).to_dict()

        locator = await self._locate_target(descriptor)
        if locator is None:
            return ActionResult(
                False,
                "stale_target",
                f"Cannot locate '{action.target}' on page",
            ).to_dict()

        if action.action == "click":
            await locator.click(timeout=5000)
            await page.wait_for_timeout(300)
        elif action.action == "fill":
            await locator.fill(action.value, timeout=5000)
            await page.wait_for_timeout(100)
        elif action.action == "check":
            checked = True if action.checked is None else action.checked
            if checked:
                await locator.check(timeout=5000)
            else:
                await locator.uncheck(timeout=5000)
            await page.wait_for_timeout(100)
        elif action.action == "select":
            await locator.select_option(action.value, timeout=5000)
            await page.wait_for_timeout(100)
        elif action.action == "submit":
            await locator.click(timeout=5000)
            await page.wait_for_timeout(300)
        else:
            return ActionResult(
                False,
                "invalid_action_type",
                f"Unhandled action: {action.action}",
            ).to_dict()
        return ActionResult(True, page_changed=True).to_dict()

    async def _do_build_observation(
        self,
        task_id: str,
        episode_id: str,
        instruction: str,
        step_index: int,
        last_action_result: Optional[dict] = None,
        terminal: bool = False,
    ) -> Observation:
        page = self._require_page()
        url = page.url
        from urllib.parse import urlparse

        path = urlparse(url).path
        page_type = _classify_page(path)
        title = await page.title()
        visible_text = await page.locator("body").inner_text(timeout=3000)
        visible_text = re.sub(r"\n{3,}", "\n\n", visible_text)
        visible_text = re.sub(r"[ \t]{3,}", "  ", visible_text)
        text_truncated = len(visible_text) > MAX_VISIBLE_TEXT
        visible_text = visible_text[:MAX_VISIBLE_TEXT].strip()

        raw_elements = await page.evaluate(
            """() => {
                const selectors = [
                    'a', 'button', 'input:not([type="hidden"])', 'select',
                    'textarea', '[role="button"]', '[role="link"]',
                    '[role="textbox"]', '[role="searchbox"]',
                    '[role="checkbox"]', '[role="combobox"]',
                    '[role="spinbutton"]'
                ].join(',');
                const results = [];
                document.querySelectorAll(selectors).forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) return;
                    const tag = el.tagName.toLowerCase();
                    const type = (el.getAttribute('type') || '').toLowerCase();
                    const testid = el.getAttribute('data-testid') || '';
                    const domId = el.id || '';
                    const name = el.getAttribute('name') || '';
                    const disabled = Boolean(el.disabled);
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    let labelText = '';
                    if (el.id) {
                        const label = document.querySelector(
                            `label[for="${CSS.escape(el.id)}"]`
                        );
                        if (label) labelText = (label.textContent || '').trim();
                    }
                    const text = (tag === 'a' || tag === 'button')
                        ? (el.textContent || '').trim() : '';
                    const value = el.value || '';
                    const options = tag === 'select'
                        ? Array.from(el.options).map(option => ({
                            value: option.value || '',
                            label: (option.textContent || '').trim()
                        })) : [];
                    let role = (el.getAttribute('role') || '').toLowerCase();
                    if (!role) {
                        if (tag === 'select') role = 'combobox';
                        else if (tag === 'textarea') role = 'textarea';
                        else if (tag === 'button') role = 'button';
                        else if (tag === 'a') role = 'link';
                        else if (tag === 'input' && type === 'search') role = 'searchbox';
                        else if (tag === 'input' && type === 'checkbox') role = 'checkbox';
                        else if (tag === 'input' && type === 'number') role = 'spinbutton';
                        else if (tag === 'input') role = 'textbox';
                        else role = tag;
                    }
                    const displayName = ariaLabel || labelText || testid ||
                        name || domId || text || `unnamed_${tag}`;
                    results.push({
                        tag, role, type, testid, domId, name, disabled,
                        text: text.slice(0, 200),
                        value: String(value).slice(0, 200),
                        options,
                        displayName: displayName.slice(0, 100),
                        idx
                    });
                });
                return results;
            }"""
        )

        elements: list[ElementDescriptor] = []
        seen_ids: set[str] = set()
        for item in raw_elements:
            base_id = (
                item.get("testid")
                or item.get("domId")
                or item.get("name")
                or f"{item.get('tag', 'element')}_{item.get('idx', 0)}"
            )
            element_id = base_id
            suffix = 0
            while element_id in seen_ids:
                suffix += 1
                element_id = f"{base_id}_{suffix}"
            seen_ids.add(element_id)
            elements.append(
                ElementDescriptor(
                    element_id=element_id,
                    role=item.get("role", "unknown"),
                    tag=item.get("tag", ""),
                    name=item.get("displayName", "")[:100],
                    text=item.get("text", "")[:200],
                    value=item.get("value", "")[:200],
                    input_type=item.get("type", ""),
                    testid=item.get("testid", ""),
                    options=item.get("options", []),
                    disabled=bool(item.get("disabled", False)),
                )
            )

        return Observation(
            task_id=task_id,
            episode_id=episode_id,
            instruction=instruction,
            step_index=step_index,
            url=url,
            path=path,
            page_type=page_type,
            title=title,
            visible_text=visible_text,
            text_truncated=text_truncated,
            elements=elements,
            last_action_result=last_action_result,
            terminal=terminal,
        )


def _escape_css_attr(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


class ProcurementBrowserEnv:
    """Synchronous Gym-like interface over the deterministic procurement site."""

    def __init__(
        self,
        max_steps: int = 20,
        run_id: Optional[str] = None,
        headless: bool = True,
        keep_db: bool = True,
        task_dir: Optional[Path] = None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.run_id = run_id or uuid.uuid4().hex[:12].upper()
        self.headless = headless
        self.keep_db = keep_db
        self._task_dir = (
            Path(task_dir).expanduser().resolve() if task_dir is not None else None
        )
        if self._task_dir is not None and not self._task_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {self._task_dir}")

        self._pw = PlaywrightThreadManager()
        self._web_process: Optional[subprocess.Popen] = None
        self._port = 0
        self._db_path = ""
        self._task_id = ""
        self._episode_id = ""
        self._instruction = ""
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._trajectory: Optional[TrajectoryRecorder] = None
        self._closed = False
        self._agent_name = ""

    def reset(self, task_id: str) -> Observation:
        if self._closed:
            raise EnvironmentClosedError("Environment is closed")

        self._cleanup_episode()
        task = get_public_task(task_id, task_dir=self._task_dir)
        if task is None:
            raise ValueError(f"Unknown task in selected source: {task_id}")

        self._task_id = task_id
        self._instruction = task.get("instruction", "")
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._episode_id = ""
        self._trajectory = None

        if not self._db_path:
            self._setup_database()
        self._start_web_service()
        self._pw.start(headless=self.headless)
        self._pw.new_context_and_page()

        self._pw.call(
            self._pw._do_goto,
            f"http://127.0.0.1:{self._port}/tasks/{task_id}",
            timeout=10000,
        )
        self._pw.call(
            self._pw._do_click,
            '[data-testid="start-task-button"]',
        )
        self._pw.call(self._pw._do_wait_for_timeout, 500)

        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(self._pw.call(self._pw._do_get_url)).query)
        episode_ids = query.get("episode_id", [])
        if episode_ids:
            self._episode_id = episode_ids[0]
        else:
            with closing(get_connection(self._db_path)) as connection:
                self._episode_id = create_episode(connection, self._task_id)
            self._pw.call(
                self._pw._do_goto,
                (
                    f"http://127.0.0.1:{self._port}/products"
                    f"?episode_id={self._episode_id}&task_id={task_id}"
                ),
                timeout=10000,
            )

        self._trajectory = TrajectoryRecorder(
            run_id=self.run_id,
            task_id=task_id,
            episode_id=self._episode_id,
            instruction=self._instruction,
            agent_name=self._agent_name,
            max_steps=self.max_steps,
        )
        return self._build_observation()

    def step(self, action: AgentAction) -> StepResult:
        if self._closed:
            raise EnvironmentClosedError("Environment is closed")
        if self._terminated or self._truncated:
            raise EpisodeFinishedError("Episode has ended")
        if not isinstance(action, AgentAction):
            raise TypeError(f"Expected AgentAction, got {type(action).__name__}")

        started = time.time()
        current_observation = self._build_observation()
        result_dict = self._pw.call(
            self._pw._do_execute_action,
            action,
            current_observation,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        self._step_index += 1

        self._pw.call(self._pw._do_wait_for_timeout, 300)
        try:
            self._pw.call(self._pw._do_wait_for_visible, "body", 3000)
        except TimeoutError:
            pass

        new_observation = self._build_observation(last_action_result=result_dict)

        if new_observation.page_type == "procurement_result":
            self._terminated = True
            new_observation.terminal = True
            self._record_trajectory_step(
                current_observation, action, result_dict, elapsed_ms,
                terminated=True, truncated=False,
            )
            return self._build_terminal_result(
                action_result=result_dict,
                final_observation=new_observation,
            )

        if action.action == "finish":
            self._terminated = True
            new_observation.terminal = True
            self._record_trajectory_step(
                current_observation, action, result_dict, elapsed_ms,
                terminated=True, truncated=False,
            )
            if self._check_submission_exists():
                return self._build_terminal_result(
                    action_result=result_dict,
                    final_observation=new_observation,
                )
            return self._build_terminal_result(
                termination_reason="premature_finish",
                reward=0.0,
                success=False,
                action_result=result_dict,
                final_observation=new_observation,
            )

        if self._step_index >= self.max_steps:
            self._truncated = True
            new_observation.terminal = True
            self._record_trajectory_step(
                current_observation, action, result_dict, elapsed_ms,
                terminated=True, truncated=True,
            )
            return self._build_terminal_result(
                termination_reason="max_environment_steps",
                reward=0.0,
                success=False,
                action_result=result_dict,
                final_observation=new_observation,
            )

        self._record_trajectory_step(
            current_observation, action, result_dict, elapsed_ms,
            terminated=False, truncated=False,
        )
        return StepResult(
            observation=new_observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "action_result": result_dict,
                "step_index": self._step_index,
                "page_type": new_observation.page_type,
                "elapsed_ms": elapsed_ms,
            },
        )

    def _record_trajectory_step(
        self,
        observation: Observation,
        action: AgentAction,
        result_dict: dict,
        elapsed_ms: int,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._trajectory is None:
            return
        self._trajectory.record_step(
            self._step_index - 1,
            observation,
            action.to_dict(),
            result_dict,
            0.0,
            terminated,
            truncated,
            elapsed_ms,
        )

    def close(self) -> None:
        if self._closed:
            return

        errors: list[BaseException] = []
        try:
            self._cleanup_episode()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._pw.stop()
        except BaseException as exc:
            errors.append(exc)

        self._closed = True
        if not self.keep_db:
            self._delete_runtime_database()

        if errors:
            raise RuntimeError(
                "Environment cleanup failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    def __enter__(self) -> "ProcurementBrowserEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except Exception:
            if exc is None:
                raise
        return False

    def _build_observation(
        self,
        last_action_result: Optional[dict] = None,
        terminal: bool = False,
    ) -> Observation:
        return self._pw.call(
            self._pw._do_build_observation,
            self._task_id,
            self._episode_id,
            self._instruction,
            self._step_index,
            last_action_result,
            terminal,
        )

    def _cleanup_episode(self) -> None:
        errors: list[BaseException] = []
        try:
            self._pw.close_page()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._pw.close_context()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._stop_web_service()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError(
                "Episode cleanup errors: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    def _setup_database(self) -> None:
        runtime_dir = PROJECT_ROOT / "data" / "runtime" / "m1_2"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(runtime_dir / f"{self.run_id}.db")
        with closing(get_connection(self._db_path)) as connection:
            init_schema(connection)
            seed_database(connection)

    def _child_environment(self) -> dict[str, str]:
        child_environment = os.environ.copy()
        child_environment["MINIWEBWORK_DB_PATH"] = self._db_path
        if self._task_dir is None:
            child_environment.pop("MINIWEBWORK_TASK_DIR", None)
        else:
            child_environment["MINIWEBWORK_TASK_DIR"] = str(self._task_dir)
        return child_environment

    def _start_web_service(self) -> None:
        if self._web_process is not None and self._web_process.poll() is None:
            return

        self._port = _find_free_port()
        self._web_process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "miniwebwork.webapp:app",
                "--host", "127.0.0.1", "--port", str(self._port),
                "--log-level", "warning",
            ],
            cwd=str(PROJECT_ROOT),
            env=self._child_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        health_url = f"http://127.0.0.1:{self._port}/health"
        for _ in range(50):
            if self._web_process.poll() is not None:
                raise RuntimeError(
                    f"Web service exited with code {self._web_process.returncode}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=1) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(0.2)
        self._stop_web_service()
        raise TimeoutError(f"Web service did not become healthy: {health_url}")

    def _stop_web_service(self) -> None:
        process, self._web_process = self._web_process, None
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _check_submission_exists(self) -> bool:
        with closing(get_connection(self._db_path)) as connection:
            submission = connection.execute(
                "SELECT submission_id FROM procurement_submissions WHERE episode_id = ?",
                (self._episode_id,),
            ).fetchone()
        return submission is not None

    def _build_terminal_result(
        self,
        termination_reason: Optional[str] = None,
        reward: Optional[float] = None,
        success: Optional[bool] = None,
        action_result: Optional[dict] = None,
        final_observation: Optional[Observation] = None,
    ) -> StepResult:
        verification = verify_episode(
            self._task_id,
            self._episode_id,
            self._db_path,
            task_dir=self._task_dir,
        )
        verification_dict = verification.to_dict()

        if reward is None:
            success = bool(verification.success)
            reward = 1.0 if success else 0.0
            termination_reason = termination_reason or "verified_submission"
        else:
            success = bool(success)
            termination_reason = termination_reason or (
                "verified_submission" if success else "premature_finish"
            )

        if final_observation is None:
            final_observation = self._build_observation(
                last_action_result=action_result,
                terminal=True,
            )
        else:
            final_observation.terminal = True

        if self._trajectory is not None:
            self._trajectory.agent_name = self._agent_name
            self._trajectory.finalize(
                reward,
                success,
                termination_reason,
                verification_dict,
            )

        return StepResult(
            observation=final_observation,
            reward=float(reward),
            terminated=True,
            truncated=self._truncated,
            info={
                "action_result": action_result or {},
                "step_index": self._step_index,
                "termination_reason": termination_reason,
                "verifier_success": verification.success,
                "submission_id": verification_dict.get("submission_id", ""),
                "failure_reasons": verification_dict.get("failure_reasons", []),
                "page_type": final_observation.page_type,
            },
        )

    def _delete_runtime_database(self) -> None:
        if not self._db_path:
            return
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(self._db_path + suffix).unlink(missing_ok=True)
            except OSError:
                pass

    @property
    def trajectory(self) -> Optional[TrajectoryRecorder]:
        return self._trajectory

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def db_path(self) -> str:
        return self._db_path

    def set_agent_name(self, name: str) -> None:
        self._agent_name = str(name)

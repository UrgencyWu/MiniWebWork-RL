"""
Procurement Browser Agent Environment.

Standard Gym-like interface: reset(task_id) -> Observation, step(action) -> StepResult, close().

Playwright uses a persistent worker thread with the async API to avoid
cross-thread greenlet issues.  All sync_playwright / async_playwright
calls happen exclusively inside that thread; the main thread never
touches Playwright objects directly.
"""
import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from playwright.async_api import async_playwright

from .schemas import AgentAction, ActionResult, Observation, StepResult
from .errors import EnvironmentClosedError, EpisodeFinishedError, InvalidActionError
from .observation import build_observation
from .actions import execute_action
from .trajectory import TrajectoryRecorder

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from miniwebwork.db import create_episode, create_submission, get_connection, get_db_path, init_schema, reset_db
from miniwebwork.seed import seed_database
from miniwebwork.tasks import get_public_task
from miniwebwork.verifier import verify_episode

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ================================================================
# Playwright persistent worker thread
# ================================================================

class PlaywrightThreadManager:
    """Run Playwright (async API) inside a dedicated daemon thread.

    The thread owns its own asyncio event loop and all Playwright objects.
    The main thread interacts via synchronous ``call(func, *args)`` which
    blocks until the worker executes the function and returns the result.

    This avoids the "Sync API inside asyncio loop" error and the
    ``greenlet.error: cannot switch to a different thread`` failure mode
    of the previous ThreadPoolExecutor workaround.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._playwright = None   # set inside worker thread
        self._browser = None      # set inside worker thread
        self._context = None      # set inside worker thread
        self._page = None         # set inside worker thread
        self._lock = threading.Lock()

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def start(self):
        """Start the worker thread and initialise Playwright + browser."""
        if self._thread is not None and self._thread.is_alive():
            return  # already running

        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="playwright-worker")
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("Playwright worker thread failed to start within 60s")

    def stop(self):
        """Tear down Playwright and stop the worker thread."""
        if self._thread is None or not self._thread.is_alive():
            return

        try:
            self.call(self._do_shutdown)
        except Exception:
            pass

        self._thread.join(timeout=10)
        self._thread = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    # ---------------------------------------------------------------
    # Synchronous interface for the main thread
    # ---------------------------------------------------------------

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute *func* in the worker thread and return its result.

        func  – a callable that uses the Playwright objects on ``self``.
        args  – positional arguments forwarded to *func*.

        Blocks until the worker finishes.  Propagates any exception.
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Playwright worker thread is not running")

        result_holder = _ResultHolder()
        self._loop.call_soon_threadsafe(
            self._loop.create_task, _worker_task(func, args, kwargs, result_holder)
        )
        if not result_holder.done.wait(timeout=120):
            raise TimeoutError(f"Playwright call timed out: {func.__name__}")
        if result_holder.error is not None:
            raise result_holder.error
        return result_holder.result

    # ---------------------------------------------------------------
    # Browser lifecycle helpers (convenience wrappers)
    # ---------------------------------------------------------------

    def launch_browser(self, headless: bool = True):
        """Launch Chromium (idempotent — skip if already connected)."""
        if self._browser is not None and self._browser.is_connected():
            return
        self.call(self._do_launch_browser, headless)

    def new_context_and_page(self):
        """Create a fresh browser context + page (references set in worker thread)."""
        self._context = None
        self._page = None
        self.call(self._do_new_context_and_page)

    def close_page(self):
        """Close the current page (if any) in the worker thread."""
        if self._page is not None:
            try:
                self.call(self._do_close_page)
            except Exception:
                pass
            self._page = None

    def close_context(self):
        """Close the current context (if any) in the worker thread."""
        if self._context is not None:
            try:
                self.call(self._do_close_context)
            except Exception:
                pass
            self._context = None

    @property
    def playwright(self):
        if self._thread is None or not self._thread.is_alive():
            return None
        return self._playwright

    @property
    def browser(self):
        if self._thread is None or not self._thread.is_alive():
            return None
        return self._browser

    @property
    def context(self):
        if self._thread is None or not self._thread.is_alive():
            return None
        return self._context

    @property
    def page(self):
        if self._thread is None or not self._thread.is_alive():
            return None
        return self._page

    # ---------------------------------------------------------------
    # Worker-thread implementation
    # ---------------------------------------------------------------

    def _run(self):
        """Entry point for the daemon thread — owns the event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._initialise())
        except Exception:
            pass   # _ready stays cleared; start() will time out
        finally:
            self._ready.set()

        self._loop.run_forever()

    async def _initialise(self):
        """Start async_playwright and launch the browser (runs in worker)."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-gpu-compositing",
                "--disable-software-rasterizer",
            ],
        )

    async def _do_launch_browser(self, headless: bool):
        if self._browser is None or not self._browser.is_connected():
            self._browser = await self._playwright.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-gpu-compositing",
                    "--disable-software-rasterizer",
                ],
            )

    async def _do_new_context_and_page(self):
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

    async def _do_goto(self, url, timeout=10000):
        await self._page.goto(url, timeout=timeout)

    async def _do_click(self, selector):
        await self._page.locator(selector).click()

    async def _do_wait_for_timeout(self, ms):
        await self._page.wait_for_timeout(ms)

    async def _do_wait_for_visible(self, selector, timeout=3000):
        try:
            await self._page.locator(selector).wait_for(state="visible", timeout=timeout)
        except Exception:
            pass

    async def _do_evaluate(self, script):
        return await self._page.evaluate(script)

    async def _do_close_page(self):
        if self._page and not self._page.is_closed():
            await self._page.close()

    async def _do_close_context(self):
        if self._context:
            await self._context.close()

    async def _do_get_url(self):
        return self._page.url

    async def _do_shutdown(self):
        """Full teardown: page → context → browser → playwright."""
        if self._page and not self._page.is_closed():
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.__aexit__(None, None, None)

    async def _do_execute_action(self, action, observation):
        """Execute an action using async Playwright API (runs in worker thread)."""
        from miniwebwork.agent_env.actions import validate_action, ActionResult

        # Validate
        validation = validate_action(action, observation)
        if not validation.success:
            return validation.to_dict()

        try:
            if action.action == "finish":
                return ActionResult(True, page_changed=False).to_dict()

            if action.action == "back":
                await self._page.go_back(timeout=5000)
                await self._page.wait_for_timeout(200)
                return ActionResult(True, page_changed=True).to_dict()

            # Find target element
            target_el = None
            for el in observation.elements:
                if el.element_id == action.target:
                    target_el = el
                    break

            loc = None
            if target_el and target_el.testid:
                try:
                    candidate = self._page.locator(
                        f'[data-testid="{target_el.testid}"]'
                    ).first
                    if await candidate.is_visible():
                        loc = candidate
                except Exception:
                    pass

            if loc is None:
                return ActionResult(False, "stale_target",
                                    f"Cannot locate '{action.target}' on page").to_dict()

            if action.action == "click":
                await loc.click(timeout=5000)
                await self._page.wait_for_timeout(300)
                return ActionResult(True, page_changed=True).to_dict()

            elif action.action == "fill":
                await loc.fill(action.value, timeout=5000)
                await self._page.wait_for_timeout(100)
                return ActionResult(True, page_changed=True).to_dict()

            elif action.action == "check":
                checked = action.checked if action.checked is not None else True
                if checked:
                    await loc.check(timeout=5000)
                else:
                    await loc.uncheck(timeout=5000)
                await self._page.wait_for_timeout(100)
                return ActionResult(True, page_changed=True).to_dict()

            elif action.action == "select":
                await loc.select_option(action.value, timeout=5000)
                await self._page.wait_for_timeout(100)
                return ActionResult(True, page_changed=True).to_dict()

            elif action.action == "submit":
                await loc.click(timeout=5000)
                await self._page.wait_for_timeout(300)
                return ActionResult(True, page_changed=True).to_dict()

        except Exception as e:
            return ActionResult(False, "browser_error", str(e)[:200],
                                page_changed=False).to_dict()

    async def _do_build_observation(self, task_id, episode_id, instruction,
                                    step_index, last_action_result=None,
                                    terminal=False):
        """Build an Observation inside the worker thread using async Playwright API."""
        from urllib.parse import urlparse, parse_qs
        from miniwebwork.agent_env.schemas import Observation, ElementDescriptor
        from miniwebwork.agent_env.observation import (
            _classify_page, INTERACTIVE_ROLES, MAX_VISIBLE_TEXT,
        )

        url = self._page.url
        parsed = urlparse(url)
        path = parsed.path
        page_type = _classify_page(path)
        title = await self._page.title()

        # Extract visible text (async)
        try:
            visible_text = await self._page.locator("body").inner_text(timeout=2000)
        except Exception:
            visible_text = ""
        import re
        visible_text = re.sub(r"\n{3,}", "\n\n", visible_text)
        visible_text = re.sub(r"[ \t]{3,}", "  ", visible_text)
        text_truncated = len(visible_text) >= MAX_VISIBLE_TEXT
        if len(visible_text) > MAX_VISIBLE_TEXT:
            visible_text = visible_text[:MAX_VISIBLE_TEXT]
        visible_text = visible_text.strip()

        # Extract elements (async evaluate)
        try:
            raw = await self._page.evaluate("""() => {
                const selectors = 'a, button, input:not([type="hidden"]), select, textarea, [role="button"], [role="link"], [role="textbox"], [role="searchbox"], [role="checkbox"], [role="combobox"], [role="spinbutton"]';
                const els = document.querySelectorAll(selectors);
                const results = [];
                els.forEach((el, idx) => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) return;
                    const tag = el.tagName.toLowerCase();
                    const type = (el.getAttribute('type') || '').toLowerCase();
                    const testid = el.getAttribute('data-testid') || '';
                    const domId = el.id || '';
                    const name = el.getAttribute('name') || '';
                    const disabled = el.disabled || false;
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    let labelText = '';
                    const lbl = document.querySelector('label[for="' + el.id + '"]');
                    if (lbl) labelText = lbl.textContent.trim().substring(0, 100);
                    let text = '';
                    if (tag === 'a' || tag === 'button') {
                        text = (el.textContent || '').trim().substring(0, 200);
                    }
                    let value = el.value || '';
                    let options = [];
                    if (tag === 'select') {
                        for (let i = 0; i < el.options.length; i++) {
                            options.push({value: el.options[i].value || '', label: el.options[i].textContent.trim()});
                        }
                    }
                    let role = tag;
                    if (tag === 'select') role = 'combobox';
                    else if (tag === 'textarea') role = 'textarea';
                    else if (tag === 'button') role = 'button';
                    else if (tag === 'a') role = 'link';
                    else if (tag === 'input') {
                        if (type === 'search') role = 'searchbox';
                        else if (type === 'checkbox') role = 'checkbox';
                        else if (type === 'number') role = 'spinbutton';
                        else role = 'textbox';
                    }
                    const displayName = ariaLabel || labelText || testid || name || domId || text || 'unnamed_' + tag;
                    results.push({
                        tag, role, type, testid, domId, name, disabled, ariaLabel,
                        labelText, text: text.substring(0, 200), value: value.substring(0, 200),
                        options, displayName: displayName.substring(0, 100),
                        idx
                    });
                });
                return results;
            }""")
        except Exception:
            raw = []

        from miniwebwork.agent_env.observation import INTERACTIVE_ROLES
        elements = []
        seen_ids = set()
        for item in (raw or []):
            role = item.get("role", "unknown")
            testid = item.get("testid", "")
            dom_id = item.get("domId", "")
            name_attr = item.get("name", "")
            tag = item.get("tag", "")
            base_id = testid or dom_id or name_attr or f"{tag}_{item['idx']}"
            element_id = base_id
            dedup = 0
            while element_id in seen_ids:
                dedup += 1
                element_id = f"{base_id}_{dedup}"
            seen_ids.add(element_id)

            elements.append(ElementDescriptor(
                element_id=element_id,
                role=role,
                tag=tag,
                name=item.get("displayName", "")[:100],
                text=item.get("text", "")[:200],
                value=item.get("value", "")[:200],
                input_type=item.get("type", ""),
                testid=testid,
                options=item.get("options", []),
                disabled=item.get("disabled", False),
            ))

        return Observation(
            task_id=task_id, episode_id=episode_id, instruction=instruction,
            step_index=step_index, url=url, path=path, page_type=page_type,
            title=title, visible_text=visible_text, text_truncated=text_truncated,
            elements=elements, last_action_result=last_action_result, terminal=terminal,
        )


# ================================================================
# Task / Result helpers for the worker thread
# ================================================================

class _ResultHolder:
    """Thread-safe result container for PlaywrightThreadManager.call()."""
    __slots__ = ("result", "error", "done")

    def __init__(self):
        self.result = None
        self.error = None
        self.done = threading.Event()


async def _worker_task(func, args, kwargs, holder: _ResultHolder):
    """Execute *func* inside the worker thread's event loop.

    If *func* returns a coroutine (async function), await it.
    """
    try:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        holder.result = result
    except BaseException as exc:
        holder.error = exc
    finally:
        holder.done.set()


# ================================================================
# Find free port helper
# ================================================================

def _find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ================================================================
# ProcurementBrowserEnv
# ================================================================

class ProcurementBrowserEnv:
    """Browser-based procurement agent environment.

    Usage::

        env = ProcurementBrowserEnv()
        obs = env.reset("TASK-001")
        result = env.step(AgentAction(action="click", target="e0"))
        env.close()

    All Playwright operations run inside a dedicated daemon thread with
    the async Playwright API, avoiding asyncio-loop conflicts.
    """

    def __init__(self, max_steps: int = 20, run_id: str = None,
                 headless: bool = True, keep_db: bool = True,
                 task_dir: Path = None):
        self.max_steps = max_steps
        self.run_id = run_id or uuid.uuid4().hex[:12].upper()
        self.headless = headless
        self.keep_db = keep_db
        self._task_dir = task_dir  # explicit task directory; falls back to env var

        # Playwright manager (persistent worker thread)
        self._pw = PlaywrightThreadManager()

        # Internal state
        self._web_process = None
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

    # ================================================================
    # reset
    # ================================================================
    def reset(self, task_id: str) -> Observation:
        """Initialize or reset the environment for a new task."""
        if self._closed:
            raise EnvironmentClosedError("Environment is closed")

        # Ensure task directory is set for this episode
        if self._task_dir is not None:
            os.environ["MINIWEBWORK_TASK_DIR"] = str(self._task_dir)

        # Clean up previous episode
        self._cleanup_episode()

        # Validate task
        task = get_public_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")

        self._task_id = task_id
        self._instruction = task.get("instruction", "")
        self._step_index = 0
        self._terminated = False
        self._truncated = False

        # Setup isolated DB (only on first reset)
        if not self._db_path:
            self._setup_database()

        # Start web service (only if not running)
        if self._web_process is None or self._web_process.poll() is not None:
            self._start_web_service()

        # Start browser (lazy — starts worker thread + launches browser)
        self._start_browser()

        # Navigate to task start page and click start (all via worker thread)
        self._pw.call(self._pw._do_goto,
                      f"http://127.0.0.1:{self._port}/tasks/{task_id}", timeout=10000)
        time.sleep(0.3)
        self._pw.call(self._pw._do_click, '[data-testid="start-task-button"]')
        self._pw.call(self._pw._do_wait_for_timeout, 500)

        # Extract episode_id from the redirect URL
        from urllib.parse import parse_qs, urlparse
        current_url = self._pw.call(self._pw._do_get_url)
        qs = parse_qs(urlparse(current_url).query)
        eids = qs.get("episode_id", [])
        if eids:
            self._episode_id = eids[0]
        else:
            # Fallback: create via DB and navigate
            conn = get_connection(self._db_path)
            self._episode_id = create_episode(conn, self._task_id)
            conn.close()
            self._pw.call(self._pw._do_goto,
                          f"http://127.0.0.1:{self._port}/products?episode_id={self._episode_id}&task_id={task_id}",
                          timeout=10000)
            time.sleep(0.3)

        # Initialize trajectory recorder
        self._trajectory = TrajectoryRecorder(
            run_id=self.run_id,
            task_id=task_id,
            episode_id=self._episode_id,
            instruction=self._instruction,
            agent_name="",
            max_steps=self.max_steps,
        )

        # Build initial observation (via worker thread)
        obs = self._pw.call(
            self._pw._do_build_observation,
            task_id, self._episode_id, self._instruction, self._step_index,
        )
        return obs

    # ================================================================
    # step
    # ================================================================
    def step(self, action: AgentAction) -> StepResult:
        """Execute one action and return the result."""
        if self._closed:
            raise EnvironmentClosedError("Environment is closed")
        if self._terminated or self._truncated:
            raise EpisodeFinishedError("Episode has ended")

        start = time.time()

        # Build current observation for action validation (via worker thread)
        current_obs = self._pw.call(
            self._pw._do_build_observation,
            self._task_id, self._episode_id, self._instruction, self._step_index,
        )

        # Execute action via manager (runs in worker thread)
        result_dict = self._pw.call(
            self._pw._do_execute_action, action, current_obs,
        )

        elapsed = int((time.time() - start) * 1000)

        # Convert dict result back to ActionResult-like object for trajectory
        from miniwebwork.agent_env.schemas import ActionResult
        result = ActionResult(
            success=result_dict.get("success", False),
            error_code=result_dict.get("error_code", ""),
            page_changed=result_dict.get("page_changed", False),
        )

        # Check if action consumed a step
        action_dict = action.to_dict()

        # Record step in trajectory
        if self._trajectory:
            self._trajectory.record_step(
                self._step_index, current_obs, action_dict,
                result_dict, 0.0, self._terminated, self._truncated, elapsed,
            )

        self._step_index += 1

        # Check if we reached max_steps
        if self._step_index >= self.max_steps and not self._terminated:
            self._truncated = True
            return self._build_terminal_result()

        # Build new observation — wait for page to stabilize
        self._pw.call(self._pw._do_wait_for_timeout, 300)
        # Wait for any visible content to appear
        try:
            self._pw.call(self._pw._do_wait_for_visible, "body", 3000)
        except Exception:
            pass
        new_obs = self._pw.call(
            self._pw._do_build_observation,
            self._task_id, self._episode_id, self._instruction, self._step_index,
            last_action_result=result_dict,
        )

        # Check for termination: page changed to procurement_result
        if new_obs.page_type == "procurement_result":
            self._terminated = True
            return self._build_terminal_result()

        # Handle finish action
        if action.action == "finish":
            # Check if a submission was made
            submission_exists = self._check_submission_exists()
            if submission_exists:
                self._terminated = True
                return self._build_terminal_result()
            else:
                # Premature finish
                self._terminated = True
                return self._build_terminal_result(
                    termination_reason="premature_finish",
                    reward=0.0,
                    success=False,
                )

        # Normal step: non-terminal
        return StepResult(
            observation=new_obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "action_result": result_dict,
                "step_index": self._step_index,
                "page_type": new_obs.page_type,
                "elapsed_ms": elapsed,
            },
        )

    # ================================================================
    # close
    # ================================================================
    def close(self):
        """Close environment and clean up all resources."""
        self._cleanup_episode()
        try:
            self._pw.stop()
        except Exception:
            pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ================================================================
    # Internal helpers
    # ================================================================
    def _cleanup_episode(self):
        """Clean up resources from current episode (keep Playwright/browser alive)."""
        # Close page and context (browser stays alive)
        try:
            self._pw.close_page()
        except Exception:
            pass
        try:
            self._pw.close_context()
        except Exception:
            pass

        # Stop web service
        if self._web_process:
            try:
                self._web_process.send_signal(signal.SIGTERM)
                self._web_process.wait(timeout=5)
            except Exception:
                try:
                    self._web_process.kill()
                except Exception:
                    pass
            self._web_process = None

    def _setup_database(self):
        """Create an isolated runtime database."""
        runtime_dir = PROJECT_ROOT / "data" / "runtime" / "m1_2"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(runtime_dir / f"{self.run_id}.db")
        os.environ["MINIWEBWORK_DB_PATH"] = self._db_path

        conn = get_connection(self._db_path)
        init_schema(conn)
        seed_database(conn)
        conn.close()

    def _start_web_service(self):
        """Start the FastAPI web service on a free port."""
        self._port = _find_free_port()
        python = sys.executable
        self._web_process = subprocess.Popen(
            [python, "-m", "uvicorn", "miniwebwork.webapp:app",
             "--host", "127.0.0.1", "--port", str(self._port),
             "--log-level", "warning"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for service to be ready
        import urllib.request
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self._port}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.3)

    def _start_browser(self):
        """Start Playwright Chromium browser (persistent worker thread)."""
        self._pw.start()                    # start worker thread (no-op if already running)
        self._pw.launch_browser(headless=self.headless)
        self._pw.new_context_and_page()

    def _check_submission_exists(self) -> bool:
        """Check if a submission exists for the current episode."""
        try:
            conn = get_connection(self._db_path)
            sub = conn.execute(
                "SELECT submission_id FROM procurement_submissions WHERE episode_id = ?",
                (self._episode_id,),
            ).fetchone()
            conn.close()
            return sub is not None
        except Exception:
            return False

    def _build_terminal_result(self, termination_reason: str = None,
                                reward: float = None, success: bool = None) -> StepResult:
        """Build the final StepResult including verifier call."""
        # Determine reward and success from verifier if not pre-set
        if reward is None:
            try:
                vr = verify_episode(self._task_id, self._episode_id, self._db_path)
                verifier_success = vr.success
            except Exception:
                verifier_success = False
            reward = 1.0 if verifier_success else 0.0
            success = verifier_success
            if termination_reason is None:
                termination_reason = "verified_submission"
        else:
            verifier_success = success
            if termination_reason is None:
                termination_reason = "premature_finish" if not success else "verified_submission"

        # Verify again for full details (if not already done)
        try:
            vr = verify_episode(self._task_id, self._episode_id, self._db_path)
            verification_dict = vr.to_dict()
        except Exception:
            verification_dict = {}

        # Build final observation (via worker thread)
        final_obs = self._pw.call(
            self._pw._do_build_observation,
            self._task_id, self._episode_id, self._instruction, self._step_index,
            terminal=True,
        )

        # Finalize trajectory
        if self._trajectory:
            self._trajectory.agent_name = getattr(self, '_agent_name', '')
            self._trajectory.finalize(reward, success or False, termination_reason, verification_dict)

        return StepResult(
            observation=final_obs,
            reward=reward,
            terminated=True,
            truncated=self._truncated,
            info={
                "step_index": self._step_index,
                "termination_reason": termination_reason,
                "verifier_success": verifier_success,
                "submission_id": verification_dict.get("submission_id", ""),
                "failure_reasons": verification_dict.get("failure_reasons", []),
                "page_type": final_obs.page_type,
            },
        )

    # ================================================================
    # Properties for agent use
    # ================================================================
    @property
    def trajectory(self) -> Optional[TrajectoryRecorder]:
        return self._trajectory

    def set_agent_name(self, name: str):
        self._agent_name = name

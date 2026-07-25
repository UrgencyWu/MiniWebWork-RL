"""
Procurement Browser Agent Environment.

Standard Gym-like interface: reset(task_id) -> Observation, step(action) -> StepResult, close().
"""

import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

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


def _find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ProcurementBrowserEnv:
    """Browser-based procurement agent environment.

    Usage:
        env = ProcurementBrowserEnv()
        obs = env.reset("TASK-001")
        result = env.step(AgentAction(action="click", target="e0"))
        env.close()
    """

    def __init__(self, max_steps: int = 20, run_id: str = None,
                 headless: bool = True, keep_db: bool = True):
        self.max_steps = max_steps
        self.run_id = run_id or uuid.uuid4().hex[:12].upper()
        self.headless = headless
        self.keep_db = keep_db

        # Internal state
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
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

        # Clean up previous state
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

        # Start browser (reuse Playwright across resets)
        self._start_browser()

        # Navigate to task start page and click start
        self._page.goto(f"http://127.0.0.1:{self._port}/tasks/{task_id}", timeout=10000)
        time.sleep(0.3)
        self._page.locator('[data-testid="start-task-button"]').click()
        self._page.wait_for_timeout(500)

        # Extract episode_id from the redirect URL
        from urllib.parse import parse_qs, urlparse
        current_url = self._page.url
        qs = parse_qs(urlparse(current_url).query)
        eids = qs.get("episode_id", [])
        if eids:
            self._episode_id = eids[0]
        else:
            # Fallback: create via DB and navigate
            conn = get_connection(self._db_path)
            self._episode_id = create_episode(conn, self._task_id)
            conn.close()
            self._page.goto(f"http://127.0.0.1:{self._port}/products?episode_id={self._episode_id}&task_id={task_id}", timeout=10000)
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

        # Build initial observation
        obs = build_observation(
            self._page, task_id, self._episode_id,
            self._instruction, self._step_index,
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

        # Build current observation for action validation
        current_obs = build_observation(
            self._page, self._task_id, self._episode_id,
            self._instruction, self._step_index,
        )

        # Execute action
        result = execute_action(action, current_obs, self._page)
        elapsed = int((time.time() - start) * 1000)

        # Check if action consumed a step
        action_dict = action.to_dict()
        result_dict = result.to_dict()

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
        self._page.wait_for_timeout(300)
        # Wait for any visible content to appear
        try:
            self._page.locator("body").wait_for(state="visible", timeout=3000)
        except Exception:
            pass
        new_obs = build_observation(
            self._page, self._task_id, self._episode_id,
            self._instruction, self._step_index,
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
        if self._page and not self._page.is_closed():
            try:
                self._page.close()
            except Exception:
                pass
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
        # Don't close browser — reuse across resets to avoid async loop issues
        if self._web_process:
            try:
                self._web_process.send_signal(signal.SIGTERM)
                self._web_process.wait(timeout=5)
            except Exception:
                try:
                    self._web_process.kill()
                except Exception:
                    pass
        self._page = None
        self._context = None
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
        """Start Playwright Chromium browser (reuses Playwright across resets)."""
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        if self._browser is None or not self._browser.is_connected():
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
        self._context = self._browser.new_context()
        self._page = self._context.new_page()

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

        # Build final observation
        final_obs = build_observation(
            self._page, self._task_id, self._episode_id,
            self._instruction, self._step_index,
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

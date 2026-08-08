"""Persistent browser lanes feeding one shared asynchronous vLLM engine."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..m4_long_horizon_protocol import (
    DATASET_ROOT,
    GROUP_SIZE,
    MAX_ENVIRONMENT_STEPS,
    MAX_MODEL_TURNS,
    PROMPT_CONTRACT,
    SEED_ROOT,
)
from .contracts import RunIdentity
from .journal import CollectionStore
from .rollout import RolloutEvidenceWriter, run_atomic_k4_group
from .sampler import TaskDescriptor
from .vllm_backend import (
    AsyncVLLMGenerationEngine,
    RolloutRequestContext,
    ThreadsafeVLLMBackend,
)

ALLOWED_BROWSER_WORKERS = (1, 2, 4, 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class BrowserWorkerPoolConfig:
    total_workers: int
    task_dir: Path = DATASET_ROOT / "train"
    seed_dir: Path = SEED_ROOT
    headless: bool = True
    generation_timeout_seconds: float = 900.0

    def validate(self) -> None:
        _require(self.total_workers in ALLOWED_BROWSER_WORKERS, "browser worker count drift")
        task_dir = Path(self.task_dir).expanduser().resolve()
        seed_dir = Path(self.seed_dir).expanduser().resolve()
        _require(task_dir == (DATASET_ROOT / "train").resolve(), "browser pool may read only train tasks")
        _require(seed_dir == SEED_ROOT.resolve(), "browser pool seed root drift")
        _require(task_dir.is_dir(), "browser pool train task directory is missing")
        _require(seed_dir.is_dir(), "browser pool seed directory is missing")
        _require(self.headless is True, "formal browser workers must be headless")
        _require(self.generation_timeout_seconds > 0, "browser generation timeout is invalid")

    @property
    def concurrent_group_slots(self) -> int:
        return 2 if self.total_workers == 8 else 1

    @property
    def workers_per_group(self) -> int:
        return min(GROUP_SIZE, self.total_workers)


EnvironmentFactory = Callable[[int], Any]
BackendFactory = Callable[[RolloutRequestContext], Any]
AgentFactory = Callable[[Any], Any]
EpisodeRunner = Callable[..., Mapping[str, Any]]


class PersistentBrowserLane:
    """Serialize episodes through one reusable environment/browser instance."""

    def __init__(
        self,
        *,
        lane_index: int,
        environment_factory: EnvironmentFactory,
        agent_factory: AgentFactory,
        episode_runner: EpisodeRunner,
    ):
        _require(isinstance(lane_index, int) and lane_index >= 0, "invalid browser lane index")
        self.lane_index = lane_index
        self._environment_factory = environment_factory
        self._agent_factory = agent_factory
        self._episode_runner = episode_runner
        self._environment: Any | None = None
        self._lock = threading.Lock()
        self._closed = False

    def _environment_or_create(self) -> Any:
        _require(not self._closed, "browser lane is closed")
        if self._environment is None:
            self._environment = self._environment_factory(self.lane_index)
        return self._environment

    def _recycle_environment(self) -> None:
        environment, self._environment = self._environment, None
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass

    def run(
        self,
        *,
        task_id: str,
        backend: Any,
        writer: RolloutEvidenceWriter,
    ) -> Mapping[str, Any]:
        with self._lock:
            environment = self._environment_or_create()
            agent = self._agent_factory(backend)
            result = self._episode_runner(
                task_id,
                environment,
                agent,
                MAX_MODEL_TURNS,
                MAX_ENVIRONMENT_STEPS,
                turn_generated_callback=writer.on_turn_generated,
                turn_completed_callback=writer.on_turn_completed,
            )
            _require(result.get("task_id") == task_id, "browser lane returned the wrong task")
            if result.get("rollout_valid") is not True:
                self._recycle_environment()
            return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._recycle_environment()
            self._closed = True


class VLLMBrowserWorkerPool:
    """Lease disjoint persistent lanes to one or two concurrent exact-K groups."""

    def __init__(
        self,
        *,
        identity: RunIdentity,
        engine: AsyncVLLMGenerationEngine,
        event_loop: asyncio.AbstractEventLoop,
        event_loop_thread_id: int,
        config: BrowserWorkerPoolConfig,
        environment_factory: EnvironmentFactory | None = None,
        backend_factory: BackendFactory | None = None,
        agent_factory: AgentFactory | None = None,
        episode_runner: EpisodeRunner | None = None,
    ):
        identity.validate()
        config.validate()
        _require(event_loop.is_closed() is False, "browser pool event loop is closed")
        self.identity = identity
        self.engine = engine
        self.event_loop = event_loop
        self.event_loop_thread_id = event_loop_thread_id
        self.config = config
        task_dir = Path(config.task_dir).expanduser().resolve()
        seed_dir = Path(config.seed_dir).expanduser().resolve()

        def default_environment_factory(lane_index: int):
            from ..agent_env.environment import ProcurementBrowserEnv

            return ProcurementBrowserEnv(
                max_steps=MAX_ENVIRONMENT_STEPS,
                run_id=(
                    f"m4lh-{identity.method}-s{identity.seed}-i{identity.iteration_index}"
                    f"-lane{lane_index}"
                ),
                headless=True,
                keep_db=False,
                task_dir=task_dir,
                seed_dir=seed_dir,
            )

        def default_backend_factory(context: RolloutRequestContext):
            return ThreadsafeVLLMBackend(
                engine=engine,
                event_loop=event_loop,
                context=context,
                timeout_seconds=config.generation_timeout_seconds,
                event_loop_thread_id=event_loop_thread_id,
            )

        def default_agent_factory(backend: Any):
            from ..model_agent import output_parser, prompt_builder
            from ..model_agent.qwen_agent import QwenBrowserAgent

            return QwenBrowserAgent(
                backend,
                prompt_builder,
                output_parser,
                prompt_version=PROMPT_CONTRACT,
            )

        if episode_runner is None:
            from ..model_agent.agent_loop import run_model_episode

            actual_episode_runner = run_model_episode
        else:
            actual_episode_runner = episode_runner

        self._backend_factory = backend_factory or default_backend_factory
        self._lanes = [
            PersistentBrowserLane(
                lane_index=index,
                environment_factory=environment_factory or default_environment_factory,
                agent_factory=agent_factory or default_agent_factory,
                episode_runner=actual_episode_runner,
            )
            for index in range(config.total_workers)
        ]
        self._available_slots: queue.Queue[tuple[int, ...]] = queue.Queue()
        if config.total_workers == 8:
            self._available_slots.put(tuple(range(0, 4)))
            self._available_slots.put(tuple(range(4, 8)))
        else:
            self._available_slots.put(tuple(range(config.total_workers)))
        self._closed = False

    def run_group(
        self,
        *,
        store: CollectionStore,
        group_id: str,
        task: TaskDescriptor,
    ) -> dict[str, Any]:
        _require(not self._closed, "browser worker pool is closed")
        _require(store.identity.sha256 == self.identity.sha256, "browser pool/store identity mismatch")
        slot = self._available_slots.get(timeout=self.config.generation_timeout_seconds)
        try:
            def worker(rollout_index: int, writer: RolloutEvidenceWriter):
                lane_index = slot[rollout_index % len(slot)]
                context = RolloutRequestContext(
                    run_seed=self.identity.seed,
                    iteration_index=self.identity.iteration_index,
                    group_id=group_id,
                    attempt_index=writer.attempt_index,
                    trajectory_id=writer.trajectory_id,
                    rollout_index=rollout_index,
                )
                backend = self._backend_factory(context)
                return self._lanes[lane_index].run(
                    task_id=task.task_id,
                    backend=backend,
                    writer=writer,
                )

            return run_atomic_k4_group(
                store=store,
                identity=self.identity,
                group_id=group_id,
                task_id=task.task_id,
                worker=worker,
                maximum_workers=self.config.workers_per_group,
            )
        finally:
            self._available_slots.put(slot)

    def close(self) -> None:
        if self._closed:
            return
        errors = []
        for lane in self._lanes:
            try:
                lane.close()
            except Exception as exc:
                errors.append(exc)
        self._closed = True
        if errors:
            raise RuntimeError(
                "browser worker cleanup failed: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    def __enter__(self) -> "VLLMBrowserWorkerPool":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except Exception:
            if exc is None:
                raise
        return False

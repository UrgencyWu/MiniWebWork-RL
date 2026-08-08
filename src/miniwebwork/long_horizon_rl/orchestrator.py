"""Deterministic, resumable orchestration for one online collection iteration."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..m4_long_horizon_protocol import (
    ACTION_TOKEN_CAP,
    DATASET_ROOT,
    MAX_TASKS_PER_ITERATION,
    SPLIT_COUNTS,
    TASK_SAMPLER_VERSION,
)
from .contracts import RunIdentity, sha256_json, validate_committed_group
from .journal import CollectionStore
from .rollout import admit_next_group
from .sampler import DeterministicSignalSampler, TaskDescriptor, TaskSignal

COLLECTION_ORCHESTRATOR_SCHEMA = "m4_long_horizon_collection_orchestrator_v1"
MAXIMUM_CONCURRENT_K4_GROUPS = 2
MAXIMUM_ZERO_TOKEN_NO_PROGRESS_BATCHES = 3
TRAIN_PUBLIC_PATH = DATASET_ROOT / "train" / "train_public.jsonl"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PlannedGroup:
    plan_index: int
    group_id: str
    task: TaskDescriptor


GroupRunner = Callable[[str, TaskDescriptor], Mapping[str, Any]]


def load_public_task_roster(
    path: Path,
    *,
    expected_count: int | None = None,
) -> tuple[TaskDescriptor, ...]:
    """Load only the three sampler fields from a public task JSONL."""

    source = Path(path).expanduser().resolve()
    _require(source.is_file(), f"public task roster is missing: {source}")
    rows: list[TaskDescriptor] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        _require(bool(line.strip()), f"blank public task row at line {line_number}")
        raw = json.loads(line)
        _require(isinstance(raw, Mapping), f"public task row {line_number} is not an object")
        for field in ("task_id", "task_family", "horizon_stratum"):
            _require(isinstance(raw.get(field), str), f"public task row {line_number} lacks {field}")
        descriptor = TaskDescriptor(
            task_id=raw["task_id"],
            task_family=raw["task_family"],
            horizon_stratum=raw["horizon_stratum"],
        )
        descriptor.validate()
        rows.append(descriptor)
    _require(bool(rows), "public task roster is empty")
    _require(len({row.task_id for row in rows}) == len(rows), "public task roster contains duplicate IDs")
    if expected_count is not None:
        _require(len(rows) == expected_count, "public task roster count drift")
    return tuple(sorted(rows))


def load_frozen_train_roster() -> tuple[TaskDescriptor, ...]:
    """Fail closed on any path other than the frozen train-public roster."""

    path = TRAIN_PUBLIC_PATH.resolve()
    _require(path.parent.name == "train", "formal roster is not the train split")
    _require(path.name == "train_public.jsonl", "formal roster is not public-only")
    return load_public_task_roster(path, expected_count=SPLIT_COUNTS["train"])


def sampler_task_order_sha256(tasks: Sequence[TaskDescriptor], *, study_seed: int) -> str:
    """Bind run identity to the exact public roster and sampler tie-break seed."""

    sampler = DeterministicSignalSampler(tasks, study_seed=study_seed)
    return sha256_json(
        {
            "sampler_version": TASK_SAMPLER_VERSION,
            "study_seed": study_seed,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_family": task.task_family,
                    "horizon_stratum": task.horizon_stratum,
                }
                for task in sampler.tasks
            ],
        }
    )


def restore_sampler(
    tasks: Sequence[TaskDescriptor],
    *,
    study_seed: int,
    state: Mapping[str, Any] | None,
) -> DeterministicSignalSampler:
    """Rebuild only durable integer signals; derived posterior fields are re-computed."""

    if state is None:
        return DeterministicSignalSampler(tasks, study_seed=study_seed)
    payload = dict(state)
    _require(
        payload.get("schema_version") == "m4_long_horizon_sampler_state_v1",
        "sampler state schema drift",
    )
    _require(payload.get("sampler_version") == TASK_SAMPLER_VERSION, "sampler version drift")
    _require(payload.get("study_seed") == study_seed, "sampler seed drift")
    _require(payload.get("task_count") == len(tasks), "sampler task-count drift")
    raw_signals = payload.get("signals")
    _require(isinstance(raw_signals, Mapping), "sampler state lacks signals")
    task_ids = {task.task_id for task in tasks}
    _require(set(raw_signals) == task_ids, "sampler signal roster drift")
    signals = {}
    for task_id, raw in raw_signals.items():
        _require(isinstance(raw, Mapping), f"sampler signal is malformed: {task_id}")
        signal = TaskSignal(
            committed_groups=raw.get("committed_groups"),
            infra_invalid_attempts=raw.get("infra_invalid_attempts"),
            valid_trajectories=raw.get("valid_trajectories"),
            successes=raw.get("successes"),
        )
        signal.validate()
        signals[task_id] = signal
    return DeterministicSignalSampler(tasks, study_seed=study_seed, signals=signals)


def _planned_groups(
    sampler: DeterministicSignalSampler,
    *,
    iteration_index: int,
) -> tuple[PlannedGroup, ...]:
    selected = sampler.select(
        iteration_index=iteration_index,
        limit=min(MAX_TASKS_PER_ITERATION, len(sampler.tasks)),
    )
    return tuple(
        PlannedGroup(
            plan_index=index,
            group_id=f"i{iteration_index:04d}-g{index:04d}",
            task=task,
        )
        for index, task in enumerate(selected)
    )


def _rebuild_current_iteration_signals(
    *,
    sampler: DeterministicSignalSampler,
    store: CollectionStore,
    identity: RunIdentity,
    plan: Sequence[PlannedGroup],
) -> dict[str, Any]:
    by_id = {entry.group_id: entry for entry in plan}
    start_tasks: dict[tuple[str, int], str] = {}
    invalid_attempts = 0
    for event in store.journal.events:
        payload = event["payload"]
        if event["event_type"] == "group_started":
            group_id = str(payload["group_id"])
            _require(group_id in by_id, "journal references a group outside the frozen plan")
            _require(payload["task_id"] == by_id[group_id].task.task_id, "journal group/task plan drift")
            start_tasks[(group_id, int(payload["attempt_index"]))] = str(payload["task_id"])
        elif event["event_type"] == "group_invalid":
            key = (str(payload["group_id"]), int(payload["attempt_index"]))
            _require(key in start_tasks, "invalid attempt lacks a planned group start")
            sampler.record_infra_invalid_attempt(start_tasks[key])
            invalid_attempts += 1
    committed: dict[str, dict[str, Any]] = {}
    for group in store.load_committed_groups():
        validated = validate_committed_group(group, identity)
        group_id = validated["group_id"]
        _require(group_id in by_id, "committed group is outside the frozen plan")
        _require(validated["task_id"] == by_id[group_id].task.task_id, "committed group/task plan drift")
        _require(group_id not in committed, "duplicate committed group")
        sampler.record_committed_group(
            validated["task_id"],
            [trajectory["reward"] for trajectory in validated["trajectories"]],
        )
        committed[group_id] = validated
    return {"committed": committed, "invalid_attempts": invalid_attempts}


def collect_iteration(
    *,
    store: CollectionStore,
    identity: RunIdentity,
    tasks: Sequence[TaskDescriptor],
    initial_sampler_state: Mapping[str, Any] | None,
    group_runner: GroupRunner,
    global_generated_action_tokens_before: int,
    maximum_concurrent_groups: int = MAXIMUM_CONCURRENT_K4_GROUPS,
    token_cap: int = ACTION_TOKEN_CAP,
) -> dict[str, Any]:
    """Resume or collect one frozen-policy iteration without exceeding the global cap."""

    identity.validate()
    _require(store.identity.sha256 == identity.sha256, "collector/store identity mismatch")
    _require(
        isinstance(global_generated_action_tokens_before, int)
        and global_generated_action_tokens_before >= 0,
        "invalid global token predecessor",
    )
    _require(
        1 <= maximum_concurrent_groups <= MAXIMUM_CONCURRENT_K4_GROUPS,
        "concurrent K4 group count exceeds the runtime contract",
    )
    _require(isinstance(token_cap, int) and token_cap > 0, "invalid action-token cap")
    sampler = restore_sampler(tasks, study_seed=identity.seed, state=initial_sampler_state)
    _require(
        sampler_task_order_sha256(tasks, study_seed=identity.seed) == identity.task_order_sha256,
        "sampler roster/run identity mismatch",
    )
    plan = _planned_groups(sampler, iteration_index=identity.iteration_index)
    store.recover_incomplete_attempts()
    rebuilt = _rebuild_current_iteration_signals(
        sampler=sampler,
        store=store,
        identity=identity,
        plan=plan,
    )
    committed: dict[str, dict[str, Any]] = rebuilt["committed"]

    manifest_path = store.root / "collection_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(
            existing.get("task_sampler_state") == sampler.audit_dict(),
            "frozen collection sampler reconstruction drift",
        )
        manifest = store.freeze_collection(
            iteration_index=identity.iteration_index,
            stopped_for_token_budget=bool(existing.get("stopped_for_token_budget")),
            task_sampler_state=sampler.audit_dict(),
        )
        return {
            "schema_version": COLLECTION_ORCHESTRATOR_SCHEMA,
            "identity_sha256": identity.sha256,
            "resumed_frozen_collection": True,
            "selected_task_ids": [entry.task.task_id for entry in plan],
            "committed_group_count": len(committed),
            "infra_invalid_attempt_count": rebuilt["invalid_attempts"],
            "stopped_for_token_budget": manifest["stopped_for_token_budget"],
            "ready_for_update": manifest["group_count"] > 0,
            "collection_manifest": manifest,
        }

    stopped_for_token_budget = False
    zero_token_no_progress_batches = 0
    while len(committed) < len(plan):
        pending = [entry for entry in plan if entry.group_id not in committed]
        batch: list[PlannedGroup] = []
        for entry in pending:
            if len(batch) >= maximum_concurrent_groups:
                break
            admission = admit_next_group(
                global_tokens_before_iteration=global_generated_action_tokens_before,
                current_iteration_tokens=store.journal.generated_action_tokens,
                reserved_inflight_groups=len(batch),
                token_cap=token_cap,
            )
            if not admission.allowed:
                stopped_for_token_budget = True
                break
            batch.append(entry)
        if not batch:
            break

        token_count_before_batch = store.journal.generated_action_tokens
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(group_runner, entry.group_id, entry.task)
                for entry in batch
            ]
            results = [future.result() for future in futures]
        for entry, result in zip(batch, results):
            _require(result.get("group_id") == entry.group_id, "group runner returned the wrong group")
            if result.get("committed") is True:
                group = validate_committed_group(result["group"], identity)
                _require(group["task_id"] == entry.task.task_id, "group runner returned the wrong task")
                sampler.record_committed_group(
                    group["task_id"],
                    [trajectory["reward"] for trajectory in group["trajectories"]],
                )
                committed[entry.group_id] = group
            else:
                sampler.record_infra_invalid_attempt(entry.task.task_id)
                rebuilt["invalid_attempts"] += 1
        token_count_after_batch = store.journal.generated_action_tokens
        if token_count_after_batch == token_count_before_batch and any(
            result.get("committed") is not True for result in results
        ):
            zero_token_no_progress_batches += 1
            _require(
                zero_token_no_progress_batches < MAXIMUM_ZERO_TOKEN_NO_PROGRESS_BATCHES,
                "repeated infrastructure failures generated no chargeable tokens",
            )
        else:
            zero_token_no_progress_batches = 0

    if not committed:
        return {
            "schema_version": COLLECTION_ORCHESTRATOR_SCHEMA,
            "identity_sha256": identity.sha256,
            "resumed_frozen_collection": False,
            "selected_task_ids": [entry.task.task_id for entry in plan],
            "committed_group_count": 0,
            "infra_invalid_attempt_count": rebuilt["invalid_attempts"],
            "stopped_for_token_budget": stopped_for_token_budget,
            "ready_for_update": False,
            "collection_manifest": None,
        }
    manifest = store.freeze_collection(
        iteration_index=identity.iteration_index,
        stopped_for_token_budget=stopped_for_token_budget,
        task_sampler_state=sampler.audit_dict(),
    )
    return {
        "schema_version": COLLECTION_ORCHESTRATOR_SCHEMA,
        "identity_sha256": identity.sha256,
        "resumed_frozen_collection": False,
        "selected_task_ids": [entry.task.task_id for entry in plan],
        "committed_group_count": len(committed),
        "infra_invalid_attempt_count": rebuilt["invalid_attempts"],
        "stopped_for_token_budget": stopped_for_token_budget,
        "ready_for_update": True,
        "collection_manifest": manifest,
    }

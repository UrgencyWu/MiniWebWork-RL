"""Minimal data-split contracts for M6 Phase10-C SFT Specialists and OPD."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .long_horizon_rl.contracts import sha256_json
from .m5_webshop_protocol import normalized_instruction, task_id_for_goal_index


EXPOSURE_SCHEMA = "m6_phase10c_exposure_union_v1"
SPLIT_SCHEMA = "m6_phase10c_specialist_data_split_v1"
SELECTION_SEED = 20260860
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PHASE10B_EXPOSURE_SCHEMA = "m6_phase10b_exposure_union_v1"
PHASE10B_SPLIT_SCHEMA = "m6_phase10b_split_lock_v1"
PHASE10B_ROLE_COUNTS = {
    "specialist_nav_qualification": 24,
    "specialist_match_qualification": 24,
    "specialist_finish_qualification": 24,
    "opd_smoke": 8,
    "opd_train": 40,
    "opd_monitor_a": 64,
    "opd_monitor_b": 64,
    "opd_final_dev": 500,
}

# Narrower finish tasks are selected first so the broader match proxy cannot
# consume them.  Reserving the complete split does not authorize training all
# of it: Phase10-C begins with a small corpus/one-update smoke from each train
# role and expands only after the teacher gains on its held-out dev role.
ROLE_COUNTS = {
    "teacher_finish_train": 128,
    "teacher_finish_dev": 32,
    "teacher_finish_qualification": 32,
    "teacher_match_train": 128,
    "teacher_match_dev": 32,
    "teacher_match_qualification": 32,
    "teacher_nav_train": 128,
    "teacher_nav_dev": 32,
    "teacher_nav_qualification": 32,
    "opd_smoke": 8,
    "opd_train": 40,
    "opd_monitor_a": 64,
    "opd_monitor_b": 64,
    "opd_final_dev": 500,
}

SPECIALIST_BY_ROLE_PREFIX = {
    "teacher_nav_": "S_nav_sft",
    "teacher_match_": "S_match_sft",
    "teacher_finish_": "S_finish_sft",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _goal_index(task_id: str) -> int:
    _require(task_id.startswith("webshop_goal_"), "Phase10-C task identity drift")
    return int(task_id.rsplit("_", 1)[1])


def _normalized_hash(instruction: str) -> str:
    return sha256_json(normalized_instruction(instruction))


def validate_phase10b_exposure_union(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the immutable fields Phase10-C consumes, without torch."""

    value = dict(payload)
    _require(value.get("schema_version") == PHASE10B_EXPOSURE_SCHEMA,
             "Phase10-C input Phase10-B exposure schema drift")
    rows = value.get("exposures")
    _require(isinstance(rows, list) and rows, "Phase10-C input Phase10-B exposure rows missing")
    _require(value.get("content_sha256") == _self_hash(value),
             "Phase10-C input Phase10-B exposure self-hash drift")
    return value


def validate_phase10b_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete Phase10-B role reservation without training imports."""

    value = dict(payload)
    _require(value.get("schema_version") == PHASE10B_SPLIT_SCHEMA,
             "Phase10-C input Phase10-B split schema drift")
    roles = value.get("roles")
    _require(isinstance(roles, Mapping) and set(roles) == set(PHASE10B_ROLE_COUNTS),
             "Phase10-C input Phase10-B roles drift")
    seen: set[str] = set()
    for role, count in PHASE10B_ROLE_COUNTS.items():
        task_ids = roles[role].get("task_ids")
        _require(isinstance(task_ids, list) and len(task_ids) == len(set(task_ids)) == count,
                 f"Phase10-C input Phase10-B role count drift: {role}")
        _require(not (set(task_ids) & seen), f"Phase10-C input Phase10-B role overlap: {role}")
        seen.update(task_ids)
    _require(value.get("content_sha256") == _self_hash(value),
             "Phase10-C input Phase10-B split self-hash drift")
    return value


def _rank(task_id: str, *, role: str, seed: int) -> bytes:
    return hashlib.sha256(f"m6-phase10c-specialist-data-v1|{seed}|{role}|{task_id}".encode()).digest()


def _constraint_count(goal: Mapping[str, Any]) -> int:
    attributes = goal.get("instruction_attributes") or goal.get("attributes") or []
    options = goal.get("goal_options") or []
    return len(attributes) + len(options) + int(isinstance(goal.get("price_upper"), (int, float)))


def _proxy(role: str, goal: Mapping[str, Any]) -> bool:
    options = goal.get("goal_options") or []
    price = goal.get("price_upper")
    constraints = _constraint_count(goal)
    if role.startswith("teacher_nav_"):
        return not options and constraints <= 4
    if role.startswith("teacher_match_"):
        return bool(options) and constraints >= 3
    if role.startswith("teacher_finish_"):
        return bool(options) and isinstance(price, (int, float)) and constraints >= 4
    return True


def _specialist_for_role(role: str) -> str | None:
    for prefix, identity in SPECIALIST_BY_ROLE_PREFIX.items():
        if role.startswith(prefix):
            return identity
    return None


def build_phase10c_exposure_union(
    *,
    goals: Sequence[Mapping[str, Any]],
    phase10b_exposure: Mapping[str, Any],
    phase10b_split: Mapping[str, Any],
    producer_git_sha: str,
) -> dict[str, Any]:
    """Burn every Phase10-B reserved role before selecting Phase10-C tasks."""

    old = validate_phase10b_exposure_union(phase10b_exposure)
    split = validate_phase10b_split(phase10b_split)
    _require(len(goals) == 12087, "Phase10-C canonical goal count drift")
    rows: dict[int, dict[str, Any]] = {}
    for row in old["exposures"]:
        rows[int(row["goal_index"])] = {
            "goal_index": int(row["goal_index"]),
            "task_id": str(row["task_id"]),
            "normalized_instruction_sha256": str(row["normalized_instruction_sha256"]),
            "reasons": list(row["reasons"]),
            "source_labels": list(row["source_labels"]),
        }
    phase10b_ids: set[str] = set()
    for role, item in split["roles"].items():
        for task_id in item["task_ids"]:
            index = _goal_index(task_id)
            goal = goals[index]
            expected = _normalized_hash(str(goal["instruction"]))
            current = rows.setdefault(index, {
                "goal_index": index,
                "task_id": task_id,
                "normalized_instruction_sha256": expected,
                "reasons": [],
                "source_labels": [],
            })
            _require(current["task_id"] == task_id, "Phase10-C task collision")
            _require(current["normalized_instruction_sha256"] == expected,
                     "Phase10-C normalized instruction collision")
            current["reasons"] = sorted(set(current["reasons"]) | {f"Phase10-B reserved/read role: {role}"})
            current["source_labels"] = sorted(set(current["source_labels"]) | {f"phase10b_{role}"})
            phase10b_ids.add(task_id)
    exposures = [rows[index] for index in sorted(rows)]
    result = {
        "schema_version": EXPOSURE_SCHEMA,
        "development_only": True,
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "promotion_and_holdout_read": False,
        "producer_git_sha": producer_git_sha,
        "phase10b_exposure_content_sha256": old["content_sha256"],
        "phase10b_split_content_sha256": split["content_sha256"],
        "phase10b_reserved_task_count": len(phase10b_ids),
        "phase10b_reserved_task_id_sha256": sha256_json(sorted(phase10b_ids)),
        "exposures": exposures,
        "exposed_task_count": len(exposures),
        "exposed_task_id_sha256": sha256_json([row["task_id"] for row in exposures]),
    }
    result["content_sha256"] = _self_hash(result)
    return validate_phase10c_exposure_union(result)


def validate_phase10c_exposure_union(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == EXPOSURE_SCHEMA, "Phase10-C exposure schema drift")
    _require(value.get("development_only") is True and value.get("complete") is True,
             "Phase10-C exposure is incomplete")
    _require(value.get("training_performed") is False and value.get("optimizer_steps") == 0,
             "Phase10-C exposure unexpectedly trained")
    _require(value.get("promotion_and_holdout_read") is False,
             "Phase10-C exposure opened promotion/holdout")
    rows = value.get("exposures")
    _require(isinstance(rows, list) and rows, "Phase10-C exposure rows missing")
    indices = [row.get("goal_index") for row in rows]
    _require(indices == sorted(set(indices)), "Phase10-C exposure rows are not sorted/unique")
    for row in rows:
        _require(row.get("task_id") == task_id_for_goal_index(row["goal_index"]),
                 "Phase10-C exposure identity drift")
        _require(SHA256_RE.fullmatch(str(row.get("normalized_instruction_sha256"))) is not None,
                 "Phase10-C exposure normalized instruction hash drift")
        _require(row.get("reasons") and row.get("source_labels"), "Phase10-C exposure provenance missing")
    _require(value.get("exposed_task_count") == len(rows), "Phase10-C exposure count drift")
    _require(value.get("exposed_task_id_sha256") == sha256_json([row["task_id"] for row in rows]),
             "Phase10-C exposure aggregate hash drift")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10-C exposure self-hash drift")
    return value


def build_phase10c_split(
    *,
    goals: Sequence[Mapping[str, Any]],
    train_task_ids: Sequence[str],
    exposure_union: Mapping[str, Any],
    base_split_content_sha256: str,
    producer_git_sha: str,
) -> dict[str, Any]:
    """Reserve disjoint teacher-SFT and OPD roles with three essential gates."""

    exposure = validate_phase10c_exposure_union(exposure_union)
    _require(len(goals) == 12087, "Phase10-C split canonical goal count drift")
    goal_map = {task_id_for_goal_index(index): goal for index, goal in enumerate(goals)}
    exposed_ids = {row["task_id"] for row in exposure["exposures"]}
    exposed_instructions = {row["normalized_instruction_sha256"] for row in exposure["exposures"]}
    eligible = [
        task_id for task_id in train_task_ids
        if task_id not in exposed_ids
        and _normalized_hash(str(goal_map[task_id]["instruction"])) not in exposed_instructions
    ]
    _require(len(eligible) >= sum(ROLE_COUNTS.values()), "Phase10-C fresh train population is too small")
    remaining = list(eligible)
    roles: dict[str, dict[str, Any]] = {}
    used_instruction_hashes: set[str] = set()
    for offset, (role, count) in enumerate(ROLE_COUNTS.items()):
        seed = SELECTION_SEED + offset
        candidates = [
            task_id for task_id in remaining
            if _proxy(role, goal_map[task_id])
            and _normalized_hash(str(goal_map[task_id]["instruction"])) not in used_instruction_hashes
        ]
        selected: list[str] = []
        role_instruction_hashes: set[str] = set()
        for task_id in sorted(candidates, key=lambda item: (_rank(item, role=role, seed=seed), item)):
            instruction_hash = _normalized_hash(str(goal_map[task_id]["instruction"]))
            if instruction_hash in role_instruction_hashes:
                continue
            selected.append(task_id)
            role_instruction_hashes.add(instruction_hash)
            if len(selected) == count:
                break
        _require(len(selected) == count, f"Phase10-C role population too small: {role}")
        selected_set = set(selected)
        remaining = [task_id for task_id in remaining if task_id not in selected_set]
        used_instruction_hashes.update(role_instruction_hashes)
        hashes = sorted(role_instruction_hashes)
        stage = role.rsplit("_", 1)[-1] if role.startswith("teacher_") else role.removeprefix("opd_")
        roles[role] = {
            "count": count,
            "selection_seed": seed,
            "selection_proxy": role.split("_", 2)[1] if role.startswith("teacher_") else "general_hash",
            "specialist": _specialist_for_role(role),
            "stage": stage,
            "task_ids": selected,
            "goal_indices": [_goal_index(task_id) for task_id in selected],
            "task_order_sha256": sha256_json(selected),
            "normalized_instruction_hashes": hashes,
            "normalized_instruction_sha256": sha256_json(hashes),
            "K": 4,
            "model_turn_cap": 18,
            "env_step_cap": 15,
            "training_role": role.endswith("_train"),
        }
    result = {
        "schema_version": SPLIT_SCHEMA,
        "development_only": True,
        "complete": True,
        "promotion_and_holdout_read": False,
        "producer_git_sha": producer_git_sha,
        "selection_seed": SELECTION_SEED,
        "source_role": "train",
        "source_train_task_count": len(train_task_ids),
        "eligible_after_exposure_count": len(eligible),
        "base_split_content_sha256": base_split_content_sha256,
        "exposure_union_content_sha256": exposure["content_sha256"],
        "teacher_sft_quick_start": {
            "initial_tasks_per_specialist": 16,
            "source": "first_16_from_each_frozen_teacher_train_role",
            "expand_only_after_dev_gain": True,
        },
        "roles": roles,
        "checks": {
            "role_task_overlap_count": 0,
            "role_goal_index_overlap_count": 0,
            "role_normalized_instruction_overlap_count": 0,
            "exposure_task_overlap_count": 0,
            "exposure_normalized_instruction_overlap_count": 0,
            "promotion_and_holdout_read": False,
        },
    }
    result["content_sha256"] = _self_hash(result)
    return validate_phase10c_split(result)


def validate_phase10c_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SPLIT_SCHEMA, "Phase10-C split schema drift")
    _require(value.get("development_only") is True and value.get("complete") is True,
             "Phase10-C split incomplete")
    _require(value.get("promotion_and_holdout_read") is False, "Phase10-C split opened promotion/holdout")
    roles = value.get("roles")
    _require(isinstance(roles, Mapping) and set(roles) == set(ROLE_COUNTS), "Phase10-C split roles drift")
    task_union: set[str] = set()
    index_union: set[int] = set()
    instruction_union: set[str] = set()
    for role, count in ROLE_COUNTS.items():
        item = roles[role]
        task_ids = item.get("task_ids")
        indices = item.get("goal_indices")
        hashes = item.get("normalized_instruction_hashes")
        _require(isinstance(task_ids, list) and len(task_ids) == len(set(task_ids)) == count,
                 f"Phase10-C task count drift: {role}")
        _require(isinstance(indices, list) and len(indices) == len(set(indices)) == count,
                 f"Phase10-C goal count drift: {role}")
        _require(task_ids == [task_id_for_goal_index(index) for index in indices],
                 f"Phase10-C task/goal identity drift: {role}")
        _require(isinstance(hashes, list) and len(hashes) == len(set(hashes)) == count,
                 f"Phase10-C instruction count drift: {role}")
        _require(not (set(task_ids) & task_union), f"Phase10-C task overlap: {role}")
        _require(not (set(indices) & index_union), f"Phase10-C goal overlap: {role}")
        _require(not (set(hashes) & instruction_union), f"Phase10-C instruction overlap: {role}")
        _require(item.get("task_order_sha256") == sha256_json(task_ids), f"Phase10-C task hash drift: {role}")
        _require(item.get("normalized_instruction_sha256") == sha256_json(sorted(hashes)),
                 f"Phase10-C instruction hash drift: {role}")
        _require(item.get("K") == 4 and item.get("model_turn_cap") == 18 and item.get("env_step_cap") == 15,
                 f"Phase10-C budget drift: {role}")
        _require(item.get("specialist") == _specialist_for_role(role), f"Phase10-C specialist drift: {role}")
        _require(item.get("training_role") is role.endswith("_train"),
                 f"Phase10-C training role drift: {role}")
        task_union.update(task_ids)
        index_union.update(indices)
        instruction_union.update(hashes)
    _require(value.get("teacher_sft_quick_start") == {
        "initial_tasks_per_specialist": 16,
        "source": "first_16_from_each_frozen_teacher_train_role",
        "expand_only_after_dev_gain": True,
    }, "Phase10-C quick-start contract drift")
    _require(value.get("checks") == {
        "role_task_overlap_count": 0,
        "role_goal_index_overlap_count": 0,
        "role_normalized_instruction_overlap_count": 0,
        "exposure_task_overlap_count": 0,
        "exposure_normalized_instruction_overlap_count": 0,
        "promotion_and_holdout_read": False,
    }, "Phase10-C split checks drift")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10-C split self-hash drift")
    return value

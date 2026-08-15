"""Fail-closed readiness contracts for M6 Phase10-B multi-Specialist OPD."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .long_horizon_rl.contracts import sha256_file, sha256_json
from .m5_webshop_protocol import normalized_instruction, task_id_for_goal_index
from .m6_phase10_readiness import validate_exposure_union, validate_phase10_split


EXPOSURE_SCHEMA = "m6_phase10b_exposure_union_v1"
SPLIT_SCHEMA = "m6_phase10b_split_lock_v1"
MODEL_MANIFEST_SCHEMA = "m6_phase10b_model_tokenizer_manifest_v1"
LOGIT_PROBE_SCHEMA = "m6_phase10b_logit_alignment_probe_v1"
SELECTION_SEED = 20260850
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MODEL_SPECS = {
    "student": {
        "model_name": "Qwen3.5-4B",
        "path": "/data/share/model/Qwen3.5-4B",
        "role": "unified_student",
    },
    "S_nav": {
        "model_name": "Qwen3.5-9B",
        "path": "/data/share/model/Qwen3.5-9B",
        "role": "search_navigation",
    },
    "S_match": {
        "model_name": "Qwen3.5-35B-A3B",
        "path": "/data/share/model/Qwen3.5-35B-A3B",
        "role": "constraint_option_matching",
    },
    "S_finish": {
        "model_name": "Qwen3.6-35B-A3B-FP8",
        "path": "/data/share/model/Qwen3.6-35B-A3B-FP8",
        "role": "recovery_budget_purchase",
    },
}

ROLE_COUNTS = {
    "specialist_nav_qualification": 24,
    "specialist_match_qualification": 24,
    "specialist_finish_qualification": 24,
    "opd_smoke": 8,
    "opd_train": 40,
    "opd_monitor_a": 64,
    "opd_monitor_b": 64,
    "opd_final_dev": 500,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def _normalized_hash(instruction: str) -> str:
    return sha256_json(normalized_instruction(instruction))


def _goal_index(task_id: str) -> int:
    _require(task_id.startswith("webshop_goal_"), "Phase10-B task identity drift")
    return int(task_id.rsplit("_", 1)[1])


def build_phase10b_exposure_union(
    *,
    goals: Sequence[Mapping[str, Any]],
    phase10_exposure: Mapping[str, Any],
    phase10_split: Mapping[str, Any],
    producer_git_sha: str,
) -> dict[str, Any]:
    """Extend the historical union with every reserved/executed Phase10-A role."""

    old = validate_exposure_union(phase10_exposure)
    split = validate_phase10_split(phase10_split)
    _require(len(goals) == 12087, "Phase10-B canonical goal count drift")
    rows: dict[int, dict[str, Any]] = {}
    for row in old["exposures"]:
        rows[int(row["goal_index"])] = {
            "goal_index": int(row["goal_index"]),
            "task_id": str(row["task_id"]),
            "normalized_instruction_sha256": str(row["normalized_instruction_sha256"]),
            "reasons": list(row["reasons"]),
            "source_labels": list(row["source_labels"]),
        }
    phase10a_task_ids: list[str] = []
    for role, item in split["roles"].items():
        for task_id in item["task_ids"]:
            index = _goal_index(task_id)
            goal = goals[index]
            _require(goal.get("goal_index") == index, "Phase10-B goal identity drift")
            expected = _normalized_hash(str(goal["instruction"]))
            current = rows.setdefault(index, {
                "goal_index": index,
                "task_id": task_id,
                "normalized_instruction_sha256": expected,
                "reasons": [],
                "source_labels": [],
            })
            _require(current["task_id"] == task_id, "Phase10-B task collision")
            _require(current["normalized_instruction_sha256"] == expected, "Phase10-B instruction collision")
            current["reasons"] = sorted(set(current["reasons"]) | {f"Phase10-A reserved/readiness role: {role}"})
            current["source_labels"] = sorted(set(current["source_labels"]) | {f"phase10a_{role}"})
            phase10a_task_ids.append(task_id)
    exposures = [rows[index] for index in sorted(rows)]
    result = {
        "schema_version": EXPOSURE_SCHEMA,
        "development_only": True,
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "promotion_and_holdout_read": False,
        "producer_git_sha": producer_git_sha,
        "phase10_exposure_content_sha256": old["content_sha256"],
        "phase10_split_content_sha256": split["content_sha256"],
        "phase10a_reserved_task_count": len(set(phase10a_task_ids)),
        "phase10a_reserved_task_id_sha256": sha256_json(sorted(set(phase10a_task_ids))),
        "exposures": exposures,
        "exposed_task_count": len(exposures),
        "exposed_task_id_sha256": sha256_json([row["task_id"] for row in exposures]),
    }
    result["content_sha256"] = sha256_json(result)
    return validate_phase10b_exposure_union(result)


def validate_phase10b_exposure_union(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == EXPOSURE_SCHEMA, "Phase10-B exposure schema drift")
    _require(value.get("development_only") is True and value.get("complete") is True, "Phase10-B exposure incomplete")
    _require(value.get("training_performed") is False and value.get("optimizer_steps") == 0, "Phase10-B exposure trained")
    _require(value.get("promotion_and_holdout_read") is False, "Phase10-B exposure opened promotion/holdout")
    rows = value.get("exposures")
    _require(isinstance(rows, list) and rows, "Phase10-B exposure rows missing")
    indices = [row.get("goal_index") for row in rows]
    _require(indices == sorted(set(indices)), "Phase10-B exposure rows are not sorted/unique")
    for row in rows:
        _require(row.get("task_id") == task_id_for_goal_index(row["goal_index"]), "Phase10-B exposure identity drift")
        _require(SHA256_RE.fullmatch(str(row.get("normalized_instruction_sha256"))) is not None,
                 "Phase10-B normalized instruction hash drift")
        _require(row.get("reasons") and row.get("source_labels"), "Phase10-B exposure provenance missing")
    _require(value.get("exposed_task_count") == len(rows), "Phase10-B exposure count drift")
    _require(value.get("exposed_task_id_sha256") == sha256_json([row["task_id"] for row in rows]),
             "Phase10-B exposure aggregate hash drift")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10-B exposure self-hash drift")
    return value


def _rank(task_id: str, *, role: str, seed: int) -> bytes:
    return hashlib.sha256(f"m6-phase10b-opd-v1|{seed}|{role}|{task_id}".encode()).digest()


def _constraint_count(goal: Mapping[str, Any]) -> int:
    attributes = goal.get("instruction_attributes") or goal.get("attributes") or []
    options = goal.get("goal_options") or []
    return len(attributes) + len(options) + int(isinstance(goal.get("price_upper"), (int, float)))


def _qualification_proxy(role: str, goal: Mapping[str, Any]) -> bool:
    options = goal.get("goal_options") or []
    price = goal.get("price_upper")
    constraints = _constraint_count(goal)
    if role == "specialist_nav_qualification":
        return not options and constraints <= 4
    if role == "specialist_match_qualification":
        return bool(options) and constraints >= 3
    if role == "specialist_finish_qualification":
        return bool(options) and isinstance(price, (int, float)) and constraints >= 4
    return True


def _select(candidates: Sequence[str], *, role: str, count: int, seed: int) -> list[str]:
    ranked = sorted(candidates, key=lambda task_id: (_rank(task_id, role=role, seed=seed), task_id))
    _require(len(ranked) >= count, f"Phase10-B role population too small: {role}")
    return ranked[:count]


def build_phase10b_split(
    *,
    goals: Sequence[Mapping[str, Any]],
    train_task_ids: Sequence[str],
    exposure_union: Mapping[str, Any],
    base_split_content_sha256: str,
    producer_git_sha: str,
) -> dict[str, Any]:
    exposure = validate_phase10b_exposure_union(exposure_union)
    _require(len(goals) == 12087, "Phase10-B split canonical goal count drift")
    goal_map = {task_id_for_goal_index(index): goal for index, goal in enumerate(goals)}
    exposed_ids = {row["task_id"] for row in exposure["exposures"]}
    exposed_instructions = {row["normalized_instruction_sha256"] for row in exposure["exposures"]}
    eligible = [
        task_id for task_id in train_task_ids
        if task_id not in exposed_ids
        and _normalized_hash(str(goal_map[task_id]["instruction"])) not in exposed_instructions
    ]
    _require(len(eligible) >= sum(ROLE_COUNTS.values()), "Phase10-B fresh train population is too small")
    remaining = list(eligible)
    roles: dict[str, dict[str, Any]] = {}
    for offset, (role, count) in enumerate(ROLE_COUNTS.items()):
        seed = SELECTION_SEED + offset
        candidates = [task_id for task_id in remaining if _qualification_proxy(role, goal_map[task_id])]
        selected = _select(candidates, role=role, count=count, seed=seed)
        selected_set = set(selected)
        remaining = [task_id for task_id in remaining if task_id not in selected_set]
        instruction_hashes = sorted(_normalized_hash(str(goal_map[task_id]["instruction"])) for task_id in selected)
        roles[role] = {
            "count": count,
            "selection_seed": seed,
            "selection_proxy": role if role.startswith("specialist_") else "general_stratified_hash",
            "task_ids": selected,
            "goal_indices": [_goal_index(task_id) for task_id in selected],
            "task_order_sha256": sha256_json(selected),
            "normalized_instruction_hashes": instruction_hashes,
            "normalized_instruction_sha256": sha256_json(instruction_hashes),
            "K": 4,
            "model_turn_cap": 18,
            "env_step_cap": 15,
            "training_role": role == "opd_train",
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
    result["content_sha256"] = sha256_json(result)
    return validate_phase10b_split(result)


def validate_phase10b_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SPLIT_SCHEMA, "Phase10-B split schema drift")
    _require(value.get("development_only") is True and value.get("complete") is True, "Phase10-B split incomplete")
    _require(value.get("promotion_and_holdout_read") is False, "Phase10-B split opened promotion/holdout")
    roles = value.get("roles")
    _require(isinstance(roles, Mapping) and set(roles) == set(ROLE_COUNTS), "Phase10-B split roles drift")
    task_union: set[str] = set()
    index_union: set[int] = set()
    instruction_union: set[str] = set()
    for role, count in ROLE_COUNTS.items():
        item = roles[role]
        task_ids = item.get("task_ids")
        indices = item.get("goal_indices")
        hashes = item.get("normalized_instruction_hashes")
        _require(isinstance(task_ids, list) and len(task_ids) == len(set(task_ids)) == count,
                 f"Phase10-B task count drift: {role}")
        _require(isinstance(indices, list) and len(indices) == len(set(indices)) == count,
                 f"Phase10-B goal count drift: {role}")
        _require(task_ids == [task_id_for_goal_index(index) for index in indices],
                 f"Phase10-B task/goal identity drift: {role}")
        _require(isinstance(hashes, list) and len(hashes) == len(set(hashes)) == count,
                 f"Phase10-B instruction count drift: {role}")
        _require(not (set(task_ids) & task_union), f"Phase10-B task overlap: {role}")
        _require(not (set(indices) & index_union), f"Phase10-B goal overlap: {role}")
        _require(not (set(hashes) & instruction_union), f"Phase10-B instruction overlap: {role}")
        _require(item.get("task_order_sha256") == sha256_json(task_ids), f"Phase10-B task hash drift: {role}")
        _require(item.get("normalized_instruction_sha256") == sha256_json(sorted(hashes)),
                 f"Phase10-B instruction hash drift: {role}")
        _require(item.get("K") == 4 and item.get("model_turn_cap") == 18 and item.get("env_step_cap") == 15,
                 f"Phase10-B rollout budget drift: {role}")
        task_union.update(task_ids)
        index_union.update(indices)
        instruction_union.update(hashes)
    _require(value.get("checks") == {
        "role_task_overlap_count": 0,
        "role_goal_index_overlap_count": 0,
        "role_normalized_instruction_overlap_count": 0,
        "exposure_task_overlap_count": 0,
        "exposure_normalized_instruction_overlap_count": 0,
        "promotion_and_holdout_read": False,
    }, "Phase10-B split checks drift")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10-B split self-hash drift")
    return value


def build_model_tokenizer_manifest(*, producer_git_sha: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for identity, spec in MODEL_SPECS.items():
        root = Path(spec["path"])
        _require(root.is_dir(), f"Phase10-B model missing: {root}")
        tokenizer = root / "tokenizer.json"
        tokenizer_config = root / "tokenizer_config.json"
        config_path = root / "config.json"
        _require(tokenizer.is_file() and tokenizer_config.is_file() and config_path.is_file(),
                 f"Phase10-B model metadata incomplete: {root}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
        rows.append({
            "identity": identity,
            **spec,
            "tokenizer_json_sha256": sha256_file(tokenizer),
            "tokenizer_config_sha256": sha256_file(tokenizer_config),
            "config_sha256": sha256_file(config_path),
            "model_type": config.get("model_type"),
            "text_model_type": text_config.get("model_type"),
            "vocab_size": text_config.get("vocab_size"),
            "hidden_size": text_config.get("hidden_size"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
        })
    tokenizer_hashes = {row["tokenizer_json_sha256"] for row in rows}
    vocab_sizes = {row["vocab_size"] for row in rows}
    result = {
        "schema_version": MODEL_MANIFEST_SCHEMA,
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": producer_git_sha,
        "student_tokenizer_is_canonical": True,
        "model_count": len(rows),
        "models": rows,
        "checks": {
            "tokenizer_json_hash_count": len(tokenizer_hashes),
            "vocab_size_count": len(vocab_sizes),
            "common_tokenizer_json": len(tokenizer_hashes) == 1,
            "common_vocab_size": len(vocab_sizes) == 1,
            "vocab_size": next(iter(vocab_sizes)) if len(vocab_sizes) == 1 else None,
            "tokenizer_config_difference_allowed_only_after_logit_probe": True,
        },
    }
    result["content_sha256"] = sha256_json(result)
    return validate_model_tokenizer_manifest(result)


def validate_model_tokenizer_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == MODEL_MANIFEST_SCHEMA, "Phase10-B model manifest schema drift")
    _require(value.get("development_only") is True, "Phase10-B model manifest left development mode")
    _require(value.get("training_performed") is False and value.get("optimizer_steps") == 0,
             "Phase10-B model manifest trained")
    models = value.get("models")
    _require(isinstance(models, list) and len(models) == len(MODEL_SPECS), "Phase10-B model inventory drift")
    _require({row.get("identity") for row in models} == set(MODEL_SPECS), "Phase10-B model identities drift")
    checks = value.get("checks")
    _require(isinstance(checks, Mapping), "Phase10-B model checks missing")
    _require(checks.get("common_tokenizer_json") is True and checks.get("tokenizer_json_hash_count") == 1,
             "Phase10-B tokenizer.json is not common")
    _require(checks.get("common_vocab_size") is True and checks.get("vocab_size") == 248320,
             "Phase10-B vocab size drift")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10-B model manifest self-hash drift")
    return value


def teacher_to_student_kl(student_logits: torch.Tensor, specialist_logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Full-vocabulary teacher-to-student KL on an exact shared token prefix."""

    _require(student_logits.shape == specialist_logits.shape and student_logits.ndim >= 2,
             "Phase10-B logit shape mismatch")
    _require(math.isfinite(float(temperature)) and temperature > 0, "Phase10-B temperature drift")
    _require(torch.isfinite(student_logits).all().item() and torch.isfinite(specialist_logits).all().item(),
             "Phase10-B non-finite logits")
    student_log_probs = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    specialist_log_probs = torch.log_softmax(specialist_logits.float() / temperature, dim=-1)
    specialist_probs = specialist_log_probs.exp()
    value = torch.sum(specialist_probs * (specialist_log_probs - student_log_probs), dim=-1).mean()
    _require(torch.isfinite(value).item() and float(value.detach()) >= -1e-6, "Phase10-B KL is invalid")
    return value * (temperature ** 2)


def validate_logit_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == LOGIT_PROBE_SCHEMA, "Phase10-B logit probe schema drift")
    _require(value.get("training_performed") is False and value.get("optimizer_steps") == 0,
             "Phase10-B logit probe trained")
    _require(value.get("student_tokenizer_used_for_all_models") is True, "Phase10-B canonical tokenizer drift")
    rows = value.get("models")
    _require(isinstance(rows, list) and {row.get("identity") for row in rows} == set(MODEL_SPECS),
             "Phase10-B logit probe model inventory drift")
    for row in rows:
        _require(row.get("vocab_size") == 248320, "Phase10-B runtime vocab drift")
        _require(row.get("finite_logits") is True, "Phase10-B runtime logits are non-finite")
        _require(row.get("canonical_action_token_ids_match") is True, "Phase10-B canonical action token drift")
        _require(row.get("prefix_token_ids_match") is True, "Phase10-B prefix token drift")
        _require(row.get("probability_sum_abs_error", 1.0) <= 1e-5, "Phase10-B probability mass drift")
    _require(value.get("all_models_pass") is True, "Phase10-B logit alignment failed")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10-B logit probe self-hash drift")
    return value

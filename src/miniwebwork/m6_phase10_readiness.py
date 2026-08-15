"""Fail-closed Phase10 exposure-union and fresh-role contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .long_horizon_rl.contracts import sha256_file, sha256_json
from .m5_webshop_protocol import normalized_instruction, task_id_for_goal_index

EXPOSURE_SCHEMA = "m6_phase10_exposure_union_v1"
SOURCE_SPEC_SCHEMA = "m6_phase10_exposure_source_spec_v1"
SPLIT_SCHEMA = "m6_phase10_split_lock_v1"
SELECTION_SEED = 20260840

REQUIRED_SOURCE_LABELS = {
    "raw_mini_train_256",
    "sft_train",
    "sft_dev",
    "mini_dev",
    "formal_dev",
    "tuning_dev2",
    "phase1_diagnostics",
    "phase2_diagnostics",
    "phase3_diagnostics",
    "phase4_student_prescan_a",
    "phase4_student_prescan_b",
    "phase4_student_extension",
    "phase4_teacher_probe_9b",
    "phase4_teacher_probe_35b",
    "phase4_online_roster",
    "phase5_credit_and_online",
    "phase6_credit_and_online",
    "phase7_targeted_prescan",
    "phase8_prefix_feasibility",
    "phase9_shared_prefix_smoke",
}

FIXED_SOURCE_TASK_COUNTS = {
    "raw_mini_train_256": 256,
    "mini_dev": 200,
    "formal_dev": 500,
    "tuning_dev2": 128,
    "phase4_student_prescan_a": 64,
    "phase4_student_prescan_b": 64,
    "phase4_student_extension": 64,
    "phase4_teacher_probe_9b": 16,
    "phase4_teacher_probe_35b": 16,
    "phase4_online_roster": 40,
    "phase7_targeted_prescan": 32,
    "phase8_prefix_feasibility": 85,
    "phase9_shared_prefix_smoke": 8,
}
EXPECTED_SFT_UNION_TASK_COUNT = 156

ROLE_COUNTS = {
    "teacher_qualification": 32,
    "correction_train": 64,
    "correction_monitor_a": 64,
    "correction_monitor_b": 64,
    "rl_train": 40,
    "dev3": 500,
}

SCANNABLE_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
EXCLUDED_DIRECTORY_PARTS = {
    "final_adapter",
    "rollout_adapter",
    "server_environment",
    "telemetry",
    "recovery",
}
FORBIDDEN_PATH_PARTS = {"promotion", "holdout"}
TASK_ID_RE = re.compile(r"\bwebshop_goal_([0-9]{5})\b")
GOAL_INDEX_RE = re.compile(
    r"[\"']?goal[_ -]?index[\"']?\s*[:=]\s*[\"']?([0-9]{1,5})",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return sha256_json(payload)


def normalized_instruction_sha256(instruction: str) -> str:
    return sha256_json(normalized_instruction(instruction))


def validate_source_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SOURCE_SPEC_SCHEMA, "Phase10 source-spec schema drift")
    _require(value.get("development_only") is True, "Phase10 source spec left development-only mode")
    _require(value.get("promotion_and_holdout_read") is False, "Phase10 source spec opened promotion/holdout")
    _require(
        value.get("required_source_labels") == sorted(REQUIRED_SOURCE_LABELS),
        "Phase10 required source labels drift",
    )
    sources = value.get("sources")
    _require(isinstance(sources, list) and sources, "Phase10 source spec is empty")
    labels: list[str] = []
    for source in sources:
        _require(isinstance(source, Mapping), "Phase10 exposure source is malformed")
        label = source.get("label")
        _require(isinstance(label, str) and label, "Phase10 exposure source label missing")
        labels.append(label)
        _require(
            isinstance(source.get("reason"), str) and source["reason"],
            f"Phase10 exposure source reason missing: {label}",
        )
        _require(
            isinstance(source.get("expected_task_count"), int)
            and not isinstance(source["expected_task_count"], bool)
            and source["expected_task_count"] > 0,
            f"Phase10 expected task count missing: {label}",
        )
        if label in FIXED_SOURCE_TASK_COUNTS:
            _require(
                source["expected_task_count"] == FIXED_SOURCE_TASK_COUNTS[label],
                f"Phase10 fixed source count drift: {label}",
            )
        paths = source.get("paths")
        _require(
            isinstance(paths, list) and paths and all(isinstance(path, str) and path for path in paths),
            f"Phase10 source paths missing: {label}",
        )
        extractor = source.get("extractor", "text_task_identity")
        _require(
            extractor in {"text_task_identity", "split_role"},
            f"Phase10 source extractor drift: {label}",
        )
        if extractor == "split_role":
            _require(
                source.get("role") in {"mini_train", "mini_dev", "formal_dev"},
                f"Phase10 split-role extractor is not an allowed historical role: {label}",
            )
        _require(
            SHA256_RE.fullmatch(str(source.get("expected_source_content_sha256"))) is not None,
            f"Phase10 expected source hash missing: {label}",
        )
    _require(len(labels) == len(set(labels)), "Phase10 source labels are not unique")
    _require(set(labels) == REQUIRED_SOURCE_LABELS, "Phase10 exposure source inventory is incomplete")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10 source-spec self-hash drift")
    return value


def _safe_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved = candidate.resolve()
    lowered = {part.casefold() for part in resolved.parts}
    _require(not (lowered & FORBIDDEN_PATH_PARTS), "Phase10 exposure source touches promotion/holdout")
    return resolved


def _expand_source_files(paths: Sequence[str], *, base_dir: Path) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = _safe_path(raw_path, base_dir=base_dir)
        _require(path.exists(), f"Phase10 exposure source is missing: {path}")
        if path.is_file():
            _require(path.name != "goals.json", "Phase10 goals.json cannot be exposure evidence")
            _require(path.suffix.casefold() in SCANNABLE_SUFFIXES, f"Phase10 source type is unsupported: {path}")
            files.append(path)
            continue
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file() or candidate.suffix.casefold() not in SCANNABLE_SUFFIXES:
                continue
            relative_parts = {part.casefold() for part in candidate.relative_to(path).parts}
            if relative_parts & EXCLUDED_DIRECTORY_PARTS:
                continue
            _require(not (relative_parts & FORBIDDEN_PATH_PARTS), "Phase10 source directory contains promotion/holdout")
            if candidate.name == "goals.json":
                continue
            files.append(candidate.resolve())
    unique = sorted(set(files))
    _require(unique, "Phase10 exposure source expansion is empty")
    return unique


def _task_indices_from_file(path: Path) -> set[int]:
    _require(path.stat().st_size <= 512 * 1024 * 1024, f"Phase10 exposure source is too large: {path}")
    text = path.read_text(encoding="utf-8", errors="strict")
    return (
        {int(match.group(1)) for match in TASK_ID_RE.finditer(text)}
        | {int(match.group(1)) for match in GOAL_INDEX_RE.finditer(text)}
    )


def _file_inventory(files: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]


def _source_task_indices(source: Mapping[str, Any], files: Sequence[Path]) -> set[int]:
    extractor = source.get("extractor", "text_task_identity")
    if extractor == "text_task_identity":
        return set().union(*(_task_indices_from_file(path) for path in files))
    _require(extractor == "split_role" and len(files) == 1, "Phase10 split-role source must bind one file")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    role = str(source["role"])
    roles = payload.get("roles")
    _require(isinstance(roles, Mapping) and isinstance(roles.get(role), Mapping), f"Phase10 split role missing: {role}")
    item = roles[role]
    indices = item.get("goal_indices")
    task_ids = item.get("task_ids")
    _require(isinstance(indices, list) and indices == sorted(set(indices)), f"Phase10 split role indices drift: {role}")
    _require(
        task_ids == [task_id_for_goal_index(index) for index in indices],
        f"Phase10 split role task identities drift: {role}",
    )
    return set(indices)


def freeze_source_spec(
    *,
    sources: Sequence[Mapping[str, Any]],
    source_base_dir: Path,
) -> dict[str, Any]:
    """Bind a complete human-authored source list to immutable file hashes."""

    frozen_sources: list[dict[str, Any]] = []
    for raw_source in sources:
        source = dict(raw_source)
        paths = source.get("paths")
        _require(
            isinstance(paths, list) and paths and all(isinstance(path, str) and path for path in paths),
            "Phase10 source draft paths missing",
        )
        files = _expand_source_files(paths, base_dir=source_base_dir)
        source["paths"] = [str(_safe_path(path, base_dir=source_base_dir)) for path in paths]
        source["expected_source_content_sha256"] = sha256_json(_file_inventory(files))
        frozen_sources.append(source)
    result = {
        "schema_version": SOURCE_SPEC_SCHEMA,
        "development_only": True,
        "promotion_and_holdout_read": False,
        "required_source_labels": sorted(REQUIRED_SOURCE_LABELS),
        "sources": sorted(frozen_sources, key=lambda item: str(item.get("label", ""))),
    }
    result["content_sha256"] = sha256_json(result)
    return validate_source_spec(result)


def build_exposure_union(
    *,
    goals: Sequence[Mapping[str, Any]],
    source_spec: Mapping[str, Any],
    source_base_dir: Path,
    producer_git_sha: str,
) -> dict[str, Any]:
    spec = validate_source_spec(source_spec)
    _require(len(goals) == 12087, "Phase10 canonical goal count drift")
    goal_map: dict[int, Mapping[str, Any]] = {}
    for index, goal in enumerate(goals):
        _require(isinstance(goal, Mapping) and goal.get("goal_index") == index, f"Phase10 goal row drift: {index}")
        instruction = goal.get("instruction")
        _require(isinstance(instruction, str) and instruction.strip(), f"Phase10 goal instruction missing: {index}")
        goal_map[index] = goal

    rows: dict[int, dict[str, set[str]]] = {}
    indices_by_label: dict[str, set[int]] = {}
    source_bindings: list[dict[str, Any]] = []
    for source in sorted(spec["sources"], key=lambda item: str(item["label"])):
        files = _expand_source_files(source["paths"], base_dir=source_base_dir)
        matched = _source_task_indices(source, files)
        file_inventory = _file_inventory(files)
        _require(all(index in goal_map for index in matched), f"Phase10 source task is outside goals: {source['label']}")
        source_content_sha256 = sha256_json(file_inventory)
        _require(
            source_content_sha256 == source["expected_source_content_sha256"],
            f"Phase10 exposure source hash drift: {source['label']}",
        )
        expected_count = int(source["expected_task_count"])
        _require(
            len(matched) == expected_count,
            f"Phase10 exposure count drift for {source['label']}: expected {expected_count}, observed {len(matched)}",
        )
        indices_by_label[str(source["label"])] = set(matched)
        for index in matched:
            row = rows.setdefault(index, {"reasons": set(), "source_labels": set()})
            row["reasons"].add(str(source["reason"]))
            row["source_labels"].add(str(source["label"]))
        source_bindings.append(
            {
                "label": source["label"],
                "reason": source["reason"],
                "expected_task_count": expected_count,
                "matched_task_count": len(matched),
                "task_id_sha256": sha256_json([task_id_for_goal_index(index) for index in sorted(matched)]),
                "file_count": len(file_inventory),
                "files": file_inventory,
                "source_content_sha256": source_content_sha256,
            }
        )

    _require(
        len(indices_by_label["sft_train"] | indices_by_label["sft_dev"]) == EXPECTED_SFT_UNION_TASK_COUNT,
        "Phase10 SFT train/dev union count drift",
    )

    indices = sorted(rows)
    result = {
        "schema_version": EXPOSURE_SCHEMA,
        "development_only": True,
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "promotion_and_holdout_read": False,
        "producer_git_sha": producer_git_sha,
        "source_spec_content_sha256": spec["content_sha256"],
        "required_source_labels": sorted(REQUIRED_SOURCE_LABELS),
        "source_bindings": source_bindings,
        "source_count": len(source_bindings),
        "exposures": [
            {
                "goal_index": index,
                "task_id": task_id_for_goal_index(index),
                "normalized_instruction_sha256": normalized_instruction_sha256(str(goal_map[index]["instruction"])),
                "reasons": sorted(rows[index]["reasons"]),
                "source_labels": sorted(rows[index]["source_labels"]),
            }
            for index in indices
        ],
        "exposed_task_count": len(indices),
        "exposed_task_id_sha256": sha256_json([task_id_for_goal_index(index) for index in indices]),
    }
    result["content_sha256"] = sha256_json(result)
    return validate_exposure_union(result)


def validate_exposure_union(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == EXPOSURE_SCHEMA, "Phase10 exposure schema drift")
    _require(value.get("complete") is True, "Phase10 exposure union is incomplete")
    _require(value.get("development_only") is True, "Phase10 exposure union left development-only mode")
    _require(value.get("training_performed") is False and value.get("optimizer_steps") == 0, "Phase10 exposure union trained")
    _require(value.get("promotion_and_holdout_read") is False, "Phase10 exposure union opened promotion/holdout")
    _require(value.get("required_source_labels") == sorted(REQUIRED_SOURCE_LABELS), "Phase10 exposure labels drift")
    bindings = value.get("source_bindings")
    _require(isinstance(bindings, list) and value.get("source_count") == len(bindings), "Phase10 source binding drift")
    _require({item.get("label") for item in bindings} == REQUIRED_SOURCE_LABELS, "Phase10 source binding inventory drift")
    for binding in bindings:
        _require(binding.get("matched_task_count") == binding.get("expected_task_count"), "Phase10 source count mismatch")
        _require(SHA256_RE.fullmatch(str(binding.get("source_content_sha256"))) is not None, "Phase10 source hash drift")
    exposures = value.get("exposures")
    _require(isinstance(exposures, list), "Phase10 exposure rows missing")
    indices = [row.get("goal_index") for row in exposures]
    _require(indices == sorted(set(indices)), "Phase10 exposure rows are not unique and sorted")
    for row in exposures:
        index = row["goal_index"]
        _require(row.get("task_id") == task_id_for_goal_index(index), "Phase10 exposure task identity drift")
        _require(SHA256_RE.fullmatch(str(row.get("normalized_instruction_sha256"))) is not None, "Phase10 normalized instruction hash drift")
        _require(row.get("reasons") and row.get("source_labels"), "Phase10 exposure provenance missing")
    _require(value.get("exposed_task_count") == len(exposures), "Phase10 exposure task count drift")
    _require(
        value.get("exposed_task_id_sha256") == sha256_json([row["task_id"] for row in exposures]),
        "Phase10 exposed task hash drift",
    )
    _require(value.get("content_sha256") == _self_hash(value), "Phase10 exposure self-hash drift")
    return value


def _constraint_count(goal: Mapping[str, Any]) -> int:
    attributes = goal.get("instruction_attributes") or goal.get("attributes") or []
    options = goal.get("goal_options") or []
    return len(attributes) + len(options) + int(isinstance(goal.get("price_upper"), (int, float)))


def _length_bucket(instruction: str) -> str:
    tokens = normalized_instruction(instruction).split()
    if len(tokens) <= 20:
        return "length_00_20"
    if len(tokens) <= 40:
        return "length_21_40"
    return "length_41_plus"


def _bucket(goal: Mapping[str, Any]) -> str:
    category = str(goal.get("category") or "unknown").strip().casefold()
    count = _constraint_count(goal)
    constraint_bucket = f"constraints_{min(count, 6)}{'_plus' if count >= 6 else ''}"
    return f"{category}|{constraint_bucket}|{_length_bucket(str(goal['instruction']))}"


def _rank(task_id: str, *, role: str, seed: int) -> bytes:
    return hashlib.sha256(f"m6-phase10-split-v1|{seed}|{role}|{task_id}".encode()).digest()


def _proportional_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    _require(population >= total > 0, "Phase10 split population is too small")
    exact = {key: total * count / population for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    _require(sum(quotas.values()) == total, "Phase10 split quota drift")
    return quotas


def _select_role(
    candidates: Sequence[str],
    *,
    goal_map: Mapping[str, Mapping[str, Any]],
    role: str,
    count: int,
    seed: int,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    by_bucket: dict[str, list[str]] = {}
    for task_id in candidates:
        by_bucket.setdefault(_bucket(goal_map[task_id]), []).append(task_id)
    population_counts = {key: len(values) for key, values in by_bucket.items()}
    quotas = _proportional_quotas(population_counts, count)
    selected: list[str] = []
    for bucket, task_ids in by_bucket.items():
        ranked = sorted(task_ids, key=lambda task_id: (_rank(task_id, role=role, seed=seed), task_id))
        selected.extend(ranked[: quotas[bucket]])
    selected.sort(key=lambda task_id: (_rank(task_id, role=f"{role}-order", seed=seed), task_id))
    _require(len(selected) == count and len(set(selected)) == count, f"Phase10 role selection drift: {role}")
    return selected, population_counts, dict(Counter(_bucket(goal_map[task_id]) for task_id in selected))


def build_phase10_split(
    *,
    goals: Sequence[Mapping[str, Any]],
    train_task_ids: Sequence[str],
    exposure_union: Mapping[str, Any],
    base_split_content_sha256: str,
    producer_git_sha: str,
) -> dict[str, Any]:
    exposure = validate_exposure_union(exposure_union)
    _require(len(goals) == 12087, "Phase10 split canonical goal count drift")
    goal_map = {task_id_for_goal_index(index): goal for index, goal in enumerate(goals)}
    _require(all(task_id in goal_map for task_id in train_task_ids), "Phase10 base train task is missing from goals")
    _require(len(train_task_ids) == len(set(train_task_ids)), "Phase10 base train task list contains duplicates")
    exposed_ids = {row["task_id"] for row in exposure["exposures"]}
    exposed_normalized = {row["normalized_instruction_sha256"] for row in exposure["exposures"]}
    eligible = [
        task_id
        for task_id in train_task_ids
        if task_id not in exposed_ids
        and normalized_instruction_sha256(str(goal_map[task_id]["instruction"])) not in exposed_normalized
    ]
    _require(len(eligible) >= sum(ROLE_COUNTS.values()), "Phase10 fresh train population is too small")

    remaining = list(eligible)
    role_payloads: dict[str, dict[str, Any]] = {}
    selected_union: set[str] = set()
    selected_normalized: set[str] = set()
    for offset, (role, count) in enumerate(ROLE_COUNTS.items()):
        role_seed = SELECTION_SEED + offset
        selected, population_counts, selected_counts = _select_role(
            remaining,
            goal_map=goal_map,
            role=role,
            count=count,
            seed=role_seed,
        )
        current = set(selected)
        current_normalized = {
            normalized_instruction_sha256(str(goal_map[task_id]["instruction"])) for task_id in selected
        }
        _require(not (current & selected_union), f"Phase10 task overlap: {role}")
        _require(not (current_normalized & selected_normalized), f"Phase10 normalized instruction overlap: {role}")
        _require(not (current & exposed_ids), f"Phase10 exposed task entered role: {role}")
        _require(not (current_normalized & exposed_normalized), f"Phase10 exposed instruction entered role: {role}")
        selected_union.update(current)
        selected_normalized.update(current_normalized)
        remaining = [task_id for task_id in remaining if task_id not in current]
        role_payloads[role] = {
            "count": count,
            "selection_seed": role_seed,
            "task_ids": selected,
            "goal_indices": [int(task_id.rsplit("_", 1)[1]) for task_id in selected],
            "task_order_sha256": sha256_json(selected),
            "normalized_instruction_sha256": sha256_json(sorted(current_normalized)),
            "normalized_instruction_hashes": sorted(current_normalized),
            "population_bucket_counts": dict(sorted(population_counts.items())),
            "selected_bucket_counts": dict(sorted(selected_counts.items())),
            "K": 4 if role == "dev3" else None,
            "training_role": role in {"correction_train", "rl_train"},
        }

    result = {
        "schema_version": SPLIT_SCHEMA,
        "development_only": True,
        "complete": True,
        "promotion_and_holdout_read": False,
        "producer_git_sha": producer_git_sha,
        "selection_contract": "category_x_constraint_count_x_normalized_instruction_length_proportional_hash_v1",
        "selection_seed": SELECTION_SEED,
        "source_role": "train",
        "source_train_task_count": len(train_task_ids),
        "eligible_after_exposure_count": len(eligible),
        "base_split_content_sha256": base_split_content_sha256,
        "exposure_union_content_sha256": exposure["content_sha256"],
        "roles": role_payloads,
        "checks": {
            "role_task_overlap_count": 0,
            "role_goal_index_overlap_count": 0,
            "role_normalized_instruction_overlap_count": 0,
            "exposure_task_overlap_count": 0,
            "exposure_goal_index_overlap_count": 0,
            "exposure_normalized_instruction_overlap_count": 0,
            "promotion_and_holdout_read": False,
        },
    }
    result["content_sha256"] = sha256_json(result)
    return validate_phase10_split(result)


def validate_phase10_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    _require(value.get("schema_version") == SPLIT_SCHEMA, "Phase10 split schema drift")
    _require(value.get("development_only") is True and value.get("complete") is True, "Phase10 split incomplete")
    _require(value.get("promotion_and_holdout_read") is False, "Phase10 split opened promotion/holdout")
    roles = value.get("roles")
    _require(isinstance(roles, Mapping) and set(roles) == set(ROLE_COUNTS), "Phase10 split role set drift")
    task_union: set[str] = set()
    index_union: set[int] = set()
    normalized_union: set[str] = set()
    for role, count in ROLE_COUNTS.items():
        item = roles[role]
        task_ids = item.get("task_ids")
        indices = item.get("goal_indices")
        _require(isinstance(task_ids, list) and len(task_ids) == len(set(task_ids)) == count, f"Phase10 task count drift: {role}")
        _require(isinstance(indices, list) and len(indices) == len(set(indices)) == count, f"Phase10 goal count drift: {role}")
        _require(not (set(task_ids) & task_union), f"Phase10 task overlap: {role}")
        _require(not (set(indices) & index_union), f"Phase10 goal overlap: {role}")
        _require(
            task_ids == [task_id_for_goal_index(index) for index in indices],
            f"Phase10 task/goal identity drift: {role}",
        )
        task_union.update(task_ids)
        index_union.update(indices)
        normalized_hashes = item.get("normalized_instruction_hashes")
        _require(
            isinstance(normalized_hashes, list)
            and len(normalized_hashes) == len(set(normalized_hashes)) == count
            and all(SHA256_RE.fullmatch(str(value)) is not None for value in normalized_hashes),
            f"Phase10 role instruction hashes drift: {role}",
        )
        _require(not (set(normalized_hashes) & normalized_union), f"Phase10 instruction overlap: {role}")
        normalized_union.update(normalized_hashes)
        _require(
            item.get("normalized_instruction_sha256") == sha256_json(sorted(normalized_hashes)),
            f"Phase10 role instruction aggregate hash drift: {role}",
        )
        _require(item.get("task_order_sha256") == sha256_json(task_ids), f"Phase10 task order hash drift: {role}")
    _require(value.get("checks") == {
        "role_task_overlap_count": 0,
        "role_goal_index_overlap_count": 0,
        "role_normalized_instruction_overlap_count": 0,
        "exposure_task_overlap_count": 0,
        "exposure_goal_index_overlap_count": 0,
        "exposure_normalized_instruction_overlap_count": 0,
        "promotion_and_holdout_read": False,
    }, "Phase10 split checks drift")
    _require(value.get("content_sha256") == _self_hash(value), "Phase10 split self-hash drift")
    return value

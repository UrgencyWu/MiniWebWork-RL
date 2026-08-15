from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.m6_phase10_readiness import (
    FIXED_SOURCE_TASK_COUNTS,
    REQUIRED_SOURCE_LABELS,
    ROLE_COUNTS,
    build_exposure_union,
    build_phase10_split,
    freeze_source_spec,
    validate_exposure_union,
    validate_phase10_split,
)


def _task_id(index: int) -> str:
    return f"webshop_goal_{index:05d}"


def _goals() -> list[dict]:
    return [
        {
            "goal_index": index,
            "instruction": f"find item {index} with option {index % 7} under budget",
            "category": ("fashion", "grocery", "electronics", "garden")[index % 4],
            "attributes": [f"attribute-{slot}" for slot in range(index % 4)],
            "goal_options": [f"option-{index % 7}"] if index % 3 == 0 else [],
            "price_upper": 50.0 if index % 2 == 0 else None,
        }
        for index in range(12087)
    ]


def _write_task_file(path: Path, indices: list[int]) -> None:
    path.write_text(json.dumps({"task_ids": [_task_id(index) for index in indices]}), encoding="utf-8")


def _source_ranges(tmp_path: Path) -> tuple[list[dict], set[int]]:
    cursor = 0
    selected: set[int] = set()
    sources: list[dict] = []
    variable_counts = {
        "sft_train": 128,
        "sft_dev": 28,
        "phase1_diagnostics": 5,
        "phase2_diagnostics": 5,
        "phase3_diagnostics": 5,
    }
    for label in sorted(REQUIRED_SOURCE_LABELS):
        count = FIXED_SOURCE_TASK_COUNTS[label] if label in FIXED_SOURCE_TASK_COUNTS else variable_counts[label]
        # Both teacher probes intentionally saw the same frozen 16-task roster.
        if label == "phase4_teacher_probe_35b":
            indices = list(range(cursor - 16, cursor))
        else:
            indices = list(range(cursor, cursor + count))
            cursor += count
        selected.update(indices)
        path = tmp_path / f"{label}.json"
        _write_task_file(path, indices)
        sources.append(
            {
                "label": label,
                "reason": f"historical exposure from {label}",
                "expected_task_count": count,
                "paths": [path.name],
            }
        )
    return sources, selected


def _build_fixture(tmp_path: Path) -> tuple[list[dict], dict, dict, set[int]]:
    goals = _goals()
    sources, selected = _source_ranges(tmp_path)
    spec = freeze_source_spec(sources=sources, source_base_dir=tmp_path)
    exposure = build_exposure_union(
        goals=goals,
        source_spec=spec,
        source_base_dir=tmp_path,
        producer_git_sha="a" * 40,
    )
    return goals, spec, exposure, selected


def test_phase10_exposure_union_is_complete_hashed_and_normalized(tmp_path: Path):
    _, spec, exposure, selected = _build_fixture(tmp_path)
    assert set(spec["required_source_labels"]) == REQUIRED_SOURCE_LABELS
    assert exposure["complete"] is True
    assert exposure["promotion_and_holdout_read"] is False
    assert exposure["exposed_task_count"] == len(selected)
    assert {row["goal_index"] for row in exposure["exposures"]} == selected
    assert len(exposure["source_bindings"]) == len(REQUIRED_SOURCE_LABELS)
    validate_exposure_union(exposure)


def test_phase10_exposure_union_fails_closed_on_source_hash_drift(tmp_path: Path):
    goals, spec, _, _ = _build_fixture(tmp_path)
    source = next(item for item in spec["sources"] if item["label"] == "phase9_shared_prefix_smoke")
    path = tmp_path / source["paths"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["note"] = "hash drift without changing task count"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash drift"):
        build_exposure_union(
            goals=goals,
            source_spec=spec,
            source_base_dir=tmp_path,
            producer_git_sha="a" * 40,
        )


def test_phase10_source_inventory_fails_closed_when_required_label_is_missing(tmp_path: Path):
    sources, _ = _source_ranges(tmp_path)
    with pytest.raises(ValueError, match="inventory is incomplete"):
        freeze_source_spec(sources=sources[:-1], source_base_dir=tmp_path)


def test_phase10_six_roles_are_deterministic_fresh_and_pairwise_disjoint(tmp_path: Path):
    goals, _, exposure, _ = _build_fixture(tmp_path)
    train_task_ids = [_task_id(index) for index in range(1000, 5000)]
    first = build_phase10_split(
        goals=goals,
        train_task_ids=train_task_ids,
        exposure_union=exposure,
        base_split_content_sha256="b" * 64,
        producer_git_sha="a" * 40,
    )
    repeated = build_phase10_split(
        goals=goals,
        train_task_ids=train_task_ids,
        exposure_union=exposure,
        base_split_content_sha256="b" * 64,
        producer_git_sha="a" * 40,
    )
    assert first == repeated
    assert {role: item["count"] for role, item in first["roles"].items()} == ROLE_COUNTS
    exposed = {row["task_id"] for row in exposure["exposures"]}
    role_sets = [set(item["task_ids"]) for item in first["roles"].values()]
    assert not (set().union(*role_sets) & exposed)
    assert sum(len(items) for items in role_sets) == len(set().union(*role_sets))
    validate_phase10_split(first)


def test_phase10_split_validator_rejects_cross_role_instruction_reuse(tmp_path: Path):
    goals, _, exposure, _ = _build_fixture(tmp_path)
    report = build_phase10_split(
        goals=goals,
        train_task_ids=[_task_id(index) for index in range(1000, 5000)],
        exposure_union=exposure,
        base_split_content_sha256="b" * 64,
        producer_git_sha="a" * 40,
    )
    changed = copy.deepcopy(report)
    left = changed["roles"]["teacher_qualification"]["normalized_instruction_hashes"][0]
    changed["roles"]["correction_train"]["normalized_instruction_hashes"][0] = left
    changed["roles"]["correction_train"]["normalized_instruction_hashes"].sort()
    changed["roles"]["correction_train"]["normalized_instruction_sha256"] = sha256_json(
        changed["roles"]["correction_train"]["normalized_instruction_hashes"]
    )
    changed["content_sha256"] = sha256_json({key: value for key, value in changed.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="instruction overlap"):
        validate_phase10_split(changed)

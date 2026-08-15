from __future__ import annotations

from collections import OrderedDict

import pytest

import miniwebwork.m6_phase10c_data as data
from miniwebwork.long_horizon_rl.contracts import sha256_json


def _task_id(index: int) -> str:
    return f"webshop_goal_{index:05d}"


def _goals() -> list[dict]:
    goals = []
    for index in range(12087):
        kind = index % 5
        if kind == 0:
            options, price, attributes = [], None, ["a", "b"]
        elif kind in {1, 2}:
            options, price, attributes = ["size"], None, ["a", "b", "c"]
        else:
            options, price, attributes = ["size", "color"], 50.0, ["a", "b"]
        goals.append({
            "goal_index": index,
            "instruction": f"find public item {index} with distinct normalized instruction",
            "category": ("fashion", "grocery", "electronics", "garden")[index % 4],
            "attributes": attributes,
            "goal_options": options,
            "price_upper": price,
        })
    return goals


def _old_payloads(goals: list[dict]) -> tuple[dict, dict]:
    exposures = [{
        "goal_index": index,
        "task_id": _task_id(index),
        "normalized_instruction_sha256": sha256_json(data.normalized_instruction(goals[index]["instruction"])),
        "reasons": ["historical"],
        "source_labels": ["historical"],
    } for index in range(10)]
    old = {"exposures": exposures, "content_sha256": "a" * 64}
    cursor = 100
    roles = {}
    for role, count in {
        "specialist_nav_qualification": 24,
        "specialist_match_qualification": 24,
        "specialist_finish_qualification": 24,
        "opd_smoke": 8,
        "opd_train": 40,
        "opd_monitor_a": 64,
        "opd_monitor_b": 64,
        "opd_final_dev": 500,
    }.items():
        roles[role] = {"task_ids": [_task_id(index) for index in range(cursor, cursor + count)]}
        cursor += count
    return old, {"roles": roles, "content_sha256": "b" * 64}


def test_phase10c_reserves_disjoint_specialist_and_opd_roles(monkeypatch: pytest.MonkeyPatch):
    goals = _goals()
    old, phase10b = _old_payloads(goals)
    monkeypatch.setattr(data, "validate_phase10b_exposure_union", lambda value: value)
    monkeypatch.setattr(data, "validate_phase10b_split", lambda value: value)
    exposure = data.build_phase10c_exposure_union(
        goals=goals,
        phase10b_exposure=old,
        phase10b_split=phase10b,
        producer_git_sha="c" * 40,
    )
    phase10b_ids = {task_id for item in phase10b["roles"].values() for task_id in item["task_ids"]}
    assert exposure["phase10b_reserved_task_count"] == 748
    assert phase10b_ids <= {row["task_id"] for row in exposure["exposures"]}

    split = data.build_phase10c_split(
        goals=goals,
        train_task_ids=[_task_id(index) for index in range(1000, 12000)],
        exposure_union=exposure,
        base_split_content_sha256="d" * 64,
        producer_git_sha="c" * 40,
    )
    assert {role: item["count"] for role, item in split["roles"].items()} == data.ROLE_COUNTS
    role_sets = [set(item["task_ids"]) for item in split["roles"].values()]
    assert sum(map(len, role_sets)) == len(set().union(*role_sets)) == 1252
    assert not (phase10b_ids & set().union(*role_sets))
    assert split["teacher_sft_quick_start"]["initial_tasks_per_specialist"] == 16
    data.validate_phase10c_split(split)


def test_specialist_proxies_and_training_roles_are_frozen():
    nav = {"goal_options": [], "attributes": ["a", "b"], "price_upper": None}
    match = {"goal_options": ["size"], "attributes": ["a", "b"], "price_upper": None}
    finish = {"goal_options": ["size"], "attributes": ["a", "b"], "price_upper": 50.0}
    assert data._proxy("teacher_nav_train", nav)
    assert data._proxy("teacher_match_train", match)
    assert data._proxy("teacher_finish_train", finish)
    assert not data._proxy("teacher_finish_train", match)
    assert data._specialist_for_role("teacher_match_dev") == "S_match_sft"
    assert data._specialist_for_role("opd_train") is None


def test_narrow_finish_roles_are_selected_before_broader_match_roles():
    assert isinstance(data.ROLE_COUNTS, dict)
    keys = list(data.ROLE_COUNTS)
    assert keys.index("teacher_finish_train") < keys.index("teacher_match_train")
    assert keys.index("teacher_match_train") < keys.index("teacher_nav_train")


def test_split_validator_rejects_role_overlap(monkeypatch: pytest.MonkeyPatch):
    goals = _goals()
    old, phase10b = _old_payloads(goals)
    monkeypatch.setattr(data, "validate_phase10b_exposure_union", lambda value: value)
    monkeypatch.setattr(data, "validate_phase10b_split", lambda value: value)
    exposure = data.build_phase10c_exposure_union(
        goals=goals,
        phase10b_exposure=old,
        phase10b_split=phase10b,
        producer_git_sha="c" * 40,
    )
    split = data.build_phase10c_split(
        goals=goals,
        train_task_ids=[_task_id(index) for index in range(1000, 12000)],
        exposure_union=exposure,
        base_split_content_sha256="d" * 64,
        producer_git_sha="c" * 40,
    )
    roles = OrderedDict((role, dict(item)) for role, item in split["roles"].items())
    source = roles["teacher_match_train"]
    target = roles["teacher_match_dev"]
    target["task_ids"] = list(source["task_ids"][:32])
    target["goal_indices"] = list(source["goal_indices"][:32])
    target["task_order_sha256"] = sha256_json(target["task_ids"])
    target["normalized_instruction_hashes"] = sorted(source["normalized_instruction_hashes"][:32])
    target["normalized_instruction_sha256"] = sha256_json(target["normalized_instruction_hashes"])
    changed = dict(split, roles=roles)
    changed["content_sha256"] = data._self_hash(changed)
    with pytest.raises(ValueError, match="task overlap"):
        data.validate_phase10c_split(changed)

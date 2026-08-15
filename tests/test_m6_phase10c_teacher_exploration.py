from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_teacher_exploration_roster_skips_prior_smoke_tasks_and_interleaves_roles():
    builder = _module("m6_phase10c_teacher_exploration_roster", "scripts/m6_phase10c_build_teacher_exploration_roster.py")
    roles = {}
    cursor = 0
    for role in builder.ROLE_COUNTS:
        tasks = [f"webshop_goal_{cursor + index:05d}" for index in range(128)]
        cursor += 128
        roles[role] = {"task_ids": tasks}
    all_tasks = {task for value in roles.values() for task in value["task_ids"]}
    roster = builder.build_roster(
        phase10c_split={"roles": roles, "content_sha256": "a" * 64},
        base_split={"roles": {"train": {"task_ids": sorted(all_tasks)}}, "content_sha256": "b" * 64},
        protocol_sha256="c" * 64,
        producer_git_sha="d" * 40,
    )
    assert roster["task_count"] == 32
    assert roster["source_role_counts"] == {"teacher_nav_train": 11, "teacher_match_train": 11, "teacher_finish_train": 10}
    assert all(task not in roster["task_ids"] for value in roles.values() for task in value["task_ids"][:16])
    assert roster["training_performed"] is False and roster["optimizer_steps"] == 0


def test_teacher_exploration_contract_is_raw_35b_full_horizon():
    pytest.importorskip("playwright")
    collector = _module("m6_collect_phase10c_teacher_exploration", "scripts/m6_collect_policy_success.py")
    args = argparse.Namespace(
        mode="phase10c_teacher_exploration",
        role="train",
        task_roster=Path("roster.json"),
        k=4,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
        shared_prefix_manifest=None,
        state_correction_manifest=None,
        base_model=Path("/data/share/model/Qwen3.5-35B-A3B"),
        adapter=None,
        tensor_parallel_size=2,
    )
    collector.validate_phase10c_teacher_exploration_contract(args)


def test_teacher_exploration_wrapper_is_two_gpu_inference_only():
    wrapper = (ROOT / "scripts/run_m6_phase10c_teacher_exploration_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2" in wrapper
    assert "--mode phase10c_teacher_exploration --role train --k 4 --seed 20260862" in wrapper
    assert "--max-model-turns 18 --max-environment-steps 15" in wrapper
    assert "--base-model /data/share/model/Qwen3.5-35B-A3B" in wrapper
    assert "optimizer" not in wrapper


def test_teacher_exploration_scale_roster_is_disjoint_and_finish_heavy():
    builder = _module(
        "m6_phase10c_teacher_exploration_scale_roster",
        "scripts/m6_phase10c_build_teacher_exploration_scale_roster.py",
    )
    roles = {}
    cursor = 0
    for role in builder.ROLE_COUNTS:
        tasks = [f"webshop_goal_{cursor + index:05d}" for index in range(128)]
        cursor += 128
        roles[role] = {"task_ids": tasks}
    all_tasks = {task for value in roles.values() for task in value["task_ids"]}
    prior_tasks = (
        roles["teacher_nav_train"]["task_ids"][16:27]
        + roles["teacher_match_train"]["task_ids"][16:27]
        + roles["teacher_finish_train"]["task_ids"][16:26]
    )
    prior = {"task_ids": prior_tasks, "content_sha256": "e" * 64}
    roster = builder.build_roster(
        phase10c_split={"roles": roles, "content_sha256": "a" * 64},
        base_split={"roles": {"train": {"task_ids": sorted(all_tasks)}}, "content_sha256": "b" * 64},
        prior_roster=prior,
        protocol_sha256="c" * 64,
        producer_git_sha="d" * 40,
    )
    assert roster["task_count"] == 128
    assert roster["K"] == 8
    assert roster["source_role_counts"] == {
        "teacher_nav_train": 32,
        "teacher_match_train": 32,
        "teacher_finish_train": 64,
    }
    assert not (set(roster["task_ids"]) & set(prior_tasks))
    assert all(
        task not in roster["task_ids"]
        for value in roles.values()
        for task in value["task_ids"][:16]
    )
    assert roster["training_performed"] is False and roster["optimizer_steps"] == 0


def test_teacher_exploration_scale_contract_is_raw_35b_k8():
    pytest.importorskip("playwright")
    collector = _module("m6_collect_phase10c_teacher_exploration_scale", "scripts/m6_collect_policy_success.py")
    args = argparse.Namespace(
        mode="phase10c_teacher_exploration_scale",
        role="train",
        task_roster=Path("roster.json"),
        k=8,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
        shared_prefix_manifest=None,
        state_correction_manifest=None,
        base_model=Path("/data/share/model/Qwen3.5-35B-A3B"),
        adapter=None,
        tensor_parallel_size=2,
    )
    collector.validate_phase10c_teacher_exploration_contract(args)


def test_teacher_exploration_scale_wrapper_is_two_gpu_k8_inference_only():
    wrapper = (ROOT / "scripts/run_m6_phase10c_teacher_exploration_scale_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2" in wrapper
    assert "--mode phase10c_teacher_exploration_scale --role train --k 8 --seed 20260863" in wrapper
    assert "--max-model-turns 18 --max-environment-steps 15" in wrapper
    assert "--base-model /data/share/model/Qwen3.5-35B-A3B" in wrapper
    assert "optimizer" not in wrapper

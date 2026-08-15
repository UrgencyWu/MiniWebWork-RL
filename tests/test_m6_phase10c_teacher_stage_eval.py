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


def test_builder_interleaves_three_disjoint_capabilities():
    builder = _module(
        "m6_phase10c_teacher_stage_rosters",
        "scripts/m6_phase10c_build_teacher_stage_eval_rosters.py",
    )
    roles = {}
    for stage in ("dev", "qualification"):
        for offset, capability in enumerate(builder.CAPABILITIES):
            roles[f"teacher_{capability}_{stage}"] = {
                "task_ids": [f"webshop_goal_{stage[0]}{offset}{index:03d}" for index in range(32)]
            }
    split = {"roles": roles, "selection_seed": 20260860, "content_sha256": "a" * 64}
    all_tasks = {task for item in roles.values() for task in item["task_ids"]}
    base = {"roles": {"train": {"task_ids": sorted(all_tasks)}}, "content_sha256": "b" * 64}
    roster = builder.build_roster(
        stage="dev",
        phase10c_split=split,
        base_split=base,
        protocol_sha256="c" * 64,
        git_sha="d" * 40,
    )
    assert roster["task_count"] == 96
    assert roster["capability_task_counts"] == {"nav": 32, "match": 32, "finish": 32}
    assert roster["task_ids"][:3] == [
        roles["teacher_nav_dev"]["task_ids"][0],
        roles["teacher_match_dev"]["task_ids"][0],
        roles["teacher_finish_dev"]["task_ids"][0],
    ]


def test_phase10c_eval_contract_freezes_three_model_identities():
    pytest.importorskip("playwright")
    collector = _module(
        "m6_collect_phase10c_teacher_stage_eval",
        "scripts/m6_collect_policy_success.py",
    )
    common = dict(
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
    )
    raw = argparse.Namespace(
        **common,
        phase10c_evaluation_identity="raw35",
        base_model=Path("/data/share/model/Qwen3.5-35B-A3B"),
        adapter=None,
        tensor_parallel_size=2,
    )
    collector.validate_phase10c_teacher_stage_evaluation_contract(raw)
    sft35 = argparse.Namespace(
        **common,
        phase10c_evaluation_identity="sft35",
        base_model=Path("/data/share/model/Qwen3.5-35B-A3B"),
        adapter=collector.PHASE10C_SFT35_ADAPTER,
        tensor_parallel_size=2,
    )
    collector.validate_phase10c_teacher_stage_evaluation_contract(sft35)
    sft4 = argparse.Namespace(
        **common,
        phase10c_evaluation_identity="sft4",
        base_model=Path("/data/share/model/Qwen3.5-4B"),
        adapter=collector.PHASE10B_STUDENT_ADAPTER,
        tensor_parallel_size=1,
    )
    collector.validate_phase10c_teacher_stage_evaluation_contract(sft4)
    sft35.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="SFT35 identity"):
        collector.validate_phase10c_teacher_stage_evaluation_contract(sft35)


def test_eval_wrapper_freezes_budget_and_model_paths():
    source = (ROOT / "scripts/run_m6_phase10c_teacher_stage_eval_job.sh").read_text()
    assert "#SBATCH --gres=gpu:2" in source
    assert "--mode phase10c_teacher_stage_evaluation --role train --k 4" in source
    assert "--max-model-turns 18 --max-environment-steps 15" in source
    assert "raw35)" in source and "sft35)" in source and "sft4)" in source
    assert "teacher_self_sft_v1/weighted_lora_v1/final_adapter" in source


def test_task_bootstrap_is_paired_and_deterministic():
    stats = _module(
        "m6_phase10c_teacher_stage_stats",
        "scripts/m6_phase10c_teacher_stage_eval_stats.py",
    )
    first = stats._bootstrap_ci([0.25, 0.0, -0.25, 0.5], seed=7, draws=1000)
    second = stats._bootstrap_ci([0.25, 0.0, -0.25, 0.5], seed=7, draws=1000)
    assert first == second
    assert first[0] <= 12.5 <= first[1]

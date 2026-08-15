from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.long_horizon_rl.vllm_backend import RawVLLMBackendConfig


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qualification_rosters_pair_student_and_specialist_on_exact_task_order():
    builder = _module("m6_phase10b_qualification_rosters", "scripts/m6_phase10b_build_qualification_rosters.py")
    tasks = [f"webshop_goal_{index:05d}" for index in range(24)]
    split = {"selection_seed": 20260850, "content_sha256": "a" * 64}
    base = {"content_sha256": "b" * 64}
    student = builder._roster(
        identity="student",
        comparison_specialist="S_match",
        task_ids=tasks,
        phase10b_split=split,
        base_split=base,
        protocol_sha256="c" * 64,
        git_sha="d" * 40,
    )
    specialist = builder._roster(
        identity="S_match",
        comparison_specialist="S_match",
        task_ids=tasks,
        phase10b_split=split,
        base_split=base,
        protocol_sha256="c" * 64,
        git_sha="d" * 40,
    )
    assert student["task_ids"] == specialist["task_ids"] == tasks
    assert student["task_order_sha256"] == specialist["task_order_sha256"] == sha256_json(tasks)
    assert student["task_count"] == specialist["task_count"] == 24
    assert student["comparison_specialist"] == specialist["comparison_specialist"] == "S_match"


def test_qualification_contract_freezes_models_horizon_and_tensor_parallelism():
    pytest.importorskip("playwright")
    collector = _module("m6_collect_phase10b_qualification", "scripts/m6_collect_policy_success.py")
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
    student = argparse.Namespace(
        **common,
        phase10b_qualification_identity="student",
        base_model=Path("/data/share/model/Qwen3.5-4B"),
        adapter=Path(
            "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter"
        ),
        tensor_parallel_size=1,
    )
    collector.validate_phase10b_specialist_qualification_contract(student)
    match = argparse.Namespace(
        **common,
        phase10b_qualification_identity="S_match",
        base_model=Path("/data/share/model/Qwen3.5-35B-A3B"),
        adapter=None,
        tensor_parallel_size=2,
    )
    collector.validate_phase10b_specialist_qualification_contract(match)
    match.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="tensor parallelism"):
        collector.validate_phase10b_specialist_qualification_contract(match)


def test_raw_vllm_config_exposes_two_gpu_tensor_parallelism():
    config = RawVLLMBackendConfig(
        base_model="/data/share/model/Qwen3.5-35B-A3B",
        base_model_manifest_sha256="a" * 64,
        base_model_functional_sha256="b" * 64,
        seed=20260851,
        tensor_parallel_size=2,
    )
    assert config.engine_kwargs()["tensor_parallel_size"] == 2


@pytest.mark.parametrize("identity", ["S_nav", "S_match", "S_finish"])
def test_specialist_decision_requires_gain_net_flips_and_specific_safety(identity: str):
    evaluator = _module(f"m6_phase10b_qualification_{identity}", "scripts/m6_phase10b_specialist_qualification.py")
    decision = evaluator.qualification_decision(
        identity=identity,
        delta_pp=6.0,
        positive_tasks=4,
        negative_tasks=1,
        classes={
            "student": {"schema_failure": 2, "action_failure": 2, "zero_score_purchase": 2, "partial_purchase": 2},
            "specialist": {"schema_failure": 1, "action_failure": 1, "zero_score_purchase": 1, "partial_purchase": 1},
        },
        same_item_option_positive_flips=1,
        finish_purchase_positive_flips=1,
        finish_recovery_positive_flips=1,
    )
    assert decision["specialist_qualified"] is True
    undergain = evaluator.qualification_decision(
        identity=identity,
        delta_pp=4.9,
        positive_tasks=4,
        negative_tasks=1,
        classes={
            "student": {"schema_failure": 2, "action_failure": 2, "zero_score_purchase": 2, "partial_purchase": 2},
            "specialist": {"schema_failure": 1, "action_failure": 1, "zero_score_purchase": 1, "partial_purchase": 1},
        },
        same_item_option_positive_flips=1,
        finish_purchase_positive_flips=1,
        finish_recovery_positive_flips=1,
    )
    assert undergain["specialist_qualified"] is False


def test_qualification_wrapper_is_inference_only_and_model_bounded():
    job = (ROOT / "scripts" / "run_m6_phase10b_specialist_qualification_job.sh").read_text(encoding="utf-8")
    assert "--mode phase10b_specialist_qualification --role train --k 4 --seed 20260851" in job
    assert "--max-model-turns 18 --max-environment-steps 15" in job
    assert "--tensor-parallel-size \"$tp\"" in job
    assert "optimizer" not in job
    assert "S_match:S_match" in job and "tp=2" in job

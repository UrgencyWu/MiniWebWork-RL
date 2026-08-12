from __future__ import annotations

import copy
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.vllm_backend import RawVLLMBackendConfig
from miniwebwork.long_horizon_rl.vllm_backend import RolloutRequestContext, derive_context_sampling_seed
from miniwebwork.webshop_rl.formal_evaluation import (
    AUTHORIZATION_SCHEMA,
    IDENTITIES,
    evaluation_run_root,
    frozen_test_task_ids,
    load_eval_plan,
    summarize_groups,
    validate_eval_authorization,
    validate_eval_plan,
)
from miniwebwork.webshop_rl.formal_training import self_hash

ROOT = Path(__file__).resolve().parents[1]


def test_eval_plan_freezes_500_k4_eight_identity_matrix_and_resources():
    plan = load_eval_plan()["payload"]
    assert tuple(item["identity"] for item in plan["identities"]) == IDENTITIES
    assert frozen_test_task_ids() == tuple(f"webshop_goal_{index:05d}" for index in range(500))
    assert plan["test"]["task_count"] == 500
    assert plan["test"]["rollouts_per_task"] == 4
    assert plan["test"]["total_trajectory_count"] == 16000
    assert plan["test"]["shared_prefix_turns"] == 0
    assert plan["resources"] == {
        "wall_time_per_allocation": "24:00:00",
        "gpus_per_identity": 1,
        "cpus_per_identity": 6,
        "memory_gib_per_identity": 24,
        "maximum_parallel_identities": 8,
        "shared_service": {"cpus": 24, "memory_gib": 96, "workers": 16, "url": "http://127.0.0.1:44151"},
    }


def test_eval_plan_fails_closed_on_test_identity_or_resource_drift():
    plan = load_eval_plan()["payload"]
    for mutate in (
        lambda value: value["test"].update(task_count=499),
        lambda value: value["identities"].pop(),
        lambda value: value["resources"].update(cpus_per_identity=12),
        lambda value: value["success_contract"].update(training_updates_allowed=True),
    ):
        changed = copy.deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_eval_plan(changed)


def test_raw_eval_backend_has_no_lora_and_uses_m5_context():
    config = RawVLLMBackendConfig(
        base_model="/model",
        base_model_manifest_sha256="a" * 64,
        base_model_functional_sha256="b" * 64,
        seed=20260812,
    )
    kwargs = config.engine_kwargs()
    assert config.max_model_len == 8192
    assert kwargs["enable_lora"] is False
    assert "enable_sleep_mode" not in kwargs
    assert kwargs["max_num_seqs"] == 32


def test_eval_infrastructure_retry_keeps_paired_sampling_seed():
    first = RolloutRequestContext(
        run_seed=20260812,
        iteration_index=0,
        group_id="e0007",
        attempt_index=0,
        trajectory_id="e0007.a0.r2",
        rollout_index=2,
        sampling_attempt_index=0,
    )
    retry = RolloutRequestContext(
        run_seed=20260812,
        iteration_index=0,
        group_id="e0007",
        attempt_index=1,
        trajectory_id="e0007.a1.r2",
        rollout_index=2,
        sampling_attempt_index=0,
    )
    assert derive_context_sampling_seed(first, turn_index=3) == derive_context_sampling_seed(retry, turn_index=3)


def test_eval_authorization_is_git_plan_and_eight_identity_bound():
    git_sha = "a" * 40
    plan_sha = "b" * 64
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "frozen_evaluation_submission_allowed": True,
        "approval_scope": "eight_frozen_evaluation_identities_only",
        "consumer_git_sha": git_sha,
        "producer_training_git_sha": "b9b5221777427b0d7decaf7b12c6572ac9a1d3f3",
        "eval_plan_sha256": plan_sha,
        "identities": list(IDENTITIES),
        "logical_job_count": 8,
    }
    payload["content_sha256"] = self_hash(payload)
    assert validate_eval_authorization(payload, consumer_git_sha=git_sha, plan_sha256=plan_sha)["logical_job_count"] == 8
    changed = dict(payload, logical_job_count=7)
    changed["content_sha256"] = self_hash(changed)
    with pytest.raises(ValueError, match="job count"):
        validate_eval_authorization(changed, consumer_git_sha=git_sha, plan_sha256=plan_sha)


def _group(index: int):
    trajectories = []
    for rollout in range(4):
        success = rollout == 0
        trajectories.append(
            {
                "success": success,
                "reward": 1.0 if success else 0.25,
                "generated_action_tokens": 2,
                "environment_steps": 1,
                "elapsed_seconds": 0.5,
                "turns": [{"schema_valid": rollout != 3}],
                "evaluation_turn_summary": [{"action_result": {"error_code": "bad" if rollout == 3 else ""}}],
            }
        )
    return {
        "task_id": f"webshop_goal_{index:05d}",
        "task_metadata": {"goal_index": index, "category": "cat", "constraint_count": index % 2},
        "trajectories": trajectories,
    }


def test_eval_summary_records_primary_secondary_cost_and_strata():
    summary = summarize_groups([_group(index) for index in range(500)])
    assert summary["trajectory_count"] == 2000
    assert summary["success_count"] == 500
    assert summary["trajectory_success_rate"] == 0.25
    assert summary["mean_dense_task_score"] == 0.4375
    assert summary["generated_action_tokens"] == 4000
    assert summary["schema_invalid_turn_fraction"] == 0.25
    assert summary["by_category"]["cat"]["task_count"] == 500
    assert summary["by_constraint_count"]["0"]["task_count"] == 250


def test_eval_slurm_is_evaluation_only_24h_recoverable_and_resource_bounded():
    script = (ROOT / "scripts" / "run_m5_webshop_frozen_eval_job.sh").read_text()
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=6" in script
    assert "#SBATCH --mem=24G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "trap submit_timeout_successor USR1" in script
    assert 'afterany:${SLURM_JOB_ID}' in script
    assert "training_updates_allowed=false" in script
    assert "M5_EVAL_AUTHORIZATION_PATH" in script
    assert 'M5_REPO_ROOT:-/home/wushaohua/data/MiniWebWork-RL' in script
    assert "M5_REPO_ROOT=$repo_root" in script
    assert 'export PYTHONPATH="$repo_root/src' in script
    assert "scancel" not in script
    assert evaluation_run_root("raw_base_model").as_posix().endswith("formal/frozen_test/raw_base_model")


def test_eval_authorization_tool_never_submits_jobs():
    tool = (ROOT / "scripts" / "m5_webshop_authorize_frozen_eval.py").read_text()
    assert "I_APPROVE_EIGHT_FROZEN_EVALS" in tool
    assert "sbatch" not in tool

import copy
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.m5_webshop_protocol import split_for_goal_index
from miniwebwork.webshop_rl.formal_training import (
    AtomicTokenBudget,
    AUTHORIZATION_SCHEMA,
    READINESS_SCHEMA,
    formal_run_root,
    formal_task_order,
    load_formal_plan,
    self_hash,
    validate_authorization,
    validate_formal_plan,
    validate_readiness,
)


ROOT = Path(__file__).resolve().parents[1]


def _hashed(payload):
    value = dict(payload)
    value["content_sha256"] = self_hash(value)
    return value


def test_formal_plan_freezes_narrow_six_run_matrix_and_budget():
    plan = load_formal_plan()["payload"]
    assert plan["formal_submission_allowed"] is False
    assert plan["matrix"]["methods"] == ["multi_turn_grpo", "anchor_gigpo"]
    assert plan["matrix"]["seeds"] == [20260801, 20260802, 20260803]
    assert plan["online_budget"]["generated_action_token_cap_per_run"] == 500000
    assert plan["online_budget"]["maximum_atomic_group_token_reservation"] == 4 * 18 * 128
    assert plan["online_budget"]["shared_prefix_turns"] == 1
    assert plan["resource_contract"]["maximum_parallel_online_runs"] == 6


def test_formal_plan_fails_closed_on_method_resource_or_authorization_drift():
    plan = load_formal_plan()["payload"]
    for mutate in (
        lambda value: value["matrix"].update(methods=["multi_turn_grpo", "ppo"]),
        lambda value: value["resource_contract"].update(cpus_per_run=16),
        lambda value: value.update(formal_submission_allowed=True),
    ):
        changed = copy.deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_formal_plan(changed)


def test_formal_task_order_is_seeded_unique_and_train_only():
    first = formal_task_order(20260801)
    second = formal_task_order(20260802)
    assert len(first) == len(set(first)) == 10885
    assert first != second
    assert all(split_for_goal_index(int(task_id.rsplit("_", 1)[1])) == "train" for task_id in first)
    assert first == formal_task_order(20260801)


def test_atomic_token_budget_never_exceeds_cap_and_releases_unused_capacity():
    budget = AtomicTokenBudget(cap=20000, spent=1000, reservation_size=9216)
    first = budget.reserve("g0")
    second = budget.reserve("g1")
    assert first is not None and second is not None
    assert budget.reserve("g2") is None
    first.charge(800)
    assert budget.spent == 1800
    first.release()
    assert budget.reserve("g2") is not None
    with pytest.raises(ValueError, match="exceeded"):
        second.charge(9217)


def test_readiness_and_authorization_are_separate_and_hash_bound(tmp_path):
    plan_sha = "a" * 64
    git_sha = "b" * 40
    readiness = _hashed(
        {
            "schema_version": READINESS_SCHEMA,
            "decision": "READY_PENDING_USER_AUTHORIZATION",
            "ready": True,
            "formal_submission_allowed": False,
            "git_sha": git_sha,
            "formal_plan_sha256": plan_sha,
            "unmet_gates": [],
        }
    )
    assert validate_readiness(readiness, git_sha=git_sha, plan_sha256=plan_sha)["ready"] is True
    readiness_sha = sha256_json(readiness)
    authorization = _hashed(
        {
            "schema_version": AUTHORIZATION_SCHEMA,
            "formal_submission_allowed": True,
            "approval_scope": "six_online_runs_only",
            "git_sha": git_sha,
            "formal_plan_sha256": plan_sha,
            "readiness_sha256": readiness_sha,
            "methods": ["multi_turn_grpo", "anchor_gigpo"],
            "seeds": [20260801, 20260802, 20260803],
        }
    )
    assert validate_authorization(
        authorization,
        git_sha=git_sha,
        plan_sha256=plan_sha,
        readiness_sha256=readiness_sha,
    )["formal_submission_allowed"] is True
    changed = dict(authorization, readiness_sha256="c" * 64)
    changed["content_sha256"] = self_hash(changed)
    with pytest.raises(ValueError, match="readiness"):
        validate_authorization(changed, git_sha=git_sha, plan_sha256=plan_sha, readiness_sha256=readiness_sha)


def test_formal_slurm_entrypoint_is_24h_single_gpu_recoverable_and_authorized():
    script = (ROOT / "scripts" / "run_m5_webshop_formal_online_job.sh").read_text()
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "trap submit_timeout_successor USR1" in script
    assert 'afterany:${SLURM_JOB_ID}' in script
    assert "M5_AUTHORIZATION_PATH" in script
    assert "formal_training=true" in script
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in script
    assert "scancel" not in script
    assert formal_run_root("anchor_gigpo", 20260803).as_posix().endswith(
        "formal/online/anchor_gigpo/seed_20260803"
    )


def test_readiness_and_authorization_tools_never_submit_training():
    readiness = (ROOT / "scripts" / "m5_webshop_formal_readiness.py").read_text()
    authorization = (ROOT / "scripts" / "m5_webshop_authorize_formal.py").read_text()
    assert "sbatch" not in readiness
    assert "sbatch" not in authorization
    assert "I_APPROVE_SIX_ONLINE_RUNS" in authorization

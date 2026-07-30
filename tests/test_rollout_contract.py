import pytest

from miniwebwork.rollout import (
    INFRASTRUCTURE_FAILURE,
    NO_FAILURE,
    POLICY_FAILURE,
    RolloutRecord,
    RolloutStep,
    derive_rollout_seed,
    summarize_group,
)


def _record(index: int, reward: float, success: bool) -> RolloutRecord:
    return RolloutRecord(
        task_id="TASK",
        task_type="no_feasible_product",
        episode_id=f"EP-{index}",
        rollout_index=index,
        rollout_seed=derive_rollout_seed(7, "TASK", index),
        policy="policy",
        temperature=0.4,
        success=success,
        reward=reward,
        rollout_valid=True,
        failure_origin=NO_FAILURE if success else POLICY_FAILURE,
        termination_reason="verified_submission" if success else "premature_finish",
        model_turns=1,
        environment_steps=1,
        schema_valid_count=1,
        schema_invalid_count=0,
        steps=[
            RolloutStep(
                turn=1,
                page_type="products",
                generated_token_ids=[1],
                token_logprobs=[-0.1],
                schema_valid=True,
                parsed_action={"action": "finish"},
                env_action_success=True,
                terminated=True,
            )
        ],
    )


def test_group_with_mixed_rewards_is_valid_for_grpo():
    summary = summarize_group([_record(0, 0.0, False), _record(1, 1.0, True)], requested_k=2)

    assert summary.reward_sequence == [0.0, 1.0]
    assert summary.has_reward_variance is True
    assert summary.valid_for_grpo_update is True


def test_infrastructure_failure_requires_null_reward():
    record = _record(0, 0.0, False)
    record.rollout_valid = False
    record.failure_origin = INFRASTRUCTURE_FAILURE

    with pytest.raises(ValueError, match="reward=None"):
        record.validate()


def test_schema_count_must_equal_model_turns():
    record = _record(0, 0.0, False)
    record.schema_invalid_count = 1

    with pytest.raises(ValueError, match="Schema counts"):
        record.validate()


def test_rollout_seed_is_stable_and_task_specific():
    assert derive_rollout_seed(7, "TASK-A", 0) == derive_rollout_seed(7, "TASK-A", 0)
    assert derive_rollout_seed(7, "TASK-A", 0) != derive_rollout_seed(7, "TASK-B", 0)

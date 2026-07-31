import math

import pytest

from miniwebwork.rollout import (
    INFRASTRUCTURE_FAILURE,
    NO_FAILURE,
    POLICY_FAILURE,
    RolloutRecord,
    RolloutStep,
    derive_rollout_seed,
    strict_raw_policy_distribution,
    summarize_group,
)


def _record(
    index: int,
    reward: float,
    success: bool,
    *,
    temperature: float = 0.4,
    top_p: float = 0.9,
    top_k: int = 0,
) -> RolloutRecord:
    return RolloutRecord(
        task_id="TASK",
        task_type="no_feasible_product",
        episode_id=f"EP-{index}",
        rollout_index=index,
        rollout_seed=derive_rollout_seed(7, "TASK", index),
        policy="policy",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
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
                sampling_logprobs=[-0.2],
                schema_valid=True,
                parsed_action={"action": "finish"},
                env_action_success=True,
                terminated=True,
            )
        ],
    )


def _strict_record(index: int, reward: float, success: bool) -> RolloutRecord:
    return _record(
        index,
        reward,
        success,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )


def test_mixed_reward_diagnostic_group_has_learning_signal_only():
    summary = summarize_group(
        [_record(0, 0.0, False), _record(1, 1.0, True)],
        requested_k=2,
    )

    assert summary.reward_sequence == [0.0, 1.0]
    assert summary.temperature == 0.4
    assert summary.top_p == 0.9
    assert summary.top_k == 0
    assert summary.has_reward_variance is True
    assert summary.has_learning_signal is True
    assert summary.update_distribution_compatible is False
    assert summary.valid_for_grpo_update is False


def test_update_compatible_mixed_group_is_valid_for_grpo():
    summary = summarize_group(
        [_strict_record(0, 0.0, False), _strict_record(1, 1.0, True)],
        requested_k=2,
        update_distribution_compatible=True,
    )

    assert summary.has_learning_signal is True
    assert summary.update_distribution_compatible is True
    assert summary.valid_for_grpo_update is True


def test_caller_cannot_mark_warped_distribution_update_compatible():
    with pytest.raises(ValueError, match="temperature=1"):
        summarize_group(
            [_record(0, 0.0, False), _record(1, 1.0, True)],
            requested_k=2,
            update_distribution_compatible=True,
        )


def test_zero_variance_strict_group_has_no_learning_signal():
    summary = summarize_group(
        [_strict_record(0, 0.0, False), _strict_record(1, 0.0, False)],
        requested_k=2,
        update_distribution_compatible=True,
    )

    assert summary.has_learning_signal is False
    assert summary.valid_for_grpo_update is False


def test_group_rejects_mixed_top_p_values():
    with pytest.raises(ValueError, match="top_p"):
        summarize_group(
            [
                _record(0, 0.0, False, top_p=0.9),
                _record(1, 1.0, True, top_p=1.0),
            ],
            requested_k=2,
        )


def test_group_rejects_mixed_top_k_values():
    with pytest.raises(ValueError, match="top_k"):
        summarize_group(
            [
                _record(0, 0.0, False, top_k=0),
                _record(1, 1.0, True, top_k=20),
            ],
            requested_k=2,
        )


def test_strict_raw_policy_distribution_requires_no_sampling_warpers():
    assert strict_raw_policy_distribution(1.0, 1.0, 0) is True
    assert strict_raw_policy_distribution(0.4, 1.0, 0) is False
    assert strict_raw_policy_distribution(1.0, 0.9, 0) is False
    assert strict_raw_policy_distribution(1.0, 1.0, 50) is False


def test_nonfinite_policy_logprob_is_rejected():
    record = _strict_record(0, 0.0, False)
    record.steps[0].token_logprobs = [math.nan]

    with pytest.raises(ValueError, match="NaN or Inf"):
        record.validate()


def test_nonfinite_sampling_logprob_is_rejected():
    record = _strict_record(0, 0.0, False)
    record.steps[0].sampling_logprobs = [math.inf]

    with pytest.raises(ValueError, match="NaN or Inf"):
        record.validate()


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

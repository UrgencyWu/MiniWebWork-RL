from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from miniwebwork.long_horizon_rl.learner import (
    TurnTrainingExample,
    audit_group_behavior_sampling_parity,
    collate_turn_training_examples,
    extract_tail_completion_logprobs,
    hierarchical_clipped_policy_loss,
    prepare_group_training_examples,
    summarize_logprob_parity,
    train_policy_groups,
)

from miniwebwork.long_horizon_rl.contracts import group_content_sha256
from test_m4_long_horizon_credit import _group, _identity


def _example(trajectory_index, turn_index, *, tokens, turns, advantage):
    return TurnTrainingExample(
        group_id="group-0000",
        trajectory_id=f"trajectory-{trajectory_index}",
        trajectory_index=trajectory_index,
        turn_index=turn_index,
        turns_in_trajectory=turns,
        prompt_token_ids=(1, 2),
        generated_token_ids=tuple(range(3, 3 + tokens)),
        behavior_logprobs=tuple(-0.5 for _ in range(tokens)),
        sampling_logprobs=tuple(-0.5 for _ in range(tokens)),
        advantage=float(advantage),
    )


def test_prepared_group_examples_count_unique_effective_tokens_only_once():
    prepared = prepare_group_training_examples(_group("multi_turn_grpo"), "multi_turn_grpo")
    assert len(prepared["examples"]) == 8
    assert prepared["generated_action_tokens"] == 16
    assert prepared["effective_optimizer_action_tokens"] == 16
    assert prepared["hierarchical_weight_sum"] == pytest.approx(1.0)
    zero = prepare_group_training_examples(
        _group("multi_turn_grpo", rewards=(0.0, 0.0, 0.0, 0.0)),
        "multi_turn_grpo",
    )
    assert zero["zero_advantage_group"] is True
    assert zero["effective_optimizer_action_tokens"] == 0


def test_hierarchical_loss_is_invariant_to_turn_token_and_trajectory_turn_lengths():
    examples = (
        _example(0, 1, tokens=1, turns=1, advantage=1.0),
        _example(1, 1, tokens=4, turns=2, advantage=1.0),
        _example(1, 2, tokens=2, turns=2, advantage=1.0),
        _example(2, 1, tokens=3, turns=1, advantage=1.0),
        _example(3, 1, tokens=2, turns=1, advantage=1.0),
    )
    batch = collate_turn_training_examples(examples, pad_token_id=0)
    replay = batch["behavior_logprobs"].clone()
    result = hierarchical_clipped_policy_loss(replay, batch)
    assert result["loss"].item() == pytest.approx(-1.0)
    assert result["mean_ratio"].item() == pytest.approx(1.0)


def test_clipped_policy_objective_handles_positive_and_negative_advantages():
    examples = tuple(
        _example(index, 1, tokens=1, turns=1, advantage=(1.0 if index < 2 else -1.0))
        for index in range(4)
    )
    batch = collate_turn_training_examples(examples, pad_token_id=0)
    replay = batch["behavior_logprobs"].clone()
    replay[0, 0] += torch.log(torch.tensor(2.0))
    replay[1, 0] += torch.log(torch.tensor(1.2))
    replay[2, 0] += torch.log(torch.tensor(0.5))
    replay[3, 0] += torch.log(torch.tensor(0.8))
    result = hierarchical_clipped_policy_loss(replay, batch, clip_epsilon=0.2)
    expected_objectives = [1.2, 1.2, -0.8, -0.8]
    assert result["loss"].item() == pytest.approx(-sum(expected_objectives) / 4)
    assert result["clip_fraction"].item() == pytest.approx(0.5)


def test_left_padded_tail_logits_align_every_completion_token():
    examples = (
        _example(0, 1, tokens=1, turns=1, advantage=1.0),
        _example(1, 1, tokens=3, turns=1, advantage=1.0),
    )
    batch = collate_turn_training_examples(examples, pad_token_id=0)
    vocab = 10
    logits = torch.full((2, batch["logits_to_keep"], vocab), -5.0)
    maximum_completion = 3
    for row, example in enumerate(examples):
        start = maximum_completion - example.completion_tokens
        for offset, token_id in enumerate(example.generated_token_ids):
            logits[row, start + offset, token_id] = 5.0
    replay, entropy = extract_tail_completion_logprobs(logits, batch)
    assert replay[0, 0] > -0.001
    assert replay[1, :3].min() > -0.001
    assert torch.isfinite(entropy[batch["completion_mask"]]).all()


def test_behavior_sampling_parity_is_explicit_and_fail_closed():
    group = _group("multi_turn_grpo")
    report = audit_group_behavior_sampling_parity(
        group,
        maximum_absolute_difference=1e-6,
    )
    assert report["passed"] is True
    drifted = copy.deepcopy(group)
    drifted["trajectories"][0]["turns"][0]["sampling_logprobs"][0] += 0.01
    drifted["group_sha256"] = "0" * 64
    from miniwebwork.long_horizon_rl.contracts import group_content_sha256

    drifted["group_sha256"] = group_content_sha256(drifted)
    with pytest.raises(ValueError, match="parity failed"):
        audit_group_behavior_sampling_parity(
            drifted,
            maximum_absolute_difference=1e-6,
        )


def test_logprob_parity_summary_reports_ratio_relevant_statistics():
    report = summarize_logprob_parity([-1.0, -2.0], [-0.99, -2.02])
    assert report["token_count"] == 2
    assert report["maximum_absolute_logprob_difference"] == pytest.approx(0.02)
    assert report["mean_importance_ratio"] == pytest.approx(
        (torch.exp(torch.tensor(0.01)).item() + torch.exp(torch.tensor(-0.02)).item()) / 2
    )


class _TinyReplayModel(torch.nn.Module):
    def __init__(self, vocabulary_size=64):
        super().__init__()
        self.token_logits = torch.nn.Parameter(torch.zeros(vocabulary_size))

    def forward(self, *, input_ids, logits_to_keep, **_kwargs):
        logits = self.token_logits.view(1, 1, -1).expand(
            input_ids.shape[0], int(logits_to_keep), -1
        )
        return SimpleNamespace(logits=logits)


def _uniform_behavior_group(method="multi_turn_grpo", rewards=(1.0, 0.0, 1.0, 0.0)):
    group = _group(method, rewards=rewards)
    uniform_logprob = -math.log(64)
    for trajectory in group["trajectories"]:
        for turn in trajectory["turns"]:
            turn["behavior_logprobs"] = [uniform_logprob] * len(turn["generated_token_ids"])
            turn["sampling_logprobs"] = [uniform_logprob] * len(turn["generated_token_ids"])
    group["group_sha256"] = group_content_sha256(group)
    return group


def _parity_thresholds():
    return {
        "behavior_sampling_maximum_absolute_difference": 1e-6,
        "replay_mean_absolute_difference": 1e-5,
        "replay_p95_absolute_difference": 1e-5,
        "replay_maximum_absolute_difference": 1e-5,
        "mean_importance_ratio_absolute_deviation": 1e-5,
    }


def test_shared_tensor_learner_updates_each_nonzero_k4_group_per_policy_epoch():
    model = _TinyReplayModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    group = _uniform_behavior_group()
    report = train_policy_groups(
        model=model,
        optimizer=optimizer,
        groups=[group],
        method="multi_turn_grpo",
        identity=_identity("multi_turn_grpo"),
        pad_token_id=0,
        microbatch_size=4,
        device=torch.device("cpu"),
        collection_sha256="d" * 64,
        all_generated_action_tokens=19,
        parity_thresholds=_parity_thresholds(),
    )
    assert report["optimizer_updates"] == 2
    assert report["effective_optimizer_action_tokens"] == 16
    assert report["optimizer_evaluated_action_tokens"] == 32
    assert report["all_generated_action_tokens"] == 19
    assert report["parameter_change_norm"] > 0
    assert report["behavior_policy_staleness"] == 0
    assert report["initial_replay_parity"]["passed"] is True


def test_shared_tensor_learner_skips_and_reports_zero_signal_group():
    model = _TinyReplayModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    group = _uniform_behavior_group(rewards=(0.0, 0.0, 0.0, 0.0))
    report = train_policy_groups(
        model=model,
        optimizer=optimizer,
        groups=[group],
        method="multi_turn_grpo",
        identity=_identity("multi_turn_grpo"),
        pad_token_id=0,
        microbatch_size=4,
        device=torch.device("cpu"),
        collection_sha256="d" * 64,
        all_generated_action_tokens=16,
        parity_thresholds=_parity_thresholds(),
    )
    assert report["optimizer_updates"] == 0
    assert report["zero_advantage_group_count"] == 1
    assert report["effective_optimizer_action_tokens"] == 0
    assert report["parameter_change_norm"] == 0.0
    assert report["mean_ratio"] == 1.0

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

import miniwebwork.long_horizon_rl.learner as learner_module

from miniwebwork.long_horizon_rl.learner import (
    TurnTrainingExample,
    audit_initial_replay_parity,
    audit_group_behavior_sampling_parity,
    build_or_load_policy_optimizer,
    create_bootstrap_optimizer_artifact,
    collate_turn_training_examples,
    extract_tail_completion_logprobs,
    hierarchical_clipped_policy_loss,
    prepare_group_training_examples,
    summarize_logprob_parity,
    train_policy_groups,
)

from miniwebwork.long_horizon_rl.contracts import group_content_sha256, token_ids_sha256
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


def test_tail_logprob_precision_is_explicit_and_validated():
    examples = (_example(0, 1, tokens=1, turns=1, advantage=1.0),)
    batch = collate_turn_training_examples(examples, pad_token_id=0)
    logits = torch.zeros((1, batch["logits_to_keep"], 10), dtype=torch.bfloat16)
    float32, _ = extract_tail_completion_logprobs(logits, batch, logprob_precision="float32")
    model, _ = extract_tail_completion_logprobs(logits, batch, logprob_precision="model")
    assert float32.dtype == model.dtype == torch.float32
    assert torch.isfinite(float32[batch["completion_mask"]]).all()
    assert torch.isfinite(model[batch["completion_mask"]]).all()
    with pytest.raises(ValueError, match="logprob precision"):
        extract_tail_completion_logprobs(logits, batch, logprob_precision="invalid")


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
    assert report["p99_absolute_logprob_difference"] == pytest.approx(0.02)
    assert report["p999_absolute_logprob_difference"] == pytest.approx(0.02)
    assert report["initial_ratio_clip_count"] == 0
    assert report["initial_ratio_clip_fraction"] == 0.0
    assert report["mean_importance_ratio"] == pytest.approx(
        (torch.exp(torch.tensor(0.01)).item() + torch.exp(torch.tensor(-0.02)).item()) / 2
    )


def test_logprob_parity_summary_exposes_sparse_initial_clip_outliers():
    reference = [0.0] * 1000
    candidate = [0.0] * 999 + [0.3]
    report = summarize_logprob_parity(reference, candidate)
    assert report["mean_absolute_logprob_difference"] == pytest.approx(0.0003)
    assert report["p99_absolute_logprob_difference"] == 0.0
    assert report["p999_absolute_logprob_difference"] == 0.0
    assert report["maximum_absolute_logprob_difference"] == pytest.approx(0.3)
    assert report["initial_ratio_clip_count"] == 1
    assert report["initial_ratio_clip_fraction"] == pytest.approx(0.001)


def test_logprob_parity_summary_p999_exposes_more_than_point_one_percent_tail():
    reference = [0.0] * 1000
    candidate = [0.0] * 998 + [0.2, 0.3]
    report = summarize_logprob_parity(reference, candidate)
    assert report["p99_absolute_logprob_difference"] == 0.0
    assert report["p999_absolute_logprob_difference"] == pytest.approx(0.2)
    assert report["maximum_absolute_log_ratio"] == pytest.approx(0.3)
    assert report["initial_ratio_clip_count"] == 2
    assert report["initial_ratio_clip_fraction"] == pytest.approx(0.002)


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
        "replay_p99_absolute_difference": 1e-5,
        "replay_p999_absolute_difference": 1e-5,
        "replay_initial_ratio_clip_fraction": 1e-5,
        "mean_importance_ratio_absolute_deviation": 1e-5,
    }


def test_initial_parity_reports_but_does_not_gate_one_finite_sparse_maximum(monkeypatch):
    group = _uniform_behavior_group()
    uniform_logprob = -math.log(64)
    for trajectory in group["trajectories"]:
        for turn in trajectory["turns"]:
            turn["generated_token_ids"] = list(range(125))
            turn["behavior_logprobs"] = [uniform_logprob] * 125
            turn["sampling_logprobs"] = [uniform_logprob] * 125
            turn["generated_token_sha256"] = token_ids_sha256(
                turn["generated_token_ids"]
            )
        trajectory["generated_action_tokens"] = 250
    group["generated_action_tokens"] = 1000
    group["group_sha256"] = group_content_sha256(group)

    candidate = [uniform_logprob] * 1000
    candidate[-1] += 2.66
    monkeypatch.setattr(learner_module, "replay_examples", lambda **_kwargs: candidate)
    thresholds = {
        "behavior_sampling_maximum_absolute_difference": 1e-6,
        "replay_mean_absolute_difference": 0.02,
        "replay_p95_absolute_difference": 0.08,
        "replay_p99_absolute_difference": 0.08,
        "replay_p999_absolute_difference": 0.5,
        "replay_initial_ratio_clip_fraction": 0.005,
        "mean_importance_ratio_absolute_deviation": 0.02,
    }
    report = audit_initial_replay_parity(
        model=None,
        groups=[group],
        method="multi_turn_grpo",
        identity=_identity("multi_turn_grpo"),
        pad_token_id=0,
        microbatch_size=8,
        device=torch.device("cpu"),
        thresholds=thresholds,
    )
    assert report["passed"] is True
    assert report["maximum_absolute_log_ratio"] == pytest.approx(2.66)
    assert "maximum_absolute_log_ratio" not in report["checks"]


def test_initial_parity_still_rejects_distribution_wide_mismatch(monkeypatch):
    group = _uniform_behavior_group()
    uniform_logprob = -math.log(64)
    monkeypatch.setattr(
        learner_module,
        "replay_examples",
        lambda **_kwargs: [uniform_logprob + 1.0] * 16,
    )
    thresholds = {
        "behavior_sampling_maximum_absolute_difference": 1e-6,
        "replay_mean_absolute_difference": 0.02,
        "replay_p95_absolute_difference": 0.08,
        "replay_p99_absolute_difference": 0.08,
        "replay_p999_absolute_difference": 0.5,
        "replay_initial_ratio_clip_fraction": 0.005,
        "mean_importance_ratio_absolute_deviation": 0.02,
    }
    with pytest.raises(ValueError, match="initial behavior/replay parity failed"):
        audit_initial_replay_parity(
            model=None,
            groups=[group],
            method="multi_turn_grpo",
            identity=_identity("multi_turn_grpo"),
            pad_token_id=0,
            microbatch_size=8,
            device=torch.device("cpu"),
            thresholds=thresholds,
        )


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
    assert report["credit_assignment"]["mixed_reward_group_count"] == 1
    assert report["credit_assignment"]["zero_variance_group_count"] == 0
    assert report["credit_assignment"]["nonzero_turn_advantage_count"] == 8
    assert sum(item["action_tokens"] for item in report["credit_assignment"]["turn_position"].values()) == 16


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
    assert report["credit_assignment"]["mixed_reward_group_count"] == 0
    assert report["credit_assignment"]["zero_variance_group_count"] == 1


@pytest.mark.parametrize(
    ("lineage_field", "error"),
    [
        ("adapter_sha256", "adapter lineage"),
        ("rollout_adapter_sha256", "rollout-adapter lineage"),
        ("adapter_semantic_sha256", "adapter-semantic lineage"),
    ],
)
def test_bootstrap_optimizer_marker_loads_fresh_and_binds_policy_lineage(
    tmp_path,
    lineage_field,
    error,
):
    identity = _identity("multi_turn_grpo")
    path = tmp_path / "optimizer.pt"
    created = create_bootstrap_optimizer_artifact(path=path, identity=identity)
    assert created["payload"]["optimizer_state_dict"] is None
    model = _TinyReplayModel()
    optimizer = build_or_load_policy_optimizer(
        model=model,
        identity=identity,
        optimizer_artifact=path,
    )
    assert optimizer.state_dict()["state"] == {}

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[lineage_field] = "f" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        build_or_load_policy_optimizer(
            model=model,
            identity=identity,
            optimizer_artifact=path,
        )

from __future__ import annotations

import math

import pytest
import torch

from miniwebwork.long_horizon_rl.sft_trainer import (
    CompletionOnlyCollator,
    LengthBucketBatchSampler,
    PlateauController,
    TokenizedSFTExample,
    completion_only_cross_entropy,
    select_sft_microbatch,
    tokenize_sft_record,
)


class _ExactTemplateTokenizer:
    """Expose a prompt prefix and an assistant suffix including a terminator."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert kwargs == {"enable_thinking": False}
        prompt = [101, 102, 103, 104]
        if messages and messages[-1]["role"] == "assistant":
            assert add_generation_prompt is False
            return prompt + [201, 202]
        assert add_generation_prompt is True
        return prompt


class _ZeroSuffixTokenizer(_ExactTemplateTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        return [101, 102, 103, 104]


class _NonPrefixTokenizer(_ExactTemplateTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        if messages and messages[-1]["role"] == "assistant":
            return [101, 999, 201]
        return [101, 102]


def _record() -> dict:
    return {
        "sample_id": "task-1:turn:1",
        "task_id": "task-1",
        "task_family": "family-a",
        "horizon_stratum": "medium",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "observe"},
            {"role": "assistant", "content": '{"action":"click","target":"x"}'},
        ],
    }


def _example(
    sample_id: str,
    input_ids: tuple[int, ...],
    labels: tuple[int, ...],
) -> TokenizedSFTExample:
    return TokenizedSFTExample(
        sample_id=sample_id,
        task_id=sample_id.split(":")[0],
        task_family="family-a",
        horizon_stratum="medium",
        input_ids=input_ids,
        labels=labels,
        assistant_content='{"action":"click","target":"x"}',
        prompt_tokens=sum(label == -100 for label in labels),
        completion_label_tokens=sum(label != -100 for label in labels),
        untruncated_forward_tokens=len(input_ids),
    )


def test_tokenization_labels_the_exact_full_template_suffix():
    tokenized = tokenize_sft_record(_record(), _ExactTemplateTokenizer())
    assert tokenized.input_ids == (101, 102, 103, 104, 201, 202)
    assert tokenized.labels == (-100, -100, -100, -100, 201, 202)
    assert tokenized.prompt_tokens == 4
    assert tokenized.completion_label_tokens == 2
    assert tokenized.untruncated_forward_tokens == 6


def test_tokenization_fails_closed_on_zero_labels_or_non_prefix_template():
    with pytest.raises(ValueError, match="zero-label"):
        tokenize_sft_record(_record(), _ZeroSuffixTokenizer())
    with pytest.raises(ValueError, match="strict chat-template continuation"):
        tokenize_sft_record(_record(), _NonPrefixTokenizer())


def test_collator_left_pads_positions_and_limits_logits_to_completion_tail():
    first = _example("a:1", (10, 11, 12, 13), (-100, -100, 12, 13))
    second = _example("b:1", (20, 21, 22), (-100, 21, 22))
    batch = CompletionOnlyCollator(pad_token_id=0)([first, second])
    assert batch["input_ids"].tolist() == [[10, 11, 12, 13], [0, 20, 21, 22]]
    assert batch["labels"].tolist() == [
        [-100, -100, 12, 13],
        [-100, -100, 21, 22],
    ]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 1, 1, 1]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3], [0, 0, 1, 2]]
    assert batch["logits_to_keep"] == 3


def test_completion_loss_uses_causal_shift_over_only_the_returned_tail():
    logits = torch.zeros((1, 3, 6), dtype=torch.float32)
    logits[0, 0, 2] = 12.0
    logits[0, 1, 3] = 12.0
    labels = torch.tensor([[99, 98, -100, 2, 3]])
    loss = completion_only_cross_entropy(logits, labels)
    assert loss.token_count == 2
    assert loss.shifted_labels.tolist() == [[2, 3]]
    assert loss.mean.item() < 1e-3
    assert math.isclose(loss.total.item(), loss.mean.item() * 2, rel_tol=1e-6)


def test_bucket_sampler_is_unique_complete_and_epoch_deterministic():
    lengths = list(range(1, 41))
    sampler = LengthBucketBatchSampler(lengths, batch_size=4, seed=123, bucket_size=8)
    epoch_zero = list(sampler)
    assert sorted(index for batch in epoch_zero for index in batch) == list(range(40))
    assert all(1 <= len(batch) <= 4 for batch in epoch_zero)
    assert epoch_zero == list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert sorted(index for batch in epoch_one for index in batch) == list(range(40))
    assert epoch_one != epoch_zero


def test_plateau_requires_minimum_epochs_and_uses_preregistered_deltas():
    controller = PlateauController()
    first = controller.update(
        {
            "dev_nll": 1.0,
            "teacher_forced_action_exact": 0.5,
            "teacher_forced_schema_valid": 0.8,
        },
        completed_epochs=1,
    )
    assert first["stale_evaluations"] == 0
    assert first["stop"] is False
    second = controller.update(
        {
            "dev_nll": 0.996,
            "teacher_forced_action_exact": 0.501,
            "teacher_forced_schema_valid": 0.801,
        },
        completed_epochs=2,
    )
    assert second["improvements"] == {
        "dev_nll": False,
        "teacher_forced_action_exact": False,
        "teacher_forced_schema_valid": False,
    }
    assert second["stale_evaluations"] == 1
    assert second["stop"] is True


def test_microbatch_selection_is_exact_and_fails_closed_on_headroom():
    results = [
        {"microbatch_size": 1, "passed": True, "vram_headroom_fraction": 0.4, "headroom_gate_passed": True},
        {"microbatch_size": 2, "passed": True, "vram_headroom_fraction": 0.2, "headroom_gate_passed": True},
        {"microbatch_size": 4, "passed": True, "vram_headroom_fraction": 0.149, "headroom_gate_passed": False},
        {"microbatch_size": 8, "passed": False, "reason": "cuda_out_of_memory"},
    ]
    assert select_sft_microbatch(results) == 2
    with pytest.raises(RuntimeError, match="15% VRAM headroom"):
        select_sft_microbatch([
            {"microbatch_size": size, "passed": False}
            for size in (1, 2, 4, 8)
        ])
    with pytest.raises(ValueError, match="exactly"):
        select_sft_microbatch(results[:-1])

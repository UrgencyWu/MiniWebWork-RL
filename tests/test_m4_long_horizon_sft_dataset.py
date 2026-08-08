from __future__ import annotations

import json

import pytest

from miniwebwork.data_generation.m4_long_horizon import build_long_horizon_dataset
from miniwebwork.sft.m4_long_horizon_dataset import (
    build_verified_sft_corpus,
    completion_only_token_statistics,
    validate_verified_sft_corpus,
)


class _PrefixTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert kwargs == {"enable_thinking": False}
        token_ids = [101]
        for message in messages:
            if message["role"] == "user":
                token_ids.extend([10, len(message["content"])])
            elif message["role"] == "assistant":
                token_ids.extend([20, len(message["content"]), 99])
            else:
                token_ids.extend([30, len(message["content"])])
        if add_generation_prompt:
            token_ids.append(20)
        return token_ids


class _MappingPrefixTokenizer(_PrefixTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        token_ids = super().apply_chat_template(*args, **kwargs)
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}


def test_completion_only_token_statistics_uses_full_template_suffix():
    messages = [
        {"role": "user", "content": "observe"},
        {"role": "assistant", "content": '{"action":"click"}'},
    ]
    statistics = completion_only_token_statistics(
        _PrefixTokenizer(),
        messages,
        chat_template_kwargs={"enable_thinking": False},
        max_length=6,
    )
    assert statistics == {
        "prompt_tokens": 4,
        "untruncated_forward_tokens": 6,
        "untruncated_completion_label_tokens": 2,
        "forward_tokens": 6,
        "effective_completion_label_tokens": 2,
        "truncated": False,
    }
    truncated = completion_only_token_statistics(
        _PrefixTokenizer(),
        messages,
        chat_template_kwargs={"enable_thinking": False},
        max_length=5,
    )
    assert truncated["effective_completion_label_tokens"] == 1
    assert truncated["truncated"] is True
    zero_label = completion_only_token_statistics(
        _PrefixTokenizer(),
        messages,
        chat_template_kwargs={"enable_thinking": False},
        max_length=4,
    )
    assert zero_label["effective_completion_label_tokens"] == 0
    mapping_result = completion_only_token_statistics(
        _MappingPrefixTokenizer(),
        messages,
        chat_template_kwargs={"enable_thinking": False},
        max_length=6,
    )
    assert mapping_result == statistics


@pytest.mark.browser
def test_verified_sft_corpus_is_unique_successful_and_reference_exact(tmp_path):
    task_root = tmp_path / "tasks"
    seed_root = tmp_path / "seed"
    output_root = tmp_path / "sft"
    build_long_horizon_dataset(task_root, seed_dir=seed_root)
    manifest = build_verified_sft_corpus(
        output_root,
        task_root=task_root,
        seed_dir=seed_root,
        train_task_limit=4,
        dev_task_limit=4,
    )
    assert manifest["repetition_policy"] == "none"
    assert manifest["prompt_contract"] == "browser_agent_v4_long_memory"
    assert manifest["context_contract"]["evidence_memory_contract"] == (
        "public_observation_v1"
    )
    assert manifest["train"]["task_count"] == 4
    assert manifest["dev"]["task_count"] == 4
    assert manifest["train"]["sample_count"] == 47
    assert manifest["dev"]["sample_count"] == 47
    validation = validate_verified_sft_corpus(
        output_root,
        task_root=task_root,
        seed_dir=seed_root,
        require_full_roster=False,
    )
    assert validation["valid"], validation["errors"]
    for split_name, filename in (("train", "train.jsonl"), ("dev", "valid.jsonl")):
        split = manifest[split_name]
        assert split["all_reference_trajectories_verified"] is True
        assert split["sample_count"] == split["unique_sample_count"]
        assert split["duplicate_sample_count"] == 0
        rows = [
            json.loads(line)
            for line in (output_root / filename).read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == len({row["sample_id"] for row in rows})
        assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
        assert all(row["split"] == split_name for row in rows)
        long_rows = [
            row
            for row in rows
            if row["task_family"] == "highest_reliability_supplier"
        ]
        assert long_rows
        richest_prompt = max(
            (row["messages"][-2]["content"] for row in long_rows),
            key=lambda content: content.count('"path": "/suppliers/'),
        )
        assert richest_prompt.count('"path": "/suppliers/') == 3
        assert "88%" in richest_prompt
        assert "93%" in richest_prompt
        assert "99%" in richest_prompt

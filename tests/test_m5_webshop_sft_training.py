from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.webshop_rl.sft_training import (
    SFTConfig,
    select_microbatch,
    tokenize_sft_row,
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert kwargs == {"enable_thinking": False}
        prompt = [10, 11, 12]
        if add_generation_prompt:
            return prompt
        return prompt + [20, 21]


def test_m5_sft_tokenization_is_action_only_and_completion_only():
    config = SFTConfig.from_protocol(load_protocol()["payload"])
    row = {
        "task_id": "webshop_goal_01000",
        "turn_index": 1,
        "messages": [{"role": "user", "content": "state"}],
        "completion": json.dumps({"command": "search[desk lamp]"}),
    }
    example = tokenize_sft_row(row, FakeTokenizer(), config)
    assert example.sample_id == "webshop_goal_01000:001"
    assert example.input_ids == (10, 11, 12, 20, 21)
    assert example.labels == (-100, -100, -100, 20, 21)
    assert example.completion_label_tokens == 2


def test_m5_sft_microbatch_selects_largest_headroom_safe_candidate():
    config = SFTConfig.from_protocol(load_protocol()["payload"])
    rows = [
        {"microbatch_size": 1, "passed": True, "vram_headroom_fraction": 0.8},
        {"microbatch_size": 2, "passed": True, "vram_headroom_fraction": 0.5},
        {"microbatch_size": 4, "passed": True, "vram_headroom_fraction": 0.16},
        {"microbatch_size": 8, "passed": True, "vram_headroom_fraction": 0.14},
    ]
    assert select_microbatch(rows, config) == 4
    rows[2]["vram_headroom_fraction"] = 0.14
    rows[1]["vram_headroom_fraction"] = 0.14
    rows[0]["vram_headroom_fraction"] = 0.14
    with pytest.raises(ValueError, match="VRAM headroom"):
        select_microbatch(rows, config)


def test_m5_sft_slurm_entry_has_one_full_gpu_and_same_root_successor():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m5_webshop_sft_preflight_job.sh").read_text()
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=48G" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert 'afterany:${SLURM_JOB_ID}' in script
    assert 'preflight/sft_gpu' in script
    assert "M5_DISABLE_SUCCESSOR" in script

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import miniwebwork.webshop_rl.online_training as online_training_module
from miniwebwork.m5_webshop_protocol import load_protocol
from miniwebwork.webshop_rl.credit import ANCHOR_METHOD, BASELINE_METHOD
from miniwebwork.webshop_rl.online_training import (
    MAX_SEQUENCE_TOKENS,
    M5VLLMBackendConfig,
    ReplayParityError,
    audit_collection,
    audit_initial_replay_parity,
    build_committed_group,
    prepare_group_training_examples,
    trajectory_from_episode,
    validate_committed_group,
)

SHA = "a" * 64


def test_m5_replay_parity_error_preserves_complete_numeric_audit(monkeypatch):
    behavior = [0.0] * 1000
    replay = [0.0] * 989 + [0.09] * 11
    prepared = [
        {
            "examples": [
                SimpleNamespace(
                    forward_tokens=1000,
                    trajectory_index=0,
                    turn_index=0,
                    behavior_logprobs=behavior,
                    sampling_logprobs=behavior,
                )
            ]
        }
    ]
    monkeypatch.setattr(online_training_module, "_replay_logprobs", lambda *_args, **_kwargs: replay)
    thresholds = {
        "behavior_sampling_maximum_absolute_difference": 1e-6,
        "replay_mean_absolute_difference": 0.02,
        "replay_p95_absolute_difference": 0.08,
        "replay_p99_absolute_difference": 0.08,
        "replay_p999_absolute_difference": 0.5,
        "replay_initial_ratio_clip_fraction": 0.005,
        "mean_importance_ratio_absolute_deviation": 0.02,
    }

    with pytest.raises(ReplayParityError) as captured:
        audit_initial_replay_parity(
            model=None,
            prepared=prepared,
            tokenizer=None,
            device=torch.device("cpu"),
            microbatch_size=4,
            thresholds=thresholds,
        )

    report = captured.value.report
    assert report["passed"] is False
    assert report["checks"] == {
        "mean": True,
        "p95": True,
        "p99": False,
        "p999": True,
        "clip_fraction": True,
        "mean_ratio": True,
    }
    assert report["p99_absolute_logprob_difference"] == pytest.approx(0.09)
    assert report["thresholds"] == thresholds
    assert json.loads(str(captured.value).split(": ", 1)[1])["p99_absolute_logprob_difference"] == pytest.approx(0.09)


def test_m5_replay_parity_uses_exact_per_group_optimizer_batch_order(monkeypatch):
    def example(forward_tokens: int, trajectory_index: int, turn_index: int):
        return SimpleNamespace(
            forward_tokens=forward_tokens,
            trajectory_index=trajectory_index,
            turn_index=turn_index,
            behavior_logprobs=[0.0],
            sampling_logprobs=[0.0],
        )

    prepared = [
        {"examples": [example(10, 0, 1), example(5, 1, 1)]},
        {"examples": [example(8, 0, 1), example(3, 1, 1)]},
    ]
    calls = []

    def replay(_model, examples, _tokenizer, _device, _microbatch_size):
        calls.append([item.forward_tokens for item in examples])
        return [0.0] * len(examples)

    monkeypatch.setattr(online_training_module, "_replay_logprobs", replay)
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
        prepared=prepared,
        tokenizer=None,
        device=torch.device("cpu"),
        microbatch_size=4,
        thresholds=thresholds,
    )

    assert report["passed"] is True
    assert calls == [[5, 10], [3, 8]]


def _observation(page: str, *, step: int, episode: str) -> dict:
    actions = ["search[<your query>]"] if page == "home" else ["click[B001]", "click[Next >]"]
    return {
        "schema_version": "m5-webshop-1.0",
        "task_id": "webshop_goal_01000",
        "episode_id": episode,
        "instruction": "buy a mug",
        "step_index": step,
        "page_type": page,
        "visible_text": page,
        "text_truncated": False,
        "available_actions": actions,
        "terminal": False,
    }


def _episode(reward: float, rollout: int, *, task_score: float | None = None) -> dict:
    turns = []
    for index, page in enumerate(("home", "search_results"), start=1):
        turns.append(
            {
                "model_turn_index": index,
                "observation": _observation(page, step=index - 1, episode=f"episode-{rollout}"),
                "prompt_token_ids": [10, 11, rollout, index],
                "generated_token_ids": [20 + index, 30 + rollout],
                "token_logprobs": [-0.1, -0.2],
                "sampling_logprobs": [-0.1, -0.2],
                "request_id": f"request-{rollout}-{index}",
                "sampling_seed": rollout * 10 + index,
                "generation_backend": "vllm_async",
                "adapter_sha256": SHA,
                "rollout_adapter_sha256": "b" * 64,
                "adapter_semantic_sha256": "c" * 64,
                "schema_valid": True,
                "action": {"command": "search[mug]"},
                "raw_output": '{"command":"search[mug]"}',
            }
        )
    return {
        "task_id": "webshop_goal_01000",
        "success": reward == 1.0,
        "reward": reward,
        "task_score": reward if task_score is None else task_score,
        "rollout_valid": True,
        "termination_reason": "purchase" if reward else "max_model_turns",
        "environment_steps": 2,
        "turns": turns,
    }


def _group(index: int = 0) -> dict:
    trajectories = [
        trajectory_from_episode(
            _episode(float(rollout % 2 == 0), rollout),
            trajectory_id=f"g{index:04d}.a0.r{rollout}",
            rollout_index=rollout,
            adapter_sha256=SHA,
            rollout_adapter_sha256="b" * 64,
            adapter_semantic_sha256="c" * 64,
        )
        for rollout in range(4)
    ]
    return build_committed_group(
        task_id="webshop_goal_01000",
        group_id=f"g{index:04d}",
        attempt_index=0,
        trajectories=trajectories,
        git_sha="d" * 40,
        protocol_sha256="e" * 64,
        adapter_sha256=SHA,
        rollout_adapter_sha256="b" * 64,
        adapter_semantic_sha256="c" * 64,
    )


def test_m5_group_binds_token_logprobs_prompt_context_and_public_state():
    group = _group()
    assert validate_committed_group(group)["generated_action_tokens"] == 16
    baseline = prepare_group_training_examples(group, BASELINE_METHOD)
    anchor = prepare_group_training_examples(group, ANCHOR_METHOD)
    assert baseline["effective_optimizer_action_tokens"] == 16
    assert anchor["credit"]["metrics"]["non_initial_shared_anchor_count"] >= 1

    changed = copy.deepcopy(group)
    changed["trajectories"][0]["turns"][0]["prompt_token_ids"][0] = 999
    with pytest.raises(ValueError, match="self-hash|prompt token hash"):
        validate_committed_group(changed)


def test_m5_uses_official_dense_task_score_but_preserves_binary_success():
    trajectory = trajectory_from_episode(
        _episode(0.0, 0, task_score=0.625),
        trajectory_id="g0000.a0.r0",
        rollout_index=0,
        adapter_sha256=SHA,
        rollout_adapter_sha256="b" * 64,
        adapter_semantic_sha256="c" * 64,
    )
    assert trajectory["reward"] == 0.625
    assert trajectory["binary_reward"] == 0.0
    assert trajectory["success"] is False


def test_m5_collection_gate_counts_k4_signal_and_noninitial_anchor():
    groups = [_group(index) for index in range(32)]
    audit = audit_collection(groups, all_generated_action_tokens=32 * 16)
    assert audit["success_rate"] == 0.5
    assert audit["mixed_reward_group_fraction"] == 1.0
    assert audit["initial_shared_anchor_group_fraction"] == 1.0
    assert audit["shared_noninitial_group_fraction"] == 1.0
    assert audit["informative_micro_turn_fraction"] > 0.02


def test_m5_vllm_contract_uses_8192_context_and_frozen_sampling():
    config = M5VLLMBackendConfig(
        base_model="/model",
        adapter_path="/adapter",
        adapter_sha256=SHA,
        rollout_adapter_path="/view",
        rollout_adapter_sha256="b" * 64,
        adapter_semantic_sha256="c" * 64,
        seed=20260810,
    )
    config.validate(check_adapter_files=False)
    assert config.max_model_len == MAX_SEQUENCE_TOKENS == 8192
    assert config.engine_kwargs()["max_model_len"] == 8192
    assert config.max_num_seqs == 32


def test_m5_protocol_freezes_parity_and_online_slurm_recovery():
    protocol = load_protocol()["payload"]
    assert protocol["online"]["parity_contract"]["replay_p99_absolute_difference"] == 0.08
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_m5_webshop_online_preflight_job.sh").read_text()
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "trap submit_timeout_successor USR1" in script
    assert 'afterany:${SLURM_JOB_ID}' in script
    assert "unset PYTORCH_CUDA_ALLOC_CONF" in script
    assert "export PYTORCH_CUDA_ALLOC_CONF" not in script
    assert "preflight/online_gpu" in script
    assert "formal_training=false" in script
    assert 'scancel "$SLURM_JOB_ID"' not in script
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "m5_webshop_online_preflight.py").read_text()
    assert "_minimal_sft_compatibility" in runner
    assert "c97c9265fe0043a8cda59908713429eef9e13d9619261a144c13cfa5fb4d7334" in runner
    assert "_request_real_interruption" not in runner

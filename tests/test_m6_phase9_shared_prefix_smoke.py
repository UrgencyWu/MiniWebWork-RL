from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stratified_selection_is_deterministic_and_bounded(monkeypatch):
    module = _module("m6_phase9_builder", "scripts/m6_phase9_build_shared_prefix_manifest.py")
    monkeypatch.setattr(module, "TASK_COUNT", 4)
    candidates = {f"task_{index}": {} for index in range(10)}
    goals = {
        task_id: {"category": f"category_{index % 3}", "goal_options": ["size"]}
        for index, task_id in enumerate(candidates)
    }
    first = module._select_stratified(candidates, goal_map=goals)
    second = module._select_stratified(candidates, goal_map=goals)
    assert first == second
    assert len(first) == 4 and len(set(first)) == 4


def test_prefix_binding_hashes_only_the_frozen_source_prefix():
    module = _module("m6_phase9_builder_binding", "scripts/m6_phase9_build_shared_prefix_manifest.py")

    def turn(command: str, token: int):
        return {
            "action": {"command": command},
            "prompt_token_ids": [1, token],
            "generated_token_ids": [token],
            "behavior_logprobs": [-0.1],
            "sampling_logprobs": [-0.1],
        }

    trajectory = {"turns": [turn("search[x]", 2), turn("click[B00000000A]", 3), turn("click[red]", 4)]}
    binding = module._prefix_binding(
        root=Path("/tmp/source"),
        report={"content_sha256": "a" * 64, "policy_lineage": {}, "git_sha": "b" * 40},
        group={"task_id": "task", "group_id": "g0001", "content_sha256": "c" * 64},
        trajectory=trajectory,
        rollout_index=2,
        prefix_turn_count=2,
        public_option_action_count=3,
    )
    assert binding["prefix_turn_count"] == 2
    assert binding["source_rollout_index"] == 2
    assert binding["prefix_action_sequence_sha256"] == module.sha256_json(
        ["search[x]", "click[B00000000A]"]
    )


def test_phase9_collection_contract_is_inference_only():
    pytest.importorskip("playwright")
    module = _module("m6_collect_phase9", "scripts/m6_collect_policy_success.py")
    args = argparse.Namespace(
        role="train",
        task_roster=Path("roster.json"),
        adapter=Path("adapter"),
        k=4,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
        shared_prefix_manifest=Path("manifest.json"),
    )
    module.validate_phase9_shared_prefix_smoke_contract(args)
    args.maximum_action_tokens = 1
    try:
        module.validate_phase9_shared_prefix_smoke_contract(args)
    except ValueError as exc:
        assert "training token budget" in str(exc)
    else:
        raise AssertionError("Phase9 accepted a training token budget")


def test_phase9_records_suffix_only_policy_boundary_and_shared_source():
    collector = (ROOT / "scripts" / "m6_collect_policy_success.py").read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "m6_phase9_build_shared_prefix_manifest.py").read_text(encoding="utf-8")
    job = (ROOT / "scripts" / "run_m6_phase9_shared_prefix_smoke_job.sh").read_text(encoding="utf-8")
    assert 'shared_prefix_turns=len(source_turns)' in collector
    assert 'source_group["trajectories"][int(shared_prefix_spec["source_rollout_index"])]' in collector
    assert "prefix_policy_loss_eligible=False" in collector
    assert '"future_policy_loss_scope": "suffix_tokens_only"' in collector
    assert '"training_performed": False' in builder and '"optimizer_steps": 0' in builder
    assert '"target_asin_or_hidden_answer_used": False' in builder
    assert '"post_action_internal_state_used": False' in builder
    assert "promotion" in builder and "holdout" in builder
    assert '"env_state"' not in builder
    assert 'trajectory["target_asin"]' not in builder and 'goal["target_asin"]' not in builder
    assert "optimizer.step(" not in collector and "optimizer" not in job


def test_phase9_job_is_one_bounded_gpu_inference_smoke():
    job = (ROOT / "scripts" / "run_m6_phase9_shared_prefix_smoke_job.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in job
    assert "#SBATCH --cpus-per-task=4" in job and "#SBATCH --mem=24G" in job
    assert "#SBATCH --time=24:00:00" in job
    assert "--mode phase9_shared_prefix_smoke --role train --k 4 --seed 20260833" in job
    assert "--max-model-turns 18 --max-environment-steps 15" in job
    assert "--shared-prefix-manifest" in job

from __future__ import annotations

import argparse
import copy
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


def _turn(command: str, token: int, *, buy_available: bool = False):
    actions = ["click[Buy Now]"] if buy_available else ["search[x]"]
    return {
        "action": {"command": command},
        "action_result": {"success": True},
        "observation": {"available_actions": actions},
        "pre_action_public_state_sha256": f"{token:064x}",
        "prompt_token_ids": [1, token],
        "generated_token_ids": [token],
        "behavior_logprobs": [-0.1],
        "sampling_logprobs": [-0.1],
    }


def _purchase_failure(*, score: float, trajectory_id: str = "t0"):
    return {
        "trajectory_id": trajectory_id,
        "success": False,
        "task_score": score,
        "turns": [
            _turn("search[x]", 2),
            _turn("click[B00000000A]", 3),
            _turn("click[red]", 4),
            _turn("click[Buy Now]", 5, buy_available=True),
        ],
    }


def test_public_correction_index_is_invariant_to_terminal_score_and_hidden_fields():
    module = _module("m6_phase10_builder_selector", "scripts/m6_phase10_build_smoke_manifest.py")
    original = _purchase_failure(score=0.5)
    stripped = copy.deepcopy(original)
    stripped.pop("task_score")
    stripped.pop("success")
    stripped["hidden_goal"] = {"target_asin": "forbidden"}
    assert module.public_purchase_correction_prefix(original) == module.public_purchase_correction_prefix(stripped)
    assert module.public_purchase_correction_prefix(original)["prefix_turn_count"] == 3


def test_candidate_choice_does_not_prefer_partial_over_zero_score():
    module = _module("m6_phase10_builder_candidate", "scripts/m6_phase10_build_smoke_manifest.py")
    partial = _purchase_failure(score=0.5, trajectory_id="later")
    zero = _purchase_failure(score=0.0, trajectory_id="earlier")
    group = {
        "task_id": "webshop_goal_00001",
        "group_id": "g0",
        "content_sha256": "a" * 64,
        "trajectories": [partial, zero],
    }
    selected = module._best_candidates([group])["webshop_goal_00001"]
    assert selected["source_trajectory_id"] == "earlier"
    assert selected["source_failure_class"] == "zero_score_purchase"


def test_phase10_collection_contract_freezes_k2_and_identity():
    pytest.importorskip("playwright")
    module = _module("m6_collect_phase10", "scripts/m6_collect_policy_success.py")
    common = dict(
        role="train",
        task_roster=Path("roster.json"),
        k=2,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
        shared_prefix_manifest=None,
        state_correction_manifest=Path("manifest.json"),
    )
    student = argparse.Namespace(
        **common,
        phase10_smoke_identity="student",
        base_model=Path("/data/share/model/Qwen3.5-4B"),
        adapter=Path("pi0"),
    )
    module.validate_phase10_state_suffix_smoke_contract(student)
    teacher = argparse.Namespace(
        **common,
        phase10_smoke_identity="teacher",
        base_model=Path("/data/share/model/Qwen3.6-35B-A3B-FP8"),
        adapter=None,
    )
    module.validate_phase10_state_suffix_smoke_contract(teacher)
    teacher.k = 4
    with pytest.raises(ValueError, match="K2"):
        module.validate_phase10_state_suffix_smoke_contract(teacher)


def test_phase10_smoke_code_has_suffix_only_and_no_training_contracts():
    builder = (ROOT / "scripts" / "m6_phase10_build_smoke_manifest.py").read_text(encoding="utf-8")
    collector = (ROOT / "scripts" / "m6_collect_policy_success.py").read_text(encoding="utf-8")
    job = (ROOT / "scripts" / "run_m6_phase10_state_suffix_smoke_job.sh").read_text(encoding="utf-8")
    assert '"task_score_used_for_selector": False' in builder
    assert '"target_asin_or_hidden_answer_used": False' in builder
    assert '"prefix_policy_loss_eligible": False' in builder
    assert "state_correction_manifest" in collector
    assert "prefix_policy_loss_eligible=False" in collector
    assert "phase10_state_suffix_smoke" in collector
    assert "--mode phase10_state_suffix_smoke" in job
    assert "--role train --k 2 --seed 20260841" in job
    assert "#SBATCH --gres=gpu:1" in job and "#SBATCH --array=0-1%2" in job
    assert "optimizer.step(" not in collector and "optimizer" not in job

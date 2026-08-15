from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json
from miniwebwork.m6_phase10b_opd import compress_topk_logprobs, frozen_public_router

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _turn(page_type: str, actions: list[str], *, visible_text: str = ""):
    return {
        "turn_index": 3,
        "pre_action_public_state_sha256": "a" * 64,
        "observation": {
            "page_type": page_type,
            "available_actions": actions,
            "visible_text": visible_text,
        },
    }


def test_public_router_has_frozen_precedence_and_never_falls_back_to_unqualified_model():
    nav = frozen_public_router(
        _turn("search_results", ["click[item]", "click[Next]"]),
        qualified_specialists=("S_nav", "S_match"),
        previous_action_success=True,
        prior_public_state_sha256=(),
    )
    assert nav["assigned_specialist"] == "S_nav"
    assert nav["route_rule"] == "search_or_navigation_page"

    match = frozen_public_router(
        _turn("item", ["click[Red]", "click[Blue]", "click[Buy Now]"]),
        qualified_specialists=("S_nav", "S_match"),
        previous_action_success=True,
        prior_public_state_sha256=(),
    )
    assert match["assigned_specialist"] == "S_match"

    finish_unavailable = frozen_public_router(
        _turn("item", ["click[Buy Now]"], visible_text="Color: Red [selected]"),
        qualified_specialists=("S_nav", "S_match"),
        previous_action_success=True,
        prior_public_state_sha256=(),
    )
    assert finish_unavailable["desired_specialist"] == "S_finish"
    assert finish_unavailable["assigned_specialist"] is None
    assert finish_unavailable["forbidden_fields_used"] is False


def test_public_router_recovery_precedes_page_type_and_uses_only_prior_public_evidence():
    recovery = frozen_public_router(
        _turn("search_results", ["click[item]"]),
        qualified_specialists=("S_nav", "S_finish"),
        previous_action_success=False,
        prior_public_state_sha256=(),
    )
    assert recovery["assigned_specialist"] == "S_finish"
    assert recovery["route_rule"] == "recovery_or_budget_boundary"


def test_topk_compression_is_sorted_finite_and_mass_closed():
    result = compress_topk_logprobs({2: -0.5, 1: -1.5, 3: -3.0}, k=2)
    assert [row["token_id"] for row in result["topk"]] == [2, 1]
    assert 0.0 < result["topk_probability_mass"] <= 1.0
    assert result["probability_sum_abs_error"] <= 1e-12
    with pytest.raises(ValueError, match="token/logprob"):
        compress_topk_logprobs({248320: -1.0}, k=1)


def test_opd_smoke_roster_is_exactly_the_frozen_fresh_role(monkeypatch: pytest.MonkeyPatch):
    builder = _module("m6_phase10b_opd_smoke_roster", "scripts/m6_phase10b_build_opd_smoke_roster.py")
    tasks = [f"webshop_goal_{index:05d}" for index in range(8)]
    monkeypatch.setattr(builder, "validate_phase10b_split", lambda value: value)
    monkeypatch.setattr(builder, "validate_split_lock", lambda value: value)
    roster = builder.build_roster(
        phase10b_split={
            "selection_seed": 20260850,
            "content_sha256": "a" * 64,
            "roles": {"opd_smoke": {"task_ids": tasks}},
        },
        base_split={"content_sha256": "b" * 64, "roles": {"train": {"task_ids": tasks}}},
        protocol_sha256="c" * 64,
        git_sha="d" * 40,
    )
    assert roster["task_ids"] == tasks
    assert roster["task_order_sha256"] == sha256_json(tasks)
    assert roster["specialist_actions_executed"] is False
    assert roster["optimizer_steps"] == 0


def test_opd_behavior_contract_and_wrappers_are_zero_update_and_student_only():
    pytest.importorskip("playwright")
    collector = _module("m6_collect_phase10b_opd_smoke", "scripts/m6_collect_policy_success.py")
    args = argparse.Namespace(
        role="train",
        task_roster=Path("roster.json"),
        k=4,
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
        shared_prefix_manifest=None,
        state_correction_manifest=None,
        base_model=Path("/data/share/model/Qwen3.5-4B"),
        adapter=Path(
            "/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter"
        ),
        tensor_parallel_size=1,
    )
    collector.validate_phase10b_opd_smoke_behavior_contract(args)
    behavior_job = (ROOT / "scripts" / "run_m6_phase10b_opd_smoke_behavior_job.sh").read_text(encoding="utf-8")
    target_job = (ROOT / "scripts" / "run_m6_phase10b_opd_smoke_target_job.sh").read_text(encoding="utf-8")
    source = (ROOT / "scripts" / "m6_phase10b_opd_smoke.py").read_text(encoding="utf-8")
    assert "--mode phase10b_opd_smoke_behavior --role train --k 4 --seed 20260852" in behavior_job
    assert "--adapter \"$study_root/mini/pilot_sft/final_adapter\"" in behavior_job
    assert "prompt_logprobs=TOPK" in source
    assert "student_behavior_action_unchanged" in source
    assert "--base-model-manifest \"$model_manifest\"" in target_job
    assert "optimizer" not in target_job

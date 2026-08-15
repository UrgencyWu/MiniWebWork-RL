from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import miniwebwork.m6_phase10b_opd as opd
from miniwebwork.long_horizon_rl.contracts import sha256_json


def _task_id(index: int) -> str:
    return f"webshop_goal_{index:05d}"


def _goals() -> list[dict]:
    return [
        {
            "goal_index": index,
            "instruction": f"find public item {index} option {index % 5} under budget",
            "category": ("fashion", "grocery", "electronics", "garden")[index % 4],
            "attributes": [f"attribute-{slot}" for slot in range(3)],
            "goal_options": [f"option-{index % 5}"] if index % 2 == 0 else [],
            "price_upper": 50.0 if index % 3 == 0 else None,
        }
        for index in range(12087)
    ]


def _old_payloads(goals: list[dict]) -> tuple[dict, dict]:
    exposures = []
    for index in range(10):
        exposures.append({
            "goal_index": index,
            "task_id": _task_id(index),
            "normalized_instruction_sha256": sha256_json(opd.normalized_instruction(goals[index]["instruction"])),
            "reasons": ["historical"],
            "source_labels": ["historical"],
        })
    old = {"exposures": exposures, "content_sha256": "a" * 64}
    cursor = 100
    roles = {}
    for role, count in {
        "teacher_qualification": 32,
        "correction_train": 64,
        "correction_monitor_a": 64,
        "correction_monitor_b": 64,
        "rl_train": 40,
        "dev3": 500,
    }.items():
        task_ids = [_task_id(index) for index in range(cursor, cursor + count)]
        roles[role] = {"task_ids": task_ids}
        cursor += count
    return old, {"roles": roles, "content_sha256": "b" * 64}


def test_phase10b_exposure_and_roles_are_fresh_and_disjoint(monkeypatch: pytest.MonkeyPatch):
    goals = _goals()
    old, split = _old_payloads(goals)
    monkeypatch.setattr(opd, "validate_exposure_union", lambda value: value)
    monkeypatch.setattr(opd, "validate_phase10_split", lambda value: value)
    exposure = opd.build_phase10b_exposure_union(
        goals=goals,
        phase10_exposure=old,
        phase10_split=split,
        producer_git_sha="c" * 40,
    )
    reserved = {task_id for role in split["roles"].values() for task_id in role["task_ids"]}
    assert exposure["phase10a_reserved_task_count"] == len(reserved)
    train_task_ids = [_task_id(index) for index in range(1000, 8000)]
    result = opd.build_phase10b_split(
        goals=goals,
        train_task_ids=train_task_ids,
        exposure_union=exposure,
        base_split_content_sha256="d" * 64,
        producer_git_sha="c" * 40,
    )
    assert {role: item["count"] for role, item in result["roles"].items()} == opd.ROLE_COUNTS
    role_sets = [set(item["task_ids"]) for item in result["roles"].values()]
    assert sum(map(len, role_sets)) == len(set().union(*role_sets))
    assert not (reserved & set().union(*role_sets))
    assert result["roles"]["specialist_nav_qualification"]["selection_proxy"] == "specialist_nav_qualification"
    opd.validate_phase10b_split(result)


def _write_model(root: Path, *, tokenizer: str, tokenizer_config: str) -> None:
    root.mkdir()
    (root / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    (root / "tokenizer_config.json").write_text(tokenizer_config, encoding="utf-8")
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 256,
            "num_hidden_layers": 2,
        },
    }), encoding="utf-8")


def test_model_manifest_allows_config_difference_but_requires_common_vocab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    specs = {}
    for offset, identity in enumerate(opd.MODEL_SPECS):
        root = tmp_path / identity
        _write_model(root, tokenizer="same-tokenizer", tokenizer_config=f"config-{offset // 3}")
        specs[identity] = {**opd.MODEL_SPECS[identity], "path": str(root)}
    monkeypatch.setattr(opd, "MODEL_SPECS", specs)
    report = opd.build_model_tokenizer_manifest(producer_git_sha="a" * 40)
    assert report["checks"]["common_tokenizer_json"] is True
    assert report["checks"]["common_vocab_size"] is True
    assert len({row["tokenizer_config_sha256"] for row in report["models"]}) == 2


def test_model_manifest_rejects_tokenizer_index_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    specs = {}
    for offset, identity in enumerate(opd.MODEL_SPECS):
        root = tmp_path / identity
        _write_model(root, tokenizer=f"tokenizer-{offset}", tokenizer_config="same-config")
        specs[identity] = {**opd.MODEL_SPECS[identity], "path": str(root)}
    monkeypatch.setattr(opd, "MODEL_SPECS", specs)
    with pytest.raises(ValueError, match="tokenizer.json is not common"):
        opd.build_model_tokenizer_manifest(producer_git_sha="a" * 40)


def test_teacher_to_student_kl_updates_only_student():
    student = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    specialist = torch.tensor([[0.0, 2.0, -1.0]])
    value = opd.teacher_to_student_kl(student, specialist, temperature=1.0)
    assert value.item() > 0
    value.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert specialist.grad is None


def test_logit_probe_validator_fails_closed_on_one_model():
    rows = [
        {
            "identity": identity,
            "vocab_size": 248320,
            "finite_logits": True,
            "canonical_action_token_ids_match": True,
            "prefix_token_ids_match": True,
            "probability_sum_abs_error": 0.0,
        }
        for identity in opd.MODEL_SPECS
    ]
    rows[-1]["finite_logits"] = False
    report = {
        "schema_version": opd.LOGIT_PROBE_SCHEMA,
        "training_performed": False,
        "optimizer_steps": 0,
        "student_tokenizer_used_for_all_models": True,
        "models": rows,
        "all_models_pass": False,
    }
    report["content_sha256"] = sha256_json(report)
    with pytest.raises(ValueError, match="non-finite"):
        opd.validate_logit_probe(report)


def test_logit_model_report_binds_student_adapter_and_self_hash():
    report = {
        "schema_version": opd.LOGIT_MODEL_SCHEMA,
        "training_performed": False,
        "optimizer_steps": 0,
        "identity": "student",
        "model_path": opd.MODEL_SPECS["student"]["path"],
        "input_adapter": opd.PI0_ADAPTER_PATH,
        "student_tokenizer_used": True,
        "vocab_size": 248320,
        "finite_logits": True,
        "canonical_action_token_ids_match": True,
        "prefix_token_ids_match": True,
        "probability_sum_abs_error": 1e-7,
    }
    report["content_sha256"] = sha256_json(report)
    assert opd.validate_logit_model_report(report)["identity"] == "student"
    changed = dict(report, input_adapter=None)
    changed["content_sha256"] = sha256_json({key: value for key, value in changed.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="adapter drift"):
        opd.validate_logit_model_report(changed)


def test_s_finish_fp8_kernel_trust_is_scoped_to_frozen_mapping():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "m6_phase10b_logit_alignment_probe.py").read_text(
        encoding="utf-8"
    )
    assert '"repo_id": "kernels-community/finegrained-fp8", "version": 4' in source
    assert "with kernel_context:" in source
    assert "allow_all_hub_kernels()" in source


def test_s_finish_has_native_vllm_alignment_fallback_without_hub_kernel():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "m6_phase10b_logit_alignment_probe.py").read_text(encoding="utf-8")
    wrapper = (root / "scripts" / "run_m6_phase10b_logit_alignment_job.sh").read_text(encoding="utf-8")
    assert '"backend": "vllm_native_fp8"' in source
    assert '"topk_plus_rest_probability_mass_safe"' in source
    assert "probe_command=vllm-model" in wrapper
    assert "runtime_deps/kernels" not in wrapper

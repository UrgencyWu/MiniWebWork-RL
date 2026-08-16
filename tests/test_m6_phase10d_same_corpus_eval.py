from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/m6_monotonic_posttraining_v1"


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_roster_inputs():
    builder = _module(
        "m6_phase10d_same_corpus_rosters",
        "scripts/m6_phase10d_build_same_corpus_eval_roster.py",
    )
    sys.path.insert(0, str(ROOT / "src"))
    from miniwebwork.m6_phase10c_data import (  # noqa: E402
        validate_phase10b_exposure_union,
        validate_phase10c_exposure_union,
        validate_phase10c_split,
    )
    from miniwebwork.m6_posttraining_protocol import (  # noqa: E402
        validate_exposure_registry,
        validate_split_lock,
    )

    def read(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    base = validate_split_lock(read(STUDY / "locks/m6_webshop_split_v1.json"))
    goals = read(ROOT / "outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json")
    phase10c_split = validate_phase10c_split(
        read(STUDY / "phase10c_qwen35_sft_specialist_opd_v1/split_lock.json")
    )
    phase10c_roles: set[str] = set()
    for role in phase10c_split["roles"].values():
        phase10c_roles.update(role["task_ids"])
    exclusions = [
        {
            "name": "m5_goal_exposure_registry_v1",
            "paths": [STUDY / "locks/m5_goal_exposure_registry_v1.json"],
            "file_sha256": ["a" * 64],
            "task_ids": builder._validate_exposure_rows(
                validate_exposure_registry(read(STUDY / "locks/m5_goal_exposure_registry_v1.json")),
                "M5 exposure registry",
            ),
        },
        {
            "name": "phase10_exposure_union",
            "paths": [STUDY / "phase10_student_state_teacher_correction_v1/phase10_exposure_union.json"],
            "file_sha256": ["a" * 64],
            "task_ids": builder._validate_exposure_rows(
                builder._validate_phase10_exposure_union(
                    read(STUDY / "phase10_student_state_teacher_correction_v1/phase10_exposure_union.json")
                ),
                "Phase10 exposure union",
            ),
        },
        {
            "name": "phase10b_exposure_union",
            "paths": [STUDY / "phase10b_multi_specialist_opd_v1/exposure_union.json"],
            "file_sha256": ["a" * 64],
            "task_ids": builder._validate_exposure_rows(
                validate_phase10b_exposure_union(
                    read(STUDY / "phase10b_multi_specialist_opd_v1/exposure_union.json")
                ),
                "Phase10-B exposure union",
            ),
        },
        {
            "name": "phase10c_exposure_union",
            "paths": [STUDY / "phase10c_qwen35_sft_specialist_opd_v1/exposure_union.json"],
            "file_sha256": ["a" * 64],
            "task_ids": builder._validate_exposure_rows(
                validate_phase10c_exposure_union(
                    read(STUDY / "phase10c_qwen35_sft_specialist_opd_v1/exposure_union.json")
                ),
                "Phase10-C exposure union",
            ),
        },
        {
            "name": "phase10c_split_all_roles",
            "paths": [STUDY / "phase10c_qwen35_sft_specialist_opd_v1/split_lock.json"],
            "file_sha256": ["a" * 64],
            "task_ids": phase10c_roles,
        },
        {
            "name": "sft_d4_corpus",
            "paths": [STUDY / "mini/corpus_v2/train.jsonl", STUDY / "mini/corpus_v2/dev.jsonl"],
            "file_sha256": ["a" * 64, "a" * 64],
            "task_ids": builder._corpus_task_ids(STUDY / "mini/corpus_v2/train.jsonl")
            | builder._corpus_task_ids(STUDY / "mini/corpus_v2/dev.jsonl"),
        },
        {
            "name": "sft_d35_corpus",
            "paths": [
                STUDY / "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/replay_weighted_corpus_v1/train.jsonl",
                STUDY / "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/replay_weighted_corpus_v1/dev.jsonl",
            ],
            "file_sha256": ["a" * 64, "a" * 64],
            "task_ids": builder._corpus_task_ids(
                STUDY / "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/replay_weighted_corpus_v1/train.jsonl"
            )
            | builder._corpus_task_ids(
                STUDY / "phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/replay_weighted_corpus_v1/dev.jsonl"
            ),
        },
    ]
    return builder, base, goals, exclusions


def test_same_corpus_roster_is_fresh_stratified_and_deterministic():
    builder, base, goals, exclusions = _load_roster_inputs()
    kwargs = dict(
        goals=goals,
        base_split=base,
        exclusions=exclusions,
        protocol_sha256="c" * 64,
        git_sha="d" * 40,
    )
    roster = builder.build_roster(**kwargs)
    assert roster["task_count"] == 96
    assert roster["phase10c_evaluation_stage"] == "same_corpus"
    assert roster["purpose"] == "phase10d_same_corpus_scale_paired_evaluation"
    excluded_union = set()
    for item in exclusions:
        excluded_union.update(item["task_ids"])
    selected = set(roster["task_ids"])
    train = set(base["roles"]["train"]["task_ids"])
    assert selected.isdisjoint(excluded_union)
    assert selected <= train
    assert roster["excluded_from_train_task_count"] == len(excluded_union & train)
    assert sum(roster["category_task_counts"].values()) == 96
    cells = roster["stratification"]
    assert sum(cell["selected_task_count"] for cell in cells.values()) == 96
    for key, cell in cells.items():
        assert cell["selected_task_count"] <= cell["pool_task_count"]
        assert cell["category"] == key.split("|", 1)[0]
        assert cell["constraint_bucket"] == int(key.split("|", 1)[1])
    assert roster["K"] == 4
    assert roster["max_model_turns"] == 18
    assert roster["max_environment_steps"] == 15
    assert roster["rollout_seed"] == 20260867
    assert roster["checks"]["selected_has_excluded_overlap"] == 0
    assert roster["checks"]["promotion_and_holdout_read"] is False
    assert builder.build_roster(**kwargs)["content_sha256"] == roster["content_sha256"]


def test_phase10d_sft35_d4_contract_freezes_merged_identity():
    pytest.importorskip("playwright")
    collector = _module(
        "m6_collect_phase10d_same_corpus_eval",
        "scripts/m6_collect_policy_success.py",
    )
    common = dict(
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
    )
    arm = argparse.Namespace(
        **common,
        phase10c_evaluation_identity="sft35_d4",
        base_model=collector.PHASE10D_SFT35_D4_MERGED_MODEL,
        adapter=None,
        tensor_parallel_size=2,
    )
    collector.validate_phase10c_teacher_stage_evaluation_contract(arm)
    arm.adapter = collector.PHASE10B_STUDENT_ADAPTER
    with pytest.raises(ValueError, match="SFT35 identity"):
        collector.validate_phase10c_teacher_stage_evaluation_contract(arm)
    arm.adapter = None
    arm.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="SFT35 identity"):
        collector.validate_phase10c_teacher_stage_evaluation_contract(arm)


def test_same_corpus_eval_wrapper_freezes_merge_and_budget():
    source = (ROOT / "scripts/run_m6_phase10d_same_corpus_eval_job.sh").read_text()
    assert "#SBATCH --gres=gpu:2" in source
    assert "--mode phase10c_teacher_stage_evaluation --role train --k 4" in source
    assert "--max-model-turns 18 --max-environment-steps 15" in source
    assert "sft4)" in source and "sft35_d4)" in source
    assert "M6_MIN_FREE_MIB_PER_GPU" in source
    assert "refusing to start" in source
    assert 'merge_device="$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)"' in source
    assert "assigned_devices=" in source
    assert "--expected-lora-r 16" in source
    assert "--expected-lora-alpha 32" in source
    assert "--expected-target-count 310" in source
    assert "seed=20260867" in source
    assert "formal_v1/merged_model_v1" in source


def test_same_corpus_pairing_and_seed_are_frozen():
    stats = _module(
        "m6_phase10d_same_corpus_stats",
        "scripts/m6_phase10c_teacher_stage_eval_stats.py",
    )
    assert stats.PAIR_BY_STAGE["same_corpus"] == ("sft4", "sft35_d4")
    assert stats.EXPECTED_SEEDS["same_corpus"] == 20260867
    assert stats.EXPECTED_SEEDS["dev"] == 20260864
    assert stats.EXPECTED_SEEDS["qualification"] == 20260865

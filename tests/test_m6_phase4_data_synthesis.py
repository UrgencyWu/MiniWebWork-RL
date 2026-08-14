from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from miniwebwork.long_horizon_rl.contracts import sha256_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module(name: str):
    return importlib.import_module(name)


def _task_id(index: int) -> str:
    return f"webshop_goal_{index:05d}"


def _goal(index: int) -> dict:
    return {
        "goal_index": index,
        "category": ("fashion", "grocery", "electronics", "garden")[index % 4],
        "attributes": [f"attribute-{slot}" for slot in range(index % 4)],
        "goal_options": ["blue"] if index % 3 == 0 else [],
        "price_upper": 50.0 if index % 2 == 0 else None,
    }


def test_phase4_partition_selection_is_deterministic_disjoint_and_stratified():
    module = _module("m6_phase4_build_data_rosters")
    goals = {_task_id(index): _goal(index) for index in range(1000, 1400)}
    eligible = list(goals)
    first, _, _, first_gap = module._select_partition(eligible, goals, seed=20260824)
    repeated, _, _, _ = module._select_partition(eligible, goals, seed=20260824)
    remaining = [task_id for task_id in eligible if task_id not in set(first)]
    second, _, _, second_gap = module._select_partition(remaining, goals, seed=20260825)
    assert first == repeated
    assert len(first) == len(second) == 64
    assert not (set(first) & set(second))
    assert first_gap <= 0.05
    assert second_gap <= 0.05


def test_phase4_roster_builder_excludes_sft_prior_and_nontrain_tasks(tmp_path: Path, monkeypatch):
    module = _module("m6_phase4_build_data_rosters")
    goals = [_goal(index) for index in range(1000, 1400)]
    goal_path = tmp_path / "goals.json"
    goal_path.write_text(json.dumps(goals), encoding="utf-8")
    sft_ids = {_task_id(index) for index in range(1000, 1156)}
    prior_ids = {_task_id(index) for index in range(1156, 1220)}
    train_ids = [_task_id(index) for index in range(1000, 1400)]
    split = {
        "content_sha256": "s" * 64,
        "protocol_sha256": "p" * 64,
        "goals_canonical_sha256": sha256_json(goals),
        "roles": {
            "train": {"task_ids": train_ids},
            "mini_train": {"task_ids": sorted(sft_ids)},
            "promotion": {"task_ids": []},
            "holdout": {"task_ids": []},
        },
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    sft_train = tmp_path / "sft_train.jsonl"
    sft_dev = tmp_path / "sft_dev.jsonl"
    ordered_sft = sorted(sft_ids)
    sft_train.write_text("\n".join(json.dumps({"task_id": value}) for value in ordered_sft[:100]) + "\n", encoding="utf-8")
    sft_dev.write_text("\n".join(json.dumps({"task_id": value}) for value in ordered_sft[100:]) + "\n", encoding="utf-8")
    prior = {
        "schema_version": "m6_rl_curriculum_v1",
        "task_ids": sorted(prior_ids),
    }
    prior["content_sha256"] = sha256_json(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    output = tmp_path / "rosters"

    monkeypatch.setattr(module, "validate_split_lock", lambda value: value)
    monkeypatch.setattr(module, "load_protocol", lambda: {
        "sha256": "p" * 64,
        "git_sha": "a" * 40,
        "payload": {"study_id": "fixture"},
    })
    monkeypatch.setattr(sys, "argv", [
        "m6_phase4_build_data_rosters.py",
        "--split-lock", str(split_path),
        "--goals", str(goal_path),
        "--sft-jsonl", str(sft_train),
        "--sft-jsonl", str(sft_dev),
        "--exclude-roster", str(prior_path),
        "--output-dir", str(output),
    ])
    module.main()

    first = json.loads((output / "roster_a.json").read_text(encoding="utf-8"))
    second = json.loads((output / "roster_b.json").read_text(encoding="utf-8"))
    selected = set(first["task_ids"]) | set(second["task_ids"])
    assert len(selected) == 128
    assert not (set(first["task_ids"]) & set(second["task_ids"]))
    assert not (selected & sft_ids)
    assert not (selected & prior_ids)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["partition_overlap_count"] == 0
    assert manifest["sft_overlap_count"] == 0
    assert manifest["prior_roster_overlap_count"] == 0


def test_phase4_collector_contract_is_full_horizon_sft_student_and_zero_update():
    pytest.importorskip("playwright")
    module = _module("m6_collect_policy_success")
    args = Namespace(
        role="train",
        task_roster=Path("roster.json"),
        adapter=Path("sft_adapter"),
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
    )
    module.validate_phase4_data_synthesis_contract(args)
    args.max_model_turns = 6
    with pytest.raises(ValueError, match="full evaluation horizon"):
        module.validate_phase4_data_synthesis_contract(args)
    args.max_model_turns = 18
    args.maximum_action_tokens = 10_000
    with pytest.raises(ValueError, match="cannot claim a training token budget"):
        module.validate_phase4_data_synthesis_contract(args)


def test_phase4_teacher_collector_contract_is_raw_qwen9b_and_zero_update():
    pytest.importorskip("playwright")
    module = _module("m6_collect_policy_success")
    args = Namespace(
        role="train",
        task_roster=Path("teacher_roster.json"),
        adapter=None,
        base_model=Path("/data/share/model/Qwen3.5-9B"),
        max_model_turns=18,
        max_environment_steps=15,
        maximum_tasks=None,
        task_offset=0,
        maximum_action_tokens=None,
        replay_prefix_root=None,
    )
    module.validate_phase4_teacher_probe_contract(args)
    args.adapter = Path("student_adapter")
    with pytest.raises(ValueError, match="must not reuse the student adapter"):
        module.validate_phase4_teacher_probe_contract(args)
    args.adapter = None
    args.base_model = Path("/data/share/model/Qwen3.5-4B")
    with pytest.raises(ValueError, match="teacher model drift"):
        module.validate_phase4_teacher_probe_contract(args)


def test_phase4_teacher_selection_is_deterministic_and_failure_stratified():
    module = _module("m6_phase4_build_teacher_roster")
    rows = []
    for bucket_index in range(8):
        failure_class = ("partial_purchase", "zero_match_purchase", "horizon_exhaustion")[
            bucket_index % 3
        ]
        for row_index in range(6):
            row = {
                "task_id": _task_id(4000 + 10 * bucket_index + row_index),
                "bucket": f"category_{bucket_index % 4}|constraints_{2 + bucket_index // 4}",
                "failure_classes": {failure_class: 4},
            }
            rows.append(row)
    first, population, selected_counts, maximum_gap = module._select(rows)
    repeated, _, _, _ = module._select(rows)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in repeated]
    assert len(first) == len({row["task_id"] for row in first}) == 16
    assert len(population) == len(selected_counts) == 8
    assert set(selected_counts.values()) == {2}
    assert maximum_gap == 0.0


def test_phase4_teacher_supplement_is_distinct_and_capped_per_task():
    module = _module("m6_phase4_audit_teacher_probe")
    rows = []
    for task_index in range(2):
        task_id = _task_id(5000 + task_index)
        for trajectory_index, sequence_sha in enumerate(("a" * 64, "a" * 64, "b" * 64, "c" * 64)):
            rows.append({
                "task_id": task_id,
                "trajectory_id": f"{task_id}-{trajectory_index}",
                "environment_steps": trajectory_index + 1,
                "generated_action_tokens": 10 + trajectory_index,
                "normalized_command_sequence_sha256": sequence_sha,
            })
    selected = module._select_distinct(rows)
    assert len(selected) == 4
    for task_index in range(2):
        task_id = _task_id(5000 + task_index)
        task_rows = [row for row in selected if row["task_id"] == task_id]
        assert len(task_rows) == 2
        assert len({row["normalized_command_sequence_sha256"] for row in task_rows}) == 2


@pytest.mark.parametrize(
    ("rewards", "expected_type", "expected_flag"),
    [
        ([1.0, 0.0, 1.0, 0.0], "mixed", "future_online_recollection_candidate"),
        ([1.0, 1.0, 1.0, 1.0], "all_success", "retention_and_cost_candidate"),
        ([0.0, 0.0, 0.0, 0.0], "all_failure", "conditional_teacher_supplement_candidate"),
    ],
)
def test_phase4_three_way_subset_index_preserves_policy_roles(rewards, expected_type, expected_flag):
    module = _module("m6_phase4_audit_student_prescan")
    rows = [{"strict_reward": value} for value in rewards]
    group = {
        "group_id": "g0000",
        "task_id": "webshop_goal_02000",
        "content_sha256": "f" * 64,
    }
    result = module._subset_index_row(partition="a", group=group, rows=rows)
    assert result["group_type"] == expected_type
    assert result[expected_flag] is True
    assert result["prescan_rollouts_reusable_as_online_batches_after_policy_update"] is False
    assert result["teacher_generated_rollouts_allowed_for_on_policy_grpo"] is False


def test_phase4_jobs_and_audit_freeze_resources_dependency_and_teacher_separation():
    job = (SCRIPTS / "run_m6_phase4_student_prescan_job.sh").read_text(encoding="utf-8")
    audit_job = (SCRIPTS / "run_m6_phase4_student_prescan_audit_job.sh").read_text(encoding="utf-8")
    submit = (SCRIPTS / "submit_m6_phase4_student_prescan.sh").read_text(encoding="utf-8")
    audit = (SCRIPTS / "m6_phase4_audit_student_prescan.py").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1%2" in job
    assert "#SBATCH --gres=gpu:1" in job
    assert "#SBATCH --cpus-per-task=4" in job
    assert "#SBATCH --mem=24G" in job
    assert "--mode phase4_data_synthesis --role train --k 4" in job
    assert "--max-model-turns 18 --max-environment-steps 15" in job
    assert "seed=20260824" in job and "seed=20260825" in job
    assert "#SBATCH --gres=gpu" not in audit_job
    assert 'dependency="afterok:$prescan_job"' in submit
    assert "optimizer.step(" not in audit
    assert '"optimizer_steps": 0' in audit
    assert '"teacher_trajectories_allowed_for_on_policy_grpo": False' in audit
    assert '"future_online_updates_require_current_policy_recollection": True' in audit
    assert 'output_dir / "subset_index.json"' in audit


def test_phase4_teacher_jobs_are_bounded_dependent_and_never_on_policy():
    prep_job = (SCRIPTS / "run_m6_phase4_teacher_prep_job.sh").read_text(encoding="utf-8")
    probe_job = (SCRIPTS / "run_m6_phase4_teacher_probe_job.sh").read_text(encoding="utf-8")
    audit_job = (SCRIPTS / "run_m6_phase4_teacher_probe_audit_job.sh").read_text(encoding="utf-8")
    submit = (SCRIPTS / "submit_m6_phase4_teacher_probe.sh").read_text(encoding="utf-8")
    audit = (SCRIPTS / "m6_phase4_audit_teacher_probe.py").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu" not in prep_job
    assert "#SBATCH --gres=gpu:1" in probe_job
    assert "#SBATCH --cpus-per-task=4" in probe_job
    assert "#SBATCH --mem=32G" in probe_job
    assert "--mode phase4_teacher_probe --role train --k 4" in probe_job
    assert "--base-model /data/share/model/Qwen3.5-9B" in probe_job
    assert "--max-model-turns 18 --max-environment-steps 15" in probe_job
    assert "#SBATCH --gres=gpu" not in audit_job
    assert 'dependency="afterok:$prep_job"' in submit
    assert 'dependency="afterok:$probe_job"' in submit
    assert "optimizer.step(" not in audit
    assert '"optimizer_steps": 0' in audit
    assert '"teacher_data_allowed_for_on_policy_grpo": False' in audit
    assert '"maximum_trajectories_per_task": MAXIMUM_SUPPLEMENT_PER_TASK' in audit

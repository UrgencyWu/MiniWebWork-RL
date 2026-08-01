import importlib.util
import json
import random
from pathlib import Path

import pytest

from miniwebwork.m4_protocol import M4RunConfig, build_m4_run_manifest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "m4_analyze_final.py"
    spec = importlib.util.spec_from_file_location("m4_analyze_final", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_analysis_input_parser_requires_declared_method_seed_matrix_syntax(tmp_path: Path):
    module = _load_module()
    algorithm, seed, path = module._parse_input_spec(f"gspo:20260801:{tmp_path / 'artifact.json'}")
    assert (algorithm, seed, path.name) == ("gspo", 20260801, "artifact.json")
    with pytest.raises(Exception, match="ALGORITHM"):
        module._parse_input_spec("gspo:20260801")


def test_final_analysis_requires_collector_compatible_combined_task_source_hash():
    module = _load_module()
    task_root = ROOT / "data" / "tasks" / "m4_rlvr_v1"
    seed_dir = ROOT / "data" / "seed_m4_rlvr_v1"
    manifest = build_m4_run_manifest(
        M4RunConfig("gspo", 20260801, "final_test"),
        task_root=task_root,
        seed_dir=seed_dir,
    )
    artifact = {
        "schema_version": "3.3",
        "complete": True,
        "study_id": "m4_rlvr_v1",
        "git_sha": manifest["git_sha"],
        "prompt_contract": manifest["prompt_contract"],
        "split": "test",
        "study_seed": 20260801,
        "K": 4,
        "task_source_sha256": manifest["hashes"]["task_source_sha256"],
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "policy": "m4_gspo_seed20260801",
    }

    module._validate_final_artifact(
        artifact, algorithm="gspo", seed=20260801, run_manifest=manifest
    )

    artifact["task_source_sha256"] = manifest["hashes"]["test_public.jsonl"]
    with pytest.raises(ValueError, match="test task hash"):
        module._validate_final_artifact(
            artifact, algorithm="gspo", seed=20260801, run_manifest=manifest
        )

    artifact["task_source_sha256"] = manifest["hashes"]["task_source_sha256"]
    artifact["prompt_contract"] = "browser_agent_v2"
    with pytest.raises(ValueError, match="prompt contract"):
        module._validate_final_artifact(
            artifact, algorithm="gspo", seed=20260801, run_manifest=manifest
        )


def test_final_analysis_rejects_frozen_test_roster_substitution_or_task_type_drift():
    module = _load_module()
    ordered_task_ids = ["M4-TEST-A", "M4-TEST-B"]
    task_types = {"M4-TEST-A": "cheapest_feasible", "M4-TEST-B": "no_feasible_product"}
    shuffled = list(ordered_task_ids)
    random.Random(20260801).shuffle(shuffled)
    artifact = {
        "task_order_seed": 20260801,
        "task_order_sha256": module._task_order_sha256(shuffled),
        "full_task_order_sha256": module._task_order_sha256(shuffled),
        "available_task_count": 2,
        "max_tasks": None,
        "requested_task_count": 2,
        "completed_task_count": 2,
        "stopped_for_action_token_budget": False,
        "records": [
            {
                "task_id": task_id,
                "task_type": task_types[task_id],
                "rollout_index": rollout_index,
            }
            for task_id in ordered_task_ids
            for rollout_index in range(4)
        ],
    }

    assert module._validate_frozen_record_roster(
        artifact,
        seed=20260801,
        ordered_task_ids=ordered_task_ids,
        task_types=task_types,
    )

    artifact["records"][0]["task_type"] = "substituted_type"
    with pytest.raises(ValueError, match="task_type"):
        module._validate_frozen_record_roster(
            artifact,
            seed=20260801,
            ordered_task_ids=ordered_task_ids,
            task_types=task_types,
        )


def test_final_analysis_closes_offline_and_online_adapter_hash_lineage(tmp_path: Path):
    module = _load_module()
    training_root = tmp_path / "runs"

    offline_adapter = training_root / "sft" / "seed_20260801" / "training" / "seed_20260801" / "final_adapter"
    offline_adapter.mkdir(parents=True)
    (offline_adapter / "adapter.bin").write_bytes(b"offline-final")
    (offline_adapter.parent / "metrics.json").write_text(json.dumps({"seed": 20260801}))
    offline_run = offline_adapter.parents[2]
    (offline_run / "resolved_run_manifest.json").write_text(
        json.dumps({"initial_adapter_sha256": "shared-initial"})
    )
    offline = module._expected_final_adapter_lineage("sft", 20260801, training_root)
    assert offline["initial_adapter_sha256"] == "shared-initial"
    assert offline["final_adapter_sha256"] == module._directory_sha256(offline_adapter)

    initial = tmp_path / "initial_adapter"
    initial.mkdir()
    (initial / "adapter.bin").write_bytes(b"initial")
    first_next = training_root / "grpo" / "seed_20260801" / "pass_1" / "update" / "updated_adapter"
    second_next = training_root / "grpo" / "seed_20260801" / "pass_2" / "update" / "updated_adapter"
    first_next.mkdir(parents=True)
    second_next.mkdir(parents=True)
    (first_next / "adapter.bin").write_bytes(b"pass-one")
    (second_next / "adapter.bin").write_bytes(b"pass-two")
    first_artifact = training_root / "grpo" / "seed_20260801" / "pass_1" / "collection" / "collector" / "first.json"
    second_artifact = training_root / "grpo" / "seed_20260801" / "pass_2" / "collection" / "collector" / "second.json"
    first_artifact.parent.mkdir(parents=True)
    second_artifact.parent.mkdir(parents=True)
    first_artifact.write_text(json.dumps({"adapter_sha256": module._directory_sha256(initial)}))
    second_artifact.write_text(json.dumps({"adapter_sha256": module._directory_sha256(first_next)}))
    for index, artifact, source, target in (
        (1, first_artifact, initial, first_next),
        (2, second_artifact, first_next, second_next),
    ):
        report = {
            "source_artifact_sha256": module._sha256(artifact),
            "source_adapter": str(source),
            "source_adapter_sha256": module._directory_sha256(source),
            "next_adapter": str(target),
            "next_adapter_sha256": module._directory_sha256(target),
        }
        report_path = training_root / "grpo" / "seed_20260801" / f"pass_{index}" / "update" / "online_update_report.json"
        report_path.write_text(json.dumps(report))
    summary_path = training_root / "grpo" / "seed_20260801" / "online_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "algorithm": "grpo",
                "seed": 20260801,
                "passes": [
                    {"pass_index": 1, "artifact": str(first_artifact), "next_adapter": str(first_next)},
                    {"pass_index": 2, "artifact": str(second_artifact), "next_adapter": str(second_next)},
                ],
            }
        )
    )
    online = module._expected_final_adapter_lineage("grpo", 20260801, training_root)
    assert online["initial_adapter_sha256"] == module._directory_sha256(initial)
    assert online["final_adapter_sha256"] == module._directory_sha256(second_next)


def test_strict_online_lineage_requires_designated_start_and_completed_standard_updates(tmp_path: Path):
    module = _load_module()
    training_root = tmp_path / "runs"
    run_dir = training_root / "grpo" / "seed_20260801"
    initial = tmp_path / "designated_initial_adapter"
    initial.mkdir()
    (initial / "adapter.bin").write_bytes(b"initial")
    canonical = {
        "path": str(initial.resolve()),
        "sha256": module._directory_sha256(initial),
        "study_manifest_sha256": "study-manifest",
    }
    expected_train_manifest = {"frozen": "train-manifest"}
    first_next = run_dir / "pass_1" / "update" / "updated_adapter"
    second_next = run_dir / "pass_2" / "update" / "updated_adapter"
    first_next.mkdir(parents=True)
    second_next.mkdir(parents=True)
    (first_next / "adapter.bin").write_bytes(b"pass-one")
    (second_next / "adapter.bin").write_bytes(b"pass-two")
    artifacts = []
    for pass_index, adapter in ((1, initial), (2, first_next)):
        artifact = run_dir / f"pass_{pass_index}" / "collection" / "collector" / f"single_probe_{pass_index}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"adapter_sha256": module._directory_sha256(adapter)}))
        artifacts.append(artifact)
    reports = []
    for pass_index, artifact, source, target in (
        (1, artifacts[0], initial, first_next),
        (2, artifacts[1], first_next, second_next),
    ):
        report = {
            "complete": True,
            "passed": True,
            "algorithm_id": "grpo",
            "study_seed": 20260801,
            "pass_index": pass_index,
            "run_manifest": expected_train_manifest,
            "source_artifact": str(artifact),
            "source_artifact_sha256": module._sha256(artifact),
            "source_adapter": str(source),
            "source_adapter_sha256": module._directory_sha256(source),
            "next_adapter": str(target),
            "next_adapter_sha256": module._directory_sha256(target),
        }
        report_path = run_dir / f"pass_{pass_index}" / "update" / "online_update_report.json"
        report_path.write_text(json.dumps(report))
        reports.append((report_path, report))
    (run_dir / "online_run_summary.json").write_text(
        json.dumps(
            {
                "algorithm": "grpo",
                "seed": 20260801,
                "initial_adapter": str(initial),
                "initial_adapter_sha256": canonical["sha256"],
                "canonical_initial_adapter": canonical,
                "passes": [
                    {"pass_index": 1, "artifact": str(artifacts[0]), "next_adapter": str(first_next)},
                    {"pass_index": 2, "artifact": str(artifacts[1]), "next_adapter": str(second_next)},
                ],
            }
        )
    )

    lineage = module._expected_final_adapter_lineage(
        "grpo",
        20260801,
        training_root,
        canonical_initial_adapter=canonical,
        expected_train_manifest=expected_train_manifest,
    )
    assert lineage["initial_adapter_sha256"] == canonical["sha256"]

    wrong_start = {**canonical, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="initial adapter hash"):
        module._expected_final_adapter_lineage(
            "grpo",
            20260801,
            training_root,
            canonical_initial_adapter=wrong_start,
            expected_train_manifest=expected_train_manifest,
        )

    reports[1][1]["complete"] = False
    reports[1][0].write_text(json.dumps(reports[1][1]))
    with pytest.raises(ValueError, match="incomplete or failed"):
        module._expected_final_adapter_lineage(
            "grpo",
            20260801,
            training_root,
            canonical_initial_adapter=canonical,
            expected_train_manifest=expected_train_manifest,
        )


def test_final_analysis_rejects_archived_or_nonstandard_final_artifact_paths(tmp_path: Path):
    module = _load_module()
    final_root = tmp_path / "m4_final_eval"
    valid = final_root / "sft" / "seed_20260801" / "collector" / "single_probe_final.json"
    valid.parent.mkdir(parents=True)
    valid.write_text("{}")
    assert module._require_standard_final_artifact_path(
        valid, algorithm="sft", seed=20260801, final_eval_root=final_root
    ) == valid.resolve()
    archived = tmp_path / "m4_invalidated" / "sft" / "seed_20260801" / "collector" / "single_probe_final.json"
    archived.parent.mkdir(parents=True)
    archived.write_text("{}")
    with pytest.raises(ValueError, match="outside the standard"):
        module._require_standard_final_artifact_path(
            archived, algorithm="sft", seed=20260801, final_eval_root=final_root
        )


def test_final_analysis_requires_sft_fixed_effective_label_target_evidence(tmp_path: Path):
    module = _load_module()
    initial = tmp_path / "initial"
    initial.mkdir()
    adapter = tmp_path / "training" / "seed_20260801" / "final_adapter"
    adapter.mkdir(parents=True)
    source_statistics = {
        "sample_count": 2460,
        "effective_supervision_sample_count": 2460,
        "zero_completion_label_sample_count": 0,
        "zero_completion_label_sample_fraction": 0.0,
        "completion_tokens_per_epoch": 48660,
        "min_nonzero_completion_label_tokens": 8,
        "max_nonzero_completion_label_tokens": 36,
        "at_max_length_sample_count": 0,
        "zero_completion_label_at_max_length_sample_count": 0,
    }
    statistics = {
        "sample_count": 12690,
        "effective_supervision_sample_count": 12690,
        "zero_completion_label_sample_count": 0,
        "zero_completion_label_sample_fraction": 0.0,
        "completion_tokens_per_epoch": 249997,
        "min_nonzero_completion_label_tokens": 8,
        "max_nonzero_completion_label_tokens": 36,
        "at_max_length_sample_count": 0,
        "zero_completion_label_at_max_length_sample_count": 0,
    }
    selection = {
        "mode": "seeded_repeated_full_example_permutations",
        "seed": 20260801,
        "target_supervised_completion_tokens": 250000,
        "realized_supervised_completion_tokens": 249997,
        "shortfall_supervised_completion_tokens": 3,
        "min_nonzero_source_completion_label_tokens": 8,
        "selected_occurrence_count": 12690,
        "complete_examples_only": True,
    }
    semantics = "fixed effective completion-only supervised label-token target"
    audit = {
        "schema_version": "m4_sft_supervision_audit_v2",
        "seed": 20260801,
        "max_length": 6144,
        "num_epochs": 1,
        "statistics": statistics,
        "source_statistics": source_statistics,
        "budget_selection": selection,
        "target_supervised_completion_tokens": 250000,
        "max_zero_completion_label_fraction": 0.0,
        "planned_supervised_completion_tokens": 249997,
        "max_supervised_completion_tokens": 250000,
        "budget_semantics": semantics,
        "initial_adapter": str(initial),
        "initial_adapter_sha256": "initial-hash",
    }
    audit_path = adapter.parent / "supervision_audit.json"
    audit_path.write_text(json.dumps(audit))
    metrics = {
        "seed": 20260801,
        "initial_adapter": str(initial),
        "initial_adapter_sha256": "initial-hash",
        "supervision_audit": str(audit_path),
        "supervision_audit_sha256": module._sha256(audit_path),
        "supervision_budget_semantics": semantics,
        "completion_tokens_per_epoch": 249997,
        "planned_supervised_completion_tokens": 249997,
        "max_supervised_completion_tokens": 250000,
        "target_supervised_completion_tokens": 250000,
        "budget_selection": selection,
        "source_supervision_statistics": source_statistics,
        "max_zero_completion_label_fraction": 0.0,
    }
    run_manifest = {
        "supervision_passes": 1,
        "max_supervised_completion_tokens": 250000,
        "training_budget": {
            "mode": "fixed_effective_completion_label_token_target",
            "target_supervised_completion_tokens": 250000,
            "max_sequence_length": 6144,
            "max_zero_completion_label_fraction": 0.0,
        },
    }
    canonical = {"path": str(initial), "sha256": "initial-hash"}
    result = module._validate_offline_supervision_audit(
        algorithm="sft",
        run_manifest=run_manifest,
        metrics=metrics,
        adapter=adapter,
        canonical_initial_adapter=canonical,
    )
    assert result["mode"] == "fixed_effective_label_target"
    assert result["shortfall_supervised_completion_tokens"] == 3

    audit["budget_selection"]["shortfall_supervised_completion_tokens"] = 8
    audit_path.write_text(json.dumps(audit))
    metrics["supervision_audit_sha256"] = module._sha256(audit_path)
    with pytest.raises(ValueError, match="fixed token-target"):
        module._validate_offline_supervision_audit(
            algorithm="sft",
            run_manifest=run_manifest,
            metrics=metrics,
            adapter=adapter,
            canonical_initial_adapter=canonical,
        )

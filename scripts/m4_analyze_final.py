#!/usr/bin/env python3
"""Build the frozen M4 5x3 final-test statistical report from rollout artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ALL_ALGORITHMS, ONLINE_ALGORITHMS
from miniwebwork.m4_analysis import (
    aggregate_m4_seed_summaries,
    paired_task_permutation_analysis,
    summarize_m4_evaluation,
)
from miniwebwork.m4_protocol import (
    COLLECTED_ACTION_TOKEN_CAP,
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    STUDY_SEEDS,
    assert_m4_canonical_initial_adapter,
    build_m4_run_manifest,
    load_m4_study_manifest,
    m4_adapter_directory_sha256,
    m4_task_roster_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(path: Path) -> str:
    """Match the adapter-directory hash algorithm used by all M4 runners."""
    return m4_adapter_directory_sha256(path)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require_path(value: Any, expected: Path, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} is missing")
    actual = Path(value).expanduser().resolve()
    if actual != expected.resolve():
        raise ValueError(f"{label} does not match the canonical M4 lineage path")
    return actual


def _require_canonical_initial_metadata(
    value: dict[str, Any],
    *,
    canonical_initial_adapter: dict[str, str],
    label: str,
    require_nested_metadata: bool,
) -> None:
    if value.get("initial_adapter_sha256") != canonical_initial_adapter["sha256"]:
        raise ValueError(f"{label} initial adapter hash does not match the designated canonical adapter")
    _require_path(
        value.get("initial_adapter"),
        Path(canonical_initial_adapter["path"]),
        label=f"{label} initial adapter path",
    )
    if not require_nested_metadata:
        return
    nested = value.get("canonical_initial_adapter")
    if not isinstance(nested, dict):
        raise ValueError(f"{label} lacks canonical initial-adapter metadata")
    if nested.get("sha256") != canonical_initial_adapter["sha256"]:
        raise ValueError(f"{label} canonical initial-adapter hash mismatch")
    _require_path(
        nested.get("path"),
        Path(canonical_initial_adapter["path"]),
        label=f"{label} canonical initial-adapter path",
    )
    if nested.get("study_manifest_sha256") != canonical_initial_adapter["study_manifest_sha256"]:
        raise ValueError(f"{label} study-manifest hash mismatch")


def _validate_offline_supervision_audit(
    *,
    algorithm: str,
    run_manifest: dict[str, Any],
    metrics: dict[str, Any],
    adapter: Path,
    canonical_initial_adapter: dict[str, str],
) -> dict[str, Any]:
    """Require realized label-token and truncation evidence for offline runs."""
    if metrics.get("no_signal"):
        if algorithm != "rsft":
            raise ValueError("only RSFT may use the audited no-signal fallback")
        if metrics.get("final_adapter_sha256") != canonical_initial_adapter["sha256"]:
            raise ValueError("RSFT no-signal final adapter must equal the canonical initial adapter")
        return {
            "mode": "no_signal",
            "reason": metrics.get("reason"),
            "final_adapter_sha256": metrics.get("final_adapter_sha256"),
        }

    audit_path = adapter.parent / "supervision_audit.json"
    audit = _load_json_object(audit_path, label="offline supervision audit")
    if audit.get("schema_version") != "m4_sft_supervision_audit_v1":
        raise ValueError("offline supervision audit schema is unsupported")
    if audit.get("seed") != metrics.get("seed"):
        raise ValueError("offline supervision audit seed does not match training metrics")
    _require_canonical_initial_metadata(
        audit,
        canonical_initial_adapter=canonical_initial_adapter,
        label="offline supervision audit",
        require_nested_metadata=False,
    )
    if Path(metrics.get("supervision_audit", "")).expanduser().resolve() != audit_path.resolve():
        raise ValueError("offline metrics do not point to the expected supervision audit")
    if metrics.get("supervision_audit_sha256") != _sha256(audit_path):
        raise ValueError("offline metrics supervision-audit hash mismatch")
    if metrics.get("supervision_budget_semantics") != audit.get("budget_semantics"):
        raise ValueError("offline metrics supervision-budget semantics mismatch")
    statistics = audit.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("offline supervision audit lacks label statistics")
    labels_per_epoch = statistics.get("completion_tokens_per_epoch")
    sample_count = statistics.get("sample_count")
    zero_label_count = statistics.get("zero_completion_label_sample_count")
    zero_label_at_limit = statistics.get("zero_completion_label_at_max_length_sample_count")
    if (
        not isinstance(labels_per_epoch, int)
        or labels_per_epoch <= 0
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or not isinstance(zero_label_count, int)
        or not 0 <= zero_label_count <= sample_count
        or not isinstance(zero_label_at_limit, int)
        or not 0 <= zero_label_at_limit <= zero_label_count
    ):
        raise ValueError("offline supervision statistics are invalid")
    supervision_passes = run_manifest.get("supervision_passes")
    cap = run_manifest.get("max_supervised_completion_tokens")
    planned = audit.get("planned_supervised_completion_tokens")
    if supervision_passes != 2 or cap != COLLECTED_ACTION_TOKEN_CAP:
        raise ValueError("offline supervision plan does not use the preregistered two-pass upper cap")
    if planned != labels_per_epoch * supervision_passes or planned > cap:
        raise ValueError("offline realized supervised-label tokens violate the declared upper cap")
    for key, expected in (
        ("completion_tokens_per_epoch", labels_per_epoch),
        ("planned_supervised_completion_tokens", planned),
        ("max_supervised_completion_tokens", cap),
    ):
        if metrics.get(key) != expected:
            raise ValueError(f"offline metrics {key} does not match the supervision audit")
    return {
        "mode": "supervised",
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "completion_tokens_per_epoch": labels_per_epoch,
        "planned_supervised_completion_tokens": planned,
        "max_supervised_completion_tokens": cap,
        "sample_count": sample_count,
        "zero_completion_label_sample_count": zero_label_count,
        "zero_completion_label_at_max_length_sample_count": zero_label_at_limit,
        "budget_semantics": audit.get("budget_semantics"),
    }


def _expected_final_adapter_lineage(
    algorithm: str,
    seed: int,
    training_root: Path,
    *,
    canonical_initial_adapter: dict[str, str] | None = None,
    expected_train_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close the training-to-frozen-evaluation adapter hash chain on disk.

    In its strict mode (used by ``main``), this verifies the designated start,
    update identity and path containment rather than accepting merely
    hash-consistent artifacts supplied from arbitrary directories.
    """
    strict = canonical_initial_adapter is not None
    if strict and expected_train_manifest is None:
        raise ValueError("strict M4 lineage validation requires the frozen train manifest")
    run_dir = Path(training_root).expanduser().resolve() / algorithm / f"seed_{seed}"
    if algorithm not in ONLINE_ALGORITHMS:
        run_manifest = _load_json_object(
            run_dir / "resolved_run_manifest.json", label="offline resolved run manifest"
        )
        initial_hash = run_manifest.get("initial_adapter_sha256")
        if not isinstance(initial_hash, str) or not initial_hash:
            raise ValueError("offline resolved run manifest lacks initial_adapter_sha256")
        adapter = run_dir / "training" / f"seed_{seed}" / "final_adapter"
        metrics = _load_json_object(adapter.parent / "metrics.json", label="offline training metrics")
        if metrics.get("seed") != seed:
            raise ValueError("offline training metrics seed does not match final analysis request")
        if strict:
            assert canonical_initial_adapter is not None and expected_train_manifest is not None
            _require_canonical_initial_metadata(
                run_manifest,
                canonical_initial_adapter=canonical_initial_adapter,
                label="offline resolved run manifest",
                require_nested_metadata=True,
            )
            _require_canonical_initial_metadata(
                metrics,
                canonical_initial_adapter=canonical_initial_adapter,
                label="offline training metrics",
                require_nested_metadata=False,
            )
            if run_manifest.get("no_signal"):
                if any(run_manifest.get(key) != value for key, value in expected_train_manifest.items()):
                    raise ValueError("RSFT no-signal manifest does not contain the frozen train manifest")
            elif run_manifest.get("run_manifest") != expected_train_manifest:
                raise ValueError("offline resolved run manifest does not contain the frozen train manifest")
            supervision_audit = _validate_offline_supervision_audit(
                algorithm=algorithm,
                run_manifest=run_manifest,
                metrics=metrics,
                adapter=adapter,
                canonical_initial_adapter=canonical_initial_adapter,
            )
        else:
            supervision_audit = None
        final_hash = _directory_sha256(adapter)
        return {
            "verified": True,
            "regime": "offline",
            "initial_adapter": run_manifest.get("initial_adapter"),
            "initial_adapter_sha256": initial_hash,
            "final_adapter": str(adapter.resolve()),
            "final_adapter_sha256": final_hash,
            "resolved_run_manifest_sha256": _sha256(run_dir / "resolved_run_manifest.json"),
            "supervision_audit": supervision_audit,
        }

    summary = _load_json_object(run_dir / "online_run_summary.json", label="online run summary")
    if summary.get("algorithm") != algorithm or summary.get("seed") != seed:
        raise ValueError("online run summary algorithm/seed mismatch")
    if strict:
        assert canonical_initial_adapter is not None and expected_train_manifest is not None
        _require_canonical_initial_metadata(
            summary,
            canonical_initial_adapter=canonical_initial_adapter,
            label="online run summary",
            require_nested_metadata=True,
        )
    passes = summary.get("passes")
    if not isinstance(passes, list) or [item.get("pass_index") for item in passes if isinstance(item, dict)] != [1, 2]:
        raise ValueError("online final lineage requires exactly ordered passes 1 and 2")
    if any(not isinstance(item, dict) for item in passes):
        raise ValueError("online run summary pass entry must be a mapping")
    previous_hash: str | None = canonical_initial_adapter["sha256"] if strict else None
    previous_adapter: Path | None = (
        Path(canonical_initial_adapter["path"]).resolve() if strict else None
    )
    checked_passes: list[dict[str, Any]] = []
    for pass_index, summary_item in enumerate(passes, start=1):
        raw_artifact = summary_item.get("artifact")
        if not isinstance(raw_artifact, str):
            raise ValueError("online run summary collection artifact is missing")
        artifact_path = Path(raw_artifact).expanduser().resolve()
        if strict:
            expected_artifact_dir = run_dir / f"pass_{pass_index}" / "collection" / "collector"
            if (
                artifact_path.parent != expected_artifact_dir.resolve()
                or not artifact_path.name.startswith("single_probe_")
                or artifact_path.suffix != ".json"
            ):
                raise ValueError("online collection artifact is outside the standard collector location")
        artifact = _load_json_object(artifact_path, label=f"online pass {pass_index} collection artifact")
        update_path = run_dir / f"pass_{pass_index}" / "update" / "online_update_report.json"
        update = _load_json_object(update_path, label=f"online pass {pass_index} update report")
        source_hash = artifact.get("adapter_sha256")
        if not isinstance(source_hash, str) or not source_hash:
            raise ValueError("online collection artifact lacks adapter_sha256")
        if previous_hash is not None and source_hash != previous_hash:
            raise ValueError("online pass adapter lineage does not continue from the previous update")
        if strict:
            assert expected_train_manifest is not None and previous_adapter is not None
            if not update.get("complete") or not update.get("passed"):
                raise ValueError("online update report is incomplete or failed")
            if (
                update.get("algorithm_id") != algorithm
                or update.get("study_seed") != seed
                or update.get("pass_index") != pass_index
            ):
                raise ValueError("online update report identity does not match final analysis request")
            if update.get("run_manifest") != expected_train_manifest:
                raise ValueError("online update report run manifest does not match the frozen train manifest")
            _require_path(
                update.get("source_artifact"), artifact_path, label="online update source artifact"
            )
        if update.get("source_artifact_sha256") != _sha256(artifact_path):
            raise ValueError("online update report source artifact hash does not match collection artifact")
        if update.get("source_adapter_sha256") != source_hash:
            raise ValueError("online update source adapter hash does not match collection artifact")
        source_adapter = Path(update.get("source_adapter", "")).expanduser().resolve()
        if strict:
            assert previous_adapter is not None
            _require_path(update.get("source_adapter"), previous_adapter, label="online update source adapter")
        if _directory_sha256(source_adapter) != source_hash:
            raise ValueError("online update source adapter contents do not match its declared hash")
        next_adapter = Path(update.get("next_adapter", "")).expanduser().resolve()
        next_hash = update.get("next_adapter_sha256")
        if strict:
            expected_next = source_adapter if update.get("no_signal") else update_path.parent / "updated_adapter"
            _require_path(update.get("next_adapter"), expected_next, label="online update next adapter")
        if not isinstance(next_hash, str) or _directory_sha256(next_adapter) != next_hash:
            raise ValueError("online update next adapter contents do not match its declared hash")
        if Path(summary_item.get("next_adapter", "")).expanduser().resolve() != next_adapter:
            raise ValueError("online run summary next adapter does not match the update report")
        checked_passes.append(
            {
                "pass_index": pass_index,
                "collection_artifact_sha256": _sha256(artifact_path),
                "source_adapter_sha256": source_hash,
                "next_adapter_sha256": next_hash,
            }
        )
        previous_hash = next_hash
        previous_adapter = next_adapter
    assert previous_hash is not None and previous_adapter is not None
    return {
        "verified": True,
        "regime": "online",
        "initial_adapter": summary.get("initial_adapter") if strict else None,
        "initial_adapter_sha256": checked_passes[0]["source_adapter_sha256"],
        "final_adapter": str(previous_adapter),
        "final_adapter_sha256": previous_hash,
        "passes": checked_passes,
    }


def _load_frozen_task_roster(run_manifest: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    task_dir = Path(run_manifest.get("task_dir", "")).expanduser().resolve()
    public_path = task_dir / "test_public.jsonl"
    if not public_path.is_file():
        raise FileNotFoundError(f"frozen M4 test public roster missing: {public_path}")
    order: list[str] = []
    task_types: dict[str, str] = {}
    for line_number, raw_line in enumerate(public_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        task_id = row.get("task_id")
        task_type = row.get("task_type")
        if not isinstance(task_id, str) or not task_id or not isinstance(task_type, str) or not task_type:
            raise ValueError(f"invalid frozen task roster row: {public_path}:{line_number}")
        if task_id in task_types:
            raise ValueError(f"duplicate frozen task ID: {task_id}")
        order.append(task_id)
        task_types[task_id] = task_type
    expected_count = run_manifest.get("split_manifest", {}).get("task_count")
    if not isinstance(expected_count, int) or len(order) != expected_count:
        raise ValueError("frozen test roster count does not match its split manifest")
    return order, task_types


def _task_order_sha256(task_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()


def _validate_frozen_record_roster(
    artifact: dict[str, Any],
    *,
    seed: int,
    ordered_task_ids: list[str],
    task_types: dict[str, str],
    expected_rollouts_per_task: int = 4,
) -> str:
    """Reject substitution, type drift, duplicate, missing, or reordered test evidence."""
    if artifact.get("task_order_seed") != seed:
        raise ValueError("final artifact task order seed does not match the study seed")
    shuffled = list(ordered_task_ids)
    random.Random(seed).shuffle(shuffled)
    expected_order_hash = _task_order_sha256(shuffled)
    if artifact.get("task_order_sha256") != expected_order_hash:
        raise ValueError("final artifact task order hash does not match the frozen seeded roster")
    if artifact.get("full_task_order_sha256") != expected_order_hash:
        raise ValueError("final artifact full task order hash does not match the frozen seeded roster")
    if artifact.get("available_task_count") != len(ordered_task_ids):
        raise ValueError("final artifact available task count does not match the frozen roster")
    if artifact.get("max_tasks") is not None:
        raise ValueError("final artifact must not truncate the frozen test roster")
    if artifact.get("requested_task_count") != len(ordered_task_ids):
        raise ValueError("final artifact requested task count does not match the frozen roster")
    if artifact.get("completed_task_count") != len(ordered_task_ids):
        raise ValueError("final artifact completed task count does not match the frozen roster")
    if artifact.get("stopped_for_action_token_budget") is not False:
        raise ValueError("final artifact must not stop at a training action-token budget")
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("final artifact records must be a list")
    expected_pairs = {
        (task_id, rollout_index)
        for task_id in ordered_task_ids
        for rollout_index in range(expected_rollouts_per_task)
    }
    observed_pairs: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("final artifact record must be a mapping")
        task_id = record.get("task_id")
        rollout_index = record.get("rollout_index")
        if task_id not in task_types:
            raise ValueError("final artifact contains a task outside the frozen roster")
        if record.get("task_type") != task_types[task_id]:
            raise ValueError("final artifact task_type does not match the frozen public roster")
        if not isinstance(rollout_index, int) or not 0 <= rollout_index < expected_rollouts_per_task:
            raise ValueError("final artifact rollout index is outside the frozen K-way roster")
        pair = (task_id, rollout_index)
        if pair in observed_pairs:
            raise ValueError("final artifact contains a duplicate task/rollout identity")
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        raise ValueError("final artifact task/rollout roster is missing, substituted, or incomplete")
    return m4_task_roster_sha256(ordered_task_ids)


def _require_standard_final_artifact_path(
    path: Path, *, algorithm: str, seed: int, final_eval_root: Path
) -> Path:
    """Accept exactly one artifact from the current standard final-eval tree."""
    path = Path(path).expanduser().resolve()
    collector_dir = (
        Path(final_eval_root).expanduser().resolve() / algorithm / f"seed_{seed}" / "collector"
    )
    if (
        path.parent != collector_dir
        or not path.name.startswith("single_probe_")
        or path.suffix != ".json"
    ):
        raise ValueError("final artifact is outside the standard frozen-evaluation collector path")
    candidates = sorted(collector_dir.glob("single_probe_*.json"))
    if candidates != [path]:
        raise ValueError("final collector directory must contain exactly the supplied artifact")
    return path


def _parse_input_spec(value: str) -> tuple[str, int, Path]:
    try:
        algorithm, raw_seed, raw_path = value.split(":", 2)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input must be ALGORITHM:SEED:ARTIFACT_PATH") from exc
    if algorithm not in ALL_ALGORITHMS:
        raise argparse.ArgumentTypeError(f"unknown M4 algorithm: {algorithm}")
    if seed not in STUDY_SEEDS:
        raise argparse.ArgumentTypeError(f"unknown M4 study seed: {seed}")
    return algorithm, seed, Path(raw_path).expanduser().resolve()


def _validate_final_artifact(
    artifact: dict[str, Any],
    *,
    algorithm: str,
    seed: int,
    run_manifest: dict[str, Any],
    expected_adapter_lineage: dict[str, Any] | None = None,
) -> None:
    if artifact.get("schema_version") != "3.3" or not artifact.get("complete"):
        raise ValueError("final analysis requires complete schema-3.3 artifacts")
    if artifact.get("study_id") != "m4_rlvr_v1" or artifact.get("split") != "test":
        raise PermissionError("final analysis accepts only frozen M4 test artifacts")
    if artifact.get("git_sha") != run_manifest.get("git_sha"):
        raise ValueError("final artifact git SHA does not match the frozen M4 test manifest")
    if artifact.get("study_seed") != seed or artifact.get("K") != 4:
        raise ValueError("final artifact has the wrong study seed or rollout count")
    if artifact.get("task_source_sha256") != run_manifest["hashes"]["task_source_sha256"]:
        raise ValueError("final artifact test task hash does not match the frozen manifest")
    if (artifact.get("temperature"), artifact.get("top_p"), artifact.get("top_k")) != (1.0, 1.0, 0):
        raise ValueError("final artifact sampling settings drifted from the preregistered rule")
    expected_label = f"m4_{algorithm}_seed{seed}"
    if artifact.get("policy") != expected_label:
        raise ValueError("final artifact policy label does not match its declared method/seed")
    if expected_adapter_lineage is not None:
        expected_hash = expected_adapter_lineage["final_adapter_sha256"]
        expected_path = Path(expected_adapter_lineage["final_adapter"]).resolve()
        if artifact.get("adapter_sha256") != expected_hash:
            raise ValueError("final artifact adapter hash does not match its verified training lineage")
        if Path(artifact.get("adapter_path", "")).expanduser().resolve() != expected_path:
            raise ValueError("final artifact adapter path does not match its verified training lineage")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=_parse_input_spec,
        action="append",
        required=True,
        help="Repeat ALGORITHM:SEED:ARTIFACT_PATH for all 15 final artifacts.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--training-root", type=Path, default=Path("outputs/m4_runs"))
    parser.add_argument("--final-eval-root", type=Path, default=Path("outputs/m4_final_eval"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=20_000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError("bootstrap and permutation sample counts must be positive")
    expected = {(algorithm, seed) for algorithm in ALL_ALGORITHMS for seed in STUDY_SEEDS}
    task_root = args.task_root.expanduser().resolve()
    seed_dir = args.seed_dir.expanduser().resolve()
    study_manifest = load_m4_study_manifest(task_root)
    canonical_initial_adapter = assert_m4_canonical_initial_adapter(
        Path(study_manifest["canonical_initial_adapter"]["path"]), task_root=task_root
    )
    final_eval_root = args.final_eval_root.expanduser().resolve()
    supplied: dict[tuple[str, int], Path] = {}
    for algorithm, seed, path in args.input:
        key = (algorithm, seed)
        if key in supplied:
            raise ValueError(f"duplicate final artifact input: {key}")
        if not path.is_file():
            raise FileNotFoundError(f"final artifact not found: {path}")
        supplied[key] = _require_standard_final_artifact_path(
            path, algorithm=algorithm, seed=seed, final_eval_root=final_eval_root
        )
    if set(supplied) != expected:
        missing = sorted(expected - set(supplied))
        extra = sorted(set(supplied) - expected)
        raise ValueError(f"final matrix must contain every method/seed exactly once; missing={missing}, extra={extra}")

    summaries: dict[str, dict[int, dict[str, Any]]] = {algorithm: {} for algorithm in sorted(ALL_ALGORITHMS)}
    rows: list[dict[str, Any]] = []
    for algorithm, seed in sorted(expected):
        path = supplied[(algorithm, seed)]
        artifact = json.loads(path.read_text(encoding="utf-8"))
        run_manifest = build_m4_run_manifest(
            M4RunConfig(algorithm, seed, "final_test"),
            task_root=task_root,
            seed_dir=seed_dir,
        )
        expected_train_manifest = build_m4_run_manifest(
            M4RunConfig(algorithm, seed, "train"),
            task_root=task_root,
            seed_dir=seed_dir,
        )
        lineage = _expected_final_adapter_lineage(
            algorithm,
            seed,
            args.training_root,
            canonical_initial_adapter=canonical_initial_adapter,
            expected_train_manifest=expected_train_manifest,
        )
        _validate_final_artifact(
            artifact,
            algorithm=algorithm,
            seed=seed,
            run_manifest=run_manifest,
            expected_adapter_lineage=lineage,
        )
        frozen_order, frozen_task_types = _load_frozen_task_roster(run_manifest)
        frozen_roster_hash = _validate_frozen_record_roster(
            artifact,
            seed=seed,
            ordered_task_ids=frozen_order,
            task_types=frozen_task_types,
        )
        summary = summarize_m4_evaluation(
            artifact.get("records", []),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=seed,
            reported_wall_seconds=float(artifact.get("elapsed_s", 0.0)),
        )
        if not summary["complete"]:
            raise ValueError(f"final artifact is incomplete: {algorithm}/{seed}: {summary['incomplete_task_ids'][:5]}")
        summary["algorithm_id"] = algorithm
        summary["study_seed"] = seed
        summary["artifact_path"] = str(path)
        summary["artifact_sha256"] = _sha256(path)
        summary["adapter_lineage"] = lineage
        summary["frozen_task_roster_sha256"] = frozen_roster_hash
        summaries[algorithm][seed] = summary
        rows.append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "task_macro_success": summary["primary_task_macro_success"],
                "task_cluster_ci_low": summary["primary_task_cluster_bootstrap_95ci"][0],
                "task_cluster_ci_high": summary["primary_task_cluster_bootstrap_95ci"][1],
                "raw_attempt_success": summary["raw_attempt_success_rate"],
                "action_tokens": summary["cost"]["action_tokens"],
                "model_turns": summary["cost"]["model_turns"],
                "environment_steps": summary["cost"]["environment_steps"],
                "wall_seconds": summary["cost"]["reported_wall_seconds"],
                "artifact_sha256": summary["artifact_sha256"],
                "final_adapter_sha256": lineage["final_adapter_sha256"],
            }
        )

    aggregates = {
        algorithm: aggregate_m4_seed_summaries([summaries[algorithm][seed] for seed in STUDY_SEEDS])
        for algorithm in sorted(ALL_ALGORITHMS)
    }
    initial_adapter_hashes = {
        summary["adapter_lineage"]["initial_adapter_sha256"]
        for by_seed in summaries.values()
        for summary in by_seed.values()
    }
    if initial_adapter_hashes != {canonical_initial_adapter["sha256"]}:
        raise ValueError("final matrix does not start from the designated canonical initial adapter hash")
    frozen_roster_hashes = {
        summary["frozen_task_roster_sha256"]
        for by_seed in summaries.values()
        for summary in by_seed.values()
    }
    if len(frozen_roster_hashes) != 1:
        raise ValueError("final matrix does not share one frozen task roster hash")
    comparisons: dict[str, Any] = {}
    for first, second in combinations(sorted(ALL_ALGORITHMS), 2):
        comparisons[f"{second}_minus_{first}"] = {
            str(seed): paired_task_permutation_analysis(
                summaries[first][seed],
                summaries[second][seed],
                permutation_samples=args.permutation_samples,
                permutation_seed=seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            for seed in STUDY_SEEDS
        }

    report = {
        "schema_version": "m4_final_report_v3",
        "complete": True,
        "matrix": {algorithm: {str(seed): summaries[algorithm][seed] for seed in STUDY_SEEDS} for algorithm in sorted(ALL_ALGORITHMS)},
        "aggregates": aggregates,
        "pairwise_task_clustered_comparisons": comparisons,
        "audit": {
            "common_initial_adapter_sha256": canonical_initial_adapter["sha256"],
            "canonical_initial_adapter": canonical_initial_adapter,
            "frozen_task_roster_sha256": next(iter(frozen_roster_hashes)),
            "adapter_lineage_verified": True,
            "designated_initial_adapter_verified": True,
            "standard_final_artifact_path_verified": True,
            "frozen_record_roster_verified": True,
        },
        "reporting_rule": (
            "Task-level rollouts are clustered; incomplete test artifacts are rejected rather than "
            "zero-filled. Online/RSFT generated action-token caps and SFT completion-label token "
            "upper bounds are reported with their distinct realized-token semantics."
        ),
    }
    output_dir = args.output_dir.expanduser().resolve()
    _write_json(output_dir / "m4_final_report.json", report)
    with (output_dir / "m4_final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"report": str(output_dir / "m4_final_report.json"), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

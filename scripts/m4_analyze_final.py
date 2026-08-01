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
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    STUDY_SEEDS,
    build_m4_run_manifest,
    m4_task_roster_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(path: Path) -> str:
    """Match the adapter-directory hash algorithm used by the rollout probe."""
    path = Path(path).expanduser().resolve()
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError(f"adapter directory has no files: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(_sha256(file).encode("ascii"))
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _expected_final_adapter_lineage(
    algorithm: str, seed: int, training_root: Path
) -> dict[str, Any]:
    """Close the training-to-frozen-evaluation adapter hash chain on disk."""
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
        final_hash = _directory_sha256(adapter)
        return {
            "verified": True,
            "regime": "offline",
            "initial_adapter_sha256": initial_hash,
            "final_adapter": str(adapter.resolve()),
            "final_adapter_sha256": final_hash,
            "resolved_run_manifest_sha256": _sha256(run_dir / "resolved_run_manifest.json"),
        }

    summary = _load_json_object(run_dir / "online_run_summary.json", label="online run summary")
    if summary.get("algorithm") != algorithm or summary.get("seed") != seed:
        raise ValueError("online run summary algorithm/seed mismatch")
    passes = summary.get("passes")
    if not isinstance(passes, list) or [item.get("pass_index") for item in passes] != [1, 2]:
        raise ValueError("online final lineage requires exactly ordered passes 1 and 2")
    previous_hash: str | None = None
    checked_passes: list[dict[str, Any]] = []
    for pass_index, summary_item in enumerate(passes, start=1):
        if not isinstance(summary_item, dict):
            raise ValueError("online run summary pass entry must be a mapping")
        artifact_path = Path(summary_item.get("artifact", "")).expanduser().resolve()
        artifact = _load_json_object(artifact_path, label=f"online pass {pass_index} collection artifact")
        update_path = run_dir / f"pass_{pass_index}" / "update" / "online_update_report.json"
        update = _load_json_object(update_path, label=f"online pass {pass_index} update report")
        source_hash = artifact.get("adapter_sha256")
        if not isinstance(source_hash, str) or not source_hash:
            raise ValueError("online collection artifact lacks adapter_sha256")
        if previous_hash is not None and source_hash != previous_hash:
            raise ValueError("online pass adapter lineage does not continue from the previous update")
        if update.get("source_artifact_sha256") != _sha256(artifact_path):
            raise ValueError("online update report source artifact hash does not match collection artifact")
        if update.get("source_adapter_sha256") != source_hash:
            raise ValueError("online update source adapter hash does not match collection artifact")
        source_adapter = Path(update.get("source_adapter", "")).expanduser().resolve()
        if _directory_sha256(source_adapter) != source_hash:
            raise ValueError("online update source adapter contents do not match its declared hash")
        next_adapter = Path(update.get("next_adapter", "")).expanduser().resolve()
        next_hash = update.get("next_adapter_sha256")
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
    assert previous_hash is not None
    return {
        "verified": True,
        "regime": "online",
        "initial_adapter_sha256": checked_passes[0]["source_adapter_sha256"],
        "final_adapter": str(
            Path(passes[-1]["next_adapter"]).expanduser().resolve()
        ),
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
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=20_000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0 or args.permutation_samples <= 0:
        raise ValueError("bootstrap and permutation sample counts must be positive")
    expected = {(algorithm, seed) for algorithm in ALL_ALGORITHMS for seed in STUDY_SEEDS}
    supplied: dict[tuple[str, int], Path] = {}
    for algorithm, seed, path in args.input:
        key = (algorithm, seed)
        if key in supplied:
            raise ValueError(f"duplicate final artifact input: {key}")
        if not path.is_file():
            raise FileNotFoundError(f"final artifact not found: {path}")
        supplied[key] = path
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
            task_root=args.task_root,
            seed_dir=args.seed_dir,
        )
        lineage = _expected_final_adapter_lineage(algorithm, seed, args.training_root)
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
    if len(initial_adapter_hashes) != 1:
        raise ValueError("final matrix does not share one common initial adapter hash")
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
        "schema_version": "m4_final_report_v2",
        "complete": True,
        "matrix": {algorithm: {str(seed): summaries[algorithm][seed] for seed in STUDY_SEEDS} for algorithm in sorted(ALL_ALGORITHMS)},
        "aggregates": aggregates,
        "pairwise_task_clustered_comparisons": comparisons,
        "audit": {
            "common_initial_adapter_sha256": next(iter(initial_adapter_hashes)),
            "frozen_task_roster_sha256": next(iter(frozen_roster_hashes)),
            "adapter_lineage_verified": True,
            "frozen_record_roster_verified": True,
        },
        "reporting_rule": "Task-level rollouts are clustered; incomplete test artifacts are rejected rather than zero-filled.",
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

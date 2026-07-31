#!/usr/bin/env python3
"""Build the frozen M4 5x3 final-test statistical report from rollout artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ALL_ALGORITHMS
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
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    artifact: dict[str, Any], *, algorithm: str, seed: int, run_manifest: dict[str, Any]
) -> None:
    if artifact.get("schema_version") != "3.3" or not artifact.get("complete"):
        raise ValueError("final analysis requires complete schema-3.3 artifacts")
    if artifact.get("study_id") != "m4_rlvr_v1" or artifact.get("split") != "test":
        raise PermissionError("final analysis accepts only frozen M4 test artifacts")
    if artifact.get("study_seed") != seed or artifact.get("K") != 4:
        raise ValueError("final artifact has the wrong study seed or rollout count")
    if artifact.get("task_source_sha256") != run_manifest["hashes"]["test_public.jsonl"]:
        raise ValueError("final artifact test task hash does not match the frozen manifest")
    if (artifact.get("temperature"), artifact.get("top_p"), artifact.get("top_k")) != (1.0, 1.0, 0):
        raise ValueError("final artifact sampling settings drifted from the preregistered rule")
    expected_label = f"m4_{algorithm}_seed{seed}"
    if artifact.get("policy") != expected_label:
        raise ValueError("final artifact policy label does not match its declared method/seed")


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
        _validate_final_artifact(artifact, algorithm=algorithm, seed=seed, run_manifest=run_manifest)
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
            }
        )

    aggregates = {
        algorithm: aggregate_m4_seed_summaries([summaries[algorithm][seed] for seed in STUDY_SEEDS])
        for algorithm in sorted(ALL_ALGORITHMS)
    }
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
        "schema_version": "m4_final_report_v1",
        "complete": True,
        "matrix": {algorithm: {str(seed): summaries[algorithm][seed] for seed in STUDY_SEEDS} for algorithm in sorted(ALL_ALGORITHMS)},
        "aggregates": aggregates,
        "pairwise_task_clustered_comparisons": comparisons,
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

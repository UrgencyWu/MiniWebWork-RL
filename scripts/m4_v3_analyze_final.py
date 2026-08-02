#!/usr/bin/env python3
"""Build the v3 5x3 frozen-test report after all lineage gates pass."""

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

from miniwebwork.m4_algorithms import ALL_ALGORITHMS
from miniwebwork.m4_analysis import aggregate_m4_seed_summaries, paired_task_permutation_analysis, summarize_m4_evaluation
from miniwebwork.m4_protocol import DEFAULT_SEED_DIR, DEFAULT_TASK_ROOT, M4RunConfig, STUDY_SEEDS, m4_adapter_directory_sha256, m4_task_roster_sha256
from miniwebwork.m4_v3_protocol import V3_STUDY_ID, assert_v3_initial_adapter, build_v3_run_manifest, load_v3_study_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _input(value: str) -> tuple[str, int, Path]:
    try:
        algorithm, raw_seed, raw_path = value.split(":", 2)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input must be ALGORITHM:SEED:ARTIFACT_PATH") from exc
    if algorithm not in ALL_ALGORITHMS or seed not in STUDY_SEEDS:
        raise argparse.ArgumentTypeError("input algorithm or seed is not in the v3 matrix")
    return algorithm, seed, Path(raw_path).expanduser().resolve()


def _test_roster(run_manifest: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    path = Path(run_manifest["task_dir"]) / "test_public.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["task_id"] for row in rows]
    types = {row["task_id"]: row["task_type"] for row in rows}
    if len(ids) != 120 or len(set(ids)) != 120:
        raise ValueError("v3 frozen test roster must contain exactly 120 unique tasks")
    return ids, types


def _validate_roster(artifact: dict[str, Any], seed: int, ids: list[str], types: dict[str, str]) -> str:
    expected = list(ids)
    random.Random(seed).shuffle(expected)
    expected_hash = hashlib.sha256("\n".join(expected).encode("utf-8")).hexdigest()
    for key, value in (("task_order_seed", seed), ("task_order_sha256", expected_hash), ("full_task_order_sha256", expected_hash), ("available_task_count", 120), ("requested_task_count", 120), ("completed_task_count", 120), ("max_tasks", None), ("stopped_for_action_token_budget", False)):
        if artifact.get(key) != value:
            raise ValueError(f"frozen artifact roster field {key} is invalid")
    pairs = set()
    for row in artifact.get("records", []):
        task_id = row.get("task_id")
        rollout = row.get("rollout_index")
        if task_id not in types or row.get("task_type") != types[task_id] or not isinstance(rollout, int) or not 0 <= rollout < 4:
            raise ValueError("frozen artifact contains task/type/rollout substitution")
        pair = (task_id, rollout)
        if pair in pairs:
            raise ValueError("frozen artifact contains duplicate task/rollout evidence")
        pairs.add(pair)
    expected_pairs = {(task_id, rollout) for task_id in ids for rollout in range(4)}
    if pairs != expected_pairs:
        raise ValueError("frozen artifact task/rollout roster is incomplete")
    return m4_task_roster_sha256(ids)


def _lineage(algorithm: str, seed: int, training_root: Path, canonical: dict[str, str]) -> dict[str, Any]:
    run_dir = training_root / algorithm / f"seed_{seed}"
    if algorithm in {"rloo", "grpo", "gspo"}:
        summary = _load(run_dir / "online_run_summary.json")
        if summary.get("study_id") != V3_STUDY_ID or summary.get("algorithm") != algorithm or summary.get("seed") != seed or summary.get("complete") is not True:
            raise ValueError("online v3 lineage summary is incomplete or mismatched")
        if summary.get("initial_adapter_sha256") != canonical["sha256"]:
            raise ValueError("online v3 lineage does not start from canonical adapter")
        previous_hash = canonical["sha256"]
        previous_adapter = Path(canonical["path"]).resolve()
        checked = []
        for index, item in enumerate(summary.get("passes", []), start=1):
            if item.get("pass_index") != index:
                raise ValueError("online v3 passes are not ordered")
            artifact_path = Path(item.get("artifact", "")).expanduser().resolve()
            update_path = run_dir / f"pass_{index}" / "update" / "online_update_report.json"
            update = _load(update_path)
            if update.get("study_id") not in (None, V3_STUDY_ID) or not update.get("complete") or not update.get("passed"):
                raise ValueError("online v3 update report is incomplete")
            if update.get("source_artifact_sha256") != _sha256(artifact_path) or update.get("source_adapter_sha256") != previous_hash:
                raise ValueError("online v3 adapter/artifact lineage hash mismatch")
            if Path(update.get("source_adapter", "")).resolve() != previous_adapter:
                raise ValueError("online v3 source adapter path does not continue lineage")
            next_adapter = Path(update.get("next_adapter", "")).expanduser().resolve()
            next_hash = update.get("next_adapter_sha256")
            if not isinstance(next_hash, str) or m4_adapter_directory_sha256(next_adapter) != next_hash:
                raise ValueError("online v3 next adapter hash mismatch")
            if Path(item.get("next_adapter", "")).resolve() != next_adapter:
                raise ValueError("online v3 summary adapter path mismatch")
            checked.append({"pass_index": index, "artifact_sha256": _sha256(artifact_path), "next_adapter_sha256": next_hash})
            previous_adapter, previous_hash = next_adapter, next_hash
        if len(checked) != 2:
            raise ValueError("online v3 lineage requires two passes")
        return {"regime": "online", "initial_adapter_sha256": canonical["sha256"], "final_adapter": str(previous_adapter), "final_adapter_sha256": previous_hash, "passes": checked}
    gate = _load(run_dir / "v3_training_gate.json")
    if gate.get("study_id") != V3_STUDY_ID or gate.get("algorithm") != algorithm or gate.get("seed") != seed:
        raise ValueError("offline v3 gate is incomplete or mismatched")
    adapter = (run_dir / "training" / f"seed_{seed}" / "final_adapter").resolve()
    final_hash = m4_adapter_directory_sha256(adapter)
    manifest = _load(run_dir / "resolved_run_manifest.json")
    if manifest.get("initial_adapter_sha256") != canonical["sha256"] or Path(manifest.get("initial_adapter", "")).resolve() != Path(canonical["path"]).resolve():
        raise ValueError("offline v3 lineage does not start from canonical adapter")
    return {"regime": "offline", "initial_adapter_sha256": canonical["sha256"], "final_adapter": str(adapter), "final_adapter_sha256": final_hash, "training_gate": gate}


def _validate_artifact(artifact: dict[str, Any], algorithm: str, seed: int, manifest: dict[str, Any], lineage: dict[str, Any]) -> None:
    if artifact.get("schema_version") != "3.3" or artifact.get("study_id") != V3_STUDY_ID or artifact.get("split") != "test" or not artifact.get("complete"):
        raise ValueError("v3 final artifact schema/study/split gate failed")
    if artifact.get("git_sha") != manifest.get("git_sha") or artifact.get("prompt_contract") != manifest.get("prompt_contract") or artifact.get("study_seed") != seed or artifact.get("K") != 4:
        raise ValueError("v3 final artifact identity gate failed")
    if artifact.get("policy") != f"m4_v3_{algorithm}_seed{seed}":
        raise ValueError("v3 final artifact policy label mismatch")
    if artifact.get("adapter_sha256") != lineage["final_adapter_sha256"] or Path(artifact.get("adapter_path", "")).resolve() != Path(lineage["final_adapter"]).resolve():
        raise ValueError("v3 final artifact adapter lineage mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=_input, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("outputs/m4_v3_runs"))
    parser.add_argument("--final-eval-root", type=Path, default=Path("outputs/m4_v3_final_eval"))
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    args = parser.parse_args()
    expected = {(algorithm, seed) for algorithm in ALL_ALGORITHMS for seed in STUDY_SEEDS}
    supplied = {(algorithm, seed): path for algorithm, seed, path in args.input}
    if set(supplied) != expected or len(args.input) != len(expected):
        raise ValueError("v3 final analysis requires exactly one artifact for every 5x3 method/seed")
    study = load_v3_study_manifest()
    canonical = assert_v3_initial_adapter(PROJECT_ROOT / study["payload"]["canonical_initial_adapter"]["relative_path"])
    training_root = args.training_root.expanduser().resolve()
    eval_root = args.final_eval_root.expanduser().resolve()
    summaries: dict[str, dict[int, dict[str, Any]]] = {algorithm: {} for algorithm in ALL_ALGORITHMS}
    rows: list[dict[str, Any]] = []
    roster_hashes = set()
    for algorithm, seed in sorted(expected):
        path = supplied[(algorithm, seed)].expanduser().resolve()
        expected_dir = eval_root / algorithm / f"seed_{seed}" / "collection" / "collector"
        if path.parent != expected_dir.resolve() or len(list(expected_dir.glob("single_probe_*.json"))) != 1:
            raise ValueError("v3 final artifact is outside the standard immutable eval directory")
        artifact = _load(path)
        test_manifest = build_v3_run_manifest(M4RunConfig(algorithm, seed, "final_test"), task_root=args.task_root, seed_dir=args.seed_dir)
        lineage = _lineage(algorithm, seed, training_root, canonical)
        _validate_artifact(artifact, algorithm, seed, test_manifest, lineage)
        ids, types = _test_roster(test_manifest)
        roster_hash = _validate_roster(artifact, seed, ids, types)
        roster_hashes.add(roster_hash)
        summary = summarize_m4_evaluation(artifact["records"], bootstrap_samples=args.bootstrap_samples, bootstrap_seed=seed, reported_wall_seconds=float(artifact.get("elapsed_s", 0.0)))
        if not summary.get("complete"):
            raise ValueError(f"incomplete frozen evaluation: {algorithm}/{seed}")
        summary.update({"algorithm_id": algorithm, "study_seed": seed, "artifact_path": str(path), "artifact_sha256": _sha256(path), "adapter_lineage": lineage, "frozen_task_roster_sha256": roster_hash})
        summaries[algorithm][seed] = summary
        cost = summary["cost"]
        rows.append({"algorithm": algorithm, "seed": seed, "success_rate": summary["primary_task_macro_success"], "ci_low": summary["primary_task_cluster_bootstrap_95ci"][0], "ci_high": summary["primary_task_cluster_bootstrap_95ci"][1], "action_tokens": cost["action_tokens"], "model_turns": cost["model_turns"], "environment_steps": cost["environment_steps"], "wall_seconds": cost["reported_wall_seconds"], "artifact_sha256": summary["artifact_sha256"], "final_adapter_sha256": lineage["final_adapter_sha256"]})
    if len(roster_hashes) != 1:
        raise ValueError("v3 final evaluations do not share one frozen roster hash")
    aggregates = {algorithm: aggregate_m4_seed_summaries([summaries[algorithm][seed] for seed in STUDY_SEEDS]) for algorithm in sorted(ALL_ALGORITHMS)}
    comparisons = {}
    for first, second in combinations(sorted(ALL_ALGORITHMS), 2):
        comparisons[f"{second}_minus_{first}"] = {str(seed): paired_task_permutation_analysis(summaries[first][seed], summaries[second][seed], permutation_samples=args.permutation_samples, permutation_seed=seed, bootstrap_samples=args.bootstrap_samples) for seed in STUDY_SEEDS}
    report = {
        "schema_version": "m4_final_report_v3",
        "complete": True,
        "matrix": {algorithm: {str(seed): summaries[algorithm][seed] for seed in STUDY_SEEDS} for algorithm in sorted(ALL_ALGORITHMS)},
        "aggregates": aggregates,
        "pairwise_task_clustered_comparisons": comparisons,
        "audit": {"study_id": V3_STUDY_ID, "common_initial_adapter_sha256": canonical["sha256"], "frozen_task_roster_sha256": next(iter(roster_hashes)), "adapter_lineage_verified": True, "frozen_test_roster_verified": True, "excluded_v2_artifacts": True},
        "reporting_rule": "Task-level outcomes are clustered; incomplete, substituted, or cross-version artifacts are rejected. Action-token and effective supervised-label budgets retain distinct units and are reported separately.",
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "m4_v3_final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "m4_v3_final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"report": str(output / "m4_v3_final_report.json"), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

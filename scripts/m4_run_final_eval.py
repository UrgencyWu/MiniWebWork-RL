#!/usr/bin/env python3
"""Resolve a completed M4 run's final adapter and collect its frozen test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ALL_ALGORITHMS, ONLINE_ALGORITHMS
from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    assert_m4_canonical_initial_adapter,
    build_m4_run_manifest,
    load_m4_study_manifest,
    m4_adapter_directory_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _validate_online_final_summary(
    algorithm: str,
    seed: int,
    run_dir: Path,
    *,
    canonical_initial_adapter: dict[str, str],
    expected_train_manifest: dict[str, Any],
) -> Path:
    """Close enough lineage to prevent frozen collection from starting early."""
    summary = _load_json_object(run_dir / "online_run_summary.json", label="online run summary")
    if summary.get("algorithm") != algorithm or summary.get("seed") != seed:
        raise ValueError("online summary algorithm/seed does not match final evaluation request")
    _require_canonical_initial_metadata(
        summary,
        canonical_initial_adapter=canonical_initial_adapter,
        label="online run summary",
        require_nested_metadata=True,
    )
    passes = summary.get("passes")
    if not isinstance(passes, list) or [item.get("pass_index") for item in passes if isinstance(item, dict)] != [1, 2]:
        raise ValueError("online pass order is incomplete or invalid")
    if any(not isinstance(item, dict) for item in passes):
        raise ValueError("online pass summary entries must be mappings")
    previous_adapter = Path(canonical_initial_adapter["path"]).resolve()
    previous_hash = canonical_initial_adapter["sha256"]
    for pass_index, summary_item in enumerate(passes, start=1):
        artifact_dir = run_dir / f"pass_{pass_index}" / "collection" / "collector"
        artifact_path = _require_path(
            summary_item.get("artifact"), artifact_dir / Path(str(summary_item.get("artifact", ""))).name,
            label=f"online pass {pass_index} collection artifact",
        )
        if artifact_path.parent != artifact_dir.resolve() or not artifact_path.name.startswith("single_probe_") or artifact_path.suffix != ".json":
            raise ValueError("online collection artifact is outside the standard collector location")
        update_path = run_dir / f"pass_{pass_index}" / "update" / "online_update_report.json"
        update = _load_json_object(update_path, label=f"online pass {pass_index} update report")
        if not update.get("complete") or not update.get("passed"):
            raise ValueError("online update report is incomplete or failed")
        if (
            update.get("algorithm_id") != algorithm
            or update.get("study_seed") != seed
            or update.get("pass_index") != pass_index
        ):
            raise ValueError("online update report identity does not match final evaluation request")
        if update.get("run_manifest") != expected_train_manifest:
            raise ValueError("online update report run manifest does not match the frozen train manifest")
        _require_path(update.get("source_artifact"), artifact_path, label="online update source artifact")
        if update.get("source_artifact_sha256") != _sha256(artifact_path):
            raise ValueError("online update source artifact hash mismatch")
        _require_path(update.get("source_adapter"), previous_adapter, label="online update source adapter")
        if update.get("source_adapter_sha256") != previous_hash:
            raise ValueError("online update source adapter hash does not continue the lineage")
        if m4_adapter_directory_sha256(previous_adapter) != previous_hash:
            raise ValueError("online update source adapter contents do not match its declared hash")
        expected_next = previous_adapter if update.get("no_signal") else update_path.parent / "updated_adapter"
        next_adapter = _require_path(update.get("next_adapter"), expected_next, label="online update next adapter")
        next_hash = update.get("next_adapter_sha256")
        if not isinstance(next_hash, str) or m4_adapter_directory_sha256(next_adapter) != next_hash:
            raise ValueError("online update next adapter contents do not match its declared hash")
        _require_path(
            summary_item.get("next_adapter"), next_adapter, label="online summary next adapter"
        )
        previous_adapter, previous_hash = next_adapter, next_hash
    return previous_adapter


def _resolve_final_adapter(
    algorithm: str,
    seed: int,
    training_root: Path,
    *,
    canonical_initial_adapter: dict[str, str] | None = None,
    expected_train_manifest: dict[str, Any] | None = None,
) -> Path:
    """Find the only auditable final adapter for a completed M4 method/seed."""
    run_dir = training_root / algorithm / f"seed_{seed}"
    if algorithm in ONLINE_ALGORITHMS:
        if canonical_initial_adapter is not None and expected_train_manifest is not None:
            adapter = _validate_online_final_summary(
                algorithm,
                seed,
                run_dir,
                canonical_initial_adapter=canonical_initial_adapter,
                expected_train_manifest=expected_train_manifest,
            )
        else:
            summary_path = run_dir / "online_run_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"completed online summary missing: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("algorithm") != algorithm or summary.get("seed") != seed:
                raise ValueError("online summary algorithm/seed does not match final evaluation request")
            passes = summary.get("passes")
            if not isinstance(passes, list) or len(passes) != 2:
                raise ValueError("final evaluation requires exactly two completed online passes")
            if [item.get("pass_index") for item in passes] != [1, 2]:
                raise ValueError("online pass order is incomplete or invalid")
            raw_adapter = passes[-1].get("next_adapter")
            if not isinstance(raw_adapter, str):
                raise ValueError("online summary does not record the final adapter")
            adapter = Path(raw_adapter).expanduser().resolve()
    else:
        adapter = (run_dir / "training" / f"seed_{seed}" / "final_adapter").resolve()
        metrics_path = adapter.parent / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"completed offline metrics missing: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("seed") != seed:
            raise ValueError("offline metrics seed does not match final evaluation request")
        if canonical_initial_adapter is not None:
            resolved_manifest = _load_json_object(
                run_dir / "resolved_run_manifest.json", label="offline resolved run manifest"
            )
            _require_canonical_initial_metadata(
                resolved_manifest,
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
            if expected_train_manifest is None:
                raise ValueError("strict offline final evaluation requires the frozen train manifest")
            if resolved_manifest.get("no_signal"):
                if any(
                    resolved_manifest.get(key) != value
                    for key, value in expected_train_manifest.items()
                ):
                    raise ValueError("RSFT no-signal manifest does not contain the frozen train manifest")
            elif resolved_manifest.get("run_manifest") != expected_train_manifest:
                raise ValueError("offline resolved run manifest does not contain the frozen train manifest")
    if not adapter.is_dir():
        raise FileNotFoundError(f"completed final adapter missing: {adapter}")
    return adapter


def _command(
    config: M4RunConfig, *, adapter: Path, output_dir: Path, task_root: Path, seed_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m4_collect_rollouts.py"),
        "--algorithm", config.algorithm_id,
        "--seed", str(config.seed),
        "--phase", "final_test",
        "--adapter", str(adapter),
        "--output-dir", str(output_dir),
        "--task-root", str(task_root),
        "--seed-dir", str(seed_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ALL_ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("outputs/m4_runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = M4RunConfig(args.algorithm, args.seed, "final_test")
    config.validate(task_root=args.task_root)
    training_root = args.training_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    seed_dir = args.seed_dir.expanduser().resolve()
    study_manifest = load_m4_study_manifest(task_root)
    canonical_initial_adapter = assert_m4_canonical_initial_adapter(
        Path(study_manifest["canonical_initial_adapter"]["path"]), task_root=task_root
    )
    expected_train_manifest = build_m4_run_manifest(
        M4RunConfig(args.algorithm, args.seed, "train"), task_root=task_root, seed_dir=seed_dir
    )
    adapter = _resolve_final_adapter(
        args.algorithm,
        args.seed,
        training_root,
        canonical_initial_adapter=canonical_initial_adapter,
        expected_train_manifest=expected_train_manifest,
    )
    command = _command(
        config, adapter=adapter, output_dir=output_dir, task_root=task_root, seed_dir=seed_dir
    )
    print(
        json.dumps(
            {
                "adapter": str(adapter),
                "canonical_initial_adapter": canonical_initial_adapter,
                "collector_command": command,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one preflighted M4 strict multi-turn rollout collection.

This wrapper is intentionally the only supported M4 path into the historical
collector.  It resolves the versioned task world, checks the frozen-split gate,
writes the immutable run manifest, and then delegates token-level collection to
the already audited M3 implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ALL_ALGORITHMS
from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4_MAX_NEW_TOKENS,
    M4RunConfig,
    ONLINE_PASS_ACTION_TOKEN_CAP,
    RSFT_TRAIN_TASKS_PER_PASS,
    build_m4_run_manifest,
    write_m4_run_manifest,
)


def _collection_seed(study_seed: int, pass_index: int | None) -> int:
    """Derive reproducible but distinct RNG streams for ordered M4 passes."""
    if pass_index is None:
        return study_seed
    digest = hashlib.sha256(f"m4_rlvr_v1:{study_seed}:pass:{pass_index}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _action_token_budget(config: M4RunConfig) -> int | None:
    """Split the fixed generation budget for on-policy and RSFT train passes."""
    return (
        ONLINE_PASS_ACTION_TOKEN_CAP
        if config.phase == "train" and config.algorithm_id != "sft"
        else None
    )


def _collector_command(
    config: M4RunConfig,
    *,
    adapter: Path,
    output_dir: Path,
    task_root: Path,
    seed_dir: Path,
    max_tasks: int | None,
    train_pass_index: int | None,
) -> list[str]:
    if config.phase == "train":
        if train_pass_index not in range(1, config.online_passes + 1):
            raise ValueError(f"train_pass_index must be in [1, {config.online_passes}]")
    elif train_pass_index is not None:
        raise ValueError("train_pass_index is only valid for M4 train collection")
    split = config.split
    k = config.group_size if config.phase == "train" else config.eval_rollouts_per_task
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "m2_3_mini_single_probe.py"),
        "--policy",
        "custom",
        "--policy-label",
        f"m4_{config.algorithm_id}_seed{config.seed}",
        "--adapter",
        str(adapter),
        "--base-model",
        config.base_model,
        "--task-dir",
        str(task_root / split),
        "--seed-dir",
        str(seed_dir),
        "--split",
        split,
        "--task-order-seed",
        str(config.seed),
        "--K",
        str(k),
        "--seed",
        str(_collection_seed(config.seed, train_pass_index)),
        "--study-seed",
        str(config.seed),
        "--temperature",
        str(config.temperature),
        "--top-p",
        str(config.top_p),
        "--top-k",
        str(config.top_k),
        "--max-model-turns",
        str(config.max_model_turns),
        "--max-env-steps",
        str(config.max_environment_steps),
        "--max-new-tokens",
        str(M4_MAX_NEW_TOKENS),
        "--output-dir",
        str(output_dir / "collector"),
        "--study-id",
        "m4_rlvr_v1",
    ]
    if train_pass_index is not None:
        command.extend(["--collection-pass-index", str(train_pass_index)])
    action_token_budget = _action_token_budget(config)
    if action_token_budget is not None:
        command.extend(["--max-collected-action-tokens", str(action_token_budget)])
    if max_tasks is not None:
        command.extend(["--max-tasks", str(max_tasks)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALL_ALGORITHMS),
        required=True,
        help="Collect a preregistered M4 algorithm's train or frozen evaluation rollouts.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--phase", choices=("train", "dev", "final_test"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--train-pass-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("max-tasks must be positive")
    if args.algorithm == "sft" and args.phase == "train":
        raise ValueError("SFT uses the audited offline corpus and cannot collect M4 train rollouts")
    if args.algorithm == "rsft" and args.phase == "train" and args.max_tasks != RSFT_TRAIN_TASKS_PER_PASS:
        raise ValueError(
            f"RSFT train collection requires --max-tasks {RSFT_TRAIN_TASKS_PER_PASS} "
            "to keep both fixed-budget passes on the same roster"
        )
    adapter = args.adapter.expanduser().resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter}")
    output_dir = args.output_dir.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    seed_dir = args.seed_dir.expanduser().resolve()
    config = M4RunConfig(args.algorithm, args.seed, args.phase)
    manifest = build_m4_run_manifest(config, task_root=task_root, seed_dir=seed_dir)
    command = _collector_command(
        config,
        adapter=adapter,
        output_dir=output_dir,
        task_root=task_root,
        seed_dir=seed_dir,
        max_tasks=args.max_tasks,
        train_pass_index=args.train_pass_index,
    )
    manifest["collection"] = {
        "study_seed": config.seed,
        "collection_seed": _collection_seed(config.seed, args.train_pass_index),
        "train_pass_index": args.train_pass_index,
        "max_collected_action_tokens": _action_token_budget(config),
        "task_order_seed": config.seed,
        "max_tasks": args.max_tasks,
    }
    print(json.dumps({"run_manifest": manifest, "collector_command": command}, indent=2))
    if args.dry_run:
        return 0
    write_m4_run_manifest(output_dir / "resolved_run_manifest.json", manifest)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

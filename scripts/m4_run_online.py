#!/usr/bin/env python3
"""Execute two audited M4 online passes for one method/seed on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.m4_algorithms import ONLINE_ALGORITHMS
from miniwebwork.m4_protocol import (
    DEFAULT_SEED_DIR,
    DEFAULT_TASK_ROOT,
    M4RunConfig,
    assert_m4_canonical_initial_adapter,
)


def _single_artifact(directory: Path) -> Path:
    artifacts = sorted((directory / "collector").glob("single_probe_*.json"))
    if len(artifacts) != 1:
        raise ValueError(f"expected exactly one complete collector artifact in {directory}, got {artifacts}")
    return artifacts[0]


def _commands(config: M4RunConfig, *, initial_adapter: Path, output_dir: Path) -> list[dict]:
    """Resolve the ordered collect/update steps without performing GPU work."""
    adapter = initial_adapter
    plan: list[dict] = []
    for pass_index in range(1, config.online_passes + 1):
        pass_dir = output_dir / f"pass_{pass_index}"
        collect = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m4_collect_rollouts.py"),
            "--algorithm", config.algorithm_id,
            "--seed", str(config.seed),
            "--phase", "train",
            "--train-pass-index", str(pass_index),
            "--adapter", str(adapter),
            "--output-dir", str(pass_dir / "collection"),
        ]
        update = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m4_apply_online_update.py"),
            "--algorithm", config.algorithm_id,
            "--seed", str(config.seed),
            "--pass-index", str(pass_index),
            "--artifact", str(pass_dir / "collection" / "collector" / "single_probe_*.json"),
            "--adapter", str(adapter),
            "--output-dir", str(pass_dir / "update"),
        ]
        plan.append({"pass_index": pass_index, "input_adapter": str(adapter), "collect": collect, "update": update})
        # The concrete next adapter is resolved from the update report at runtime.
        adapter = pass_dir / "update" / "updated_adapter"
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ONLINE_ALGORITHMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = M4RunConfig(args.algorithm, args.seed, "train")
    config.validate(task_root=args.task_root)
    canonical_initial_adapter = assert_m4_canonical_initial_adapter(
        args.initial_adapter, task_root=args.task_root
    )
    initial_adapter = Path(canonical_initial_adapter["path"])
    output_dir = args.output_dir.expanduser().resolve()
    plan = _commands(config, initial_adapter=initial_adapter, output_dir=output_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config.__dict__,
                    "canonical_initial_adapter": canonical_initial_adapter,
                    "plan": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    adapter = initial_adapter
    completed: list[dict] = []
    for pass_index in range(1, config.online_passes + 1):
        pass_dir = output_dir / f"pass_{pass_index}"
        collect = plan[pass_index - 1]["collect"]
        subprocess.run(collect + ["--task-root", str(args.task_root), "--seed-dir", str(args.seed_dir)], check=True)
        artifact = _single_artifact(pass_dir / "collection")
        update = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m4_apply_online_update.py"),
            "--algorithm", config.algorithm_id,
            "--seed", str(config.seed),
            "--pass-index", str(pass_index),
            "--artifact", str(artifact),
            "--adapter", str(adapter),
            "--output-dir", str(pass_dir / "update"),
            "--task-root", str(args.task_root),
            "--seed-dir", str(args.seed_dir),
        ]
        subprocess.run(update, check=True)
        update_report = json.loads((pass_dir / "update" / "online_update_report.json").read_text(encoding="utf-8"))
        if not update_report.get("complete") or not update_report.get("passed"):
            raise RuntimeError(f"M4 pass {pass_index} update did not complete")
        adapter = Path(update_report["next_adapter"]).resolve()
        if not adapter.is_dir():
            raise FileNotFoundError(f"M4 pass {pass_index} next adapter missing: {adapter}")
        completed.append({"pass_index": pass_index, "artifact": str(artifact), "next_adapter": str(adapter)})
    (output_dir / "online_run_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "online_run_summary.json").write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "seed": args.seed,
                "initial_adapter": str(initial_adapter),
                "initial_adapter_sha256": canonical_initial_adapter["sha256"],
                "canonical_initial_adapter": canonical_initial_adapter,
                "passes": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

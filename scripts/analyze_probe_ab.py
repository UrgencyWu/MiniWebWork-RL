#!/usr/bin/env python3
"""Analyze two canonical rollout artifacts using paired task/rollout keys."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from miniwebwork.probe_analysis import analyze_probe_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired analysis of rollout A/B artifacts")
    parser.add_argument("--a", required=True, type=Path, help="Policy A artifact")
    parser.add_argument("--b", required=True, type=Path, help="Policy B artifact")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    args = parser.parse_args()

    artifact_a = json.loads(args.a.read_text(encoding="utf-8"))
    artifact_b = json.loads(args.b.read_text(encoding="utf-8"))
    result = analyze_probe_pair(
        artifact_a,
        artifact_b,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

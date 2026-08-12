#!/usr/bin/env python3
"""Enforce the paired mini-SFT > Raw gate before any online RL update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json  # noqa: E402
from miniwebwork.m6_mini import build_mini_sft_gate  # noqa: E402


def _load(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"M6 gate input is not an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--sft-eval", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_mini_sft_gate(
        raw=_load(args.raw_eval),
        sft=_load(args.sft_eval),
        corpus_audit=_load(args.corpus_audit),
        bootstrap_samples=args.bootstrap_samples,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

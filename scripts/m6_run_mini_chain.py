#!/usr/bin/env python3
"""Finalize the development-only Raw -> SFT -> RL M6-mini gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json  # noqa: E402
from miniwebwork.m6_mini import build_mini_chain_report  # noqa: E402


def _json(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"M6 mini artifact is not an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval", type=Path, required=True)
    parser.add_argument("--sft-eval", type=Path, required=True)
    parser.add_argument("--rl-eval", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--rl-audit", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-nonpassing",
        action="store_true",
        help="write the complete diagnostic report and exit zero even when the promotion gate fails",
    )
    args = parser.parse_args()
    report = build_mini_chain_report(
        raw=_json(args.raw_eval),
        sft=_json(args.sft_eval),
        rl=_json(args.rl_eval),
        corpus_audit=_json(args.corpus_audit),
        rl_audit=_json(args.rl_audit),
        bootstrap_samples=args.bootstrap_samples,
    )
    atomic_write_json(args.output, report)
    print(json.dumps({
        "path": str(args.output.expanduser().resolve()),
        "passed": report["passed"],
        "decision": report["decision"],
        "content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))
    if not report["passed"] and not args.allow_nonpassing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze M6 N_eval from historical task-level paired-difference files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json, sha256_file  # noqa: E402
from miniwebwork.m6_power import build_power_report  # noqa: E402


def _input(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("input must be COMPARISON=/path/to/paired.csv")
    return name, Path(path)


def _read(path: Path) -> list[float]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty paired-difference CSV: {resolved}")
    candidate_fields = (
        "success_delta_second_minus_first",
        "success_delta",
        "paired_success_difference",
        "difference",
    )
    field = next((name for name in candidate_fields if name in rows[0]), None)
    if field is None:
        raise ValueError(f"paired-difference field missing: {resolved}")
    return [float(row[field]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=_input, action="append", required=True)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = {name: _read(path) for name, path in args.input}
    report = build_power_report(
        comparisons=comparisons,
        simulations=args.simulations,
        seed=args.seed,
    )
    report["input_files"] = {
        name: {
            "path": str(path.expanduser().resolve()),
            "sha256": sha256_file(path.expanduser().resolve()),
        }
        for name, path in sorted(args.input)
    }
    # Recompute after binding exact input files.
    report.pop("content_sha256")
    from miniwebwork.long_horizon_rl.contracts import sha256_json

    report["content_sha256"] = sha256_json(report)
    publish_immutable_json(args.output, report)
    print(json.dumps({
        "path": str(args.output.expanduser().resolve()),
        "passed": report["passed"],
        "selected_n_eval": report["selected_n_eval"],
        "content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the M6 train/dev/promotion/holdout role lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import build_split_lock  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument(
        "--exposure-registry",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "m6_monotonic_posttraining_v1"
            / "locks"
            / "m5_goal_exposure_registry_v1.json"
        ),
    )
    parser.add_argument("--power-report", type=Path)
    parser.add_argument("--n-eval", type=int, choices=(1000, 1500, 2000), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "m6_monotonic_posttraining_v1"
            / "locks"
            / "m6_webshop_split_v1.json"
        ),
    )
    args = parser.parse_args()
    if args.power_report is None:
        raise ValueError("M6 split publication requires a passing prospective power report")
    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    registry = json.loads(args.exposure_registry.expanduser().resolve().read_text(encoding="utf-8"))
    power_sha256 = None
    if args.power_report is not None:
        from miniwebwork.m6_power import validate_power_report

        power = validate_power_report(
            json.loads(args.power_report.expanduser().resolve().read_text(encoding="utf-8"))
        )
        if power.get("passed") is not True or power.get("selected_n_eval") != args.n_eval:
            raise ValueError("M6 split N_eval is not authorized by the prospective power report")
        power_sha256 = power["content_sha256"]
    report = build_split_lock(
        goals=goals,
        exposure_registry=registry,
        n_eval=args.n_eval,
        power_report_content_sha256=power_sha256,
    )
    publish_immutable_json(args.output, report)
    print(json.dumps({
        "path": str(args.output.expanduser().resolve()),
        "n_eval": report["n_eval"],
        "role_counts": {role: value["count"] for role, value in report["roles"].items()},
        "content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

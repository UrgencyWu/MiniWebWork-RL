#!/usr/bin/env python3
"""Build the fail-closed M5 task-specific exposure registry for M6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import build_exposure_registry  # noqa: E402

SCANNABLE_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
EXCLUDED_PARTS = {
    "upstream",
    "server_environment",
    "rollout_adapter",
    "final_adapter",
    "recovery",
    "telemetry",
}


def _evidence(value: str) -> tuple[str, Path]:
    reason, separator, path = value.partition("=")
    if not separator or not reason or not path:
        raise argparse.ArgumentTypeError("evidence must be REASON=/absolute/or/relative/path")
    return reason, Path(path)


def _expand(values: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for reason, raw in values:
        path = raw.expanduser().resolve()
        if path.is_file():
            output.append((reason, path))
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for candidate in sorted(path.rglob("*")):
            relative = candidate.relative_to(path)
            if (
                candidate.is_file()
                and candidate.suffix.casefold() in SCANNABLE_SUFFIXES
                and not set(relative.parts) & EXCLUDED_PARTS
                and candidate.name != "goals.json"
                and candidate.stat().st_size <= 512 * 1024 * 1024
            ):
                output.append((reason, candidate))
    if not output:
        raise ValueError("M6 exposure evidence expansion is empty")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=_evidence,
        action="append",
        required=True,
        help="Repeat REASON=PATH for result-bearing M5 evidence only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "m6_monotonic_posttraining_v1"
            / "locks"
            / "m5_goal_exposure_registry_v1.json"
        ),
    )
    args = parser.parse_args()
    report = build_exposure_registry(evidence_files=_expand(args.evidence))
    publish_immutable_json(args.output, report)
    print(json.dumps({
        "path": str(args.output.expanduser().resolve()),
        "exposed_goal_count": report["exposed_goal_count"],
        "content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

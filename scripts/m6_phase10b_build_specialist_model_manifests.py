#!/usr/bin/env python3
"""Build or validate full-file manifests for the frozen Phase10-B Specialists."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.model_manifest import (  # noqa: E402
    build_base_model_manifest,
    validate_base_model_manifest,
)

MODELS = {
    "S_nav": Path("/data/share/model/Qwen3.5-9B"),
    "S_match": Path("/data/share/model/Qwen3.5-35B-A3B"),
    "S_finish": Path("/data/share/model/Qwen3.6-35B-A3B-FP8"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-git-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _require(args.producer_git_sha == _git_sha(), "Phase10-B model-manifest Git drift")
    root = args.output_root.expanduser().resolve()
    rows = {}
    for identity, model in MODELS.items():
        destination = root / f"{identity}.json"
        if destination.is_file():
            audit = validate_base_model_manifest(
                path=destination,
                expected_base_model=model,
                verify_files=True,
            )
        else:
            audit = build_base_model_manifest(base_model=model, destination=destination)
            audit = validate_base_model_manifest(
                path=destination,
                expected_base_model=model,
                verify_files=False,
            )
        rows[identity] = {
            "path": audit["path"],
            "manifest_sha256": audit["sha256"],
            "functional_file_set_sha256": audit["payload"]["functional_file_set_sha256"],
            "file_count": audit["payload"]["file_count"],
            "total_bytes": audit["payload"]["total_bytes"],
        }
    print(json.dumps({"models": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze Phase10's explicit historical-exposure source inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_json  # noqa: E402
from miniwebwork.m6_phase10_readiness import freeze_source_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draft_path = args.draft.expanduser().resolve()
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    sources = draft.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Phase10 source draft must contain a sources list")
    report = freeze_source_spec(sources=sources, source_base_dir=draft_path.parent)
    publish_immutable_json(args.output, report)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "source_count": len(report["sources"]),
        "content_sha256": report["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

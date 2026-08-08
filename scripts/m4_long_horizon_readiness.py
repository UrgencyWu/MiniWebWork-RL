#!/usr/bin/env python3
"""Generate the final fail-closed readiness manifest without submitting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from miniwebwork.long_horizon_rl.readiness import (
    READINESS_ROOT,
    build_readiness_manifest,
    write_readiness_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output-dir", type=Path, default=READINESS_ROOT)
    args = parser.parse_args()
    manifest = build_readiness_manifest(expected_git_sha=args.expected_git_sha)
    path = write_readiness_manifest(manifest, args.output_dir)
    print(json.dumps({"path": str(path), "decision": manifest["decision"], "manifest_content_sha256": manifest["manifest_content_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

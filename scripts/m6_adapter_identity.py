#!/usr/bin/env python3
"""Build/revalidate the deterministic vLLM view for one M6 adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.adapter_view import build_vllm_adapter_view  # noqa: E402
from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = args.adapter.expanduser().resolve()
    audit = build_vllm_adapter_view(
        source_adapter=adapter,
        destination=args.view.expanduser().resolve(),
        base_model=args.base_model.expanduser().resolve(),
    )
    report = {
        "schema_version": "m6_adapter_identity_v1",
        "adapter": str(adapter),
        "adapter_sha256": directory_sha256(adapter),
        "rollout_adapter": str(args.view.expanduser().resolve()),
        "rollout_adapter_sha256": audit["view_directory_sha256"],
        "adapter_semantic_sha256": audit["semantic_tensor_sha256"],
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

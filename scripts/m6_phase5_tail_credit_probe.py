#!/usr/bin/env python3
"""Run two frozen-panel, zero-update tail-two-turn gradient counterfactuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import validate_committed_group  # noqa: E402
from miniwebwork.webshop_rl.phase5_tail_credit import probe_tail2_panel  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_panel(collection_root: Path, input_adapter: Path) -> tuple[list[dict], str, dict]:
    root = collection_root.expanduser().resolve()
    report = json.loads((root / "collection_report.json").read_text(encoding="utf-8"))
    _require(
        report.get("complete") is True
        and report.get("mode") == "phase4_online_rl_collection"
        and report.get("role") == "train"
        and report.get("K") == 4
        and report.get("task_count") == 4
        and report.get("training_updates_allowed") is True,
        "Phase5 source collection contract drift",
    )
    adapter = input_adapter.expanduser().resolve()
    _require(report.get("policy_lineage", {}).get("adapter_sha256") == directory_sha256(adapter), "Phase5 behavior adapter drift")
    semantic = str(report.get("policy_lineage", {}).get("adapter_semantic_sha256", ""))
    _require(len(semantic) == 64, "Phase5 behavior semantic identity is missing")
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=4)
        for path in sorted((root / "groups").glob("g*.json"))
    ]
    _require(len(groups) == 4, "Phase5 source group count drift")
    _require(report.get("group_content_sha256") == [group["content_sha256"] for group in groups], "Phase5 group/report drift")
    return groups, semantic, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-collection-root", action="append", type=Path, required=True)
    parser.add_argument("--panel-input-adapter", action="append", type=Path, required=True)
    parser.add_argument("--reference-sft-adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    _require(len(args.panel_collection_root) == len(args.panel_input_adapter) == 2, "Phase5 requires two panels")
    panels = []
    for index, (collection_root, input_adapter) in enumerate(zip(args.panel_collection_root, args.panel_input_adapter)):
        groups, semantic, source_report = _load_panel(collection_root, input_adapter)
        result = probe_tail2_panel(
            groups=groups,
            base_model=args.base_model.expanduser().resolve(),
            input_adapter=input_adapter.expanduser().resolve(),
            input_adapter_semantic_sha256=semantic,
            reference_sft_adapter=args.reference_sft_adapter.expanduser().resolve(),
            seed=args.seed + index,
        )
        result["panel_index"] = index
        result["source_collection_report_sha256"] = source_report["content_sha256"]
        panels.append(result)
    finite_nonzero = all(
        panel[arm]["gradient_finite"] and panel[arm]["gradient_norm"] > 0.0
        for panel in panels
        for arm in ("full", "tail2")
    )
    report = {
        "schema_version": "m6_phase5_tail_credit_probe_v1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "hypothesis": "last_two_action_turn_credit_changes_the_strict_binary_gradient_direction",
        "panels": panels,
        "decision": {
            "finite_nonzero_gradients": finite_nonzero,
            "at_least_one_panel_below_0_98_cosine": any(panel["full_vs_tail2_gradient_cosine"] < 0.98 for panel in panels),
        },
    }
    report["decision"]["tail2_probe_passed"] = all(report["decision"].values())
    report["content_sha256"] = sha256_json(report)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase5 output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

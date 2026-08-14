#!/usr/bin/env python3
"""Compare tail-two credit with buy-excluding preterminal credit without updating."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.phase5_tail_credit import probe_tail2_panel  # noqa: E402
from m6_phase5_tail_credit_probe import _load_panel  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-collection-root", action="append", type=Path, required=True)
    parser.add_argument("--panel-input-adapter", action="append", type=Path, required=True)
    parser.add_argument("--reference-sft-adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    _require(len(args.panel_collection_root) == len(args.panel_input_adapter) == 2, "Phase6 requires two panels")
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
            baseline_window="tail2",
            candidate_window="preterminal1",
        )
        result["panel_index"] = index
        result["source_collection_report_sha256"] = source_report["content_sha256"]
        panels.append(result)
    finite_nonzero = all(
        panel[arm]["gradient_finite"] and panel[arm]["gradient_norm"] > 0.0
        for panel in panels
        for arm in ("tail2", "preterminal1")
    )
    report = {
        "schema_version": "m6_phase6_preterminal_credit_probe_v1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "hypothesis": "excluding_terminal_buy_now_focuses_credit_on_product_and_option_selection",
        "panels": panels,
        "decision": {
            "finite_nonzero_gradients": finite_nonzero,
            "at_least_one_panel_below_0_98_cosine": any(
                panel["tail2_vs_preterminal1_gradient_cosine"] < 0.98 for panel in panels
            ),
        },
    }
    report["decision"]["preterminal_probe_passed"] = all(report["decision"].values())
    report["content_sha256"] = sha256_json(report)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Phase6 output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

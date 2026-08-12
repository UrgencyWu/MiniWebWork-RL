#!/usr/bin/env python3
"""Extract three historical paired-task distributions from M5 K4 groups."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import publish_immutable_bytes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _task_success(groups_dir: Path) -> dict[str, float]:
    output: dict[str, float] = {}
    paths = sorted(groups_dir.expanduser().resolve().glob("*.json"))
    _require(paths, f"M5 group directory is empty: {groups_dir}")
    for path in paths:
        group = json.loads(path.read_text(encoding="utf-8"))
        task_id = group.get("task_id")
        trajectories = group.get("trajectories")
        if not (isinstance(task_id, str) and isinstance(trajectories, list) and len(trajectories) == 4):
            continue
        _require(task_id not in output, f"duplicate M5 task group: {task_id}")
        values = []
        for trajectory in trajectories:
            score = trajectory.get("task_score", trajectory.get("reward"))
            _require(isinstance(score, (int, float)), f"M5 task score is missing: {path}")
            success = trajectory.get("success")
            # Older M5 frozen groups stored dense score in ``reward`` and the
            # strict verifier outcome in ``success``.  Newer ones also carry
            # ``task_score``.  Never reconstruct strict outcome by thresholding
            # the dense fallback when the canonical bool is present.
            _require(isinstance(success, bool), f"M5 strict success is missing: {path}")
            if "task_score" in trajectory:
                _require(success == (float(score) >= 0.999), f"M5 strict success drift: {path}")
            values.append(float(success))
        output[task_id] = fmean(values)
    _require(len(output) >= 100, f"too few M5 paired tasks in {groups_dir}")
    return output


def _aggregate(roots: list[Path]) -> dict[str, float]:
    values = [_task_success(root) for root in roots]
    roster = tuple(sorted(values[0]))
    _require(all(tuple(sorted(item)) == roster for item in values), "M5 historical task roster differs across seeds")
    return {task_id: fmean(item[task_id] for item in values) for task_id in roster}


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    destination = path.expanduser().resolve()
    temporary = destination.with_name(f".{destination.name}.render-{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        publish_immutable_bytes(destination, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-groups", type=Path, required=True)
    parser.add_argument("--sft-groups", type=Path, required=True)
    parser.add_argument("--rl-groups", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = _task_success(args.raw_groups)
    sft = _task_success(args.sft_groups)
    rl = _aggregate(args.rl_groups)
    roster = tuple(sorted(raw))
    _require(tuple(sorted(sft)) == tuple(sorted(rl)) == roster, "M5 Raw/SFT/RL task rosters differ")
    output = args.output_dir.expanduser().resolve()
    comparisons = {
        "raw_sft": (raw, sft),
        "sft_rl": (sft, rl),
        "raw_rl": (raw, rl),
    }
    for name, (first, second) in comparisons.items():
        rows = [
            {
                "task_id": task_id,
                "first_task_success": first[task_id],
                "second_task_success": second[task_id],
                "success_delta_second_minus_first": second[task_id] - first[task_id],
            }
            for task_id in roster
        ]
        _atomic_csv(output / f"{name}.csv", rows)
    print(json.dumps({"output_dir": str(output), "task_count": len(raw), "comparisons": sorted(comparisons)}, indent=2))


if __name__ == "__main__":
    main()

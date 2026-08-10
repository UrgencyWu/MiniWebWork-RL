#!/usr/bin/env python3
"""Build the deterministic, public-action-valid M5 WebShop SFT corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import (  # noqa: E402
    audit_goals,
    deterministic_candidate_order,
    eligible_goal_indices,
    load_protocol,
    sha256_file,
    task_id_for_goal_index,
)
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.oracle import OraclePolicyFailure, build_verified_oracle_trajectory  # noqa: E402
from miniwebwork.webshop_rl import prompt as webshop_prompt  # noqa: E402
from miniwebwork.webshop_rl.actions import parse_command_output  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    content = b"".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return sha256_file(destination)


def _record_path(output_root: Path, split: str, goal_index: int) -> Path:
    return output_root / "records" / split / f"{task_id_for_goal_index(goal_index)}.json"


def _validate_record(
    record: Mapping[str, Any],
    *,
    split: str,
    goal_index: int,
    protocol_sha256: str,
    git_sha: str,
    goals_sha256: str,
) -> dict[str, Any]:
    payload = dict(record)
    expected_record_keys = {
        "schema_version",
        "split",
        "goal_index",
        "task_id",
        "protocol_sha256",
        "git_sha",
        "goals_sha256",
        "status",
        "exclusion_reason",
        "trajectory",
        "content_sha256",
    }
    _require(set(payload) == expected_record_keys, "oracle record keys drift")
    _require(payload.get("schema_version") == "m5_webshop_oracle_record_v1", "oracle record schema drift")
    _require(payload.get("split") == split and payload.get("goal_index") == goal_index, "oracle record identity drift")
    _require(payload.get("task_id") == task_id_for_goal_index(goal_index), "oracle record task id drift")
    _require(payload.get("protocol_sha256") == protocol_sha256, "oracle record protocol drift")
    _require(payload.get("git_sha") == git_sha, "oracle record Git lineage drift")
    _require(payload.get("goals_sha256") == goals_sha256, "oracle record goal source drift")
    _require(payload.get("status") in {"verified", "policy_excluded"}, "oracle record status drift")
    expected = dict(payload)
    content_sha = expected.pop("content_sha256", None)
    _require(content_sha == sha256_json(expected), "oracle record self-hash drift")
    if payload["status"] == "verified":
        _require(payload.get("exclusion_reason") == "", "verified oracle record has an exclusion reason")
        trajectory = payload.get("trajectory")
        _require(isinstance(trajectory, Mapping), "verified oracle record lacks trajectory")
        trajectory_without_hash = dict(trajectory)
        trajectory_hash = trajectory_without_hash.pop("content_sha256", None)
        _require(trajectory_hash == sha256_json(trajectory_without_hash), "oracle trajectory self-hash drift")
        _require(trajectory.get("verified_reward") == 1.0, "oracle trajectory reward drift")
        _require(trajectory.get("split") == split and trajectory.get("goal_index") == goal_index, "oracle trajectory identity drift")
        turns = trajectory.get("turns")
        _require(isinstance(turns, list) and turns, "verified oracle trajectory is empty")
        expected_turn_keys = {
            "schema_version",
            "task_id",
            "split",
            "goal_index",
            "turn_index",
            "messages",
            "prompt_sha256",
            "completion",
            "command",
            "public_state_anchor_sha256",
        }
        for turn_index, turn in enumerate(turns, start=1):
            _require(isinstance(turn, Mapping) and set(turn) == expected_turn_keys, "oracle turn schema drift")
            _require(turn.get("turn_index") == turn_index, "oracle turn index drift")
            _require(turn.get("split") == split and turn.get("goal_index") == goal_index, "oracle turn role drift")
            messages = turn.get("messages")
            _require(isinstance(messages, list) and messages, "oracle turn prompt is missing")
            _require(
                turn.get("prompt_sha256") == webshop_prompt.compute_message_hash(messages),
                "oracle prompt hash drift",
            )
            parsed = parse_command_output(str(turn.get("completion") or ""))
            _require(parsed.strict_json_success and parsed.schema_valid, "oracle trajectory has an invalid label")
            _require(parsed.action is not None and parsed.action.command == turn.get("command"), "oracle command drift")
            anchor = str(turn.get("public_state_anchor_sha256") or "")
            _require(len(anchor) == 64 and all(character in "0123456789abcdef" for character in anchor), "oracle anchor drift")
    else:
        _require(payload.get("trajectory") is None, "excluded oracle record contains a trajectory")
        _require(isinstance(payload.get("exclusion_reason"), str) and payload["exclusion_reason"], "excluded oracle reason is missing")
    return payload


def _build_record(
    *,
    goal: Mapping[str, Any],
    split: str,
    base_url: str,
    output_root: Path,
    protocol_sha256: str,
    git_sha: str,
    goals_sha256: str,
) -> dict[str, Any]:
    goal_index = int(goal["goal_index"])
    path = _record_path(output_root, split, goal_index)
    if path.is_file():
        return _validate_record(
            json.loads(path.read_text(encoding="utf-8")),
            split=split,
            goal_index=goal_index,
            protocol_sha256=protocol_sha256,
            git_sha=git_sha,
            goals_sha256=goals_sha256,
        )
    environment = WebShopHTTPEnvironment(base_url=base_url, split=split)
    try:
        try:
            trajectory = build_verified_oracle_trajectory(environment, goal).to_dict()
            status = "verified"
            reason = ""
        except OraclePolicyFailure as exc:
            trajectory = None
            status = "policy_excluded"
            reason = str(exc)[:200]
    finally:
        environment.close()
    record = {
        "schema_version": "m5_webshop_oracle_record_v1",
        "split": split,
        "goal_index": goal_index,
        "task_id": task_id_for_goal_index(goal_index),
        "protocol_sha256": protocol_sha256,
        "git_sha": git_sha,
        "goals_sha256": goals_sha256,
        "status": status,
        "exclusion_reason": reason,
        "trajectory": trajectory,
    }
    record["content_sha256"] = sha256_json(record)
    atomic_write_json(path, record)
    return _validate_record(
        record,
        split=split,
        goal_index=goal_index,
        protocol_sha256=protocol_sha256,
        git_sha=git_sha,
        goals_sha256=goals_sha256,
    )


def _collect_split(
    *,
    split: str,
    target_count: int,
    goals: list[Mapping[str, Any]],
    base_url: str,
    output_root: Path,
    protocol_sha256: str,
    git_sha: str,
    goals_sha256: str,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    order = deterministic_candidate_order(
        candidates=eligible_goal_indices(split),
        seed=seed,
        namespace=f"m5-sft-{split}",
    )
    selected: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    batch_size = max(workers * 4, 32)
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(
                executor.map(
                    lambda index: _build_record(
                        goal=goals[index],
                        split=split,
                        base_url=base_url,
                        output_root=output_root,
                        protocol_sha256=protocol_sha256,
                        git_sha=git_sha,
                        goals_sha256=goals_sha256,
                    ),
                    batch,
                )
            )
        for record in records:
            attempted.append(record)
            if record["status"] == "verified" and len(selected) < target_count:
                selected.append(record)
        if len(selected) == target_count:
            break
    _require(len(selected) == target_count, f"insufficient verified {split} oracle trajectories")
    selected_goal_indices = [int(record["goal_index"]) for record in selected]
    _require(len(selected_goal_indices) == len(set(selected_goal_indices)), f"duplicate selected {split} goal")
    return {
        "split": split,
        "target_count": target_count,
        "attempted_count": len(attempted),
        "selected_records": selected,
        "selected_goal_indices": selected_goal_indices,
        "selected_goal_order_sha256": sha256_json(selected_goal_indices),
        "exclusion_reasons": dict(
            sorted(Counter(record["exclusion_reason"] for record in attempted if record["status"] != "verified").items())
        ),
    }


def build_corpus(
    *,
    goals_path: Path,
    output_root: Path,
    base_url: str,
    workers: int,
    health_audit_path: Path,
) -> dict[str, Any]:
    _require(1 <= workers <= 64, "SFT oracle worker count must be in [1,64]")
    protocol = load_protocol()
    health_path = Path(health_audit_path).expanduser().resolve()
    _require(health_path.is_file(), "M5 WebShop service health audit is missing")
    health_audit = json.loads(health_path.read_text(encoding="utf-8"))
    _require(isinstance(health_audit, dict), "M5 WebShop service health audit is malformed")
    health_without_hash = dict(health_audit)
    health_hash = health_without_hash.pop("content_sha256", None)
    _require(health_hash == sha256_json(health_without_hash), "M5 WebShop service health self-hash drift")
    _require(health_audit.get("passed") is True, "M5 WebShop service health did not pass")
    _require(health_audit.get("protocol_sha256") == protocol["sha256"], "M5 WebShop service protocol drift")
    _require(health_audit.get("git_sha") == protocol["git_sha"], "M5 WebShop service Git lineage drift")
    _require(health_audit.get("base_url") == base_url.rstrip("/"), "M5 WebShop service URL drift")
    _require(
        health_audit.get("expected_workers")
        in protocol["payload"]["slurm"]["shared_environment_service"]["worker_candidates"],
        "M5 WebShop service worker count drift",
    )
    _require(
        health_audit.get("request_concurrency_mode") == "process_serialized_asgi_v1",
        "M5 WebShop service request-concurrency drift",
    )
    goal_audit = audit_goals(goals_path, protocol["payload"])
    goals = json.loads(Path(goals_path).expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(goals, list), "WebShop goals root is not a list")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sft = protocol["payload"]["sft"]
    seed = int(sft["selection_seed"])
    selections = {
        split: _collect_split(
            split=split,
            target_count=int(sft[f"{split}_task_count"]),
            goals=goals,
            base_url=base_url,
            output_root=root,
            protocol_sha256=protocol["sha256"],
            git_sha=protocol["git_sha"],
            goals_sha256=goal_audit["goals_sha256"],
            seed=seed,
            workers=workers,
        )
        for split in ("train", "dev")
    }
    _require(
        set(selections["train"]["selected_goal_indices"]).isdisjoint(selections["dev"]["selected_goal_indices"]),
        "selected SFT train/dev goal overlap",
    )
    corpus_files = {}
    total_turns = 0
    for split, selection in selections.items():
        rows = []
        for record in selection["selected_records"]:
            trajectory = record["trajectory"]
            for turn in trajectory["turns"]:
                row = dict(turn)
                row["git_sha"] = protocol["git_sha"]
                row["trajectory_content_sha256"] = trajectory["content_sha256"]
                rows.append(row)
        _require(all(row["completion"].strip() for row in rows), f"{split} corpus contains a zero label")
        path = root / f"{split}.jsonl"
        digest = _atomic_jsonl(path, rows)
        corpus_files[split] = {"path": str(path), "sha256": digest, "rows": len(rows)}
        total_turns += len(rows)
    audit = {
        "schema_version": "m5_webshop_sft_corpus_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "formal_training": False,
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "goals_sha256": goal_audit["goals_sha256"],
        "selection_seed": seed,
        "base_url": base_url,
        "service_workers": health_audit["expected_workers"],
        "service_health_audit_path": str(health_path),
        "service_health_audit_sha256": sha256_file(health_path),
        "workers": workers,
        "train_task_count": selections["train"]["target_count"],
        "dev_task_count": selections["dev"]["target_count"],
        "train_goal_order_sha256": selections["train"]["selected_goal_order_sha256"],
        "dev_goal_order_sha256": selections["dev"]["selected_goal_order_sha256"],
        "attempted_counts": {split: value["attempted_count"] for split, value in selections.items()},
        "exclusion_reasons": {split: value["exclusion_reasons"] for split, value in selections.items()},
        "total_verified_turns": total_turns,
        "zero_label_fraction": 0.0,
        "test_goal_count_used": 0,
        "corpus_files": corpus_files,
    }
    audit["content_sha256"] = sha256_json(audit)
    atomic_write_json(root / "corpus_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--health-audit", type=Path, required=True)
    args = parser.parse_args()
    report = build_corpus(
        goals_path=args.goals,
        output_root=args.output_dir,
        base_url=args.base_url,
        workers=args.workers,
        health_audit_path=args.health_audit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

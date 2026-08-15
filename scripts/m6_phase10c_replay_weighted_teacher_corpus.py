#!/usr/bin/env python3
"""Fresh-replay Raw35 strict paths and build a capability-weighted SFT corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.webshop_rl.actions import WebShopCommand, normalize_command  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.m6_corpus import (  # noqa: E402
    audit_conditional_learnability,
    build_policy_visible_corpus,
    flatten_corpus_rows,
)
from miniwebwork.webshop_rl.verifier_td import public_state_anchor_signature  # noqa: E402

CAPABILITY_ROLES = {
    "nav": "teacher_nav_train",
    "match": "teacher_match_train",
    "finish": "teacher_finish_train",
}
CAPABILITY_WEIGHTS = {"nav": 0.20, "match": 0.40, "finish": 0.40}
SPLIT_SEED = 20260864
EXPECTED_MODES = {
    "phase10c_teacher_exploration": (32, 4),
    "phase10c_teacher_exploration_scale": (128, 8),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Phase10-C artifact is not an object: {path}")
    expected = dict(value)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), f"Phase10-C artifact self-hash drift: {path}")
    return value


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    with temporary.open("xb") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _commands(episode: Mapping[str, Any]) -> list[str]:
    return [
        normalize_command(str(turn["action"]["command"]))
        for turn in episode["turns"]
        if turn.get("schema_valid") is True
        and isinstance(turn.get("action"), Mapping)
        and isinstance(turn.get("action_result"), Mapping)
    ]


def fresh_replay_episode(
    episode: Mapping[str, Any],
    *,
    base_url: str,
    environment_factory: Any = WebShopHTTPEnvironment,
) -> dict[str, Any]:
    """Replay one full command sequence and require exact public-state parity."""

    environment = environment_factory(base_url=base_url, split="train", timeout_seconds=120)
    executed = 0
    score = 0.0
    terminated = False
    public_state_exact = True
    action_result_exact = True
    error = None
    try:
        observation = environment.reset(str(episode["task_id"]))
        for turn in episode["turns"]:
            recorded_pre = turn.get("observation")
            _require(isinstance(recorded_pre, Mapping), "Phase10-C replay turn lacks public observation")
            if public_state_anchor_signature(observation.to_dict()) != public_state_anchor_signature(recorded_pre):
                public_state_exact = False
                break
            action = turn.get("action")
            recorded_result = turn.get("action_result")
            if not isinstance(action, Mapping) or not isinstance(recorded_result, Mapping):
                continue
            result = environment.step(WebShopCommand(normalize_command(str(action["command"]))))
            executed += 1
            observed_result = result.info.get("action_result", {})
            if bool(observed_result.get("success", False)) is not bool(recorded_result.get("success", False)):
                action_result_exact = False
                break
            recorded_post = turn.get("post_action_observation")
            _require(isinstance(recorded_post, Mapping), "Phase10-C replay turn lacks post-action observation")
            observation = result.observation
            if public_state_anchor_signature(observation.to_dict()) != public_state_anchor_signature(recorded_post):
                public_state_exact = False
                break
            score = float(result.info.get("task_score", score))
            terminated = bool(result.terminated)
            if result.terminated or result.truncated:
                break
    except Exception as exc:  # keep every failed replay as evidence
        error = f"{type(exc).__name__}: {exc}"
    finally:
        environment.close()
    strict = bool(error is None and public_state_exact and action_result_exact and terminated and score >= 0.999)
    return {
        "fresh_session": True,
        "public_state_exact": public_state_exact,
        "action_result_exact": action_result_exact,
        "executed_command_count": executed,
        "expected_command_count": len(_commands(episode)),
        "terminated": terminated,
        "task_score": score,
        "strict_success": strict,
        "error": error,
        "command_sequence_sha256": sha256_json(_commands(episode)),
    }


def capability_for_task(task_id: str, role_tasks: Mapping[str, set[str]]) -> str:
    matches = [capability for capability, tasks in role_tasks.items() if task_id in tasks]
    _require(len(matches) == 1, f"Phase10-C task capability is ambiguous: {task_id}")
    return matches[0]


def stratified_task_split(task_capabilities: Mapping[str, str]) -> tuple[set[str], set[str]]:
    train: set[str] = set()
    dev: set[str] = set()
    for capability in CAPABILITY_WEIGHTS:
        tasks = [task_id for task_id, current in task_capabilities.items() if current == capability]
        tasks.sort(key=lambda task_id: sha256_json({"seed": SPLIT_SEED, "task_id": task_id}))
        _require(len(tasks) >= 2, f"Phase10-C {capability} has too few replay-strict tasks")
        dev_count = max(1, len(tasks) // 10)
        dev.update(tasks[:dev_count])
        train.update(tasks[dev_count:])
    _require(train and dev and not (train & dev), "Phase10-C train/dev task split drift")
    return train, dev


def build_weighted_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_capabilities: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Assign source->task->path->action-row weights that sum to one."""

    rows_by_capability_task_path: dict[str, dict[str, dict[str, list[Mapping[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        task_id = str(row["task_id"])
        capability = task_capabilities[task_id]
        rows_by_capability_task_path[capability][task_id][str(row["trajectory_id"])].append(row)
    output: list[dict[str, Any]] = []
    for capability, capability_weight in CAPABILITY_WEIGHTS.items():
        tasks = rows_by_capability_task_path[capability]
        _require(tasks, f"Phase10-C weighted split lacks {capability}")
        for task_id in sorted(tasks):
            paths = tasks[task_id]
            task_weight = capability_weight / len(tasks)
            for trajectory_id in sorted(paths):
                path_rows = paths[trajectory_id]
                path_weight = task_weight / len(paths)
                for source in path_rows:
                    row = dict(source)
                    row.update({
                        "source": "raw35_replay_strict_self_exploration",
                        "capability": capability,
                        "capability_loss_mass": capability_weight,
                        "task_loss_mass": task_weight,
                        "path_loss_mass": path_weight,
                        "row_loss_weight": path_weight / len(path_rows),
                        "loss_hierarchy": "capability_task_path_action_row_token_mean_v1",
                    })
                    output.append(row)
    _require(math.isclose(sum(float(row["row_loss_weight"]) for row in output), 1.0, abs_tol=1e-12),
             "Phase10-C weighted row mass drift")
    return output


def _load_sources(roots: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for root in roots:
        report = _load_hashed(root / "collection_report.json")
        invocation = _load_hashed(root / "invocation.json")
        mode = str(report.get("mode"))
        _require(mode in EXPECTED_MODES, "Phase10-C replay source mode drift")
        expected_tasks, expected_k = EXPECTED_MODES[mode]
        _require(
            report.get("complete") is True
            and report.get("task_count") == expected_tasks
            and report.get("trajectory_count") == expected_tasks * expected_k
            and report.get("K") == expected_k
            and report.get("training_updates_allowed") is False,
            "Phase10-C replay source collection contract drift",
        )
        _require(invocation.get("content_sha256") == report.get("invocation_content_sha256"),
                 "Phase10-C replay source invocation binding drift")
        task_ids = set()
        episode_paths = sorted((root / "episodes").glob("*.json"))
        _require(len(episode_paths) == expected_tasks * expected_k, "Phase10-C source episode count drift")
        for path in episode_paths:
            episode = _load_hashed(path)
            task_ids.add(str(episode["task_id"]))
            if episode.get("success") is True and float(episode.get("task_score", 0.0)) >= 0.999:
                episode["_source_episode_path"] = str(path)
                episode["_source_episode_content_sha256"] = episode["content_sha256"]
                episodes.append(episode)
        _require(len(task_ids) == expected_tasks and not (task_ids & seen_tasks),
                 "Phase10-C source task identity overlap/drift")
        seen_tasks.update(task_ids)
        reports.append({
            "root": str(root),
            "mode": mode,
            "collection_report_content_sha256": report["content_sha256"],
            "invocation_content_sha256": invocation["content_sha256"],
            "task_count": expected_tasks,
            "trajectory_count": expected_tasks * expected_k,
            "strict_success_trajectory_count": report["strict_success_trajectory_count"],
            "strict_success_task_count": report["strict_success_task_count"],
            "policy_lineage": report["policy_lineage"],
        })
    _require(len({sha256_json(item["policy_lineage"]) for item in reports}) == 1,
             "Phase10-C source Raw35 lineage drift")
    _require(len(episodes) == sum(int(item["strict_success_trajectory_count"]) for item in reports),
             "Phase10-C source strict trajectory count drift")
    return episodes, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, action="append", required=True)
    parser.add_argument("--phase10c-split", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-35B-A3B"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(len(args.collection_root) == 2, "Phase10-C replay expects the initial and scale collections")
    _require(1 <= args.workers <= 8, "Phase10-C replay workers must be in [1,8]")
    output = args.output_dir.expanduser().resolve()
    _require(not (output / "replay_report.json").exists(), "Phase10-C replay output already exists")
    output.mkdir(parents=True, exist_ok=True)

    split = _load_hashed(args.phase10c_split)
    role_tasks = {
        capability: set(split["roles"][role]["task_ids"])
        for capability, role in CAPABILITY_ROLES.items()
    }
    goals = json.loads(args.goals.expanduser().resolve().read_text(encoding="utf-8"))
    goal_map = {f"webshop_goal_{int(item['goal_index']):05d}": item for item in goals}
    roots = [path.expanduser().resolve() for path in args.collection_root]
    episodes, sources = _load_sources(roots)
    episodes.sort(key=lambda episode: sha256_json({
        "seed": SPLIT_SEED,
        "task_id": episode["task_id"],
        "trajectory_id": episode["trajectory_id"],
        "commands": _commands(episode),
    }))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        replay_values = list(executor.map(
            lambda episode: fresh_replay_episode(episode, base_url=args.base_url),
            episodes,
        ))
    replay_rows: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for episode, replay in zip(episodes, replay_values):
        capability = capability_for_task(str(episode["task_id"]), role_tasks)
        replay_rows.append({
            "task_id": episode["task_id"],
            "trajectory_id": episode["trajectory_id"],
            "capability": capability,
            "source_episode_path": episode["_source_episode_path"],
            "source_episode_content_sha256": episode["_source_episode_content_sha256"],
            "replay": replay,
        })
        if replay["strict_success"] is True:
            current = dict(episode)
            current["replay_success"] = True
            verified.append(current)

    replay_report = {
        "schema_version": "m6_phase10c_raw35_fresh_replay_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "fresh_session_per_trajectory": True,
        "public_state_exact_required": True,
        "action_result_exact_required": True,
        "source_collections": sources,
        "attempted_strict_trajectory_count": len(episodes),
        "replay_strict_trajectory_count": len(verified),
        "replay_strict_task_count": len({item["task_id"] for item in verified}),
        "replay_strict_by_capability": dict(sorted(Counter(
            capability_for_task(str(item["task_id"]), role_tasks) for item in verified
        ).items())),
        "rejection_counts": {
            "public_state_mismatch": sum(not row["replay"]["public_state_exact"] for row in replay_rows),
            "action_result_mismatch": sum(not row["replay"]["action_result_exact"] for row in replay_rows),
            "non_strict_outcome": sum(not row["replay"]["strict_success"] for row in replay_rows),
            "exception": sum(row["replay"]["error"] is not None for row in replay_rows),
        },
        "trajectories": replay_rows,
    }
    replay_report["content_sha256"] = sha256_json(replay_report)
    atomic_write_json(output / "replay_report.json", replay_report)
    _require(verified, "Phase10-C fresh replay retained no strict trajectories")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    corpus = build_policy_visible_corpus(
        trajectories=verified,
        goal_by_task_id=goal_map,
        completion_token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    audit = audit_conditional_learnability(corpus=corpus, goal_by_task_id=goal_map, mini=True)
    _require(audit["checks"]["strict_success_fraction"] and audit["checks"]["replay_success_fraction"],
             "Phase10-C corpus lost strict/replay purity")
    _require(audit["checks"]["hidden_policy_fields"] and audit["checks"]["target_asin_search_labels"],
             "Phase10-C corpus contains hidden/target label leakage")
    atomic_write_json(output / "corpus.json", corpus)
    atomic_write_json(output / "corpus_audit.json", audit)

    rows = flatten_corpus_rows(corpus)
    task_capabilities = {
        str(trajectory["task_id"]): capability_for_task(str(trajectory["task_id"]), role_tasks)
        for trajectory in corpus["trajectories"]
    }
    train_tasks, dev_tasks = stratified_task_split(task_capabilities)
    train_rows = build_weighted_rows(
        [row for row in rows if row["task_id"] in train_tasks],
        task_capabilities=task_capabilities,
    )
    dev_rows = build_weighted_rows(
        [row for row in rows if row["task_id"] in dev_tasks],
        task_capabilities=task_capabilities,
    )
    _atomic_jsonl(output / "train.jsonl", train_rows)
    _atomic_jsonl(output / "dev.jsonl", dev_rows)

    manifest = {
        "schema_version": "m6_phase10c_weighted_raw35_sft_corpus_v1",
        "development_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "base_model": str(args.base_model.expanduser().resolve()),
        "replay_report_content_sha256": replay_report["content_sha256"],
        "corpus_content_sha256": corpus["content_sha256"],
        "corpus_audit_content_sha256": audit["content_sha256"],
        "capability_weights": dict(CAPABILITY_WEIGHTS),
        "weight_rationale": "inverse_difficulty_rounded_v1: nav strong; match and finish weak",
        "loss_hierarchy": "capability_task_path_action_row_token_mean_v1",
        "maximum_distinct_paths_per_task": 4,
        "train_task_count": len(train_tasks),
        "dev_task_count": len(dev_tasks),
        "task_overlap": len(train_tasks & dev_tasks),
        "train_action_row_count": len(train_rows),
        "dev_action_row_count": len(dev_rows),
        "train_completion_label_tokens": sum(int(row["completion_label_tokens"]) for row in train_rows),
        "dev_completion_label_tokens": sum(int(row["completion_label_tokens"]) for row in dev_rows),
        "train_row_loss_weight_sum": sum(float(row["row_loss_weight"]) for row in train_rows),
        "dev_row_loss_weight_sum": sum(float(row["row_loss_weight"]) for row in dev_rows),
        "train_task_counts_by_capability": dict(sorted(Counter(task_capabilities[task] for task in train_tasks).items())),
        "dev_task_counts_by_capability": dict(sorted(Counter(task_capabilities[task] for task in dev_tasks).items())),
        "train_path_counts_by_capability": dict(sorted(Counter(row["capability"] for row in {
            (row["trajectory_id"], row["capability"]): row for row in train_rows
        }.values()).items())),
        "split_seed": SPLIT_SEED,
        "split_rule": "capability_stratified_task_hash_90_10_v1",
        "source_collections": sources,
    }
    _require(manifest["task_overlap"] == 0, "Phase10-C weighted corpus task leakage")
    _require(math.isclose(manifest["train_row_loss_weight_sum"], 1.0, abs_tol=1e-12),
             "Phase10-C train loss mass drift")
    _require(math.isclose(manifest["dev_row_loss_weight_sum"], 1.0, abs_tol=1e-12),
             "Phase10-C dev loss mass drift")
    manifest["content_sha256"] = sha256_json(manifest)
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "attempted_strict_trajectories": len(episodes),
        "replay_strict_trajectories": len(verified),
        "replay_strict_tasks": replay_report["replay_strict_task_count"],
        "selected_corpus_trajectories": corpus["trajectory_count"],
        "selected_corpus_tasks": corpus["task_count"],
        "capability_weights": CAPABILITY_WEIGHTS,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "manifest_content_sha256": manifest["content_sha256"],
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

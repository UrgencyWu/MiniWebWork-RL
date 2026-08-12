#!/usr/bin/env python3
"""Replay Raw successes and publish M6-mini SFT/retention artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from miniwebwork.m6_posttraining_protocol import load_protocol, validate_split_lock  # noqa: E402
from miniwebwork.m6_pilot import build_pilot_authorization  # noqa: E402
from miniwebwork.webshop_rl.actions import WebShopCommand, normalize_command  # noqa: E402
from miniwebwork.webshop_rl.environment import WebShopHTTPEnvironment  # noqa: E402
from miniwebwork.webshop_rl.m6_corpus import (  # noqa: E402
    audit_conditional_learnability,
    build_policy_visible_corpus,
    build_retention_states,
    flatten_corpus_rows,
)
from miniwebwork.webshop_rl.m6_sft_training import (  # noqa: E402
    M6SFTConfig,
    build_token_audit,
    load_jsonl_examples,
    load_retention_examples,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    with temporary.open("xb") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _replay_success(episode: Mapping[str, Any], *, base_url: str) -> bool:
    environment = WebShopHTTPEnvironment(base_url=base_url, split="train", timeout_seconds=120)
    try:
        environment.reset(str(episode["task_id"]))
        last = None
        for turn in episode["turns"]:
            action = turn.get("action")
            if turn.get("schema_valid") is not True or not isinstance(action, Mapping):
                continue
            last = environment.step(WebShopCommand(normalize_command(str(action["command"]))))
            if last.terminated or last.truncated:
                break
        return bool(last is not None and last.terminated and last.info.get("success") is True)
    finally:
        environment.close()


def _collection_identity(root: Path) -> dict[str, Any]:
    report = _load(root / "collection_report.json")
    invocation = _load(root / "invocation.json")
    for value, label in ((report, "collection report"), (invocation, "invocation")):
        expected = dict(value)
        observed = expected.pop("content_sha256", None)
        _require(observed == sha256_json(expected), f"M6 source {label} self-hash drift: {root}")
    _require(report.get("complete") is True and report.get("mode") == "raw_collection", "M6 corpus source is not complete Raw collection")
    _require(report.get("K") == 8 and invocation.get("K") == 8, "M6 corpus source is not an independent K8 collection")
    _require(invocation.get("mode") == "raw_collection", "M6 corpus invocation mode drift")
    _require(report.get("invocation_content_sha256") == invocation.get("content_sha256"), "M6 collection/invocation binding drift")
    return {
        "root": str(root),
        "collection_report_content_sha256": report["content_sha256"],
        "invocation_content_sha256": invocation["content_sha256"],
        "seed": int(invocation["seed"]),
        "git_sha": report["git_sha"],
        "protocol_sha256": report["protocol_sha256"],
        "split_lock_content_sha256": report["split_lock_content_sha256"],
        "task_order_sha256": report["task_order_sha256"],
        "policy_lineage": report["policy_lineage"],
    }


def _load_episodes(roots: list[Path], *, base_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    identities = [_collection_identity(root) for root in roots]
    _require(len({item["seed"] for item in identities}) == len(identities), "M6 Raw K8 supplement reused a sampling seed")
    for field in ("git_sha", "protocol_sha256", "split_lock_content_sha256", "task_order_sha256", "policy_lineage"):
        _require(len({sha256_json(item[field]) for item in identities}) == 1, f"M6 Raw collection roots disagree on {field}")
    seen_trajectory_ids: set[str] = set()
    for root, identity in zip(roots, identities):
        namespace = identity["collection_report_content_sha256"][:12]
        paths = sorted((root / "episodes").glob("*.json"))
        _require(paths, f"M6 source collection contains no episodes: {root}")
        for path in paths:
            value = _load(path)
            expected = dict(value)
            observed = expected.pop("content_sha256", None)
            _require(observed == sha256_json(expected), f"M6 source episode self-hash drift: {path}")
            value.pop("content_sha256", None)
            source_trajectory_id = str(value.get("trajectory_id", path.stem))
            value["trajectory_id"] = f"{namespace}:{source_trajectory_id}"
            _require(value["trajectory_id"] not in seen_trajectory_ids, "M6 merged Raw trajectory identity collision")
            seen_trajectory_ids.add(value["trajectory_id"])
            value["replay_success"] = bool(
                value.get("success") is True and _replay_success(value, base_url=base_url)
            )
            episodes.append(value)
    _require(episodes, "M6 source collection contains no episodes")
    return episodes, identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, action="append", required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--base-url", default="http://127.0.0.1:44151")
    parser.add_argument("--retention-states", type=int, default=20_000)
    parser.add_argument(
        "--pilot-waiver",
        type=Path,
        help="Approved development-only waiver; required when the original corpus gate fails",
    )
    args = parser.parse_args()

    protocol = load_protocol()
    split = validate_split_lock(_load(args.split_lock))
    _require(split["protocol_sha256"] == protocol["sha256"], "M6 corpus split/protocol drift")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    goals = _load(args.goals)
    _require(sha256_json(goals) == split["goals_canonical_sha256"], "M6 corpus goals/split drift")
    goal_map = {f"webshop_goal_{int(item['goal_index']):05d}": item for item in goals}
    collection_roots = [path.expanduser().resolve() for path in args.collection_root]
    _require(1 <= len(collection_roots) <= 2, "M6 mini corpus permits one Raw K8 collection and at most one K8 supplement")
    episodes, source_collections = _load_episodes(
        collection_roots,
        base_url=args.base_url,
    )
    _require(
        all(item["split_lock_content_sha256"] == split["content_sha256"] for item in source_collections),
        "M6 corpus collection/split drift",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        _require(tokenizer.eos_token_id is not None, "M6 tokenizer lacks pad/EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def count_completion(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    corpus = build_policy_visible_corpus(
        trajectories=episodes,
        goal_by_task_id=goal_map,
        completion_token_counter=count_completion,
    )
    corpus_audit = audit_conditional_learnability(corpus=corpus, goal_by_task_id=goal_map, mini=True)
    atomic_write_json(output / "corpus.json", corpus)
    atomic_write_json(output / "corpus_audit.json", corpus_audit)
    pilot_authorization = None
    if corpus_audit["passed"] is not True:
        _require(args.pilot_waiver is not None, f"M6 mini corpus gate failed: {corpus_audit['checks']}")
        pilot_authorization = build_pilot_authorization(
            corpus_audit=corpus_audit,
            source_collections=source_collections,
            waiver_path=args.pilot_waiver,
        )
        atomic_write_json(output / "pilot_authorization.json", pilot_authorization)

    rows = flatten_corpus_rows(corpus)
    task_ids = sorted({row["task_id"] for row in rows})
    # Bind the 90/10 split to the frozen protocol seed rather than input file
    # order.  The same source corpus always yields the same task-level split,
    # while one task can never leak across train/dev.
    split_seed = int(protocol["payload"]["split"]["selection_seed"])
    ranked_tasks = sorted(
        task_ids,
        key=lambda task_id: (
            sha256_json({"namespace": "m6-sft-internal-dev-v1", "seed": split_seed, "task_id": task_id}),
            task_id,
        ),
    )
    dev_count = max(1, len(ranked_tasks) // 10)
    dev_tasks = set(ranked_tasks[:dev_count])
    train_rows = [row for row in rows if row["task_id"] not in dev_tasks]
    dev_rows = [row for row in rows if row["task_id"] in dev_tasks]
    _require(train_rows and dev_rows, "M6 SFT internal train/dev partition is empty")
    _atomic_jsonl(output / "train.jsonl", train_rows)
    _atomic_jsonl(output / "dev.jsonl", dev_rows)
    retention = build_retention_states(
        trajectories=episodes,
        maximum_states=args.retention_states,
    )
    atomic_write_json(output / "retention.json", retention)

    config = M6SFTConfig.from_protocol(protocol["payload"])
    train = load_jsonl_examples(output / "train.jsonl", tokenizer, config)
    dev = load_jsonl_examples(output / "dev.jsonl", tokenizer, config)
    retention_examples = load_retention_examples(output / "retention.json", config)
    token_audit = build_token_audit(
        train_examples=train,
        dev_examples=dev,
        retention_examples=retention_examples,
        base_model=args.base_model,
        input_files={
            "corpus": output / "corpus.json",
            "corpus_audit": output / "corpus_audit.json",
            "train": output / "train.jsonl",
            "dev": output / "dev.jsonl",
            "retention": output / "retention.json",
            **(
                {"pilot_authorization": output / "pilot_authorization.json"}
                if pilot_authorization is not None
                else {}
            ),
        },
        config=config,
    )
    token_audit["protocol_sha256"] = protocol["sha256"]
    token_audit["git_sha"] = protocol["git_sha"]
    token_audit["corpus_audit_sha256"] = sha256_file(output / "corpus_audit.json")
    token_audit["source_collections"] = source_collections
    token_audit["split_lock_content_sha256"] = split["content_sha256"]
    token_audit["corpus_gate_mode"] = (
        "approved_156_task_pilot_waiver" if pilot_authorization is not None else "original_protocol_gate"
    )
    token_audit["pilot_authorization_content_sha256"] = (
        pilot_authorization["content_sha256"] if pilot_authorization is not None else None
    )
    token_audit.pop("content_sha256")
    token_audit["content_sha256"] = sha256_json(token_audit)
    atomic_write_json(output / "token_audit.json", token_audit)
    _require(token_audit["passed"] is True, f"M6 token audit failed: {token_audit['checks']}")
    print(json.dumps({
        "output_dir": str(output),
        "corpus_passed": corpus_audit["passed"],
        "pilot_authorized": pilot_authorization is not None,
        "token_audit_passed": token_audit["passed"],
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "retention_states": retention["state_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

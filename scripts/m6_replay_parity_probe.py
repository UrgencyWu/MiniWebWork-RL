#!/usr/bin/env python3
"""Compare frozen vLLM behavior logprobs with HF replay across microbatches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import (  # noqa: E402
    atomic_write_json,
    directory_sha256,
    sha256_json,
)
from miniwebwork.long_horizon_rl.learner import (  # noqa: E402
    load_trainable_policy_model,
    replay_examples,
    summarize_logprob_parity,
)
from miniwebwork.m6_posttraining_protocol import load_protocol  # noqa: E402
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    M6TurnTrainingExample,
    replay_parity_checks,
    validate_committed_group,
)
from miniwebwork.webshop_rl.verifier_td import METHODS  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--collection-producer-git-sha", required=True)
    parser.add_argument("--input-adapter", type=Path, required=True)
    parser.add_argument("--input-adapter-semantic-sha256", required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--expected-k", type=int, choices=(4, 8), default=8)
    parser.add_argument("--maximum-groups", type=int)
    parser.add_argument("--microbatch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--logprob-precisions", choices=("float32", "model"), nargs="+", default=["float32"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 parity probe requires one GPU")
    _require(args.microbatch_sizes and set(args.microbatch_sizes) <= {1, 2, 4, 8}, "invalid parity-probe microbatch")
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "M6 parity probe output already exists")

    collection = json.loads(args.collection_report.read_text(encoding="utf-8"))
    expected_collection = dict(collection)
    observed_collection = expected_collection.pop("content_sha256", None)
    _require(observed_collection == sha256_json(expected_collection), "M6 parity probe collection self-hash drift")
    _require(collection.get("git_sha") == args.collection_producer_git_sha, "M6 parity probe producer Git drift")
    _require(collection.get("complete") is True and collection.get("mode") == "rl_collection", "M6 parity probe collection drift")

    group_paths = sorted(args.groups_dir.glob("g*.json"))
    if args.maximum_groups is not None:
        _require(args.maximum_groups > 0, "invalid parity-probe group limit")
        group_paths = group_paths[: args.maximum_groups]
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=args.expected_k)
        for path in group_paths
    ]
    _require(
        groups and collection.get("group_content_sha256") == [item["content_sha256"] for item in groups],
        "M6 parity probe group binding drift",
    )
    adapter_sha256 = directory_sha256(args.input_adapter)
    _require(all(item["adapter_sha256"] == adapter_sha256 for item in groups), "M6 parity probe adapter byte drift")
    _require(
        all(item["adapter_semantic_sha256"] == args.input_adapter_semantic_sha256 for item in groups),
        "M6 parity probe adapter semantic drift",
    )

    examples_by_group = []
    for group in groups:
        group_examples = []
        for trajectory_index, trajectory in enumerate(group["trajectories"]):
            turns = trajectory["turns"]
            for turn in turns:
                group_examples.append(
                    M6TurnTrainingExample(
                        group_id=group["group_id"],
                        trajectory_id=trajectory["trajectory_id"],
                        trajectory_index=trajectory_index,
                        turn_index=turn["turn_index"],
                        turns_in_trajectory=len(turns),
                        prompt_token_ids=tuple(turn["prompt_token_ids"]),
                        generated_token_ids=tuple(turn["generated_token_ids"]),
                        behavior_logprobs=tuple(turn["behavior_logprobs"]),
                        sampling_logprobs=tuple(turn["sampling_logprobs"]),
                        advantage=0.0,
                    )
                )
        examples_by_group.append(
            tuple(
                sorted(
                    group_examples,
                    key=lambda example: (
                        example.forward_tokens,
                        example.trajectory_index,
                        example.turn_index,
                    ),
                )
            )
        )
    examples = tuple(example for group_examples in examples_by_group for example in group_examples)
    behavior = [value for example in examples for value in example.behavior_logprobs]
    token_metadata = [
        {
            "trajectory_index": example.trajectory_index,
            "turn_index": example.turn_index,
            "completion_token_index": token_index,
            "prompt_tokens": len(example.prompt_token_ids),
            "completion_tokens": example.completion_tokens,
            "generated_token_id": example.generated_token_ids[token_index],
        }
        for example in examples
        for token_index in range(example.completion_tokens)
    ]

    model, tokenizer = load_trainable_policy_model(base_model=args.base_model, adapter_path=args.input_adapter)
    protocol = load_protocol()
    thresholds = protocol["payload"]["rl"]["parity_contract"]
    results = []
    for logprob_precision in args.logprob_precisions:
        for microbatch_size in args.microbatch_sizes:
            replay = []
            for group_examples in examples_by_group:
                replay.extend(
                    replay_examples(
                        model=model,
                        examples=group_examples,
                        pad_token_id=tokenizer.pad_token_id,
                        microbatch_size=microbatch_size,
                        device=torch.device("cuda:0"),
                        logprob_precision=logprob_precision,
                    )
                )
            parity = summarize_logprob_parity(behavior, replay)
            checks = replay_parity_checks(parity, contract=thresholds)
            ranked = sorted(
                range(len(behavior)),
                key=lambda index: abs(replay[index] - behavior[index]),
                reverse=True,
            )[:20]
            outliers = []
            for index in ranked:
                item = dict(token_metadata[index])
                item.update(
                    {
                        "flat_token_index": index,
                        "behavior_logprob": behavior[index],
                        "replay_logprob": replay[index],
                        "absolute_difference": abs(replay[index] - behavior[index]),
                    }
                )
                outliers.append(item)
            results.append(
                {
                    "logprob_precision": logprob_precision,
                    "microbatch_size": microbatch_size,
                    "passed": all(checks.values()),
                    "checks": checks,
                    "parity": parity,
                    "top_outliers": outliers,
                }
            )

    report = {
        "schema_version": "m6_replay_parity_probe_v1",
        "development_only": True,
        "training_updates_performed": 0,
        "producer_git_sha": args.collection_producer_git_sha,
        "consumer_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "protocol_sha256": protocol["sha256"],
        "collection_report_content_sha256": collection["content_sha256"],
        "group_content_sha256": [item["content_sha256"] for item in groups],
        "input_adapter_sha256": adapter_sha256,
        "input_adapter_semantic_sha256": args.input_adapter_semantic_sha256,
        "method": args.method,
        "expected_K": args.expected_k,
        "group_count": len(groups),
        "token_count": len(behavior),
        "thresholds": thresholds,
        "results": results,
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

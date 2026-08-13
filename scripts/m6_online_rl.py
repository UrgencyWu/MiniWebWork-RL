#!/usr/bin/env python3
"""Apply one recoverable M6-mini K8 GRPO-family update."""

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
    sha256_file,
    sha256_json,
)
from miniwebwork.m6_posttraining_protocol import load_protocol  # noqa: E402
from miniwebwork.m6_pilot import (  # noqa: E402
    load_pilot_replay_parity_calibration,
    validate_artifact_git_compatibility,
    validate_pilot_authorization,
    validate_pilot_method,
)
from miniwebwork.webshop_rl.m6_online_training import (  # noqa: E402
    ReplayParityError,
    train_mini_policy_iteration,
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
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--input-adapter", type=Path, required=True)
    parser.add_argument("--input-adapter-semantic-sha256", required=True)
    parser.add_argument("--reference-sft-adapter", type=Path, required=True)
    parser.add_argument("--reference-sft-adapter-semantic-sha256", required=True)
    parser.add_argument("--input-optimizer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--pilot-authorization", type=Path, required=True)
    parser.add_argument("--replay-parity-calibration", type=Path, required=True)
    parser.add_argument("--collection-producer-git-sha")
    parser.add_argument("--input-state-producer-git-sha")
    parser.add_argument("--parity-rejection-report", type=Path)
    args = parser.parse_args()
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "M6 mini RL requires one Slurm GPU")
    protocol = load_protocol()
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    _require(git_sha == protocol["git_sha"], "M6 mini RL Git/protocol drift")
    authorization = validate_pilot_authorization(
        json.loads(args.pilot_authorization.read_text(encoding="utf-8"))
    )
    validate_pilot_method(args.method, authorization)
    _require(
        authorization.get("source_protocol_sha256") == protocol["sha256"],
        "M6 pilot authorization/protocol drift",
    )
    collection = json.loads(args.collection_report.read_text(encoding="utf-8"))
    _require(collection.get("schema_version") == "m6_rollout_collection_report_v1", "M6 RL collection report drift")
    expected_collection = dict(collection)
    observed_collection_hash = expected_collection.pop("content_sha256", None)
    from miniwebwork.long_horizon_rl.contracts import sha256_json

    _require(observed_collection_hash == sha256_json(expected_collection), "M6 RL collection report self-hash drift")
    _require(collection.get("complete") is True and collection.get("mode") == "rl_collection", "M6 RL collection is incomplete")
    _require(collection.get("role") == "mini_train" and collection.get("K") == 8, "M6 RL collection role/K drift")
    _require(collection.get("action_token_budget_respected") is True, "M6 RL collection exceeded its token budget")
    _require(collection.get("training_updates_allowed") is True, "M6 collection report is not update-authorized")
    collection_producer_git_sha = validate_artifact_git_compatibility(
        artifact_name="RL recovery collection",
        producer_git_sha=collection.get("git_sha"),
        consumer_git_sha=git_sha,
        explicitly_authorized_producer_git_sha=args.collection_producer_git_sha,
    )
    _require(collection.get("protocol_sha256") == protocol["sha256"], "M6 collection report protocol drift")
    groups = [
        validate_committed_group(json.loads(path.read_text(encoding="utf-8")), require_k=8)
        for path in sorted(args.groups_dir.glob("g*.json"))
    ]
    _require(groups and all(group.get("training_updates_allowed") is True for group in groups), "M6 RL groups are not update-authorized")
    _require(
        collection.get("group_content_sha256") == [group["content_sha256"] for group in groups],
        "M6 collection report/group roster drift",
    )
    _require(
        all(group.get("adapter_semantic_sha256") == args.input_adapter_semantic_sha256 for group in groups),
        "M6 RL behavior adapter does not match learner input",
    )
    _require(
        isinstance(collection.get("policy_lineage"), dict)
        and collection["policy_lineage"].get("adapter_sha256") == groups[0]["adapter_sha256"]
        and collection["policy_lineage"].get("rollout_adapter_sha256") == groups[0]["rollout_adapter_sha256"]
        and collection["policy_lineage"].get("adapter_semantic_sha256") == groups[0]["adapter_semantic_sha256"],
        "M6 collection report policy lineage drift",
    )
    input_adapter_sha256 = directory_sha256(args.input_adapter)
    reference_sft_adapter_sha256 = directory_sha256(args.reference_sft_adapter)
    replay_calibration = load_pilot_replay_parity_calibration(
        args.replay_parity_calibration,
        protocol_sha256=protocol["sha256"],
        parity_contract=protocol["payload"]["rl"]["parity_contract"],
        learner_microbatch_size=args.microbatch_size,
        frozen_sft_adapter_sha256=reference_sft_adapter_sha256,
        frozen_sft_adapter_semantic_sha256=args.reference_sft_adapter_semantic_sha256,
    )
    _require(
        all(group.get("adapter_sha256") == input_adapter_sha256 for group in groups),
        "M6 RL behavior adapter bytes do not match learner input",
    )
    _require(
        all(
            group.get("protocol_sha256") == protocol["sha256"]
            and group.get("git_sha") == collection_producer_git_sha
            for group in groups
        ),
        "M6 RL group protocol/Git lineage drift",
    )
    _require(
        collection.get("policy_lineage", {}).get("adapter_sha256") == input_adapter_sha256
        and collection.get("policy_lineage", {}).get("adapter_semantic_sha256")
        == args.input_adapter_semantic_sha256,
        "M6 collection policy does not match learner input",
    )
    try:
        report = train_mini_policy_iteration(
            groups=groups,
            all_generated_action_tokens=int(collection["all_attempt_generated_action_tokens"]),
            base_model=args.base_model,
            input_adapter=args.input_adapter,
            input_adapter_semantic_sha256=args.input_adapter_semantic_sha256,
            reference_sft_adapter=args.reference_sft_adapter,
            reference_sft_adapter_semantic_sha256=args.reference_sft_adapter_semantic_sha256,
            input_optimizer=args.input_optimizer,
            output_root=args.output_dir,
            git_sha=git_sha,
            protocol_sha256=protocol["sha256"],
            iteration_index=args.iteration_index,
            seed=args.seed,
            method=args.method,
            pilot_authorization_sha256=authorization["content_sha256"],
            replay_parity_calibration=replay_calibration["payload"],
            replay_parity_calibration_file_sha256=sha256_file(
                args.replay_parity_calibration
            ),
            input_state_producer_git_sha=args.input_state_producer_git_sha,
            microbatch_size=args.microbatch_size,
        )
    except ReplayParityError as error:
        if args.parity_rejection_report is None:
            raise
        rejection = {
            "schema_version": "m6_replay_parity_rejection_v1",
            "complete": True,
            "decision": "SKIP_UNTRUSTED_GROUP_WITHOUT_UPDATE",
            "method": args.method,
            "seed": args.seed,
            "iteration_index": args.iteration_index,
            "git_sha": git_sha,
            "protocol_sha256": protocol["sha256"],
            "pilot_authorization_sha256": authorization["content_sha256"],
            "collection_report_content_sha256": collection["content_sha256"],
            "collection_producer_git_sha": collection_producer_git_sha,
            "input_adapter_sha256": input_adapter_sha256,
            "input_adapter_semantic_sha256": args.input_adapter_semantic_sha256,
            "all_attempt_generated_action_tokens": int(
                collection["all_attempt_generated_action_tokens"]
            ),
            "optimizer_updates_performed": 0,
            "failure": error.evidence,
        }
        rejection["content_sha256"] = sha256_json(rejection)
        atomic_write_json(args.parity_rejection_report, rejection)
        print(json.dumps(rejection, indent=2, sort_keys=True))
        raise SystemExit(42) from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

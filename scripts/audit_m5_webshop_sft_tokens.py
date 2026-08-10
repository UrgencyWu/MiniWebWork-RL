#!/usr/bin/env python3
"""Bind the M5 SFT corpus to the local tokenizer and completion-label budget."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import load_protocol, sha256_file  # noqa: E402
from miniwebwork.sft.m4_long_horizon_dataset import completion_only_token_statistics  # noqa: E402
from miniwebwork.webshop_rl import prompt as webshop_prompt  # noqa: E402
from miniwebwork.webshop_rl.actions import parse_command_output  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tokenizer_hashes(model_root: Path) -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
    hashes = {name: sha256_file(model_root / name) for name in names if (model_root / name).is_file()}
    _require(bool(hashes), "M5 base model has no tokenizer files")
    return hashes


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(bool(rows) and all(isinstance(row, dict) for row in rows), f"M5 SFT file is empty or malformed: {path}")
    return rows


def _verify_self_hash(payload: dict[str, Any], *, label: str) -> None:
    expected = dict(payload)
    observed = expected.pop("content_sha256", None)
    _require(observed == sha256_json(expected), f"{label} self-hash drift")


def _audit_split(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    git_sha: str,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = []
    labels = []
    forwards = []
    untruncated_forwards = []
    truncated = 0
    turns_by_task: Counter[str] = Counter()
    expected_keys = {
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
        "git_sha",
        "trajectory_content_sha256",
    }
    for row in rows:
        _require(set(row) == expected_keys, "M5 SFT row schema drift")
        _require(row.get("git_sha") == git_sha, "M5 SFT row Git lineage drift")
        sample_id = f"{row['task_id']}:{int(row['turn_index']):03d}"
        sample_ids.append(sample_id)
        parsed = parse_command_output(row.get("completion", ""))
        _require(parsed.schema_valid and parsed.strict_json_success, f"invalid strict SFT completion: {sample_id}")
        _require(parsed.action is not None and parsed.action.command == row.get("command"), f"SFT command drift: {sample_id}")
        prompt_messages = list(row.get("messages") or [])
        _require(
            row.get("prompt_sha256") == webshop_prompt.compute_message_hash(prompt_messages),
            f"SFT prompt hash drift: {sample_id}",
        )
        for hash_field in ("public_state_anchor_sha256", "trajectory_content_sha256"):
            value = str(row.get(hash_field) or "")
            _require(
                len(value) == 64 and all(character in "0123456789abcdef" for character in value),
                f"invalid {hash_field}: {sample_id}",
            )
        messages = prompt_messages + [{"role": "assistant", "content": row["completion"]}]
        statistics = completion_only_token_statistics(
            tokenizer,
            messages,
            chat_template_kwargs=chat_template_kwargs,
            max_length=max_length,
        )
        labels.append(int(statistics["effective_completion_label_tokens"]))
        forwards.append(int(statistics["forward_tokens"]))
        untruncated_forwards.append(int(statistics["untruncated_forward_tokens"]))
        truncated += int(bool(statistics["truncated"]))
        turns_by_task[str(row["task_id"])] += 1
    zero = sum(value == 0 for value in labels)
    _require(len(sample_ids) == len(set(sample_ids)), "M5 SFT sample IDs are not unique")
    _require(zero == 0, "M5 SFT contains zero-label samples")
    _require(truncated == 0, "M5 SFT contains truncated samples")
    return {
        "sample_count": len(rows),
        "unique_sample_count": len(set(sample_ids)),
        "task_count": len(turns_by_task),
        "effective_completion_label_tokens_per_epoch": sum(labels),
        "two_epoch_completion_label_token_exposure": 2 * sum(labels),
        "three_epoch_completion_label_token_exposure": 3 * sum(labels),
        "total_forward_tokens_per_epoch": sum(forwards),
        "zero_completion_label_sample_count": zero,
        "zero_completion_label_sample_fraction": zero / len(labels),
        "truncated_sample_count": truncated,
        "minimum_completion_label_tokens": min(labels),
        "maximum_completion_label_tokens": max(labels),
        "maximum_forward_sequence_tokens": max(forwards),
        "maximum_untruncated_forward_sequence_tokens": max(untruncated_forwards),
        "minimum_turns_per_task": min(turns_by_task.values()),
        "maximum_turns_per_task": max(turns_by_task.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_root = args.data_dir.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    protocol = load_protocol()
    sft = protocol["payload"]["sft"]
    max_length = int(sft["maximum_sequence_tokens"])
    chat_template_kwargs = dict(sft["chat_template_kwargs"])
    corpus_audit_path = data_root / "corpus_audit.json"
    _require(corpus_audit_path.is_file(), "M5 SFT corpus audit is missing")
    corpus_audit = json.loads(corpus_audit_path.read_text(encoding="utf-8"))
    _require(isinstance(corpus_audit, dict), "M5 SFT corpus audit is malformed")
    _verify_self_hash(corpus_audit, label="M5 SFT corpus audit")
    _require(corpus_audit.get("passed") is True, "M5 SFT corpus audit did not pass")
    _require(corpus_audit.get("protocol_sha256") == protocol["sha256"], "M5 SFT corpus protocol drift")
    _require(corpus_audit.get("git_sha") == protocol["git_sha"], "M5 SFT corpus Git lineage drift")
    corpus_files = corpus_audit.get("corpus_files")
    _require(isinstance(corpus_files, dict), "M5 SFT corpus file manifest is missing")
    for split in ("train", "dev"):
        split_path = data_root / f"{split}.jsonl"
        entry = corpus_files.get(split)
        _require(isinstance(entry, dict), f"M5 SFT {split} manifest is missing")
        _require(entry.get("sha256") == sha256_file(split_path), f"M5 SFT {split} file hash drift")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True, trust_remote_code=True)
    splits = {
        split: _audit_split(
            _load_rows(data_root / f"{split}.jsonl"),
            tokenizer,
            max_length=max_length,
            git_sha=protocol["git_sha"],
            chat_template_kwargs=chat_template_kwargs,
        )
        for split in ("train", "dev")
    }
    _require(splits["train"]["task_count"] == int(sft["train_task_count"]), "M5 SFT train task count drift")
    _require(splits["dev"]["task_count"] == int(sft["dev_task_count"]), "M5 SFT dev task count drift")
    token_floor = int(sft["minimum_effective_completion_label_token_exposure"])
    _require(
        splits["train"]["three_epoch_completion_label_token_exposure"] >= token_floor,
        "M5 SFT cannot reach the completion-label token floor within three epochs",
    )
    report = {
        "schema_version": "m5_webshop_sft_token_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "protocol_sha256": protocol["sha256"],
        "git_sha": protocol["git_sha"],
        "corpus_audit_sha256": sha256_file(corpus_audit_path),
        "train_sha256": sha256_file(data_root / "train.jsonl"),
        "dev_sha256": sha256_file(data_root / "dev.jsonl"),
        "base_model": str(base_model),
        "tokenizer_file_sha256": _tokenizer_hashes(base_model),
        "maximum_sequence_tokens": max_length,
        "chat_template_kwargs": chat_template_kwargs,
        "minimum_completion_label_token_exposure": token_floor,
        "label_tokenization_contract": "full chat template minus generation-prompt prefix, including assistant terminator, then right truncate",
        "splits": splits,
    }
    report["content_sha256"] = sha256_json(report)
    output = (args.output or data_root / "token_audit.json").expanduser().resolve()
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

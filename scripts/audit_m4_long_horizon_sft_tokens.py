#!/usr/bin/env python3
"""Audit unique SFT completion labels without repeating source examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.sft.m4_long_horizon_dataset import (
    DEFAULT_OUTPUT_DIR,
    SFT_DATASET_ID,
    completion_only_token_statistics,
    validate_verified_sft_corpus,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokenizer_hashes(model_root: Path) -> dict[str, str]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    hashes = {
        name: _sha256(model_root / name)
        for name in names
        if (model_root / name).is_file()
    }
    if not hashes:
        raise ValueError(f"no tokenizer files found in {model_root}")
    return hashes


def _audit_rows(rows, tokenizer, max_length: int) -> dict:
    sample_ids = []
    completion_counts = []
    untruncated_completion_counts = []
    forward_counts = []
    untruncated_forward_counts = []
    truncated_count = 0
    horizon_labels = Counter()
    family_labels = Counter()
    for row in rows:
        sample_ids.append(row["sample_id"])
        statistics = completion_only_token_statistics(
            tokenizer,
            row["messages"],
            chat_template_kwargs=row.get("chat_template_kwargs", {}),
            max_length=max_length,
        )
        label_count = int(statistics["effective_completion_label_tokens"])
        forward_count = int(statistics["forward_tokens"])
        completion_counts.append(label_count)
        untruncated_completion_counts.append(
            int(statistics["untruncated_completion_label_tokens"])
        )
        forward_counts.append(forward_count)
        untruncated_forward_counts.append(int(statistics["untruncated_forward_tokens"]))
        truncated_count += int(bool(statistics["truncated"]))
        horizon_labels[row.get("horizon_stratum", "unknown")] += label_count
        family_labels[row.get("task_family", "unknown")] += label_count
    zero_count = sum(count == 0 for count in completion_counts)
    return {
        "sample_count": len(rows),
        "unique_sample_count": len(set(sample_ids)),
        "duplicate_sample_count": len(rows) - len(set(sample_ids)),
        "effective_completion_label_tokens": sum(completion_counts),
        "untruncated_completion_label_tokens": sum(untruncated_completion_counts),
        "total_forward_tokens": sum(forward_counts),
        "total_untruncated_forward_tokens": sum(untruncated_forward_counts),
        "truncated_sample_count": truncated_count,
        "zero_completion_label_sample_count": zero_count,
        "zero_completion_label_sample_fraction": zero_count / len(rows) if rows else 0.0,
        "minimum_completion_label_tokens": min(completion_counts) if completion_counts else None,
        "maximum_completion_label_tokens": max(completion_counts) if completion_counts else None,
        "maximum_forward_sequence_tokens": max(forward_counts) if forward_counts else None,
        "maximum_untruncated_forward_sequence_tokens": (
            max(untruncated_forward_counts) if untruncated_forward_counts else None
        ),
        "completion_label_tokens_by_horizon": dict(sorted(horizon_labels.items())),
        "completion_label_tokens_by_task_family": dict(sorted(family_labels.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_length != 6144:
        raise ValueError("focused SFT token audit is frozen at max_length=6144")
    data_dir = args.data_dir.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    validation = validate_verified_sft_corpus(data_dir, require_full_roster=True)
    if not validation["valid"]:
        raise ValueError(f"cannot token-audit invalid SFT corpus: {validation['errors']}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), local_files_only=True, trust_remote_code=True
    )
    splits = {}
    for split, filename in (("train", "train.jsonl"), ("dev", "valid.jsonl")):
        rows = [
            json.loads(line)
            for line in (data_dir / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        splits[split] = _audit_rows(rows, tokenizer, args.max_length)
        if splits[split]["duplicate_sample_count"] != 0:
            raise ValueError(f"{split} SFT rows are not unique")
        if splits[split]["zero_completion_label_sample_fraction"] != 0.0:
            raise ValueError(f"{split} SFT rows contain zero-label examples")
    report = {
        "schema_version": "m4_long_horizon_sft_token_audit_v1",
        "dataset_id": SFT_DATASET_ID,
        "corpus_manifest_sha256": _sha256(data_dir / "manifest.json"),
        "train_sha256": _sha256(data_dir / "train.jsonl"),
        "valid_sha256": _sha256(data_dir / "valid.jsonl"),
        "base_model": str(base_model),
        "tokenizer_file_sha256": _tokenizer_hashes(base_model),
        "max_length": args.max_length,
        "label_tokenization_contract": "full chat template minus generation prompt prefix, including assistant terminator, then right truncate",
        "repetition_policy": "none",
        "splits": splits,
        "passed": True,
    }
    output = (args.output or (data_dir / "token_audit.json")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

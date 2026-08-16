#!/usr/bin/env python3
"""Merge Phase10-C LoRA deltas directly into the complete Raw35 shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, directory_sha256  # noqa: E402
from miniwebwork.long_horizon_rl.model_manifest import (  # noqa: E402
    build_base_model_manifest,
    validate_base_model_manifest,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def portable_lora_target(key: str) -> tuple[str, str]:
    """Return the complete-model weight key and whether this is A or B."""

    prefix = "base_model.model.model."
    _require(key.startswith(prefix), f"Phase10-C merge LoRA prefix drift: {key}")
    for component in ("A", "B"):
        suffix = f".lora_{component}.weight"
        if key.endswith(suffix):
            module = key[len(prefix) : -len(suffix)]
            return f"model.language_model.{module}.weight", component
    raise ValueError(f"Phase10-C merge found a non-LoRA tensor: {key}")


def _adapter_pairs(adapter: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    portable = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    staged: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in portable.items():
        target, component = portable_lora_target(key)
        staged[target][component] = tensor
    _require(all(set(value) == {"A", "B"} for value in staged.values()), "Phase10-C incomplete LoRA pair")
    return {key: (value["A"], value["B"]) for key, value in staged.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-lora-r", type=int, default=8)
    parser.add_argument("--expected-lora-alpha", type=int, default=16)
    parser.add_argument("--expected-target-count", type=int, default=190)
    args = parser.parse_args()

    base_model = args.base_model.expanduser().resolve()
    base_manifest = args.base_model_manifest.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    report = args.report.expanduser().resolve()
    _require(args.expected_git_sha == _git_sha(), "Phase10-C merge Git drift")
    _require(base_model.is_dir() and adapter.is_dir(), "Phase10-C merge input is missing")
    _require(not output.exists() and not manifest.exists() and not report.exists(), "Phase10-C merge refuses overwrite")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Phase10-C shard merge requires one GPU")
    source_audit = validate_base_model_manifest(
        path=base_manifest, expected_base_model=base_model, verify_files=True
    )
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    _require(
        adapter_config.get("r") == args.expected_lora_r
        and adapter_config.get("lora_alpha") == args.expected_lora_alpha,
             "Phase10-C merge LoRA scale drift")
    scale = float(adapter_config["lora_alpha"]) / float(adapter_config["r"])
    pairs = _adapter_pairs(adapter)
    _require(len(pairs) == args.expected_target_count, "Phase10-C merge target count drift")
    weight_index = json.loads((base_model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = weight_index["weight_map"]
    _require(set(pairs).issubset(weight_map), "Phase10-C merge target missing from Raw35")
    by_shard: dict[str, set[str]] = defaultdict(set)
    for target in pairs:
        by_shard[weight_map[target]].add(target)

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.mkdir(parents=True)
    delta_norms: dict[str, float] = {}
    try:
        for source in sorted(base_model.iterdir()):
            destination = temporary / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.suffix != ".safetensors":
                shutil.copy2(source, destination)
        for shard in sorted(set(weight_map.values())):
            source = base_model / shard
            tensors = load_file(str(source), device="cpu")
            for target in sorted(by_shard.get(shard, ())):
                a, b = pairs[target]
                base = tensors[target]
                delta = torch.matmul(b.float().cuda(), a.float().cuda()).mul_(scale)
                delta_norms[target] = float(torch.linalg.vector_norm(delta).cpu())
                tensors[target] = (base.cuda() + delta.to(base.dtype)).cpu()
                del delta
            with safe_open(source, framework="pt", device="cpu") as opened:
                metadata = opened.metadata()
            save_file(tensors, str(destination := temporary / shard), metadata=metadata)
            del tensors
            torch.cuda.empty_cache()
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    built = build_base_model_manifest(base_model=output, destination=manifest)
    checked = validate_base_model_manifest(path=manifest, expected_base_model=output, verify_files=False)
    payload = {
        "schema_version": "m6_phase10c_merged_teacher_lora_v2",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "producer_git_sha": args.expected_git_sha,
        "base_model": str(base_model),
        "base_model_manifest_sha256": source_audit["sha256"],
        "base_model_functional_sha256": source_audit["payload"]["functional_file_set_sha256"],
        "adapter": str(adapter),
        "adapter_directory_sha256": directory_sha256(adapter),
        "merged_model": str(output),
        "merged_model_manifest": str(manifest),
        "merged_model_manifest_sha256": built["sha256"],
        "merged_model_functional_sha256": checked["payload"]["functional_file_set_sha256"],
        "merge_method": "complete_raw35_shard_lora_delta_v2",
        "adapter_tensor_count": len(pairs) * 2,
        "merged_target_count": len(pairs),
        "expected_lora_r": args.expected_lora_r,
        "expected_lora_alpha": args.expected_lora_alpha,
        "expected_target_count": args.expected_target_count,
        "lora_scale": scale,
        "all_delta_norms_finite_positive": all(value > 0 and torch.isfinite(torch.tensor(value)) for value in delta_norms.values()),
        "unchanged_weight_count": len(weight_map) - len(pairs),
    }
    _require(payload["all_delta_norms_finite_positive"], "Phase10-C merge produced an invalid delta")
    atomic_write_json(report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

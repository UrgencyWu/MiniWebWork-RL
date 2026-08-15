"""Deterministic PEFT-to-vLLM adapter views for Qwen3.5.

Transformers loads the text-only Qwen3.5 policy as ``Qwen3_5ForCausalLM`` and
therefore saves PEFT keys below ``model.layers``.  vLLM 0.17 resolves the same
checkpoint as ``Qwen3_5ForConditionalGeneration``; its language module lives
below ``language_model.model.layers``.  vLLM's inherited HF mapper does not
rewrite the text-only PEFT prefix, so feeding the canonical adapter directory
directly can result in a successfully registered but ineffective adapter.

The canonical PEFT adapter remains the policy artifact used by the learner.
This module creates a small, deterministic inference view that changes names
only.  Every tensor, layer, A/B pair, source hash, base-model config hash, and
name mapping is audited before the view can be used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    atomic_write_json,
    canonical_json_bytes,
    directory_sha256,
    sha256_file,
    sha256_json,
)

VLLM_ADAPTER_VIEW_SCHEMA = "m4_qwen35_vllm_adapter_view_v1"
VLLM_ADAPTER_MAPPING_CONTRACT = (
    "qwen35_peft_model_layers_to_conditional_language_model_v1"
)
SOURCE_KEY_PREFIX = "base_model.model.model."
VLLM_KEY_PREFIX = "base_model.model.model.language_model."
VIEW_MANIFEST_NAME = "m4_vllm_adapter_view.json"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_TENSORS_NAME = "adapter_model.safetensors"

_LORA_KEY = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>[0-9]+)\."
    r"(?P<family>mlp|self_attn|linear_attn)\.(?P<target>[A-Za-z0-9_]+)\."
    r"lora_(?P<side>A|B)\.weight$"
)
_VLLM_LORA_KEY = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\."
    r"(?P<layer>[0-9]+)\.(?P<family>mlp|self_attn|linear_attn)\."
    r"(?P<target>[A-Za-z0-9_]+)\.lora_(?P<side>A|B)\.weight$"
)
_MLP_TARGETS = frozenset({"down_proj", "gate_proj", "up_proj"})
_ATTENTION_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_EXPECTED_TARGETS = _MLP_TARGETS | _ATTENTION_TARGETS
SUPPORTED_LORA_RANK = 16
_PHASE10C_LINEAR_TARGETS = frozenset(
    {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
)
_PHASE10C_TARGETS = _ATTENTION_TARGETS | _PHASE10C_LINEAR_TARGETS
PHASE10C_LORA_RANK = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON payload must be an object: {path}")
    return payload


def _model_layout(base_model: Path) -> dict[str, Any]:
    config_path = Path(base_model).expanduser().resolve() / "config.json"
    payload = _load_json(config_path)
    text = payload.get("text_config", payload)
    _require(isinstance(text, dict), "Qwen3.5 text config is malformed")
    layer_types = text.get("layer_types")
    number = text.get("num_hidden_layers")
    _require(
        isinstance(layer_types, list)
        and bool(layer_types)
        and all(value in {"linear_attention", "full_attention"} for value in layer_types),
        "Qwen3.5 layer_types contract drift",
    )
    _require(number == len(layer_types), "Qwen3.5 hidden-layer count drift")
    full_attention = [
        index for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]
    _require(bool(full_attention), "Qwen3.5 config has no full-attention layers")
    return {
        "base_model_config_sha256": sha256_file(config_path),
        "architectures": payload.get("architectures", []),
        "number_of_layers": number,
        "layer_types": layer_types,
        "full_attention_layer_indices": full_attention,
    }


def _adapter_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    targets = config.get("target_modules")
    _require(isinstance(targets, list), "adapter target_modules contract drift")
    target_set = frozenset(str(value) for value in targets)
    rank = config.get("r")
    if rank == SUPPORTED_LORA_RANK and target_set == _EXPECTED_TARGETS:
        return {
            "name": "dense_mlp_full_attention_v1",
            "rank": SUPPORTED_LORA_RANK,
            "targets": _EXPECTED_TARGETS,
        }
    if rank == PHASE10C_LORA_RANK and target_set == _PHASE10C_TARGETS:
        return {
            "name": "phase10c_text_token_mixers_v1",
            "rank": PHASE10C_LORA_RANK,
            "targets": _PHASE10C_TARGETS,
        }
    raise ValueError(
        f"unsupported adapter profile: rank={rank} targets={sorted(target_set)}"
    )


def _expected_modules(layout: Mapping[str, Any], *, profile: Mapping[str, Any]) -> set[str]:
    modules: set[str] = set()
    if profile["name"] == "dense_mlp_full_attention_v1":
        for layer in range(int(layout["number_of_layers"])):
            modules.update(
                f"layers.{layer}.mlp.{target}" for target in sorted(_MLP_TARGETS)
            )
        for layer in layout["full_attention_layer_indices"]:
            modules.update(
                f"layers.{layer}.self_attn.{target}"
                for target in sorted(_ATTENTION_TARGETS)
            )
    else:
        for layer, layer_type in enumerate(layout["layer_types"]):
            family = "self_attn" if layer_type == "full_attention" else "linear_attn"
            targets = _ATTENTION_TARGETS if family == "self_attn" else _PHASE10C_LINEAR_TARGETS
            modules.update(
                f"layers.{layer}.{family}.{target}" for target in sorted(targets)
            )
    return modules


def _load_tensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - frozen training env has it
        raise ValueError("safetensors is required for adapter-view validation") from exc
    _require(path.is_file(), f"adapter tensor file is missing: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def _logical_key(name: str, *, view: bool) -> str:
    prefix = VLLM_KEY_PREFIX if view else SOURCE_KEY_PREFIX
    _require(name.startswith(prefix), f"adapter tensor prefix drift: {name}")
    return name[len(prefix) :]


def _semantic_tensor_sha256(tensors: Mapping[str, Any], *, view: bool) -> str:
    """Hash tensor meaning independently of the PEFT/vLLM namespace."""

    import torch

    digest = hashlib.sha256()
    logical_names: set[str] = set()
    for name in sorted(tensors):
        logical = _logical_key(name, view=view)
        _require(logical not in logical_names, f"duplicate logical adapter tensor: {logical}")
        logical_names.add(logical)
        tensor = tensors[name].detach().cpu().contiguous()
        metadata = canonical_json_bytes(
            {
                "logical_key": logical,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _audit_tensor_structure(
    tensors: Mapping[str, Any],
    *,
    layout: Mapping[str, Any],
    profile: Mapping[str, Any],
    view: bool,
) -> dict[str, Any]:
    pattern = _VLLM_LORA_KEY if view else _LORA_KEY
    prefix = VLLM_KEY_PREFIX if view else SOURCE_KEY_PREFIX
    modules: dict[str, dict[str, Any]] = {}
    rank = int(profile["rank"])
    for name, tensor in tensors.items():
        match = pattern.fullmatch(name)
        _require(match is not None, f"unsupported adapter tensor key: {name}")
        layer = int(match.group("layer"))
        family = match.group("family")
        target = match.group("target")
        side = match.group("side")
        _require(0 <= layer < layout["number_of_layers"], f"adapter layer out of range: {name}")
        allowed = {
            "mlp": _MLP_TARGETS,
            "self_attn": _ATTENTION_TARGETS,
            "linear_attn": _PHASE10C_LINEAR_TARGETS,
        }[family]
        _require(target in allowed, f"adapter target/family mismatch: {name}")
        _require(getattr(tensor, "ndim", None) == 2, f"LoRA tensor is not rank two: {name}")
        if side == "A":
            _require(tensor.shape[0] == rank, f"LoRA A rank drift: {name}")
        else:
            _require(tensor.shape[1] == rank, f"LoRA B rank drift: {name}")
        logical_module = name[len(prefix) :].rsplit(".lora_", 1)[0]
        sides = modules.setdefault(logical_module, {})
        _require(side not in sides, f"duplicate LoRA side: {name}")
        sides[side] = tensor

    expected = _expected_modules(layout, profile=profile)
    actual = set(modules)
    _require(
        actual == expected,
        "adapter module set drift: "
        f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}",
    )
    for module, sides in modules.items():
        _require(set(sides) == {"A", "B"}, f"incomplete LoRA A/B module pair: {module}")
        _require(
            sides["A"].shape[0] == sides["B"].shape[1] == rank,
            f"LoRA module rank mismatch: {module}",
        )
    _require(len(tensors) == len(expected) * 2, "adapter tensor count drift")
    return {
        "tensor_count": len(tensors),
        "module_count": len(modules),
        "semantic_tensor_sha256": _semantic_tensor_sha256(tensors, view=view),
    }


def _validate_adapter_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    _adapter_profile(payload)
    _require(payload.get("peft_type") == "LORA", "adapter PEFT type drift")
    return payload


def _view_manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    return sha256_json(value)


def validate_vllm_adapter_view(
    *,
    source_adapter: Path,
    view_directory: Path,
    base_model: Path,
) -> dict[str, Any]:
    """Validate a persisted view and prove byte-identical tensor semantics."""

    source = Path(source_adapter).expanduser().resolve()
    view = Path(view_directory).expanduser().resolve()
    base = Path(base_model).expanduser().resolve()
    _require(source.is_dir(), "canonical PEFT adapter directory is missing")
    _require(view.is_dir(), "vLLM adapter view directory is missing")
    manifest_path = view / VIEW_MANIFEST_NAME
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == VLLM_ADAPTER_VIEW_SCHEMA,
        "vLLM adapter-view schema drift",
    )
    _require(
        manifest.get("mapping_contract") == VLLM_ADAPTER_MAPPING_CONTRACT,
        "vLLM adapter-view mapping contract drift",
    )
    _require(
        manifest.get("manifest_sha256") == _view_manifest_content_sha256(manifest),
        "vLLM adapter-view manifest hash drift",
    )
    _require(
        manifest.get("source_adapter_directory_sha256") == directory_sha256(source),
        "vLLM adapter-view source_adapter_directory_sha256 drift",
    )
    _require(
        manifest.get("source_adapter_config_sha256")
        == sha256_file(source / ADAPTER_CONFIG_NAME),
        "vLLM adapter-view source_adapter_config_sha256 drift",
    )
    _require(
        manifest.get("source_adapter_tensor_file_sha256")
        == sha256_file(source / ADAPTER_TENSORS_NAME),
        "vLLM adapter-view source_adapter_tensor_file_sha256 drift",
    )
    source_config = _validate_adapter_config(source / ADAPTER_CONFIG_NAME)
    view_config = _validate_adapter_config(view / ADAPTER_CONFIG_NAME)
    profile = _adapter_profile(source_config)
    _require(_adapter_profile(view_config) == profile, "vLLM adapter profile drift")
    _require(
        (source / ADAPTER_CONFIG_NAME).read_bytes()
        == (view / ADAPTER_CONFIG_NAME).read_bytes(),
        "vLLM view adapter_config differs from canonical PEFT adapter",
    )
    layout = _model_layout(base)
    source_tensors = _load_tensors(source / ADAPTER_TENSORS_NAME)
    view_tensors = _load_tensors(view / ADAPTER_TENSORS_NAME)
    source_audit = _audit_tensor_structure(
        source_tensors, layout=layout, profile=profile, view=False
    )
    view_audit = _audit_tensor_structure(
        view_tensors, layout=layout, profile=profile, view=True
    )
    expected_mapping = {
        SOURCE_KEY_PREFIX + _logical_key(name, view=False):
        VLLM_KEY_PREFIX + _logical_key(name, view=False)
        for name in source_tensors
    }
    _require(set(view_tensors) == set(expected_mapping.values()), "vLLM tensor key mapping drift")
    _require(
        source_audit["semantic_tensor_sha256"]
        == view_audit["semantic_tensor_sha256"],
        "PEFT/vLLM adapter tensor semantics differ",
    )

    expected_manifest = {
        "source_adapter_directory_sha256": directory_sha256(source),
        "source_adapter_config_sha256": sha256_file(source / ADAPTER_CONFIG_NAME),
        "source_adapter_tensor_file_sha256": sha256_file(source / ADAPTER_TENSORS_NAME),
        "base_model_config_sha256": layout["base_model_config_sha256"],
        "source_key_prefix": SOURCE_KEY_PREFIX,
        "vllm_key_prefix": VLLM_KEY_PREFIX,
        "number_of_layers": layout["number_of_layers"],
        "full_attention_layer_indices": layout["full_attention_layer_indices"],
        "target_modules": sorted(profile["targets"]),
        "tensor_count": source_audit["tensor_count"],
        "module_count": source_audit["module_count"],
        "semantic_tensor_sha256": source_audit["semantic_tensor_sha256"],
        "vllm_adapter_tensor_file_sha256": sha256_file(view / ADAPTER_TENSORS_NAME),
        "complete": True,
    }
    if profile["name"] == "phase10c_text_token_mixers_v1":
        expected_manifest.update({
            "adapter_profile": profile["name"],
            "lora_rank": profile["rank"],
        })
    for field, expected in expected_manifest.items():
        _require(manifest.get(field) == expected, f"vLLM adapter-view {field} drift")
    return {
        "source_adapter_directory_sha256": expected_manifest[
            "source_adapter_directory_sha256"
        ],
        "view_directory_sha256": directory_sha256(view),
        "semantic_tensor_sha256": source_audit["semantic_tensor_sha256"],
        "tensor_count": source_audit["tensor_count"],
        "module_count": source_audit["module_count"],
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


def build_vllm_adapter_view(
    *,
    source_adapter: Path,
    destination: Path,
    base_model: Path,
) -> dict[str, Any]:
    """Build or revalidate an atomic deterministic vLLM inference view."""

    source = Path(source_adapter).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    base = Path(base_model).expanduser().resolve()
    _require(source.is_dir(), "canonical PEFT adapter directory is missing")
    _require(base.is_dir(), "base model directory is missing")
    if target.exists():
        return validate_vllm_adapter_view(
            source_adapter=source,
            view_directory=target,
            base_model=base,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{os.getpid()}-{time.time_ns()}"
    _require(not staging.exists(), "vLLM adapter-view staging collision")
    staging.mkdir()
    try:
        source_config = source / ADAPTER_CONFIG_NAME
        source_tensor_path = source / ADAPTER_TENSORS_NAME
        source_config_payload = _validate_adapter_config(source_config)
        profile = _adapter_profile(source_config_payload)
        layout = _model_layout(base)
        source_tensors = _load_tensors(source_tensor_path)
        source_audit = _audit_tensor_structure(
            source_tensors,
            layout=layout,
            profile=profile,
            view=False,
        )
        mapped = {
            VLLM_KEY_PREFIX + _logical_key(name, view=False): tensor
            for name, tensor in sorted(source_tensors.items())
        }
        _audit_tensor_structure(mapped, layout=layout, profile=profile, view=True)

        shutil.copy2(source_config, staging / ADAPTER_CONFIG_NAME)
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - frozen training env has it
            raise ValueError("safetensors is required to build a vLLM adapter view") from exc
        save_file(
            mapped,
            staging / ADAPTER_TENSORS_NAME,
            # A single metadata key avoids Rust HashMap ordering differences
            # between otherwise byte-identical safetensors serializations.
            # The full mapping contract remains hash-bound in the manifest.
            metadata={"format": "pt"},
        )
        manifest = {
            "schema_version": VLLM_ADAPTER_VIEW_SCHEMA,
            "mapping_contract": VLLM_ADAPTER_MAPPING_CONTRACT,
            "source_adapter_directory_sha256": directory_sha256(source),
            "source_adapter_config_sha256": sha256_file(source_config),
            "source_adapter_tensor_file_sha256": sha256_file(source_tensor_path),
            "base_model_config_sha256": layout["base_model_config_sha256"],
            "source_key_prefix": SOURCE_KEY_PREFIX,
            "vllm_key_prefix": VLLM_KEY_PREFIX,
            "number_of_layers": layout["number_of_layers"],
            "full_attention_layer_indices": layout[
                "full_attention_layer_indices"
            ],
            "target_modules": sorted(profile["targets"]),
            "tensor_count": source_audit["tensor_count"],
            "module_count": source_audit["module_count"],
            "semantic_tensor_sha256": source_audit["semantic_tensor_sha256"],
            "vllm_adapter_tensor_file_sha256": sha256_file(
                staging / ADAPTER_TENSORS_NAME
            ),
            "complete": True,
        }
        if profile["name"] == "phase10c_text_token_mixers_v1":
            manifest.update({
                "adapter_profile": profile["name"],
                "lora_rank": profile["rank"],
            })
        manifest["manifest_sha256"] = _view_manifest_content_sha256(manifest)
        atomic_write_json(staging / VIEW_MANIFEST_NAME, manifest)
        for file_path in staging.iterdir():
            descriptor = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(staging)
        os.rename(staging, target)
        _fsync_directory(target.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return validate_vllm_adapter_view(
        source_adapter=source,
        view_directory=target,
        base_model=base,
    )

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file
safe_open = pytest.importorskip("safetensors").safe_open

from miniwebwork.long_horizon_rl.adapter_view import (
    ADAPTER_TENSORS_NAME,
    SOURCE_KEY_PREFIX,
    VLLM_KEY_PREFIX,
    build_vllm_adapter_view,
    validate_vllm_adapter_view,
)


TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _write_base_model(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 2,
                    "layer_types": ["linear_attention", "full_attention"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _source_tensors(*, include_extra_attention: bool = False) -> dict[str, torch.Tensor]:
    modules = []
    for layer in range(2):
        modules.extend(
            f"layers.{layer}.mlp.{target}"
            for target in ("down_proj", "gate_proj", "up_proj")
        )
    modules.extend(
        f"layers.1.self_attn.{target}"
        for target in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    if include_extra_attention:
        modules.append("layers.0.self_attn.q_proj")
    tensors = {}
    for index, module in enumerate(modules):
        tensors[f"{SOURCE_KEY_PREFIX}{module}.lora_A.weight"] = (
            torch.arange(32, dtype=torch.float32).reshape(16, 2) + index
        )
        tensors[f"{SOURCE_KEY_PREFIX}{module}.lora_B.weight"] = (
            torch.arange(48, dtype=torch.float32).reshape(3, 16) - index
        )
    return tensors


def _write_source(path: Path, *, tensors=None) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": TARGETS,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    save_file(
        tensors or _source_tensors(),
        path / ADAPTER_TENSORS_NAME,
        metadata={"format": "pt"},
    )
    return path


def _write_phase10c_source(path: Path) -> Path:
    path.mkdir()
    targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    ]
    (path / "adapter_config.json").write_text(
        json.dumps({
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": targets,
        }, sort_keys=True),
        encoding="utf-8",
    )
    modules = [
        f"layers.0.linear_attn.{target}"
        for target in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
    ] + [
        f"layers.1.self_attn.{target}"
        for target in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    tensors = {}
    for index, module in enumerate(modules):
        tensors[f"{SOURCE_KEY_PREFIX}{module}.lora_A.weight"] = (
            torch.arange(16, dtype=torch.float32).reshape(8, 2) + index
        )
        tensors[f"{SOURCE_KEY_PREFIX}{module}.lora_B.weight"] = (
            torch.arange(24, dtype=torch.float32).reshape(3, 8) - index
        )
    save_file(tensors, path / ADAPTER_TENSORS_NAME, metadata={"format": "pt"})
    return path


def test_builds_deterministic_semantically_identical_vllm_view(tmp_path):
    base = _write_base_model(tmp_path / "base")
    source = _write_source(tmp_path / "source")
    first = build_vllm_adapter_view(
        source_adapter=source,
        destination=tmp_path / "view-a",
        base_model=base,
    )
    second = build_vllm_adapter_view(
        source_adapter=source,
        destination=tmp_path / "view-b",
        base_model=base,
    )
    assert first["tensor_count"] == 20
    assert first["module_count"] == 10
    assert first["semantic_tensor_sha256"] == second["semantic_tensor_sha256"]
    assert first["view_directory_sha256"] == second["view_directory_sha256"]
    with safe_open(tmp_path / "view-a" / ADAPTER_TENSORS_NAME, framework="pt") as handle:
        keys = list(handle.keys())
    assert len(keys) == 20
    assert all(key.startswith(VLLM_KEY_PREFIX) for key in keys)
    assert not any(key.startswith(SOURCE_KEY_PREFIX + "layers") for key in keys)


def test_builds_phase10c_rank8_text_token_mixer_view(tmp_path):
    base = _write_base_model(tmp_path / "base")
    source = _write_phase10c_source(tmp_path / "source")
    audit = build_vllm_adapter_view(
        source_adapter=source,
        destination=tmp_path / "view",
        base_model=base,
    )
    assert audit["tensor_count"] == 18
    assert audit["module_count"] == 9
    assert audit["manifest"]["adapter_profile"] == "phase10c_text_token_mixers_v1"
    assert audit["manifest"]["lora_rank"] == 8


def test_existing_view_is_revalidated_and_source_drift_fails_closed(tmp_path):
    base = _write_base_model(tmp_path / "base")
    source = _write_source(tmp_path / "source")
    view = tmp_path / "view"
    first = build_vllm_adapter_view(
        source_adapter=source,
        destination=view,
        base_model=base,
    )
    assert build_vllm_adapter_view(
        source_adapter=source,
        destination=view,
        base_model=base,
    )["view_directory_sha256"] == first["view_directory_sha256"]
    with (source / "adapter_config.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="source_adapter_.*sha256 drift"):
        validate_vllm_adapter_view(
            source_adapter=source,
            view_directory=view,
            base_model=base,
        )


def test_rejects_missing_lora_pair_without_publishing_view(tmp_path):
    base = _write_base_model(tmp_path / "base")
    tensors = _source_tensors()
    tensors.pop(next(key for key in tensors if ".lora_B." in key))
    source = _write_source(tmp_path / "source", tensors=tensors)
    view = tmp_path / "view"
    with pytest.raises(ValueError, match="module set drift|A/B module pair|tensor count"):
        build_vllm_adapter_view(
            source_adapter=source,
            destination=view,
            base_model=base,
        )
    assert not view.exists()


def test_rejects_attention_lora_on_linear_attention_layer(tmp_path):
    base = _write_base_model(tmp_path / "base")
    source = _write_source(
        tmp_path / "source",
        tensors=_source_tensors(include_extra_attention=True),
    )
    with pytest.raises(ValueError, match="module set drift"):
        build_vllm_adapter_view(
            source_adapter=source,
            destination=tmp_path / "view",
            base_model=base,
        )

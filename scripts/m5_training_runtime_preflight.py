#!/usr/bin/env python3
"""Audit the shared M5 learner/rollout environment without allocating a GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from miniwebwork.long_horizon_rl.contracts import atomic_write_json, sha256_json  # noqa: E402
from miniwebwork.m5_webshop_protocol import load_protocol, sha256_file  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_runtime(*, base_model: Path, output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    contract = protocol["payload"]["training_runtime"]
    python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    _require(python_major_minor == contract["python_major_minor"], "M5 training Python drift")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU runtime audit must hide every GPU")
    packages = {
        name: importlib.metadata.version(name)
        for name in contract["critical_packages"]
    }
    _require(packages == contract["critical_packages"], "M5 training critical-package drift")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import requests  # noqa: F401

    request_warnings = [
        f"{item.category.__name__}: {item.message}"
        for item in caught
        if item.category.__name__ == "RequestsDependencyWarning"
    ]
    _require(not request_warnings, "RequestsDependencyWarning remains in the M5 training runtime")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    pip_lines = [
        line.strip()
        for line in (pip_check.stdout + pip_check.stderr).splitlines()
        if line.strip()
    ]
    _require(pip_check.returncode == 1, "M5 pip check unexpectedly changed status")
    _require(pip_lines == [contract["known_pip_check_exception"]], "M5 pip check has an unregistered conflict")

    model_root = Path(base_model).expanduser().resolve()
    config_path = model_root / "config.json"
    _require(config_path.is_file(), "M5 base-model config is missing")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture = "Qwen3_5ForConditionalGeneration"
    _require(model_config.get("model_type") == "qwen3_5", "M5 base-model type drift")
    _require(model_config.get("architectures") == [architecture], "M5 base-model architecture drift")
    from vllm.model_executor.models.registry import ModelRegistry

    supported_architectures = ModelRegistry.get_supported_archs()
    _require(architecture in supported_architectures, "vLLM does not natively register Qwen3.5")

    import torch

    _require(torch.cuda.is_available() is False, "CPU runtime audit unexpectedly initialized a GPU")
    report = {
        "schema_version": "m5_training_runtime_audit_v1",
        "study_id": protocol["payload"]["study_id"],
        "passed": True,
        "formal_training": False,
        "protocol_sha256": protocol["sha256"],
        "python": sys.version,
        "packages": packages,
        "torch_runtime_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "requests_dependency_warnings": request_warnings,
        "pip_check_returncode": pip_check.returncode,
        "pip_check_known_exception": pip_lines,
        "base_model": str(model_root),
        "base_model_config_sha256": sha256_file(config_path),
        "model_architecture": architecture,
        "vllm_native_architecture_registered": True,
        "gpu_generation_still_required": True,
        "repair_requirements_sha256": sha256_file(PROJECT_ROOT / "requirements.m5-training-runtime.txt"),
    }
    report["content_sha256"] = sha256_json(report)
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("/data/share/model/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_runtime(base_model=args.base_model, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

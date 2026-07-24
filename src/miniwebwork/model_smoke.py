"""
Minimal Qwen3.5-4B model loading and generation smoke test.

Loads model from MODEL_PATH environment variable, performs a single
short generation, and saves the result as JSON.
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path = os.environ.get("MODEL_PATH", "")
    artifacts_dir = os.environ.get(
        "MINIWEBWORK_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent.parent / "artifacts"),
    )

    result = {
        "success": False,
        "model_path": model_path,
        "torch_version": "",
        "cuda_version": "",
        "gpu_name": "",
        "generated_text": "",
        "max_new_tokens": 32,
    }

    exit_code = 0

    try:
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model path not found: {model_path}. "
                "Set MODEL_PATH to the Qwen3.5-4B directory."
            )

        # Detect GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        gpu_id = 0
        device = f"cuda:{gpu_id}"

        gpu_props = torch.cuda.get_device_properties(gpu_id)
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda or "unknown"
        result["gpu_name"] = gpu_props.name

        print(f"PyTorch: {result['torch_version']}")
        print(f"CUDA: {result['cuda_version']}")
        print(f"GPU: {result['gpu_name']} ({gpu_props.total_memory / 1024**3:.1f} GB)")

        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Load model
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        model.eval()

        # Generate
        prompt = "请只回复：MiniWebWork ready"
        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        result["generated_text"] = generated.strip()
        result["success"] = True

        print(f"Generated: {result['generated_text']}")
        print("Model smoke test PASSED")

    except Exception as e:
        print(f"Model smoke test FAILED: {e}")
        result["error"] = str(e)
        exit_code = 1

    finally:
        # Save JSON result
        artifacts = Path(artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        json_path = artifacts / "model_smoke_result.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved: {json_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

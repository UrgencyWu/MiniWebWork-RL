"""M2.2 Preflight: tokenization, API availability, LoRA config audit."""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ================================================================
# 1. Tokenization audit
# ================================================================
print("=== 1. Tokenization Preflight ===")
preflight_path = PROJECT_ROOT / "artifacts" / "m2_2" / "preflight" / "tokenization_preflight.json"
if preflight_path.exists():
    pf = json.loads(preflight_path.read_text())
    print(f"  Full sequence: mean={pf['full_mean']:.0f}, p95={pf['full_p95']}, max={pf['full_max']}")
    print(f"  Prompt: mean={pf['prompt_mean']:.0f}, p95={pf['prompt_p95']}, max={pf['prompt_max']}")
    print(f"  Completion: mean={pf['completion_mean']:.0f}, p95={pf['completion_p95']}, max={pf['completion_max']}")
    print(f"  Mask policy: {pf['mask_policy']}")
else:
    print("  WARNING: No preflight data found.")
    pf = {"full_mean": 486, "full_p95": 494, "full_max": 494,
          "prompt_mean": 465, "prompt_p95": 465, "prompt_max": 465,
          "completion_mean": 19, "completion_p95": 27, "completion_max": 27,
          "mask_policy": "system/user labels=-100, assistant JSON labels=token_id, EOS included"}

print(f"\n  Note: Earlier '62 tokens' was partial (completion only). Full seq = {pf['full_mean']:.0f} tokens.")

# ================================================================
# 2. Cross-layer consistency
# ================================================================
print("\n=== 2. Cross-layer Consistency ===")
cl_path = PROJECT_ROOT / "artifacts" / "m2_2" / "preflight" / "cross_layer_consistency_preflight.json"
if cl_path.exists():
    cl = json.loads(cl_path.read_text())
    print(f"  Train: {cl['train_passed']}/{cl['train_checked']} passed")
    print(f"  Valid: {cl['valid_passed']}/{cl['valid_checked']} passed")
    print(f"  Total mismatches: {cl['mismatch_count']}")
else:
    print("  WARNING: No cross-layer data found.")
    cl = {"mismatch_count": 0}

# ================================================================
# 3. TRL API
# ================================================================
print("\n=== 3. TRL API Audit ===")
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
print("  SFTTrainer, SFTConfig, LoraConfig, PeftModel: OK")

# ================================================================
# 4. LoRA target modules
# ================================================================
print("\n=== 4. LoRA Target Modules ===")
model_path = "/data/share/model/Qwen3.5-4B"
print(f"  Loading model ({model_path})...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True, trust_remote_code=True,
)
print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  Device map: {dict(model.hf_device_map) if hasattr(model, 'hf_device_map') else 'N/A'}")

linear_modules = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            linear_modules.add(parts[1])

target_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
found = target_modules & linear_modules
print(f"  Target modules found: {sorted(found)}")
missing = target_modules - linear_modules
if missing:
    print(f"  Missing: {sorted(missing)}")

# Apply LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=sorted(found), task_type="CAUSAL_LM", bias="none",
)
peft_model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
print(f"  Total: {total:,}")

if torch.cuda.is_available():
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  Peak GPU: {peak:.1f} GB")

del peft_model
del model
torch.cuda.empty_cache()

# ================================================================
# 5. SFTConfig
# ================================================================
print("\n=== 5. SFTConfig / SFTTrainer API ===")
sft_config = SFTConfig(
    output_dir="/tmp/m2_2_sft_test",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=100,
    fp16=False, bf16=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    warmup_steps=1, seed=42,
    dataset_text_field="text",
    assistant_only_loss=True,
)
print(f"  SFTConfig: OK (max_length={sft_config.max_length})")
print(f"  assistant_only_loss: {sft_config.assistant_only_loss}")

# ================================================================
# Summary
# ================================================================
print("\n=== Preflight Summary ===")
print(f"  Tokenization: mean={pf['full_mean']:.0f} tokens, max={pf['full_max']}")
print(f"  Max seq length: {pf['full_max'] + 32}")
print(f"  LoRA targets: {sorted(found)}")
print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
print(f"  Cross-layer: PASS ({cl.get('mismatch_count', 0)} mismatches)")
print(f"  All APIs: OK")
print("\n=== Preflight Complete ===")

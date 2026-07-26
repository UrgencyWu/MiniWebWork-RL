# M2.0 Command Log

## Phase: Qwen3.5-4B Base Browser Agent Baseline

### Model Audit
```bash
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(...)"
# model_type: qwen3_5, tokenizer: Qwen2Tokenizer
# chat_template: present, SHA-256 a4aee8af...
# enable_thinking: supported
# eos_token_id: 248046, pad_token_id: 248044
```

### Module Creation
```bash
prompts/browser_agent_v1.txt                          # System prompt v1
src/miniwebwork/model_agent/__init__.py
src/miniwebwork/model_agent/prompt_builder.py          # build_messages()
src/miniwebwork/model_agent/output_parser.py           # strict + fallback parse
src/miniwebwork/model_agent/model_backend.py           # QwenTransformersBackend
src/miniwebwork/model_agent/qwen_agent.py              # QwenBrowserAgent
src/miniwebwork/model_agent/agent_loop.py              # run_model_episode()
src/miniwebwork/model_agent/metrics.py                 # compute_metrics()
src/miniwebwork/model_agent/failure_analysis.py        # classify_failures()
src/miniwebwork/model_agent_runner.py                  # CLI runner
scripts/slurm/m2_0_model_action_smoke.sbatch           # Smoke test (1 GPU)
scripts/slurm/m2_0_base_agent_eval.sbatch              # Full eval (1 GPU)
```

### Slurm Jobs
```bash
# Smoke test
sbatch scripts/slurm/m2_0_model_action_smoke.sbatch
# Job 946: FAILED (parse() API mismatch)
# Job 947: FAILED (empty generations — tokenize=True returns BatchEncoding)
# Job 948: COMPLETED (13s) — 3/3 valid JSON actions

# Full evaluation
sbatch scripts/slurm/m2_0_base_agent_eval.sbatch
# Job 949: FAILED (Playwright async loop conflict between tasks)
# Job 950: FAILED (same — env recreated per task)
# Job 951: COMPLETED (6m21s) — 15/15 tasks, 5/15 success (33.3%)
```

### Key Fixes
1. **Tokenizer**: `apply_chat_template(tokenize=True)` returns BatchEncoding, not token list. Fixed: render text first (tokenize=False), then `tokenizer(text, return_tensors="pt")`.
2. **Parser API**: QwenBrowserAgent expected `self._parser.parse()` but received function. Fixed: `getattr(self._parser, "parse", self._parser)(raw)`.
3. **Playwright async loop**: Creating new `ProcurementBrowserEnv` per task failed because `sync_playwright().start()` can't be called repeatedly. Fixed: create env once outside loop, reuse across all tasks, keep browser alive between resets.

### Results
```bash
python -m miniwebwork.model_agent_runner \
  --model-path /data/share/model/Qwen3.5-4B \
  --tasks all --max-model-turns 20 --max-env-steps 15 \
  --output-dir artifacts/m2_0
```

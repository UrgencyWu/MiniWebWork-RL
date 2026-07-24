# M1.0 Environment Report

## 1. Execution Context

- **Hostname**: user-NF5468M6
- **User**: wushaohua
- **Project root**: /home/wushaohua/data/MiniWebWork-RL
- **Conda environment**: miniwebwork
- **Slurm partition**: compute
- **Audit time**: 2026-07-24 17:38 CST (audit) / 2026-07-25 01:26 CST (implementation complete)

## 2. Environment Strategy

| Item | Decision |
|---|---|
| Base environment | qwen9B (PyTorch 2.10.0+cu128, Transformers 5.14.1, ModelScope 1.34.0) |
| Clone used | Yes: `conda create -n miniwebwork --clone qwen9B` |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 (PyTorch), 13.2 (Driver) |
| Transformers | 5.14.1 |
| ModelScope | 1.34.0 |
| Playwright | 1.61.0 (Python), Chromium 149.0.7827.55 |
| TRL | 1.9.0 |
| Model | Qwen3.5-4B at /data/share/model/Qwen3.5-4B (8.8 GB, pre-cached) |

## 3. Acceptance Results

| 验收项 | 状态 | 证据 | 说明 |
|---|---|---|---|
| Conda 环境 | PASS | environment.initial.yml, environment.final.yml | qwen9B cloned successfully to miniwebwork |
| Chromium 启动 | PASS | Job 926 log | Headless Chromium launched in Slurm job |
| Local Web in Slurm | PASS | Job 926 log | FastAPI app started, /health returned {"status":"ok"} |
| DOM 操作闭环 | PASS | browser_smoke_result.json | Title, input_count, button_count, result_text all verified |
| Qwen3.5-4B 加载 | PASS | model_smoke_result.json | Model loaded, tokenizer loaded, generation completed |

## 4. Browser Test

- **Slurm job ID**: 926
- **Status**: COMPLETED (ExitCode 0:0, 4s)
- **Page URL**: http://127.0.0.1:18080/
- **DOM read**: title="MiniWebWork-RL Smoke Test", input_count=1, button_count=1, initial #result="ready"
- **Interaction**: filled #query with "RTX PRO 6000", clicked #search-button, #result updated to "searched: RTX PRO 6000"
- **JSON artifact**: artifacts/browser_smoke_result.json (`success: true`)
- **Screenshot artifact**: artifacts/browser_smoke.png (17375 bytes)
- **Errors**: None
- **Warnings**: None

## 5. Model Test

- **Model ID**: Qwen3.5-4B (local copy)
- **Model path**: /data/share/model/Qwen3.5-4B
- **Slurm job ID**: 925
- **Status**: COMPLETED (ExitCode 0:0, 14s)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB), allocated via Slurm gres=gpu:1
- **CUDA_VISIBLE_DEVICES**: 0 (set by Slurm)
- **Generated text**: 32 tokens generated successfully
- **JSON artifact**: artifacts/model_smoke_result.json (`success: true`)
- **Errors**: None
- **Warnings**: `torch_dtype is deprecated, use dtype instead` (transformers warning, non-blocking)

## 6. Dependency Changes

### Added Packages
| Package | Version |
|---|---|
| playwright | 1.61.0 |
| pyee | 13.0.1 |
| trl | 1.9.0 |
| miniwebwork | 0.1.0 (editable) |

### pip check Result
```
vllm 0.17.0 has requirement transformers<5,>=4.56.0, but you have transformers 5.14.1.
```
Non-critical: vLLM is not used in M1.0. This warning existed before cloning (it's in qwen9B too).

### Compatibility Risks
- vLLM + Transformers 5.14.1: Minor version mismatch. vLLM is functional (used by GPU 0/1 in production). Not a blocker for M1.0.
- No PyTorch/CUDA/torch downgrade occurred.

## 7. Slurm Constraints

| 参数 | 值 |
|---|---|
| 是否强制 Slurm | Yes (policy requirement) |
| 单任务 GPU 上限 | 2 |
| 时间上限 | 24 hours |
| Browser smoke: GPU | 0 (not requested) |
| Browser smoke: CPU | 4 |
| Browser smoke: Memory | 16G |
| Model smoke: GPU | 1 (via --gres=gpu:1) |
| Model smoke: CPU | 8 |
| Model smoke: Memory | 64G |
| 后续训练影响 | 2-GPU limit OK for Qwen3.5-4B LoRA; single-node deployment of web+browser+model fits |

### Issue Found & Fixed
- **Issue**: `$0` in sbatch scripts resolves to Slurm spool path, not project root
- **Fix**: Use `$SLURM_SUBMIT_DIR` as primary path, with `$0`-based fallback
- **Jobs fixed**: 924 (FAILED) -> 925 (COMPLETED)

## 8. Blockers

### P0
- None

### P1
- None

### P2
- vLLM/Transformers version mismatch warning (pre-existing, non-blocking for M1.0)
- requests/urllib3/charset_normalizer version warning (pre-existing, cosmetic)

## 9. Final Decision

**M1_0_PASS**

All five core acceptance criteria are met:
1. ✅ Dedicated Conda environment (miniwebwork) created and verified
2. ✅ Playwright can launch headless Chromium in Slurm job
3. ✅ Local web service runs in Slurm job, accessible via localhost
4. ✅ Playwright can read DOM, fill inputs, click buttons, and verify page changes
5. ✅ Qwen3.5-4B loads successfully and completes generation on GPU via Slurm

## 10. Recommended Next Step (M1.1)

Do NOT execute in this phase. M1.1 should establish:
- Minimal procurement website with product listing, search, and detail pages
- Supplier database schema
- First batch of deterministic procurement tasks
- SQLite-backed data persistence
- Single-browser Playwright navigation across multiple pages

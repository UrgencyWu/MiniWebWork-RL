# M1.0 Command Log

> MiniWebWork-RL M1.0 Phase — Environment Establishment & Browser Smoke Tests

## Execution Time

2026-07-24 17:38 CST ~ 2026-07-25 01:26 CST

---

## Step 1: Environment Inspection

```bash
# Check existing conda environments
conda env list
# Direct Python path inspection of qwen9B, RiskPO, llamafactory, swift-pref, riskpo-cu13
/home/wushaohua/miniconda3/envs/qwen9B/bin/python -c "import torch, transformers, modelscope; ..."
# => qwen9B selected: Python 3.11.14, PyTorch 2.10.0+cu128, Transformers 5.14.1, ModelScope 1.34.0
```

## Step 2: Create miniwebwork Environment

```bash
conda create -n miniwebwork --clone qwen9B -y
# Result: 42 packages installed, environment at /home/wushaohua/miniconda3/envs/miniwebwork
```

## Step 3: Export Initial Environment

```bash
conda env export -n miniwebwork --no-builds > environment.initial.yml
/home/wushaohua/miniconda3/envs/miniwebwork/bin/pip freeze > requirements.initial.txt
```

## Step 4: Install Playwright

```bash
/home/wushaohua/miniconda3/envs/miniwebwork/bin/pip install playwright
# => playwright-1.61.0, pyee-13.0.1

/home/wushaohua/miniconda3/envs/miniwebwork/bin/python -m playwright install chromium
# => Chrome for Testing 149.0.7827.55 (chromium v1228) -> ~/.cache/ms-playwright/chromium-1228
# => Chrome Headless Shell 149.0.7827.55 -> ~/.cache/ms-playwright/chromium_headless_shell-1228
# => FFmpeg v1011 -> ~/.cache/ms-playwright/ffmpeg-1011
```

## Step 5: Install TRL

```bash
# Dry-run check first (no conflicts detected)
/home/wushaohua/miniconda3/envs/miniwebwork/bin/pip install --dry-run trl
# => All dependencies satisfied, only trl-1.9.0 would be added

/home/wushaohua/miniconda3/envs/miniwebwork/bin/pip install trl
# => trl-1.9.0 installed successfully
```

## Step 6: Install Package in Dev Mode

```bash
/home/wushaohua/miniconda3/envs/miniwebwork/bin/pip install -e .
# => miniwebwork-0.1.0 installed (editable)
```

## Step 7: Run Unit Tests

```bash
/home/wushaohua/miniconda3/envs/miniwebwork/bin/python -m pytest tests/ -q
# => 9 passed in 0.20s
```

## Step 8: Local Browser Smoke Test

```bash
# Start web app
python -m uvicorn miniwebwork.webapp:app --host 127.0.0.1 --port 18080 &
# Run browser smoke
MINIWEBWORK_URL=http://127.0.0.1:18080 python -m miniwebwork.browser_smoke
# => PASSED. Result: success=true, input_count=1, button_count=1, result_text="searched: RTX PRO 6000"
```

## Step 9: Slurm Browser Smoke Test

```bash
sbatch scripts/slurm/m1_browser_smoke.sbatch
# => Job 926, COMPLETED, ExitCode 0:0, Elapsed 00:00:04
# => artifacts/browser_smoke_result.json: success=true
# => artifacts/browser_smoke.png: 17375 bytes
```

## Step 10: Slurm Model Smoke Test

```bash
# First attempt: Job 924 -> FAILED (PROJECT_ROOT resolution issue due to $0 in sbatch)
# Fixed: Use $SLURM_SUBMIT_DIR

sbatch scripts/slurm/m1_model_smoke.sbatch
# => Job 925, COMPLETED, ExitCode 0:0, Elapsed 00:00:14
# => Model: /data/share/model/Qwen3.5-4B
# => GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB)
# => artifacts/model_smoke_result.json: success=true
```

## Step 11: Final Environment Export

```bash
conda env export -n miniwebwork --no-builds > environment.final.yml
/home/wushaohua/miniconda3/envs/miniwebwork/bin/pip freeze > requirements.final.txt
pip check
# => vllm 0.17.0 has requirement transformers<5,>=4.56.0 (non-critical warning)
```

## New Dependencies Added

| Package | Version | Purpose |
|---|---|---|
| playwright | 1.61.0 | Browser automation |
| pyee | 13.0.1 | Playwright dependency |
| trl | 1.9.0 | Transformer Reinforcement Learning |
| miniwebwork | 0.1.0 (editable) | Project package |

# MiniWebWork-RL

MiniWebWork-RL 是一个面向确定性采购调研流程的轻量浏览器 Agent 算法项目。Qwen3.5-4B 读取文本化浏览器观察，通过固定 JSON 动作空间操作本地采购网站，最终由非 LLM Verifier 计算任务结果。

项目目标是建立一条可运行、可训练、可审计的最小 Agentic RL 链路，而不是构建通用网页基准或第三方 Agent 编排框架。

## Current Stage

```text
M1.0–M1.2  Environment and Agent Runtime        PASS
M2.0       Canonical Base Agent                 PASS
M2.1F      Expert trajectories and SFT data     PASS
M2.2R      Canonical SFT and Frozen E2E         PASS
M3.0A      Rollout readiness audit              PASS → Route B
M2.3-mini  No-solution/recovery SFT patch       TRAINING PASS
M2.3 Probe Canonical GPU rerun                  PENDING
M3.0B      Outcome-only GRPO pilot              NOT STARTED
```

正式状态见 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)。当前仍为：

```text
READY_FOR_GRPO=false
```

M2.2R 历史 15-task E2E：Canonical Base 0/15；三个 SFT seed 分别为 9/15、10/15、12/15。该任务集已多次用于调试，仅用于阶段连续性比较，不作为最终泛化测试。

## Architecture

```text
Public Task + Private Oracle + SQLite
                 ↓
FastAPI/Jinja2 Procurement Site
                 ↓
Playwright Browser Environment
(reset / step / close / text observation / JSON action)
                 ↓
QwenBrowserAgent
(Canonical Prompt v2 → Qwen3.5-4B → Parser → Action)
                 ↓
Deterministic Verifier
                 ↓
Expert SFT / Grouped Rollout / Planned GRPO
```

项目已经实现自研的 Agent Runtime；尚未实现的是 on-policy rollout update、group-relative advantage 和策略优化循环。详见 [`docs/ARCHITECTURE_AND_CONTRACTS.md`](docs/ARCHITECTURE_AND_CONTRACTS.md)。

## Authoritative Documents

- [当前实现状态](docs/CURRENT_STATUS.md)
- [架构与运行合同](docs/ARCHITECTURE_AND_CONTRACTS.md)
- [实验与数据治理](docs/EXPERIMENT_GOVERNANCE.md)
- [M3.0 Agentic RL 计划](docs/M3_0_AGENTIC_RL_PLAN.md)
- [项目历史记录](docs/PROJECT_JOURNAL.md)

`docs/M1_*`、`docs/M2_*` 中的阶段报告作为历史证据保留；发生冲突时，以上四份权威文档优先。

## Frozen Scope

| Layer | Choice |
|---|---|
| Web service | FastAPI + Uvicorn + Jinja2 |
| Database | SQLite; 6 suppliers, 24 products |
| Browser | Playwright + Chromium, headless |
| Observation | Text-only DOM-derived state |
| Action | Fixed JSON action schema |
| Model | Qwen3.5-4B via Transformers |
| Fine-tuning | LoRA / PEFT |
| RL | Outcome-only GRPO, pending readiness gate |
| Scheduling | Slurm |
| Deployment | No Docker requirement |

Visual input, generic external websites, multi-browser distributed rollout, value-model PPO, and process reward models are outside the first project scope.

## Installation

Python 3.11 is the supported runtime. Install a PyTorch build compatible with the target server CUDA/driver first, then install the project:

```bash
conda env create -f environment.yml
conda activate miniwebwork

# Install the correct CUDA PyTorch build for the server separately.
pip install -e ".[test,training]"
python -m playwright install chromium
```

The former host-specific `requirements.final.txt` and `environment.final.yml` were removed because they contained unrelated packages and absolute local build paths.

## Quality Gate

```bash
bash scripts/run_quality_checks.sh
```

Equivalent steps:

```bash
python -m compileall -q src scripts tests
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-seed
python -m miniwebwork.cli validate-tasks
python -m pytest -q
```

GitHub Actions runs CPU syntax, dataset-contract, browser dependency, and test checks. GPU/model/Slurm tests remain cluster-only.

## Local Site

```bash
python -m miniwebwork.cli init-db
python -m miniwebwork.webapp
```

Default address: `http://127.0.0.1:18080`.

## Canonical Rollout Probe

There is one formal rollout implementation:

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
```

Old temperature-sweep and comparison scripts were removed because they used incompatible metrics and failure handling.

Single-task smoke for the patched policy:

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 1 1
```

Formal controlled A/B runs:

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8
```

Arguments:

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS]
```

Formal rollout contracts:

- Slurm owns `CUDA_VISIBLE_DEVICES`;
- missing adapters/tasks fail fast;
- prompt contract is fixed to `browser_agent_v2`;
- infrastructure failures use `reward=null`;
- Schema Valid is action-level;
- every rollout has a stable derived seed;
- prompt token IDs, completion token IDs, and raw old-policy token log-probabilities are saved;
- adapter, prompt, and task sources use content SHA-256;
- results are saved atomically and incrementally.

## Data Governance

| Source | Role |
|---|---|
| `data/tasks/tasks_public.jsonl` | historical public tasks |
| `data/tasks/tasks_oracle.jsonl` | private deterministic Oracle |
| `data/tasks/rollout_dev_no_solution_v1/` | rollout/RL development tasks |
| future `final_test_v2` | final one-time evaluation |

A process reads exactly one task source. Development tasks are never merged with the default task set, and duplicate task IDs fail fast.

Checkpoint selection uses Canonical Valid metrics only. Frozen Test results are not used for checkpoint, temperature, or hyperparameter selection.

## M3.0 Boundary

The first RL implementation will use:

```text
K grouped trajectories per task
terminal verifier reward {1, 0}
group-relative advantage
token-level clipped policy objective
optional small reference-policy KL
LoRA-only updates
```

It does not require a value model, GAE, replay buffer, or handcrafted step rewards. GRPO starts only after the canonical rollout rerun produces at least one mixed-reward no-solution group with valid token-level evidence and negligible infrastructure failures.

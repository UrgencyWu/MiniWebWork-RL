# MiniWebWork-RL

MiniWebWork-RL 是一个面向确定性采购调研流程的轻量浏览器 Agent 算法项目。Qwen3.5-4B 读取文本化浏览器观察，通过固定 JSON 动作空间操作本地采购网站，最终由非 LLM Verifier 计算任务结果。

项目目标是建立一条可运行、可训练、可审计的最小多轮 Agentic RL 链路，而不是构建通用网页基准或第三方 Agent 编排框架。

## Current Stage

```text
M1.0–M1.2  Environment and Agent Runtime        PASS
M2.0       Canonical Base Agent                 PASS
M2.1F      Expert trajectories and SFT data     PASS
M2.2R      Canonical SFT and Frozen E2E         PASS
M3.0A      Rollout readiness audit              PASS → Route B
M2.3-mini  No-solution/recovery SFT patch       PASS
M2.3 Probe Historical readiness GPU evidence   PASS
M3.0B-0A   Paired A/B + feasible gate           IMPLEMENTED / RUN PENDING
M3.0B-0C   Strict collection                    IMPLEMENTED / RUN PENDING
M3.0B-1    Single-batch gradient smoke          IMPLEMENTED / RUN PENDING
```

正式状态见 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)：

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
SCHEMA_V3_3_ROLLOUT_IMPLEMENTED=true
PAIRED_AB_ANALYSIS_IMPLEMENTED=true
FEASIBLE_ROLLOUT_DEV_IMPLEMENTED=true
M3_0B1_SINGLE_BATCH_SMOKE_IMPLEMENTED=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

Historical readiness probe 已确认基础设施错误为 0、raw/sampling logprob coverage 均为 1.0，并产生 no-solution 成功轨迹。该历史产物未显式冻结 `top_k`，因此只作为能力和基础设施证据，不作为 optimizer batch。新 schema-v3.3 artifact 显式记录 `temperature/top_p/top_k`。

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
Expert SFT / Grouped Multi-turn Rollout / GRPO-style Update
```

项目已经实现自研 Agent Runtime。由于每个浏览器 turn 都会重新构造 Prompt，正式 M3.0 使用自定义多轮 GRPO-style 目标，而不是把整条浏览器轨迹伪装成单次 completion。

## Authoritative Documents

- [当前实现状态](docs/CURRENT_STATUS.md)
- [架构与运行合同](docs/ARCHITECTURE_AND_CONTRACTS.md)
- [实验与数据治理](docs/EXPERIMENT_GOVERNANCE.md)
- [M3.0 多轮 Agentic RL 计划](docs/M3_0_AGENTIC_RL_PLAN.md)
- [项目历史记录](docs/PROJECT_JOURNAL.md)

`docs/M1_*`、`docs/M2_*` 为历史证据；发生冲突时，以上权威文档优先。

## Frozen Scope

| Layer | Choice |
|---|---|
| Web service | FastAPI + Uvicorn + Jinja2 |
| Database | SQLite; 6 suppliers, 24 products |
| Browser | Playwright + Chromium, headless |
| Observation | Text-only DOM-derived state |
| Action | Fixed JSON action schema v1.1 |
| Model | Qwen3.5-4B via Transformers |
| Fine-tuning | LoRA / PEFT |
| RL | Multi-turn outcome-only GRPO-style optimization |
| Scheduling | Slurm |
| Deployment | No Docker requirement |

Visual input、通用外部网站、分布式浏览器集群、value-model PPO 和 process reward model 不属于第一版范围。

## Installation

Python 3.11 是支持环境。先安装与服务器 CUDA/驱动匹配的 PyTorch，再安装项目：

```bash
conda env create -f environment.yml
conda activate miniwebwork

# Install the correct CUDA PyTorch build for the target node first.
pip install -e ".[test,training]"
python -m playwright install chromium
```

## Quality Gate

```bash
bash scripts/run_quality_checks.sh
```

等价步骤：

```bash
python -m compileall -q src scripts tests
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-seed
python -m miniwebwork.cli validate-tasks
python -m pytest -q
```

GitHub Actions 负责 CPU 语法、数据合同、浏览器依赖和测试；GPU/model/Slurm 验证仍需在集群执行。

## Canonical Rollout Collection

正式入口：

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
scripts/analyze_probe_ab.py
scripts/m3_0_single_batch_smoke.py
scripts/slurm/m3_0_single_batch_smoke.sbatch
```

Rollout Slurm 参数：

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P] [TOP_K] [TASK_SOURCE]
```

`TASK_SOURCE` 支持 `no_solution`、`feasible`、绝对路径或仓库相对路径。每个 Slurm job 使用独立 run 目录，任务源、seed、distribution、K 和 job ID 不会互相覆盖。

单任务 Patch Policy smoke：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 1 1 0.9 0 no_solution
```

显式 no-solution A/B 诊断：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 0.2 20260731 8 "" 0.9 0 no_solution
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 8 "" 0.9 0 no_solution
```

Feasible regression gate：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 0.2 20260731 8 "" 0.9 0 feasible
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 8 "" 0.9 0 feasible
```

严格更新分布：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 1.0 20260731 8 "" 1.0 0 no_solution
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 1.0 20260731 8 "" 1.0 0 no_solution
```

首版严格分布要求：

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

此外，raw policy 与 sampling-distribution token log-probabilities 的最大绝对差必须不超过 artifact 中声明的数值容差。Replay 层会根据 records 独立重算兼容性，调用方不能提升诊断 artifact。

A/B 配对分析：

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/runs/<A_RUN>/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/runs/<B_RUN>/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

分析使用相同 `(task_id, rollout_index)` 配对，报告 A-only/B-only 成功、精确 McNemar 检验、task-level bootstrap 区间、feasible success、false no-solution、no-solution success 和终止原因。不能仅凭一次总成功数宣称补丁增益或将差异归因于采样方差。

单 batch optimizer smoke：

```bash
sbatch scripts/slurm/m3_0_single_batch_smoke.sbatch \
  outputs/m2_3_mini/runs/<STRICT_RUN>/<STRICT_ARTIFACT>.json \
  <A_OR_B> [TASK_ID] [LEARNING_RATE]
```

该入口先无梯度重放真实逐 turn Prompt，验证 stored old-policy 与 current-policy logprob 一致；随后按 trajectory/turn 流式执行一次 LoRA-only backward 和 optimizer step，并验证有限非零梯度、参数变化、Adapter 保存及重新加载 forward。它只证明优化器链路正确，不代表策略性能提升。

## Data Governance

| Source | Role | Gradient allowed? |
|---|---|---:|
| `data/tasks/tasks_public.jsonl` | historical public tasks | no |
| `data/tasks/tasks_oracle.jsonl` | private deterministic Oracle | no |
| `data/tasks/rollout_dev_no_solution_v1/` | no-solution rollout/RL development | yes, after versioned collection |
| `data/tasks/rollout_dev_feasible_v1/` | policy-selection and false-no-solution gate | no |
| future `final_test_v2` | final one-time evaluation | no |

`rollout_dev_feasible_v1` 包含 12 个全新 select-product 任务，manifest 冻结 Public/Oracle SHA-256 与 seed-source Git blob SHA。一个进程只读取一个任务源。开发集与默认任务不合并，重复 task ID fail-fast。Frozen Test 不参与 checkpoint、sampling 或 optimizer 选择。

## M3.0 Boundary

第一版训练链：

```text
same-task K multi-turn trajectories
→ terminal verifier reward {1, 0}
→ group-relative trajectory advantage
→ per-turn prompt/action forward replay
→ token clipping, trajectory-normalized aggregation
→ optional frozen-reference KL
→ LoRA-only update
```

核心实现：

```text
src/miniwebwork/rollout.py
src/miniwebwork/rl/batch.py
src/miniwebwork/rl/objective.py
src/miniwebwork/rl/streaming.py
```

下一门禁是：完成 no-solution/feasible 多 seed 配对策略选择，schema-v3.3 严格分布产生至少一个 `valid_for_grpo_update=true` group，并在集群运行一次 LoRA-only gradient/checkpoint/reload smoke。第一版不需要 value model、GAE、replay buffer 或手工 step reward。

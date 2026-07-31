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
M3.0B-0    Schema-v3.3 strict collection       IN PROGRESS
M3.0B-1    Single-batch gradient smoke          NOT STARTED
```

正式状态见 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)：

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
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
```

Slurm 参数：

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P] [TOP_K]
```

单任务 Patch Policy smoke：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 1 1 0.9 0
```

显式 A/B readiness 分布：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8 "" 0.9 0
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8 "" 0.9 0
```

严格更新分布：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 1.0 20260731 8 "" 1.0 0
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 1.0 20260731 8 "" 1.0 0
```

首版严格分布要求：

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

此外，raw policy 与 sampling-distribution token log-probabilities 的最大绝对差必须不超过 artifact 中声明的数值容差。只有参数和概率证据均兼容且组内存在 mixed reward 时，group 才会标记为 `valid_for_grpo_update=true`。

A/B 配对分析：

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

分析使用相同 `(task_id, rollout_index)` 配对，报告 A-only/B-only 成功、精确 McNemar 检验、task-level bootstrap 区间和逐任务差异。不能仅凭一次总成功数宣称补丁增益或将差异归因于采样方差。

Rollout 合同：

- Slurm 管理 `CUDA_VISIBLE_DEVICES`；
- Adapter、任务和 Prompt 合同缺失时 fail-fast；
- 基础设施失败使用 `reward=null`；
- 每条 trajectory 和 group 保存 `temperature/top_p/top_k`；
- Strict JSON、Schema Valid、环境动作成功和任务成功使用独立分母；
- 保存 prompt/completion token IDs、raw policy log-probabilities 和 sampling-distribution log-probabilities；
- Adapter、Prompt 和任务源使用内容 SHA-256；
- 结果按任务原子化增量保存。

## Data Governance

| Source | Role |
|---|---|
| `data/tasks/tasks_public.jsonl` | historical public tasks |
| `data/tasks/tasks_oracle.jsonl` | private deterministic Oracle |
| `data/tasks/rollout_dev_no_solution_v1/` | no-solution rollout/RL development tasks |
| future feasible rollout_dev slice | false-no-solution and general capability checks |
| future `final_test_v2` | final one-time evaluation |

一个进程只读取一个任务源。开发集与默认任务不合并，重复 task ID fail-fast。Checkpoint 仅由 Canonical Valid 指标选择；Frozen Test 不参与 checkpoint、sampling 或 optimizer 选择。

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
```

下一门禁是：schema-v3.3 严格分布产生至少一个 `valid_for_grpo_update=true` group，完成 A/B 初始策略选择，并执行一次 LoRA-only gradient/checkpoint/reload smoke。第一版不需要 value model、GAE、replay buffer 或手工 step reward。

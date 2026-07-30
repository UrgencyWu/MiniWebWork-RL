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
M2.3-mini  No-solution/recovery SFT patch       TRAINING PASS
M2.3 Probe Canonical GPU rerun                  PENDING
M3.0B      Multi-turn GRPO-style pilot          NOT STARTED
```

正式状态见 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)：

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
Expert SFT / Grouped Multi-turn Rollout / GRPO-style Update
```

项目已经实现自研 Agent Runtime。尚未完成的是严格 on-policy rollout collection、逐 turn 策略重放和 LoRA 优化循环。由于每个浏览器 turn 都会重新构造 Prompt，正式 M3.0 使用自定义多轮 GRPO-style 目标，而不是把整条浏览器轨迹伪装成单次 completion。

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

原主机专用 `requirements.final.txt` 和 `environment.*.yml` 已删除，避免绝对路径、无关包和错误 CUDA wheel 污染环境。

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

## Local Site

```bash
python -m miniwebwork.cli init-db
python -m miniwebwork.webapp
```

默认地址：`http://127.0.0.1:18080`。

## Canonical Rollout Probe

正式 rollout 入口只有：

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
```

旧 temperature-sweep、comparison 和 browser-agent-v1 SFT 入口已删除。

单任务 Patch Policy smoke：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 1 1
```

受控 A/B readiness probe：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8
```

参数：

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS]
```

Rollout 合同：

- Slurm 管理 `CUDA_VISIBLE_DEVICES`；
- Adapter、任务和 Prompt 合同缺失时 fail-fast；
- 基础设施失败使用 `reward=null`；
- Strict JSON、Schema Valid、环境动作成功和任务成功使用独立分母；
- 每条 rollout 使用稳定派生 seed；
- 保存 prompt token IDs、completion token IDs、raw policy log-probabilities 和 sampling-distribution log-probabilities；
- Adapter、Prompt 和任务源使用内容 SHA-256；
- 结果按任务原子化增量保存。

Readiness probe 的 `top_p=0.9` 只判断探索和奖励方差。它产生的 mixed-reward group 具有学习信号，但不会自动标记为可直接更新。

## Data Governance

| Source | Role |
|---|---|
| `data/tasks/tasks_public.jsonl` | historical public tasks |
| `data/tasks/tasks_oracle.jsonl` | private deterministic Oracle |
| `data/tasks/rollout_dev_no_solution_v1/` | rollout/RL development tasks |
| future `final_test_v2` | final one-time evaluation |

一个进程只读取一个任务源。开发集与默认任务不合并，重复 task ID fail-fast。Checkpoint 仅由 Canonical Valid 指标选择；Frozen Test 不参与 checkpoint、temperature 或 optimizer 选择。

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

诊断 sampling 与正式 training sampling 分离。首个严格更新优先采用 `temperature=1.0, top_p=1.0`，确保 behavior distribution 与 raw policy log-probability 合同一致。第一版不需要 value model、GAE、replay buffer 或手工 step reward。

核心目标代码：

```text
src/miniwebwork/rl/objective.py
```

M3.0B 只能在 canonical rerun 证明 no-solution 成功、mixed reward、完整 token/logprob 证据和接近零基础设施错误后开始。

# MiniWebWork-RL

轻量级浏览器智能体强化训练项目，面向确定性采购调研任务。模型通过文本化浏览器观察和固定 JSON 动作空间操作本地采购网站，并由非 LLM Verifier 判定任务结果。

## 当前阶段

```text
M1.0–M1.2  Environment / Rule Agent        PASS
M2.0       Qwen3.5-4B Base Agent           PASS
M2.1       Expert trajectories / SFT data  PASS
M2.2R      Canonical SFT / Frozen E2E      PASS
M3.0A      Rollout readiness audit         PASS → Route B
M2.3-mini  No-solution recovery patch      TRAINED; rollout re-evaluation in progress
M3.0B      Outcome-only GRPO pilot          NOT STARTED
```

M2.2R 的冻结 15-task E2E 结果：Canonical Base 0/15；三个 SFT seed 分别为 9/15、10/15、12/15。M3.0A 发现 no-solution 随机 rollout 缺少奖励方差，因此先执行 M2.3-mini 数据补丁和继续训练。

## 系统架构

```text
FastAPI/Jinja2 procurement site + SQLite
                 ↓
Playwright browser environment
(reset / step / close, text observation, JSON action)
                 ↓
QwenBrowserAgent
(prompt contract → Qwen3.5-4B → parser → action)
                 ↓
Deterministic verifier
                 ↓
Expert SFT / rollout collection / planned GRPO
```

## 已冻结技术范围

| 项目 | 选型 |
|---|---|
| 本地 Web 服务 | FastAPI + Uvicorn + Jinja2 |
| 数据库 | SQLite（6 供应商、24 商品） |
| 浏览器自动化 | Playwright + Chromium（headless） |
| LLM | Qwen3.5-4B via Transformers |
| 微调 | LoRA / PEFT |
| 强化训练 | Outcome-only GRPO（规划阶段） |
| 集群调度 | Slurm |
| 容器化 | 不使用 Docker |

## 环境要求

- Python 3.11
- PyTorch 2.10.0+cu128
- CUDA 12.8+
- Transformers / PEFT
- Playwright + Chromium

```bash
conda activate miniwebwork
```

## 基础命令

```bash
# 初始化和验证数据库
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-seed
python -m miniwebwork.cli validate-tasks

# 全量测试
python -m pytest -q

# 启动采购网站
python -m miniwebwork.webapp
```

网站默认监听 `http://127.0.0.1:18080`。

## M2.3-mini Rollout Probe

每个 Slurm 作业运行一个 `(policy, temperature)` 组合：

```bash
# Policy A: M2.2R seed_1234；temperature=0.2；seed=20260731；K=8
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8

# Policy B: M2.3-mini patched seed_1234
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8
```

正式 Probe 合同：

- Slurm 决定 `CUDA_VISIBLE_DEVICES`，Python 进程不得在 CUDA 初始化后覆盖；
- Adapter 缺失时 fail-fast，不允许静默退化到 Base；
- Prompt 固定为 `browser_agent_v2`；
- 基础设施异常使用 `reward=null`，不得作为策略负奖励；
- Schema Valid 按模型动作级统计；
- 每条 rollout 使用可复现的 `(master_seed, task_id, k)` 派生种子；
- Adapter、任务文件和 PromptBuilder 使用内容 SHA-256；
- 每个任务完成后增量保存，作业中断时保留已完成结果。

输出默认位于：

```text
outputs/m2_3_mini/
```

## 数据治理

| 路径 | 内容 |
|---|---|
| `data/tasks/tasks_public.jsonl` | 原始公开任务，不含答案 |
| `data/tasks/tasks_oracle.jsonl` | 私有 Oracle，包含约束和正确答案 |
| `data/tasks/rollout_dev_no_solution_v1/` | no-solution rollout 开发集 |
| `data/seed/` | 供应商、商品及冻结 manifest |

训练与开发过程中不得将 Frozen Test 任务加入 SFT 或 RL 训练数据。现有 15-task Frozen Test 仅用于阶段连续性比较，最终 RL 评估应使用新的冻结测试集。

## 当前准入状态

```text
M2_3_MINI_DATA_PASS=true
M2_3_MINI_TRAINING_PASS=true
M2_3_MINI_ADAPTER_LOAD_CONTRACT_PASS=true
ROLLOUT_PIPELINE_SMOKE_PASS=true
FULL_ROLLOUT_RERUN_PENDING=true
READY_FOR_GRPO=false
```

进入 M3.0B 的最低条件是：环境错误接近零、至少一个任务组同时出现 reward 0/1、no-solution 产生成功 rollout、通用任务能力无明显退化。

## 已知限制

- 仅通过 Slurm 使用 GPU；
- 当前环境为本地确定性采购网站，不连接真实外部站点；
- 当前不包含视觉输入、多浏览器并发或 vLLM rollout 服务；
- 15-task 历史测试集规模较小，不作为最终统计泛化结论。

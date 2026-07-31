# MiniWebWork-RL

MiniWebWork-RL 是一个面向确定性采购调研流程的轻量浏览器 Agent 算法项目。Qwen3.5-4B 读取文本化网页状态，通过固定 JSON 动作空间操作本地采购网站，终态由非 LLM Verifier 给出确定性奖励。

项目目标是形成一条可运行、可训练、可审计的最小多轮 Agentic RL 链路，而不是复现通用 WebArena、构建视觉浏览器 Agent，或套用单次 completion 的现成 GRPO 示例。

## 当前阶段

```text
M1.0–M1.2  Environment and Agent Runtime        PASS
M2.0       Canonical Base Agent                 PASS
M2.1F      Expert trajectories and SFT data     PASS
M2.2R      Canonical SFT and Frozen E2E         PASS
M3.0A      Rollout readiness audit              PASS → Route B
M2.3-mini  No-solution/recovery SFT patch       PASS
M2.3 Probe Historical readiness GPU evidence   PASS
M3.0B-0A   Paired A/B + feasible v2 gate        IMPLEMENTED / RUN PENDING
M3.0B-0C   Strict update collection             RUN PENDING
M3.0B-1    Single-batch optimizer smoke         IMPLEMENTED / GPU RUN PENDING
```

正式状态见 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)：

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
ROLLOUT_DEV_FEASIBLE_V2_FROZEN=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

## 架构

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

每个浏览器 turn 都会基于新 Observation 重新构造 Prompt，因此正式 RL 路径按 turn 重放真实条件生成，再在 trajectory 层聚合 terminal advantage；不能把整条浏览器轨迹伪装为一次普通 completion。

## 权威文档

- [当前实现状态](docs/CURRENT_STATUS.md)
- [架构与运行合同](docs/ARCHITECTURE_AND_CONTRACTS.md)
- [实验与数据治理](docs/EXPERIMENT_GOVERNANCE.md)
- [M3.0 多轮 Agentic RL 计划](docs/M3_0_AGENTIC_RL_PLAN.md)
- [Slurm 入口](scripts/slurm/README.md)

历史 `docs/M1_*`、`docs/M2_*` 只作为阶段证据；发生冲突时，以上权威文档优先。

## 安装

Python 3.11 是当前支持环境。先安装与节点 CUDA/驱动匹配的 PyTorch，再安装项目：

```bash
conda env create -f environment.yml
conda activate miniwebwork
pip install -e ".[test,training]"
python -m playwright install chromium
```

## 质量门

```bash
bash scripts/run_quality_checks.sh
```

等价步骤：

```bash
python -m compileall -q src scripts tests
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-seed
python -m miniwebwork.cli validate-tasks
python -m pytest -q -m "not gpu and not slurm"
```

GitHub Actions 运行 CPU 语法、数据合同与测试；GPU、Playwright 长程稳定性和 Slurm 训练链仍需在集群验证。

## 数据治理

| 数据源 | 用途 | 允许进入梯度？ |
|---|---|---:|
| `data/tasks/rollout_dev_no_solution_v1/` | no-solution rollout 与 RL 开发 | 是，需版本化严格采集 |
| `data/tasks/rollout_dev_feasible_v2/` | 起始策略选择、false-no-solution 与通用能力回归 | 否 |
| historical frozen test v1 | 历史连续性 | 否 |
| future `final_test_v2` | 最终一次性评价 | 否 |

`rollout_dev_feasible_v2` 是当前唯一 feasible canonical slice。它由冻结规范和统一约束合同确定性生成：

```bash
python scripts/build_rollout_dev_feasible_v2.py \
  --output-dir data/tasks/rollout_dev_feasible_v2
```

生成器必须逐字节复现：

```text
valid_public.jsonl
valid_oracle.jsonl
dataset_manifest.json
```

其中 manifest 冻结规范 SHA-256、Public/Oracle SHA-256，以及 products/suppliers 的 Git blob identity。该集合 `may_update_model=false`。

## Canonical Rollout

正式入口：

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
scripts/analyze_probe_ab.py
```

Slurm 参数：

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P] [TOP_K] [TASK_SOURCE]
```

`TASK_SOURCE` 支持：

```text
no_solution
feasible
绝对任务目录
仓库相对任务目录
```

`feasible` 已固定映射到 `data/tasks/rollout_dev_feasible_v2`。

诊断采集：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 0.2 20260731 8 "" 0.9 0 no_solution
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 8 "" 0.9 0 no_solution

sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 0.2 20260731 8 "" 0.9 0 feasible
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 8 "" 0.9 0 feasible
```

严格更新分布：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 1.0 20260731 8 "" 1.0 0 no_solution
```

首版严格行为分布固定为：

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

只有 raw-policy 与 sampling-distribution token log-prob 对齐、组内存在 mixed reward、且基础设施错误为 0 的 group，才能标记为 `valid_for_grpo_update=true`。

A/B 配对分析：

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/runs/<A_RUN>/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/runs/<B_RUN>/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

分析按 `(task_id, rollout_index)` 配对，并报告 feasible success、false no-solution、no-solution success、McNemar 检验、task bootstrap 区间和终止原因。

## 单 batch 优化器 smoke

正式入口：

```text
scripts/m3_0_single_batch_smoke.py
scripts/slurm/m3_0_single_batch_smoke.sbatch
```

示例：

```bash
sbatch scripts/slurm/m3_0_single_batch_smoke.sbatch \
  <STRICT_ARTIFACT_JSON> \
  outputs/m2_3_mini/seed_1234/final_adapter
```

该 smoke 只执行一次 LoRA 更新，使用 `AdamW(weight_decay=0.0)`，并检查：

- artifact 与 Adapter hash 绑定；
- old/current log-prob 预更新一致性；
- LoRA-only finite/non-zero gradients；
- 一次 optimizer step 后参数发生变化；
- Adapter 保存、重载与 finite forward。

它只证明训练合同可运行，不代表策略性能提升。在 GPU smoke 通过前，`READY_FOR_GRPO_UPDATE` 保持 `false`。

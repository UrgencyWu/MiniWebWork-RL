# MiniWebWork-RL 当前实现状态

> 权威状态页。最后更新：2026-07-31。

## 项目定位

MiniWebWork-RL 是一个确定性采购网站上的文本浏览器 Agent 项目。Qwen3.5-4B 通过固定 JSON 动作空间执行多轮任务，终态由非 LLM Verifier 产生奖励。

```text
Task → Browser Environment → Qwen Policy → Multi-turn Rollout
→ Deterministic Verifier → SFT / GRPO-style Update → Frozen Evaluation
```

## 阶段状态

| 阶段 | 状态 |
|---|---|
| M1.0–M1.2 Environment / Agent Runtime | PASS |
| M2.0 Canonical Base Agent | PASS |
| M2.1F Expert trajectories / SFT data | PASS |
| M2.2R Canonical SFT / Frozen E2E | PASS |
| M3.0A Rollout readiness audit | PASS / Route B |
| M2.3-mini no-solution + recovery patch | PASS |
| M2.3 historical readiness GPU probe | PASS |
| M3.0B-0A schema-v3.3 / paired A-B / feasible v2 | IMPLEMENTED / RUN PENDING |
| M3.0B-0C strict update collection | RUN PENDING |
| M3.0B-1 one-batch LoRA smoke | IMPLEMENTED / GPU RUN PENDING |

```text
M2_3_MINI_CANONICAL_PROBE_PASS=true
SCHEMA_V3_3_ROLLOUT_IMPLEMENTED=true
PAIRED_AB_ANALYSIS_IMPLEMENTED=true
ROLLOUT_DEV_FEASIBLE_V2_FROZEN=true
M3_0B1_SINGLE_BATCH_SMOKE_IMPLEMENTED=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
```

## 冻结结果

M2.2R 在历史 15-task 集上的 Canonical E2E：

| Policy | Success |
|---|---:|
| Base v2 | 0/15 |
| SFT seed 42 | 9/15 |
| SFT seed 1234 | 10/15 |
| SFT seed 20260726 | 12/15 |

正式主 checkpoint 为 `seed_1234`，依据是预先冻结的最低 Canonical Valid Loss，而不是 Frozen Test：

```text
seed_1234      4.46e-05
seed_20260726  8.10e-05
seed_42        2.43e-03
```

M2.3-mini historical readiness probe：

```text
complete = true
infrastructure_errors = 0
raw_policy_logprob_coverage = 1.0
sampling_logprob_coverage = 1.0
no_solution_successes: A = 58, B = 43
valid_for_grpo_update = false
```

该历史产物未显式冻结 `top_k`，只能作为基础设施与能力证据，不能作为 optimizer batch。单次 58 vs 43 不能证明补丁优于 A，也不能直接归因于采样方差。

## Canonical 开发数据

No-solution RL development：

```text
data/tasks/rollout_dev_no_solution_v1
```

Feasible policy-selection gate：

```text
data/tasks/rollout_dev_feasible_v2
```

`rollout_dev_feasible_v2` 是唯一 canonical feasible slice。旧 v1 已删除。v2 包含 12 个 `select_product` 任务：3 exact、5 cheapest、4 highest-rating。

数据由冻结规范和统一约束合同确定性生成：

```text
spec.jsonl + products.json + suppliers.json + compute_unique_answer()
→ valid_public.jsonl
→ valid_oracle.jsonl
→ dataset_manifest.json
```

生成命令：

```bash
python scripts/build_rollout_dev_feasible_v2.py \
  --output-dir data/tasks/rollout_dev_feasible_v2
```

质量门要求生成结果与仓库冻结文件逐字节一致。该集合：

```text
role = policy_selection_and_regression_gate
may_update_model = false
```

## 冻结运行合同

- Prompt Contract v2；
- Action Schema v1.1；
- 一个进程只读取一个任务源；
- 基础设施失败使用 `reward=null`；
- 每个 turn 保存 prompt/completion token IDs；
- 分别保存 raw-policy 与 sampling-distribution log-prob；
- trajectory/group 显式保存 `temperature/top_p/top_k`；
- 首版严格分布为 `T=1, top_p=1, top_k=0`；
- Replay 独立重算 update compatibility；
- Slurm 管理 GPU 可见性；
- feasible v2 不允许进入梯度 batch。

## 权威入口

```text
scripts/build_rollout_dev_feasible_v2.py
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
scripts/analyze_probe_ab.py
scripts/m3_0_single_batch_smoke.py
scripts/slurm/m3_0_single_batch_smoke.sbatch
```

`TASK_SOURCE=feasible` 固定映射到：

```text
data/tasks/rollout_dev_feasible_v2
```

严格采集：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 1.0 20260731 8 "" 1.0 0 no_solution
```

单 batch optimizer smoke：

```bash
sbatch scripts/slurm/m3_0_single_batch_smoke.sbatch \
  <STRICT_ARTIFACT_JSON> \
  outputs/m2_3_mini/seed_1234/final_adapter
```

该 smoke 使用 `AdamW(weight_decay=0.0)`，执行一次 LoRA-only 更新，并验证 old/current log-prob、梯度、参数变化、保存和重载。

## 下一门禁

1. no-solution A/B 多 seed 配对复验；
2. feasible v2 上 false no-solution 与通用能力回归；
3. 冻结起始策略；
4. 产生至少一个 `valid_for_grpo_update=true` strict group；
5. single-batch GPU smoke 通过。

在第 5 项完成前：

```text
READY_FOR_GRPO_UPDATE=false
```

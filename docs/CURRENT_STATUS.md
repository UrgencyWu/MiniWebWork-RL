# MiniWebWork-RL 当前实现状态

> 权威状态页。最后更新：2026-07-31。

## 1. 项目定位

MiniWebWork-RL 是一个面向采购调研流程的轻量浏览器 Agent 算法项目。模型读取文本化浏览器状态，通过固定 JSON 动作空间操作本地确定性网站，并由非 LLM Verifier 计算终态奖励。

项目目标不是复现通用 WebArena，也不是构建第三方 Agent 编排框架。目标是形成一条可解释、可训练、可审计的最小 Agentic RL 链路：

```text
Task
→ Browser Environment
→ Qwen Policy
→ Multi-turn Rollout
→ Deterministic Verifier Reward
→ SFT / GRPO Policy Update
→ Frozen Evaluation
```

## 2. 阶段状态

| 阶段 | 交付内容 | 状态 |
|---|---|---|
| M1.0 | Conda、Playwright、Chromium、Qwen3.5-4B、Slurm smoke | PASS |
| M1.1 | SQLite 采购数据、网站、15 个任务、确定性 Verifier | PASS |
| M1.2 | `reset/step/close`、Observation/Action Schema、规则 Agent | PASS |
| M2.0 | Canonical Base Browser Agent 基线 | PASS |
| M2.1F | 专家任务、成功轨迹、completion-only SFT 数据 | PASS |
| M2.2 | 首轮 LoRA SFT；发现训练—推理合同漂移 | SUPERSEDED |
| M2.2R | Canonical Contract v2、三 seed SFT、Frozen E2E | PASS |
| M3.0A | Rollout readiness audit；no-solution 奖励方差不足 | PASS / Route B |
| M2.3-mini | no-solution + recovery 数据补丁、继续训练 | TRAINING PASS |
| M2.3-mini Probe | 唯一权威 Probe 已重构；GPU 全量重跑待执行 | PENDING |
| M3.0B | Outcome-only GRPO single-batch pilot | NOT STARTED |

正式状态：

```text
M2_3_MINI_DATA_PASS=true
M2_3_MINI_TRAINING_PASS=true
M2_3_MINI_ADAPTER_LOAD_CONTRACT_PASS=true
CANONICAL_ROLLOUT_PIPELINE_IMPLEMENTED=true
CANONICAL_ROLLOUT_GPU_RERUN_PENDING=true
READY_FOR_GRPO=false
```

## 3. 已冻结结果

### 3.1 M2.2R Frozen E2E

统一使用 Canonical Prompt Contract v2：

| Policy | Success | Schema Valid | Environment Action Success |
|---|---:|---:|---:|
| Base Canonical v2 | 0/15 | 19.0% | 13.8% |
| SFT seed 42 | 9/15 | 75.8% | 53.7% |
| SFT seed 1234 | 10/15 | 78.7% | 59.3% |
| SFT seed 20260726 | 12/15 | 74.5% | 61.2% |

这些结果只支持以下结论：在历史 15-task 集上，Canonical SFT 明显优于 Canonical Base。该任务集规模较小，且已多次用于调试，不再作为最终泛化测试集。

### 3.2 Checkpoint 选择

M3.0A/M2.3-mini 的正式主策略为 `seed_1234`，依据是预先冻结的最低 Canonical Valid Loss：

```text
seed_1234      4.46e-05
seed_20260726  8.10e-05
seed_42        2.43e-03
```

Frozen Test 得分不得用于 checkpoint 选择。

### 3.3 M3.0A

- Expert Replay：环境和 no-solution 提交合同通过；
- 确定性 no-solution 审计：三个 SFT seed 均只覆盖少数任务；
- stochastic rollout probe：`seed_1234` 与 `seed_20260726` 合计 0/16 success，组内 reward variance 为 0；
- 路由：不直接执行 outcome-only GRPO，先做 M2.3-mini 数据补丁。

### 3.4 M2.3-mini

- 新专家任务：28 train + 10 valid，Expert 100% 成功；
- Recovery states：300；
- 混合数据：900 train + 249 valid，patch 占约 25%，无 Frozen Test 泄露；
- Continued LoRA training：train loss `0.000975`；
- Patch Valid：Exact Match 100%，Schema Valid 100%；
- Adapter：CPU/CUDA forward 和 generation 审计通过。

这些指标证明数据和训练管道成立，不代表闭环 rollout 已通过。正式结论必须等待修复后的 canonical probe 重跑。

## 4. 当前唯一权威执行路径

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
```

已删除旧的 temperature-sweep 和 comparison probe。旧脚本包含硬编码路径、错误 reward 归因和 Schema-invalid fallback，不得从历史提交恢复为正式入口。

单任务 smoke：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 1 1
```

正式 A/B：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8
```

## 5. M3.0B 准入条件

只有同时满足以下条件，才能启动策略梯度更新：

1. 基础设施错误不进入 reward，正式 rollout 中基础设施错误率接近 0；
2. 至少一个任务组同时包含 reward 0 和 reward 1；
3. 至少一个 no-solution rollout 成功；
4. 每个训练 step 保存准确的 prompt token IDs、completion token IDs 和 old-policy log-probabilities；
5. M2.3-mini 不显著提高 false no-solution；
6. 通用任务能力相对 M2.2R 不明显退化；
7. Frozen Test 未用于采样温度、checkpoint 或超参数选择。

## 6. 尚未完成

- M2.3-mini canonical A/B GPU rollout；
- 独立 `rollout_dev` 上的温度选择；
- 新 `final_test_v2` 冻结集；
- GRPO 单 batch loss/gradient/checkpoint smoke；
- 多 seed RL 稳定性验证。

在这些工作完成前，不得写入 `READY_FOR_GRPO=true`。

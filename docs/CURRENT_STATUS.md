# MiniWebWork-RL 当前实现状态

> 权威状态页。最后更新：2026-07-31。

## 1. 项目定位

MiniWebWork-RL 是一个面向采购调研流程的轻量浏览器 Agent 算法项目。模型读取文本化浏览器状态，通过固定 JSON 动作空间操作本地确定性网站，并由非 LLM Verifier 计算终态奖励。

项目目标不是复现通用 WebArena，也不是构建第三方 Agent 编排框架。目标是形成一条可解释、可训练、可审计的最小多轮 Agentic RL 链路：

```text
Task
→ Browser Environment
→ Qwen Policy
→ Multi-turn Rollout
→ Deterministic Verifier Reward
→ SFT / GRPO-style Policy Update
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
| M2.3-mini | no-solution + recovery 数据补丁、继续训练 | PASS |
| M2.3-mini Readiness Probe | historical GPU readiness evidence | PASS |
| M3.0B-0 | schema-v3.3 严格分布采集与 A/B 策略选择 | IN PROGRESS |
| M3.0B-1 | 单 batch LoRA gradient/checkpoint smoke | NOT STARTED |

正式状态：

```text
M2_3_MINI_DATA_PASS=true
M2_3_MINI_TRAINING_PASS=true
M2_3_MINI_ADAPTER_LOAD_CONTRACT_PASS=true
M2_3_MINI_CANONICAL_PROBE_PASS=true
NO_SOLUTION_ROLLOUT_CAPABILITY_CONFIRMED=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=false
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

### 3.4 M2.3-mini 数据与训练

- 新专家任务：28 train + 10 valid，Expert 100% 成功；
- Recovery states：300；
- 混合数据：900 train + 249 valid，patch 占约 25%，无 Frozen Test 泄露；
- Continued LoRA training：train loss `0.000975`；
- Patch Valid：Exact Match 100%，Schema Valid 100%；
- Adapter：CPU/CUDA forward 和 generation 审计通过。

### 3.5 M2.3-mini Readiness Probe

修复 `PlaywrightThreadManager.start(headless=...)` monkey-patch 签名后，GPU readiness probe 完整通过。根据集群正式产物汇总：

```text
complete = true
infrastructure_errors = 0
raw_policy_logprob_coverage = 1.0
sampling_logprob_coverage = 1.0
no_solution_successes: A_M2.2R = 58
no_solution_successes: B_M2.3-mini = 43
valid_for_grpo_update = false
```

该历史产物记录了 `temperature=0.2, top_p=0.9`，但没有显式冻结 `top_k`。因此它足以支持基础设施、闭环能力和 logprob coverage 结论，但不是完整标识的行为分布，不能作为 optimizer batch。

当前结论边界：

- A、B 均具备 no-solution 闭环能力；
- M2.3-mini 已解决原先全零奖励、无法启动 RL 的问题；
- 单次汇总中 B 的成功数低于 A，不能在缺少配对分析和多 seed 复验时宣称补丁优于 A，也不能直接归因于采样方差；
- 当前 no-solution-only 开发集不能评估 feasible-task 退化或 false no-solution。

## 4. 已冻结运行合同

- Action Schema v1.1：`submit` 可作用于 button；
- Prompt Contract v2：Base、SFT、E2E 和 rollout 共用；
- Task source：默认集与开发集互斥加载，重复 task ID fail-fast；
- Browser：单一 async Playwright worker，显式启动、停止和 join；
- Web：episode 与 task 绑定，页面导航保持上下文；
- Verifier：检查 episode/task、单一 submission、约束和目标最优性；
- Failure：策略失败 reward 0，基础设施失败 reward null；
- Rollout：保存 prompt/completion tokens、原始策略 log-prob 和采样分布 log-prob；
- Distribution identity：每条 trajectory 和 group 显式保存 `temperature + top_p + top_k`；
- Strict distribution：`T=1, top_p=1, top_k=0`，且 raw/sampling token log-prob 在数值容差内一致；
- Metrics：Schema、环境动作和任务成功使用明确且互不混淆的分母；
- Slurm：禁止全局 `pkill`，禁止 Python 在 CUDA 初始化后改写设备可见性。

## 5. 当前权威入口

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
scripts/analyze_probe_ab.py
```

Slurm 参数：

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P] [TOP_K]
```

显式诊断分布：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 0.2 20260731 8 "" 0.9 0
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 0.2 20260731 8 "" 0.9 0
```

严格更新分布采集：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch A 1.0 20260731 8 "" 1.0 0
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch B 1.0 20260731 8 "" 1.0 0
```

A/B 配对分析：

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

## 6. M3.0B 准入条件

### 6.1 已满足：诊断 readiness

1. 基础设施失败不进入 reward；
2. historical readiness probe infrastructure error 为 0；
3. raw/sampling logprob coverage 为 1.0；
4. 至少一个 no-solution rollout 成功；
5. readiness 分布存在可学习的 mixed-reward group。

### 6.2 尚未满足：正式策略更新

1. 使用 schema-v3.3 显式记录 `temperature/top_p/top_k`；
2. 在 `temperature=1.0, top_p=1.0, top_k=0` 下采集至少一个 `valid_for_grpo_update=true` group；
3. strict group 的 raw/sampling log-prob 最大绝对差不超过预声明容差；
4. 使用配对、多 seed 结果完成 A/B 初始策略选择；
5. 在 feasible development slice 上确认 false no-solution 和通用任务能力未明显退化；
6. 执行更新前 old/current logprob 一致性审计；
7. 完成一次 LoRA-only single-batch gradient/checkpoint/reload smoke；
8. Frozen Test 不参与 checkpoint、采样或 optimizer 超参数选择。

## 7. 下一阶段

```text
M3.0B-0A  schema-v3.3 显式诊断分布复验与配对分析
M3.0B-0B  增加 feasible rollout_dev slice 并完成策略选择
M3.0B-0C  T=1, top_p=1, top_k=0 严格更新分布采集
M3.0B-1   单 batch LoRA gradient/checkpoint smoke
M3.0B-2   5–10 update 小规模 pilot
```

在 M3.0B-1 完成前，不得写入 `READY_FOR_GRPO_UPDATE=true`。

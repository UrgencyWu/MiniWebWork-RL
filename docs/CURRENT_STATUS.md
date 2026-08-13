# MiniWebWork-RL 当前实现状态

> 权威状态页。最后更新：2026-08-14。

## 项目定位

当前方向是公开 WebShop benchmark 上的文本 Agent 后训练研究。M6 已实现并完成
development-only Raw → SFT → RL 最小链和 2 方法 × 3 seed 中等规模验证；SFT 稳定超过
Raw，但 RL 未继续超过 SFT。
原确定性采购
网站实现保留为基础设施与失败诊断，不再承担 M5 正式效果结论。

```text
Task → Browser Environment → Qwen Policy → Multi-turn Rollout
→ Deterministic Verifier → SFT / GRPO-style Update → Frozen Evaluation
```

## M6 当前状态

M6.2 development-only 中等规模验证已完成，最终状态为 `STOP_MEDIUM_RL`；没有打开
promotion/holdout，也没有授权全量正式训练。六个 RL 分支为 2 methods × 3 paired
seeds，每个分支完成 20 次 K8 mixed optimizer update，全部通过有限 loss/gradient、真实
参数变化、信用守恒、成本和 lineage 审计。正式开发评测使用全新的 500 tasks × K4，8 个
身份合计 16,000 条轨迹：

| 身份 | strict success | 相对 SFT |
|---|---:|---:|
| Raw | 37.000% | -5.350 pp |
| shared SFT | 42.350% | — |
| multi-turn GRPO（3-seed mean） | 41.967% | -0.383 pp |
| Anchor-GiGPO（3-seed mean） | 42.350% | +0.000 pp |

SFT-Raw 为 `+5.35 pp`，95% CI `[+3.70, +7.00] pp`，paired permutation
`p=0.000050`。GRPO-SFT 的 crossed seed-task 95% CI 为 `[-1.033, +0.283] pp`，
Anchor-SFT 为 `[-0.633, +0.633] pp`，均未满足预注册 `+3 pp` 晋级门。结论是
**工程训练链成功、SFT 数据修复成功、RL 性能晋级失败**，不是 learner 未运行，也不是
已经证明 RL 显著损害 SFT。完整记录见
[`M6_MEDIUM_TRAINING_AND_EVALUATION_REPORT.md`](M6_MEDIUM_TRAINING_AND_EVALUATION_REPORT.md)。

M6.1 development-only mini chain 的历史状态为 `STOP_AND_BURN_MINI_DEV`。实现针对 M5
负迁移做三项核心修复：

1. SFT 从 Raw policy 在公开 observation 下产生的 strict-success/recovery 轨迹蒸馏，禁止
   hidden-title 和 target-ASIN 标签；
2. Raw、SFT、RL 之间设置闭环 strict-success 晋级门禁，前一阶段未被超过就停止；
3. RL 用 strict binary terminal reward 做 GRPO macro credit，并用 verifier progress 的
   telescoping TD 差分提供 turn credit，避免继续奖励 partial-match purchase。

M6 在全量训练前运行了一条 development-only 最小链：256-task mini-train、200-task
隔离 mini-dev，顺序验证 `Raw < mini-SFT < mini-RL`。Raw、SFT、GRPO 和
Anchor-GiGPO 在相同 200 tasks × K4 上的 strict success 分别为 35.750%、
41.625%、41.375% 和 41.375%。SFT-Raw 为 +5.875 pp；两个 RL-SFT 都是
-0.250 pp，task-cluster bootstrap 95% CI 均为 `[-1.25, +0.75] pp`，exact
McNemar `p=0.8145`。所以结果是“没有检测到 RL 净提升”，而不是已经证明 RL 显著伤害
SFT。

两个 learner 都通过 finite loss/gradient、5 次有效更新、真实 adapter 参数变化和信用覆盖
审计，但每个方法只在 5 个 task、40 条轨迹和约 2.3k action tokens 上训练，效应不足且
不稳定。Corpus 的 493 条轨迹、31,362 label tokens 等检查通过，但唯一成功 task 为
156，低于冻结门槛 160；development-only waiver 不改变正式失败状态。当前 mini-dev 已
burn，禁止在其上继续调参。完整结果与原因见
[`M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md`](M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md)，逐作业失败与
修复见 [`TRAINING_FAILURE_LEDGER.md`](TRAINING_FAILURE_LEDGER.md)。

## M5 当前状态

M5 已把范围冻结为一个 shared verified SFT、multi-turn GRPO 与 public-anchor
GiGPO-style credit 各 3 seeds。数据源为固定 Agent-R1 WebShop full：1,181,430
商品、12,087 goal。原始切分中的 203 个后出现重复 instruction 已通过
canonical-first lock 隔离；测试 500 条保持完整，eligible dev/train 为 499/10,885。

M5 已完成 shared verified SFT、2 方法 × 3 seed 的正式在线训练，以及 8 个冻结推理
身份在 500 test tasks × K=4 上的最终评测。六个正式 online Jobs `2165–2170` 均为
`COMPLETED/0:0`；最终矩阵包含 16,000 条轨迹。Raw、SFT、multi-turn GRPO 和
Anchor-GiGPO 的严格成功率分别为 33.50%、0.65%、9.42% 和 9.27%。本轮结论是：
SFT 因 privileged、闭环不可稳定预测的搜索标签发生严重负迁移；在线 RL 明显恢复
状态推进和购买行为，但没有恢复到 Raw；Anchor-GiGPO 没有在严格成功率上显著优于
GRPO，不过 dense score 更高、评测 token 平均低约 15%。

最终分析 Job `2184` 为 `COMPLETED/0:0`；统计报告、配对任务差异和 14,196 条失败
轨迹分类均已生成，共享 WebShop 服务也已停止。完整结果、统计边界、失败分析和产物
哈希见 [`M5_FINAL_TECHNICAL_REPORT.md`](M5_FINAL_TECHNICAL_REPORT.md)。

事前冻结方案见 [`M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md`](M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md)，
执行与恢复记录见 [`M5_EXECUTION_READINESS.md`](M5_EXECUTION_READINESS.md)。下文
M1–M4 状态均为历史证据。

## 阶段状态

| 阶段 | 状态 |
|---|---|
| M6.2 medium RL（2 methods × 3 seeds） | COMPLETE / 6 audits PASS; 120 effective updates |
| M6.2 formal-dev（8 identities × 500 tasks × K4） | COMPLETE / 16,000 trajectories |
| M6.2 performance gate | STOP / GRPO -0.383 pp; Anchor +0.000 pp vs SFT |
| M6 promotion / holdout | NOT OPENED |
| M6 Phase A/B implementation | COMPLETE / cluster chain exercised |
| M6 mini Raw→SFT gate | PASS / +5.875 pp |
| M6 mini SFT→RL gate | STOP / both methods -0.250 pp; not significant; promotion threshold missed |
| M6 full expansion | NOT AUTHORIZED / mini-dev and formal-dev are development-visible |
| M5 verified SFT | COMPLETE / negative closed-loop transfer identified |
| M5 online RL（2 methods × 3 seeds） | COMPLETE |
| M5 frozen test（8 identities × 500 tasks × K4） | COMPLETE |
| M5 final statistical analysis | COMPLETE / exploratory inference |
| M1.0–M1.2 Environment / Agent Runtime | PASS |
| M2.0 Canonical Base Agent | PASS |
| M2.1F Expert trajectories / SFT data | PASS |
| M2.2R Canonical SFT / Frozen E2E | PASS |
| M3.0A Rollout readiness audit | PASS / Route B |
| M2.3-mini no-solution + recovery patch | PASS |
| M2.3 historical readiness GPU probe | PASS |
| M3.0B-0A schema-v3.3 / feasible-v2 regression gate | COMPLETE / neutral result |
| M3.0B-0C strict update collection | PASS |
| M3.0B-1 one-batch LoRA smoke | PASS |
| M3.0B-2 formal one-batch GRPO update | PASS |
| M3.0C frozen paired comparison | COMPLETE / no improvement supported |

```text
M6_MEDIUM_RAW_SUCCESS_RATE=0.37000
M6_MEDIUM_SFT_SUCCESS_RATE=0.42350
M6_MEDIUM_GRPO_MEAN_SUCCESS_RATE=0.41967
M6_MEDIUM_ANCHOR_MEAN_SUCCESS_RATE=0.42350
M6_MEDIUM_RL_AUDITS_PASS=6/6
M6_MEDIUM_EFFECTIVE_UPDATES=120
M6_MEDIUM_DECISION=STOP_MEDIUM_RL
M6_PROMOTION_OPENED=false
M6_HOLDOUT_OPENED=false
M6_MINI_RAW_SUCCESS=286/800
M6_MINI_SFT_SUCCESS=333/800
M6_MINI_GRPO_SUCCESS=331/800
M6_MINI_ANCHOR_GIGPO_SUCCESS=331/800
M6_MINI_SFT_GATE=PASS
M6_MINI_RL_GATE=STOP
M6_MINI_DEV_BURNED=true
M6_FORMAL_EXPANSION_ALLOWED=false
M2_3_MINI_CANONICAL_PROBE_PASS=true
SCHEMA_V3_3_ROLLOUT_IMPLEMENTED=true
PAIRED_AB_ANALYSIS_IMPLEMENTED=true
ROLLOUT_DEV_FEASIBLE_V2_FROZEN=true
M3_0B1_SINGLE_BATCH_SMOKE_IMPLEMENTED=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=true
M3_0_STRICT_COLLECTION_PASS=true
M3_0_FORMAL_GRPO_UPDATE_PASS=true
M3_0_FROZEN_REGRESSION_COMPLETE=true
M3_0_DELIVERY_REPORT_COMPLETE=true
```

## M3.0 正式更新证据

严格训练来源为 job 1080 的完成版 no-solution artifact：

```text
Git SHA                              aa8cfb9073969deb42ba8c57199910aa7e7aba7a
selected group                       M2_3_V0001
valid trajectories / infra errors    8 / 0
reward sequence                      [1, 1, 0, 1, 1, 0, 1, 0]
max raw-vs-sampling logprob diff     0.006612047553062439 <= 0.05
strict runtime                       use_cache=false; T=1; top_p=1; top_k=0
```

正式 GRPO job 1081 在隔离目录 `outputs/m3_0_updates/A_1081/` 产生更新
checkpoint；其报告为 `complete=true`、`passed=true`、`formal_update=true`。
更新前 old/current replay 最大差异为 `0.0`，8 条轨迹包含 1,146 个 action
token、54 个 turn；256/256 个 LoRA 张量获得非零梯度并发生变化，最大参数
绝对变化为 `1.000240445137024e-06`。保存的 adapter SHA-256 为：

```text
df66dab40a54abdbcb25d4b87ff195755b842868c608394dfb6e4451eb27cfec
```

更新后重载 forward 为有限值。基础设施异常轨迹没有进入奖励、advantage 或
梯度；冻结 feasible-v2 从未进入该 batch。

## M3.0 冻结回归对照（完成）

比较的是独立于 no-solution 梯度来源的、冻结且 `may_update_model=false` 的
`rollout_dev_feasible_v2`。它是 no-gradient regression gate，而不是事后
声称的 newly-opened final hold-out。基线 job 1082 与更新策略 job 1083 串行
运行，二者完全匹配：相同 evaluation Git SHA `bf62c94`、任务源 SHA-256、
temperature `0.2`、top-p `0.9`、top-k `0`、K `8` 和 seed `20260731`；
唯一变化是 adapter。

| 指标 | M2.2R | 更新策略 |
|---|---:|---:|
| 完成 / 有效轨迹 | 96 / 96 | 96 / 96 |
| 基础设施错误 | 0 | 0 |
| 成功 | 14/96 (14.58%) | 14/96 (14.58%) |
| feasible 成功 | 14/22 (63.64%) | 14/22 (63.64%) |
| false no-solution | 0 | 0 |
| `model_output_failure_limit` | 74 | 74 |

成对表为 both-success=12、M2.2R-only=2、updated-only=2、both-fail=80，
updated minus baseline success delta = `0.0`；task-bootstrap 95% CI 为
`[-0.03125, 0.03125]`，exact McNemar `p=1.0`。该结果是可复现的中性/负
结果：**没有证据支持本次单 batch 更新提升该冻结回归门禁**。完整报告：
[`../reports/M3_0_DELIVERY_REPORT.md`](../reports/M3_0_DELIVERY_REPORT.md)。

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

## 后续研究（不改变本次结论）

1. 在新的、版本化的开发训练源上预注册多 batch / 多 seed 方案，重点诊断
   `model_output_failure_limit` 主导的失败；不得回用 feasible-v2 做梯度或调参。
2. 在训练/采样设置冻结后打开 `final_test_v2`，作为一次性最终 hold-out；其
   结果不得反向改变已报告的训练决策。
3. 保留本次中性报告和全部 hash、Slurm job、artifact 路径，作为下一轮比较的
   基线，而不是删除或重写不利结果。

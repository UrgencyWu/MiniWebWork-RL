# M6.2 中等规模 RL 训练与配对评测报告

> 状态：development-only M6.2 已结束
>
> 最终决策：`STOP_MEDIUM_RL`
>
> 评测报告自哈希：`dd6004510813e8ac46e134bd56d7b1794ffb73d318a3b9393dc4473860eb1637`
>
> 统计单位：500 个全新 `formal_dev` task，每个身份 `K=4`，共 2,000 条轨迹/身份

## 1. 结论先行

本轮完整实现并审计了 `Raw → SFT → RL` 中等规模链，但只实现了
`Raw < SFT ≈ RL`，没有达到事前要求的 `Raw < SFT < RL`：

| 身份 | strict success | 相对 Raw | 相对 SFT |
|---|---:|---:|---:|
| Raw | 37.000% | — | -5.350 pp |
| shared SFT | 42.350% | +5.350 pp | — |
| multi-turn GRPO，3-seed mean | 41.967% | +4.967 pp | -0.383 pp |
| Anchor-GiGPO，3-seed mean | 42.350% | +5.350 pp | +0.000 pp |

SFT 相对 Raw 的提升为 `+5.35 pp`，task bootstrap 95% CI 为
`[+3.70, +7.00] pp`，paired permutation `p=0.000050`。这证明 M6 的公开观察
success-replay SFT 数据方案在新的 500-task 开发集上产生了稳定闭环提升，并复现了
M6.1 的正向 SFT 结果。

两种 RL 的训练链都通过审计，但均未超过 SFT：

- GRPO 的三 seed 平均为 `-0.383 pp`，crossed seed-task 95% CI
  `[-1.033, +0.283] pp`；
- Anchor-GiGPO 的三 seed 平均为 `0.000 pp`，95% CI
  `[-0.633, +0.633] pp`；
- Anchor-GiGPO 相对 GRPO 为 `+0.383 pp`，95% CI
  `[-0.200, +0.983] pp`，不能据此声称 Anchor 更优。

因此本轮的准确结论是：**SFT 数据修复成功；六个 RL learner 都真实训练成功，但当前
奖励、数据覆盖与更新方案没有产生可泛化的 SFT 之上增益。** `promotion` 与 `holdout`
均未打开，不得把本结果表述为最终测试成绩。

## 2. 冻结训练合同

M6.2 只扩展 RL 验证，不重跑 SFT。六个分支从同一 M6 SFT adapter 开始：

```text
methods                       multi_turn_grpo, anchor_gigpo
paired training seeds         20260812, 20260813, 20260814
curriculum candidates/run     32 tasks
target optimizer updates/run  20 mixed-reward updates
group size                    K=8
action-token cap/run          50,000
learning rate                 1e-6
policy epochs                 1
Slurm/run                     1 GPU, 4 CPU, 24 GiB, 24h
scope                         development_only; formal_training=false
```

同一 seed 的两种方法共享 SFT 起点、curriculum、预算和优化超参数。GRPO 将组内
strict terminal advantage 施加到轨迹全部 action turn；Anchor-GiGPO 在同一 macro
advantage 上加入 detached、轨迹内零和的 verifier-TD turn redistribution。两者比较的
核心变量是信用分配方式，而非数据量或计算预算。

## 3. 训练执行与审计结果

初始六分支 array Job `2236` 启动了全部方法与 seed。Anchor seed 20260812 和 GRPO
seed 20260814 直接完成；其余四个分支在 15–16 次有效更新后被 replay parity 尾部门禁
拒绝。修复没有放宽已采样组的梯度准入：不可信组被结构化标为 rejection，产生零更新，
计入采样成本，然后从冻结 curriculum 的下一任务恢复。

最终恢复 Job `2255` 完成剩余四个分支。六个最终审计均为 `passed=true`：finite
loss/gradient/KL、真实参数变化、20 次 optimizer update、20 个 mixed group、信用守恒、
telescoping、partial-match 负 terminal credit、token cap 和 lineage chain 全部通过。

| 方法 | seed | updates | action tokens | credited turns | final iteration | RL audit SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| GRPO | 20260812 | 20 | 13,107 | 767 | 23 | `8fdb128b…f5110b8` |
| Anchor | 20260812 | 20 | 12,944 | 768 | 23 | `4d2ed751…7072738` |
| GRPO | 20260813 | 20 | 12,886 | 716 | 23 | `df855a5b…8c4c03b` |
| Anchor | 20260813 | 20 | 12,392 | 730 | 22 | `77a541ca…ae78b` |
| GRPO | 20260814 | 20 | 12,132 | 735 | 20 | `34304ce0…b2bcad` |
| Anchor | 20260814 | 20 | 11,369 | 732 | 20 | `2c2261cc…7acaa8b2` |

合计形成 120 个有效 K8 信用组、960 条有效更新轨迹、74,830 个审计内 action tokens
和 4,448 个 credited turns。四个恢复分支各保留 1 个 parity rejection；被拒组没有
loss、gradient 或参数更新。另有 homogeneous strict-reward collection 只计成本、不进入
optimizer，证明系统没有把无学习信号的组伪装成有效更新。

`final iteration` 大于 20 不代表做了更多 optimizer update：它包含被安全跳过的
homogeneous/parity-rejected collection。正式评测从 `rl_audit` 的 20 个 learner report
更新链解析 `iteration_{N}/learner/adapter`，而不是简单选择编号最大的目录。

## 4. 中断与恢复记录

| Job | 结果 | 直接原因 | 处置 |
|---|---|---|---|
| 2236 array | 2 completed / 4 failed | 四分支在后段触发 replay P95/P99 门禁 | 保留 15–16 个已审计更新；实现安全 rejection 后同根恢复 |
| 2242 recovery | failed in 0–1 s | 缺少版本化 SFT gate / identity 路径 | 补齐显式路径；未启动 learner |
| 2246 recovery | failed in 1–2 s | Raw/SFT identity 与 gate binding 漂移 | Job 2250 从不可变 collection 重建配对 identity 和 v4 gate |
| 2250 CPU gate | completed | SFT-Raw `+5.875 pp`，bootstrap positive `1.0` | 允许继续同根 RL 恢复 |
| 2251 recovery | failed in ~30 s | 把上游 Raw group producer 误作 curriculum producer | 改用 curriculum 自身 producer bridge；新 collection 未启动 |
| 2255 recovery | completed | 正确恢复四个分支 | 六个分支全部达到 20/20 并通过最终审计 |

这条记录区分了三类事件：真实 parity 数据拒绝、恢复配置错误、最终算法训练。失败 adapter
没有进入评测；已通过审计的前序更新没有因恢复而重算或重复消费预算。

## 5. 冻结开发评测

评测使用此前未参与 M6.2 checkpoint 选择的 `formal_dev`：

```text
identities              Raw + SFT + 2 methods × 3 seeds = 8
tasks/identity          500
K                       4
trajectories/identity   2,000
total trajectories      16,000
rollout seed            20260815
model/env step limits   18 / 15
gradient                disabled
promotion/holdout       unopened / unopened
```

GPU array Job `2259` 的 8 个身份全部 `COMPLETED/0:0`，CPU statistics Job `2260`
也为 `COMPLETED/0:0`。最终 adapter 由 `rl_audit` 的更新链解析，而不是按目录名做
字典序选择，避免把恢复过程中的旧 checkpoint 误当最终模型。

### 5.1 Raw 与 SFT 行为指标

| 指标 | Raw | SFT | 变化 |
|---|---:|---:|---:|
| strict success | 37.000% | 42.350% | +5.350 pp |
| dense score | 0.63833 | 0.66569 | +0.02737 |
| action error | 7.200% | 6.050% | -1.150 pp |
| schema error | 6.800% | 2.150% | -4.650 pp |
| partial-match purchase | 54.300% | 48.500% | -5.800 pp |
| search exhaustion | 1.750% | 2.050% | +0.300 pp |
| mean environment steps | 4.956 | 5.199 | +0.243 |
| mean action tokens | 72.089 | 74.129 | +2.040 |

SFT 最大的明确改善是 schema error 和 partial purchase 的下降；代价是平均交互步数与
token 略增。它不是靠更短轨迹取胜，而是提高了动作结构和最终商品匹配的准确性。

### 5.2 三 seed RL 结果

| 方法 | seed | strict success | RL-SFT | task bootstrap 95% CI | paired permutation p |
|---|---:|---:|---:|---:|---:|
| GRPO | 20260812 | 41.950% | -0.400 pp | [-1.05, +0.25] | 0.3029 |
| GRPO | 20260813 | 42.200% | -0.150 pp | [-0.90, +0.65] | 0.8013 |
| GRPO | 20260814 | 41.750% | -0.600 pp | [-1.30, +0.10] | 0.1245 |
| Anchor | 20260812 | 42.400% | +0.050 pp | [-0.65, +0.75] | 1.0000 |
| Anchor | 20260813 | 42.250% | -0.100 pp | [-0.90, +0.70] | 0.9013 |
| Anchor | 20260814 | 42.400% | +0.050 pp | [-0.65, +0.75] | 1.0000 |

冻结晋级门要求：三 seed 平均至少 `+3 pp`、至少 2/3 seeds 为正、bootstrap 正方向
比例至少 `0.8`、全部训练审计通过。结果如下：

| 方法 | mean RL-SFT | positive seeds | bootstrap positive | audits | 晋级 |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.383 pp | 0/3 | 0.1127 | PASS | FAIL |
| Anchor | +0.000 pp | 2/3 | 0.4875 | PASS | FAIL |

GRPO 三个 seed 的方向都为负，但幅度很小且单 seed CI 都跨 0；Anchor 三 seed 基本复制
SFT。多 seed 结果排除了“只因某一个坏 seed 才失败”的简单解释，同时也没有支持
“RL 显著损害 SFT”的强结论。

## 6. 成本记录

训练 Job `2236/2242/2246/2251/2255` 的 GPU allocation 累计约
`4 h 46 min`；正式开发评测 8 个 GPU task 累计约 `5 h 36 min`，CPU statistics
为 `1 min 27 s`。不计仍在运行的共享 WebShop service，训练与评测合计约
`10.38 GPU-hours`。

评测阶段完整生成 16,000 条轨迹。由于八个身份并行、每身份独占 1 GPU，墙钟完成时间
明显低于 GPU-hours 总和。这里报告的是 Slurm allocation 时间，不等价于 GPU 利用率、
能耗或货币成本；共享服务 Job `2230` 的最终成本应在服务停止后另行结算。

## 7. 为什么“训练成功”仍没有性能提升

证据最支持以下解释：

1. **优化目标与晋级指标仍有间隙。** strict terminal reward 能保证目标一致性，但在
   partial purchase、错误 option 和完全失败之间提供的信息很少；turn redistribution
   只改变信用位置，不能凭空增加 macro reward 的可辨识度。
2. **训练覆盖比 mini 更大，但仍只有 20 个有效 task-group/seed。** 六分支共 120 个
   更新组足以检验 seed 符号，却未覆盖 500-task formal-dev 的长尾商品和指令组合。
3. **小学习率与强 KL 保留使策略几乎停留在 SFT 邻域。** Anchor 的结果正体现为更稳地
   保留 SFT，而不是产生新的 strict-success 能力；GRPO 则出现约 0.4 pp 的局部重排损失。
4. **信用分配不是本轮首要瓶颈。** 两种方法的差异 CI 跨 0。继续堆叠算法名称的收益
   小于改进训练组选择、reward informativeness 和 SFT retention 评测。
5. **M6 SFT 已是强基线。** 从 Raw 提升 5.35 pp 已被显著性检验确认；在同样有限预算下
   再获得预注册的 +3 pp，比修复 M5 的退化 SFT 难得多。

## 8. 可复用经验与后续边界

1. 数据语义修复比更换 RL 公式产生了更大的实际收益；M6 已证明这一点可跨两个开发
   slice 复现。
2. loss、gradient、参数变化、信用覆盖和 lineage 审计只能证明训练有效执行，不能替代
   冻结闭环评测。
3. parity rejection 必须是零更新并继续预算，而不是关闭门禁或让错误组进入梯度。
4. 多 seed 是必要的：它把“单次偶然持平”提升为“两个算法都未形成稳定正效应”的证据。
5. 下一轮若继续，应优先设计更有信息量但仍与 official success 对齐的 macro reward、
   hard-task curriculum 与显式 SFT retention gate；不应立即增加第三种 RL 算法。
6. 在新的 development gate 证明稳定 `RL > SFT` 前，不打开 promotion/holdout，也不扩大
   为昂贵的全量训练。

## 9. 权威产物

远端根目录：

```text
/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1
```

关键路径：

```text
medium/online_v1/seed_20260812/{multi_turn_grpo,anchor_gigpo}/rl_audit.json
medium/online_v1/seed_20260813/{multi_turn_grpo,anchor_gigpo}/rl_audit.json
medium/online_v1/seed_20260814/{multi_turn_grpo,anchor_gigpo}/rl_audit.json
medium/formal_dev_eval_v1/{raw,sft,*_seed_*}/identity_report.json
medium/formal_dev_eval_v1/analysis_report.json
```

冻结计划为 `data/m6_medium_eval_plan_v1.json`，其 SHA-256 为
`109a062ccd33c0330b5ca864b9e50dc06da6930fc45ab9032efec02569b8b9f9`。
完整失败事件和后继关系见
[`TRAINING_FAILURE_LEDGER.md`](TRAINING_FAILURE_LEDGER.md)。

# MiniWebWork-RL：Raw → SFT → RL 迭代训练技术报告

> 报告状态：development-only 研究阶段总结
>
> 报告日期：2026-08-14
>
> 覆盖范围：M5、M6.1、M6.2 与 M6 Phase1 诊断
>
> 当前研究决策：停止原 M6 RL 配方扩量；不打开 promotion/holdout

## 摘要

本报告独立记录 MiniWebWork-RL 从 M5 到 M6 的训练、分析和优化演进。它不替代逐作业的
`TRAINING_FAILURE_LEDGER.md`，也不作为运行手册；其目标是回答四个连续问题：每一阶段试图
验证什么、得到什么证据、为什么未达到目标、下一轮为什么采用新的设计。

项目的稳定正结果来自数据语义修复。M5 的 SFT 使用了 policy prompt 不可见的精确商品信息，
导致闭环 strict success 从 Raw 的 33.50% 退化到 0.65%。M6 改为从公开 WebShop benchmark
中采集 policy-visible、strict-success、environment-replayable 的 Raw 轨迹后，SFT 在两个独立
开发切片上分别比 Raw 提升 5.875 和 5.350 个百分点。该结果证明公开状态 success distillation
能够把 Raw 模型偶发的正确行为转化为更稳定的策略。

项目尚未得到稳定的 `RL > SFT`。M6.1 的两种在线 RL 在 200-task mini-dev 上均比 SFT 低
0.25 个百分点；M6.2 在 500-task formal-dev 上，三 seed GRPO 平均比 SFT 低 0.383 个百分点，
Anchor-GiGPO 与 SFT 持平。Phase1 进一步证明，当前 Anchor 与 GRPO 的同批参数梯度 cosine
为 0.990806，学习率从 `1e-6` 增至 `1e-5` 虽能增加参数位移，却没有产生单调的行为改善。
综合证据指向数据支持集、奖励可辨识度、训练 horizon、batch 结构和训练时 policy parity，
而不是“再增加一种 RL 算法”。

## 1. 研究对象与证据边界

项目使用公开 WebShop full benchmark，包含约 118 万商品和 12,087 个 goal。任务不是项目
任意编写的合成题；项目自行生成的是模型在冻结 WebShop 环境中的交互轨迹、SFT action labels
和在线 RL rollout。主指标固定为：

```text
strict success = 1[official task_score >= 0.999]
```

连续 task score、partial purchase、失败分类、token、step 和 wall time 只作为诊断指标，除非
某个版本的训练合同明确将其写入优化目标。

M5 的正式结果已冻结。M6.1 与 M6.2 都是 development-only 验证；M6.1 mini-dev 和 M6.2
formal-dev 已经用于结果判断，后续不得继续用它们调参。M6 promotion 与 untouched holdout
从未打开，因此本报告不把任何 M6 数值包装成最终测试成绩。

## 2. M5：训练链执行成功，但 SFT 数据语义失败

M5 研究在相同 SFT 起点、K=4 在线采样、优化器和约 500,000 generated-action-token 预算下，
比较 multi-turn GRPO 与 Anchor-GiGPO-style 信用分配。训练和冻结评测均完成，但最终结果为：

| 策略 | strict success |
|---|---:|
| Raw | 33.50% |
| SFT | 0.65% |
| GRPO | 9.42% |
| Anchor-GiGPO | 9.27% |

失败的首要原因不是 learner 没有更新，而是 SFT 标签依赖 policy prompt 中不可见的精确商品
标题。Teacher-forced 指标能够下降，但闭环策略学成了长标题搜索、连续翻页和脆弱的固定行为。
这属于 verified-but-not-conditionally-learnable data：环境可以验证动作正确，却不能保证 policy
能够从输入观察稳定预测该动作。

在线 RL 相对严重退化的 SFT 恢复了约 8.6–8.8 个百分点，并改善了状态推进和购买完成率，
但仍远低于 Raw。这说明在线环境信号能够修复部分闭环行为，却无法在坏 SFT 起点和有限奖励下
恢复精确商品 grounding。Anchor 在 dense score 和部分成本指标上更好，但主指标未超过 GRPO。

由此得到 M6 的第一条设计原则：在任何在线 RL 前，SFT 必须在冻结闭环评测中稳定超过 Raw；
训练 loss、标签可执行性和 teacher-forced NLL 均不能替代该门禁。

## 3. M6.1：数据语义修复成功，最小 RL 未产生增量

### 3.1 数据重建

M6.1 在同一 256-task mini-train roster 上运行两个独立 Raw K8 seed，共得到 4,096 条 Raw
轨迹。最终 strict-success task 并集为 156，构建出 493 条 strict/replay-success 轨迹和
31,362 个 completion-label token。语料满足以下重要质量条件：

- label 只来自 policy-visible observation；
- hidden goal field 和 target-ASIN search label 为 0；
- strict success 与 environment replay success 均为 100%；
- public-query token fraction 为 0.9461；
- recovery trajectory fraction 为 0.2860；
- 最大 action-family fraction 为 0.3036。

唯一未达到原门槛的是唯一成功任务数 `156 < 160`，因此使用版本化 development-only waiver，
没有把该数据包装成正式合格 corpus。

这轮修复实现了真正的闭环增益：

| 策略 | strict success | 相对前一阶段 |
|---|---:|---:|
| Raw | 35.750% | — |
| SFT | 41.625% | +5.875 pp |
| GRPO | 41.375% | -0.250 pp vs SFT |
| Anchor-GiGPO | 41.375% | -0.250 pp vs SFT |

SFT 的正结果证明 M5 的主要问题确实是数据语义，而非 Qwen3.5-4B 无法完成 WebShop 任务。

### 3.2 最小 RL 失败分析

两种 learner 都产生 finite loss/gradient、5 次真实 optimizer update 和 adapter 参数变化，
但每种方法只训练 5 个 task、40 条轨迹和约 2,300 个 action token。200-task mini-dev 上，
SFT-only success 为 10，RL-only success 为 8，差异与采样波动相容。

这一阶段暴露出三类问题。第一，RL 暴露量远小于评测任务分布；第二，strict binary reward
无法区分 partial purchase、wrong option 和完全失败；第三，训练只保存最终 checkpoint，
没有独立 tuning-dev 学习曲线回答中间步骤是否短暂超过 SFT。M6.1 因此停止并 burn 当前
mini-dev，没有据此打开正式训练。

## 4. M6.2：多 seed 中等规模验证仍未形成泛化增益

M6.2 复用已经通过门禁的 M6 SFT adapter，只扩大在线 RL 证据。冻结配置为两个方法、三个
训练 seed、每 run 32 个候选任务、目标 20 个 mixed K8 update、LR `1e-6`、policy epoch 1、
50,000 generated-action-token cap。最终六个分支都完成 20 次有效更新并通过 loss、gradient、
KL、参数变化、信用守恒、token ledger 和恢复血缘审计。

冻结评测使用 500 个此前未参与 M6.2 checkpoint 选择的 formal-dev task，每身份 K=4、
共 2,000 条轨迹。结果为：

| 策略 | strict success | 相对 SFT |
|---|---:|---:|
| Raw | 37.000% | -5.350 pp |
| SFT | 42.350% | — |
| GRPO，3-seed mean | 41.967% | -0.383 pp |
| Anchor-GiGPO，3-seed mean | 42.350% | +0.000 pp |

SFT 相对 Raw 的 task-bootstrap 95% CI 为 `[+3.70,+7.00] pp`，paired permutation
`p=0.000050`。GRPO 相对 SFT 的 crossed seed-task 95% CI 为 `[-1.033,+0.283] pp`，
Anchor 为 `[-0.633,+0.633] pp`。两方法均未达到预注册的 `+3 pp`，最终决策为
`STOP_MEDIUM_RL`。

训练成功而性能未提升，证明工程审计只能说明“训练真实发生”，不能说明“训练目标有效”。
M6.2 也纠正了“单个坏 seed 导致失败”的解释：GRPO 三个 seed 方向均不为正，Anchor 三个
seed 基本复制 SFT。

## 5. Phase1：把负结果分解为可检验原因

Phase1 没有重新训练正式模型，而是对 M6.2 的不可变训练组、16,000 条评测轨迹和冻结 SFT
adapter 进行诊断。最终报告位于：

```text
outputs/m6_monotonic_posttraining_v1/phase1_diagnostics_v1/final_report.json
content_sha256 = 876dc8214551382fcdf101f59ee655ef03e4a576a0b547746b914d1650e46633
```

### 5.1 数据支持集

六个 run 的 120 个有效 K8 更新组只涉及 24 个 unique task，其中 15 个任务出现在全部六个
分支。每个 run 虽有 160 条更新轨迹，但独立统计单位仍只有 20 个任务。三个 seed 主要改变
rollout 和 optimizer 随机性，没有产生三套独立 task roster。

在这 24 个实际更新任务上，GRPO 三 seed 平均比 SFT 高约 0.35 个百分点，Anchor 高约
0.87 个百分点。然而它们分别只相当于 576 条轨迹净多成功 2 和 5 条，而且 seen24 本身是
“实际产生过更新”的后选择集合。该结果支持局部适配的现象假设，但不足以确认显著的局部学习；
formal-dev 无提升仍是更强证据。

### 5.2 Verifier 与信用分配

公开状态 Phi 对最终 strict success 并非完全无信息：最大/最终 preterminal Phi AUC 分别为
0.725 和 0.744。但 9,354 条失败中有 50.21% 曾达到较高 Phi，说明它不足以单独判断
buy-readiness。失败分类中 partial purchase 为 7,874 条，占约 84.18%；首要问题是末端商品、
option grounding 和购买门控，而非 schema 格式。

同一训练 batch 的参数梯度反事实为：

```text
GRPO gradient norm                 0.495869
Anchor gradient norm               0.447296
gradient cosine                    0.990806
relative norm difference           0.097956
turn advantage sign flips          0
```

这达到预注册的冗余停止门。当前 Anchor 只在轨迹内部零和重排同一个 macro advantage，没有
形成可区分的新优化方向，因此不再作为下一轮正式方法。

### 5.3 学习率与 KL

单步学习率探针结果为：

| LR | adapter relative L2 | fixed-state KL | seen-task delta |
|---:|---:|---:|---:|
| `1e-6` | 0.000175 | 0.000191 | -1.042 pp |
| `3e-6` | 0.000531 | 0.000326 | -2.604 pp |
| `1e-5` | 0.001762 | 0.001317 | +0.000 pp |

三个设置均产生 finite update 且没有越过 KL 0.01 安全上限，参数位移随 LR 增加，但行为没有
单调改善。因此“单纯提高学习率”被排除为充分修复方案。

M6.2 实际 reference KL 多在 `2e-4` 量级，adaptive coefficient 已下降到 `1e-5` floor。
旧报告中“强 KL 抑制学习”的表述应由本证据更新：策略靠近 SFT 是事实，但主要原因更可能是
任务覆盖小、单任务梯度互相抵消和奖励信息不足，而不是 KL 惩罚过强。

## 6. 跨轮次技术结论

### 6.1 数据支持集被连续筛窄

当前链路实际上是：Raw 成功拒绝采样 → SFT 只学习 Raw 已成功任务 → RL 再从 SFT 已见任务中
选择当前 policy 恰好 mixed 的 group。SFT 和 RL 没有形成互补的数据支持，RL 很难学习
Raw/SFT 从未成功的新任务，只会局部重排已会任务的概率。

语料还存在两个次要但真实的权重偏差。两个 Raw roots 顺序合并并受每任务四条上限影响，最终
493 条 corpus 中第一 root 贡献 407 条、第二 root 只贡献 86 条；按 action-turn row 训练又使
长任务获得更多 CE 权重。下一版本必须先全局合并、打散和去重，再采用 task → trajectory →
turn 的分层采样或任务总 loss 归一。

### 6.2 当前方法不是有效的算法对比

`policy_epochs=1` 且每个 group 更新前只 forward 一次，PPO clip 主要吸收 vLLM/HF replay
差异，并未形成典型的多 epoch PPO trust region。准确名称应是
`trajectory group-normalized policy gradient + SFT reference KL`；可以简称 trajectory-GRPO，
但文档必须注明 single-epoch。

当前 Anchor 的准确描述是 `verifier-TD credit-redistributed GRPO`，不是一个已经证明独立有效的
GiGPO 方法。Phase1 的梯度反事实已经足以停止该 arm，下一轮不增加第三种算法。

### 6.3 奖励缺少跨失败轨迹排序

当前 macro reward 只有 strict binary。Anchor 的 TD deviation 逐轨迹严格零和，只回答“同一条
轨迹中哪一步更好”，不能回答“partial failure 是否优于完全失败”。因此 Phi 有一定 AUC 也不会
自动转化为新的 trajectory ordering。

直接使用 official dense task score 同样不安全。WebShop 的 non-purchase/horizon failure 通常
为 0，而 partial purchase 可以获得较高分，容易进一步鼓励模型尽快购买一个相近商品。下一轮
需要 strict-dominant 的失败质量排序，而不是用 dense score 替换 strict success。

### 6.4 训练环境不是正式长程环境

M6 RL 训练固定为 6 model turns / 6 environment steps，M6.2 formal-dev 却允许 18/15。
训练无法覆盖第 7 步后的二次搜索、回退、option 修正和长程恢复。Prompt 还只保留最近 5 turn，
observation 上限为 8,000 字符，public actions 上限为 256，均可能形成部分可观测或状态别名。

该 horizon mismatch 是实质分布偏移，但不能单独解释全部失败，因为 formal-dev 失败中 84.18%
已经结束于 partial purchase。优先级应是先修 buy-readiness，再用单变量实验确认 full horizon 的
净作用。

### 6.5 训练时 policy parity 存在未验证风险

SFT LoRA dropout 为 0.05。RL parity 检查在 `model.eval()` 下通过，但真正计算 current-policy
loss 前调用 `model.train()`，重新启用 dropout；vLLM behavior policy 则是无 dropout 推理。
因此真正反向传播的 current logprob 包含 parity 门禁未覆盖的随机扰动。

这是代码合同上的高置信风险，但尚无 dropout on/off 因果结果，不能把历史性能下降全部归因于
它。下一轮 P0 必须把 RL dropout 固定为 0，并记录 training-forward ratio、clip 和梯度方差。

### 6.6 当前预算是 cap，不是优化目标

现有 token ledger、step cap、Slurm wall time 和 GPU-hours 用于公平控制、恢复和成本报告，均未
进入 reward。准确表述是 budget-matched / budget-capped RL，而不是 budget-aware RL。

当前 ledger 主要统计 generated action tokens，没有把每轮增长的 prompt/input tokens 纳入训练
预算。对于长程智能体，prefill token、environment steps 和重复模型加载同样影响真实成本。
下一版本需分别报告 attempted tasks、prompt/generated tokens、model/env steps、optimizer tokens、
GPU-hours 和 service CPU-hours，但在准确率转正前不把硬件时间或成本直接写入 reward。

## 7. 下一版本的优化设计

### 7.1 数据设计

在相同 rollout 总预算下，优先扩大独立任务而不是继续加深同一 256 个任务。建议使用约
1,024 个按 category × constraint count × 预估 horizon 分层的任务，先 K=4；只对欠覆盖或
near-miss strata 自适应追加 K8/K16。

SFT 语料应全局合并所有 seed 后再选择。每任务通常保留一条最短 strict success 和一条语义
不同的 recovery/长轨迹，按任务等权训练；retention 按 page type × success/failure × recovery
分层并实际消费至少 1,000 个不同状态。

新角色必须互斥：

```text
sft_source
sft_internal_dev
rl_train
rl_monitor
fresh tuning-dev2
promotion
holdout
```

RL curriculum 与 SFT corpus/dev 的任务身份必须为零重叠，并冻结 attempted roster、顺序和总预算，
不能继续以“直到得到 20 个 mixed group”为停止条件。

### 7.2 方法与 batch

下一轮只保留一个优化器：single-epoch trajectory-GRPO。研究对照只改变 reward，不改变算法：

```text
binary-reward trajectory-GRPO
vs
strict-dominant process-reward trajectory-GRPO
```

推荐等预算小试配置：

```text
K                              4 / task
task groups / optimizer step   4
trajectories / step            16
optimizer steps                10
unique tasks / run             40
independent task-roster seeds  2
model/env horizon              18 / 15
RL dropout                     0
learning rate                  3e-6 main, 1e-6 control
policy epochs                  1
gradient clip                  1.0
SFT retention                  about 20% stratified states
```

一次采集四个任务组、一次加载 learner 并跨组平均梯度，比每任务重复加载 vLLM/HF 模型更能利用
GPU，也能降低单任务高方差。增加 CPU 数量不会解决当前吞吐结构问题。

### 7.3 Strict-dominant 奖励

先从公开 item、selected options、price 和 instruction constraints 构造 buy-readiness
`z_t ∈ [0,1]`，不得使用 target ASIN、隐藏标题或未显示的商品信息。失败质量
`Q ∈ [-1,0]` 可按以下原则构造：partial/wrong-option purchase 根据 buy 前的 `z` 和 premature
penalty 计分；horizon 根据轨迹最大 `z` 和 horizon penalty 计分；schema/action collapse 记为
`-1`。

初始宏观奖励为：

```text
R = S * (1 - delta * C) + (1 - S) * epsilon * Q
S = strict success
epsilon = 0.1
delta = 0
```

所有 strict success 恒为 1，所有失败不大于 0，因此失败永远不会超过成功；同时 partial
failure 可以比完全失败得到更好的组内排序。第一轮不加入成本项，也不直接加入强 turn shaping。
只有 `z` 的 strict-vs-partial 校准通过后，才允许加入中心化状态差分，并把过程项 norm 限制为
macro 的 10%–25%，不得翻转 macro advantage 符号。

### 7.4 环境与预算

先用冻结 SFT、同 task/K/seed 进行 `6/6 vs 18/15` horizon 单变量实验。若 full horizon strict
提升至少 1 pp，或至少 20% 的短程 horizon failure 转为 strict success，后续训练必须匹配
18/15；若差异不足 0.5 pp 且转化不足 10%，才把 horizon 降为次要因素。

准确率转正前，token/step 只作为 cap 和报告。准确率稳定后才允许三级词典序目标：

```text
strict success > failure quality > cost
```

成本项只在 strict-success 轨迹之间比较。若加入 `delta=0.02` 的 success-only cost term，必须
同时满足 strict success 非劣下界大于 `-0.5 pp`，且 tokens 或 steps 至少下降 5%；否则删除
cost reward，只保留预算上限。

## 8. 最小验证与停止门

下一轮正式在线训练前依次完成：

1. dropout 0 vs 0.05 的同 batch 梯度方差和真实 ratio/clip 诊断；
2. buy-readiness 的 strict-vs-partial 校准，要求 AUC 至少 0.70，final buy-readiness AUC
   至少 0.80，高 buy-readiness 失败不超过 30%；
3. `K8 × 1 task` 与 `K4 × 4 tasks` 的固定轨迹预算比较；
4. binary 与 strict-dominant reward 的同批梯度反事实；
5. 两个独立 task-roster seed 的 40-task 在线小试；
6. 全新 100–200 task tuning-dev2 K4 闭环评测。

在线小试只有同时满足以下条件才允许扩量：

- mean RL-SFT 至少 `+1 pp`；
- 两个独立 roster seed 均为正；
- RL-only flips 多于 SFT-only flips；
- partial purchase 不增加超过 0.5 pp；
- schema/action error 各不增加超过 0.5 pp；
- seen-unseen gap 不超过 1 pp；
- reference KL 小于 0.01；
- replay parity rejection 小于 2%；
- training-time clip fraction 小于 10%。

若两个独立 roster 上 unseen delta 均不为正，或只在实际更新任务上提升而 matched-unseen/tuning
不提升，则判定为局部记忆或边界任务过拟合，停止扩大训练。Promotion 仍保留原门槛：均值至少
`+3 pp`、置信区间下界大于 0、全部正式 seed 为正，不因开发负结果降低标准。

## 9. 当前结论与项目价值

截至本报告，项目可以支持以下结论：

1. policy-visible、strict/replay-success 的数据重建使 SFT 稳定超过 Raw；
2. 当前 binary reward、SFT-overlapping mixed curriculum、one-task K8 update、6-step horizon 和
   single-epoch 配方不能产生可泛化的 SFT 之上增益；
3. 当前 Anchor 与 GRPO 在同批参数梯度上实质冗余；
4. 提高学习率不是充分修复；
5. 下一轮应验证奖励可辨识度、任务覆盖、horizon 和训练 parity，而不是增加算法数量。

这条迭代线体现的工程与研究经验不是“尝试过很多算法”，而是能够区分数据语义错误、工程执行
错误、优化信号不足、支持集偏差和真正的泛化失败，并用冻结评测与因果小实验决定是否继续消耗
算力。

## 10. 关联文档与产物

- M5 正式结果：`docs/M5_FINAL_TECHNICAL_REPORT.md`
- M6 原始改进计划：`docs/M6_MONOTONIC_POSTTRAINING_PLAN.md`
- M6.1 结果：`docs/M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md`
- M6.2 结果：`docs/M6_MEDIUM_TRAINING_AND_EVALUATION_REPORT.md`
- 执行合同：`docs/M6_EXECUTION_RUNBOOK.md`
- 作业失败、修复和后继：`docs/TRAINING_FAILURE_LEDGER.md`
- M6 权威远端根：`outputs/m6_monotonic_posttraining_v1`

后续每次完成一次具有新假设或新配方的训练阶段，应在本报告中追加一个叙事章节，说明研究问题、
唯一变化、冻结配置、训练证据、闭环结果、失败分类、被排除解释和下一决策；逐 Job 的退出码、
基础设施错误和恢复链仍只追加到失败账本。

## 11. M6 Phase2 P0：训练 parity 通过初筛，过程奖励校准未过门

Phase2 首批两个独立诊断于 2026-08-14 完成。P0a Job 2288 使用同一冻结 SFT adapter 和同一
预采 K8 group，对 dropout=0.05 与 dropout=0 各进行了 8 个 RNG repeat；P0b Job 2289 仅分析
冻结 SFT 的 2,000 条 formal-dev 轨迹，不执行训练。

P0b 的 public item/option buy-readiness 得到：strict-vs-all-failure final AUC=0.7862、
strict-vs-partial-purchase AUC=0.7746、高 readiness failure 比例=13.96%。后两项通过，但第一项
低于预注册 0.80 门。因此整体 `process_reward_calibration_passed=false`。这不是基础设施失败，
也不能通过把门槛降到 0.78 修正；当前 scorer 不得进入 process-reward 在线训练。它仍提供一个
有价值的局部结论：item/option evidence 对最主要的 partial-purchase failure 有区分度，但对
horizon、zero-match 和其他失败的统一排序还不够可靠。

P0a 的原始结果显示，dropout=0.05 时跨 repeat 梯度 mean cosine 约0.781、norm CV约0.0617；
dropout=0 时 training clip=0、reference KL=0、norm CV约0.0020，方向稳定性明显更高。然而 v1
报告用 float32 对超长梯度向量直接计算 cosine，产生了 1.0056 的越界值。该数值违反 cosine
范围，故 Job 2288 虽 exit 0，其 v1 报告只能作为失败尝试保留，不能成为最终证据。修复策略是
分块 float64 累积并对 [-1,1] 建立硬不变量，只重跑 P0a，不重跑 P0b。

修正版 Job 2290 以 `COMPLETED 0:0` 结束，v2 报告自哈希和 SFT adapter 哈希均通过。dropout=0.05
的 mean/minimum gradient cosine 为0.7765/0.6710，angular dispersion为0.2235，norm CV为
0.0619；dropout=0 的对应值为0.99993/0.99990、0.000074和0.00342。dropout=0 同时保持
training clip=0、reference KL=0、全部 loss/gradient finite，正式通过 P0a 门禁。这证明训练时
dropout 是显著的额外梯度噪声来源，后续 RL 必须固定 dropout=0；但该同批探针仍不能单独证明
历史 formal-dev 负结果全部由 dropout 导致。

当前研究决策为：停止基于现有 buy-readiness 的 P2 process-reward arm；继续完成与它独立的
P0c batch-structure 和 P1 horizon 因果实验。只有后续重新设计的 verifier 在全新校准数据上
通过原门槛，才可恢复 strict-dominant process reward 实验。

P1 首次提交在创建 Slurm Job 前被数据隔离门拒绝。审计确认冻结 split 的 `train` 角色本身包含
256个 `mini_train` 任务，但与 formal-dev、promotion、holdout 和 mini-dev 的交集均为0。这里不能
把角色名等同于天然隔离，也不能删除门禁。修复后的 roster 构造显式计算
`train - union(non-train roles)`，从6929个合格任务中按类别与约束数量分层、确定性选择64个，
并再次硬检验最终选择与全部已暴露角色零重叠。首次预检没有提交作业、没有产生采样或训练结果。

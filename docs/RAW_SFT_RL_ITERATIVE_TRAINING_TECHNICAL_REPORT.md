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

P0c Job 2291 随后以 `COMPLETED 0:0` 结束，报告自哈希、冻结 SFT adapter 哈希与同轨迹预算合同
均通过。K8×1 task 与 K4×4 tasks 都使用相同64条来源轨迹且不执行 optimizer step；K4 方案的
梯度方向离散度由0.9728降到0.8692，梯度范数变异系数由0.5277降到0.3203，15/16个 K4 子组
仍为 mixed。两种结构的全部 loss/gradient 有限，reference KL均为0；K4最大 clip fraction为
0.00310。该结果通过冻结工程门，后续小规模 RL 固定为 K4/task、每次跨4个task group聚合梯度。
它只证明同一批数据上的方差和覆盖改善，不单独证明 unseen 泛化提升。

P1 Jobs 2292_0/1 分别完成6/6与18/15 horizon 的64 task×K4采样，但依赖审计 Job 2293
正确拒绝了结果。1,134个可比较共享turn的 sampling seed 全部一致，然而独立并发vLLM运行仍有
100个生成token hash不一致；其中42个发生在prompt与seed都完全相同的turn，随后累积为89个
prompt分叉。因此两臂不再只相差horizon，不能据其成功率差异做因果判断。修复不重跑短程臂，
也不放宽配对门：将2292_0的真实短程action/token/state序列作为不可变前缀，在新18/15环境中
逐步重放并硬校验prompt、token、action及前后public-state哈希；只有消费完该前缀后才调用SFT
继续生成。这样新增的唯一模型变量是第7步之后是否允许继续，失败产物和原审计均保留。

精确前缀回放 Job 2295 与依赖审计 Job 2296 随后均以 `COMPLETED 0:0` 结束。64个任务、
256条配对轨迹的共享前缀 sampling seed 与 generated-token mismatch 均为0，报告及来源
collection/group/task-roster 哈希闭合，因此该结果可以作 horizon 单变量结论。6/6 strict
success 为95/256（37.11%），18/15为108/256（42.19%），绝对提升5.08 pp；51条短程
horizon failure 中13条在全长环境转为strict success，转化率25.49%。两项均超过预注册
`+1 pp / 20%` 门，后续RL必须采用18/15。代价是平均环境步数4.43升至5.29，生成action
tokens 63.63升至75.43；partial purchase也由108增至136，说明全长预算解决了“来不及完成”，
但没有自动解决末端商品/规格判断。P1不含训练，不能把这5.08 pp表述为RL增益。

P0b-r2 不降低2289的0.80门槛，也不覆盖其负结果。修复仅针对有明确语义证据的权重错误：旧
公式在存在option约束时给予item block 80%、selected-option block 20%，与“商品正确但规格错误
仍为失败”的任务合同不一致。v2固定两个必要语义块各50%；无option约束时仍只使用item/price
block，继续禁止search-result候选分、target ASIN和隐藏答案。该50/50不是从任务结果拟合的连续
超参数。冻结P1精确回放轨迹的只读开发诊断为strict-vs-all/partial AUC 0.8563/0.8500，
高readiness失败4.05%；公式冻结后，在旧500-task formal-dev轨迹上只作外部复核，得到
0.8604/0.8540和10.75%。下一步以版本化CPU-only作业生成带自哈希的正式P0b-r2报告；只有
校准集与历史外部复核都过原门，才恢复P2同batch奖励反事实。

P0b-r2 Job 2297 以 `COMPLETED 0:0` 结束，耗时15秒且没有训练。正式报告自哈希、公式版本和
public-only合同通过：64个SFT-disjoint P1任务、256条全长轨迹上的strict-vs-all/partial AUC
为0.8563/0.8500，高readiness失败4.05%；历史冻结500-task、2,000轨迹外部复核为
0.8604/0.8540和10.75%。两侧均通过原0.80/0.70/30%门，故2297恢复P2资格，但不能把离线
AUC表述为策略性能提升。P2只比较相同冻结K4轨迹上的binary与strict-dominant参数梯度，不执行
optimizer step；若梯度差异过弱、过强或破坏strict支配，仍会停止在线A/B。

P2 Job 2298 以 `COMPLETED 0:0` 结束，耗时5分46秒；报告绑定2297 readiness自哈希、8个
full-horizon K4 group哈希和同一SFT adapter，dropout=0、optimizer steps=0。两个4-task panel
的strict恒为1、failure均在[-0.1,0]，strict/failure advantage符号与排序未翻转；全部loss和
gradient有限，reference KL=0，最大clip fraction=0.328%。安全合同通过，但方法差异门明确
失败：两panel的binary-vs-quality参数梯度cosine分别为0.999861和0.999903，高于0.98冗余线；
secondary/primary gradient norm仅1.674%和1.407%，远低于10%下界。

该结果说明50/50 buy-readiness能够离线识别错误option，但当前`0.1*q`经过每个K4 group内的
标准化后，failure间只有约-0.075至-0.1的小差异，相对strict=1几乎只是binary reward的仿射
扰动，不能形成实质不同的参数更新方向。2298是有效算法负结果，不是基础设施失败；不得通过
调大epsilon、penalty或挑选group把它刷过预注册门。依据冻结停止规则，P3四作业在线A/B不提交，
本轮Phase2到此停止。后续若另立新研究轮，应优先检验不被组内标准化消去的失败排序目标或更有
差异的task/failure构成，而不是在本轮继续放大同一scaler。

## 12. M6 Phase3 预注册：标准化后的 strict-first 失败质量残差

Phase3不是对2298的epsilon调参，也不重启被停止的四作业在线A/B。它检验一个新的、可证伪的
机制假设：P2失败的原因可能是failure quality在进入K4组标准化之前被近似仿射消去，而不是
public buy-readiness本身完全无信息。唯一变化是把质量信用的插入点移到binary strict reward的
组内标准化之后；任务、轨迹、SFT adapter、dropout、K4和每次4个task group均保持不变。

对每个冻结mixed K4 group先计算：

```text
A_macro = standardize([strict_1, ..., strict_4])
q_i in [-1, 0]                  # 仅来自已冻结public buy-readiness
r_i = 0                         # strict success
r_i = 0.2 * (q_i - mean(q_failure)) / max_abs_centered_q   # failure
A_candidate = A_macro + r_i
```

若failure quality相同，则全部`r_i=0`。所有strict轨迹的`A_macro`必须逐位完全不变；failure残差
严格零均值、最大绝对值不超过0.2，且更好的failure只能获得更大的残差。mixed K4合同还要求
所有strict advantage继续为正、所有failure advantage继续为负、任意strict仍高于任意failure。
不使用official dense task score，不加入成本项，不执行optimizer step。

先在2298完全相同的8个full-horizon K4 group上做两个独立4-task panel梯度反事实。`0.2`在看到
结果前冻结，失败后不得调整scale刷门。两个panel都必须满足binary-vs-candidate参数梯度cosine
位于`[0.90,0.98)`，secondary/primary gradient norm位于`[10%,25%]`，reference KL不超过0.01、
clip fraction不超过10%、全部数值有限。cosine至少0.98仍判冗余；低于0.80或secondary norm超过
25%判过强。只有该零更新探针完整通过，才另行设计两个SFT-disjoint roster的小规模在线A/B；
探针失败即把这一信用形式记为有效负结果并停止，不消耗在线训练预算。即使探针通过，也只证明
梯度机制有区分度，不能表述为策略性能提升。

Phase3 Job 2300 随后以 `COMPLETED 0:0` 结束，耗时5分45秒。报告自哈希
`c31fef2d98c37f20b65ac3ede3230be5dbe87586e61d7afaac2d8fdcf4ca7182`闭合，并逐项绑定2298负报告、
完全相同的8个full-horizon K4 group及同一SFT adapter；dropout=0、optimizer steps=0，未产生模型
参数更新。strict advantage逐位完全不变，failure residual在每组零均值且最大绝对值不超过0.2，
strict/failure符号和支配关系保持；全部loss/gradient有限，reference KL=0，最大clip fraction为
0.328%。

残差幅度门已通过：两个panel的secondary/primary gradient norm分别为11.99%和15.59%，说明将
质量残差放在组内标准化之后确实避免了P2中仅1.4%–1.7%的幅度消失。然而方向门仍明确失败：
binary-vs-candidate gradient cosine分别为0.992890和0.990219，均高于预注册0.98冗余线。这是比
P2更有信息的负结果：失败质量排序能够形成可测的梯度分量，但在这批轨迹上该分量仍大体沿着
binary strict梯度方向，尚不足以作为一个机制上独立的在线训练arm。

依据冻结规则，不调整0.2 scale、不重排group，也不提交Phase3在线A/B。下一研究轮若继续，应把
问题转向SFT-disjoint任务覆盖、失败状态对比或能改变动作级方向的监督信号，而不是继续放大同一
failure-quality残差。Job 2300只支持“幅度修复有效、方向区分仍不足”的机制结论，不支持任何
策略性能提升结论。

## 13. M6 Phase4 预注册：学生可探索性预扫描与条件教师补充

前三阶段已经排除了“仅换信用公式即可获得提升”的简单解释。Phase4把研究对象从优化器转向
数据支持：先确认冻结SFT策略在**未进入SFT语料、未进入P1诊断、未进入非训练角色**的新任务上，
能否生成足够多的成功/失败强对比；只有学生完全无法成功的部分才考虑更强教师。该阶段是策略
相关的数据诊断与合成，不执行optimizer step，也不能表述为RL训练。

学生预扫描冻结为两个互斥分区，每区64个task，选择seed分别为20260824与20260825。任务来自
split lock的`train`角色，按category×constraint-count比例分层，每个分区相对候选总体的最大
bucket份额偏差不超过5个百分点；两个分区、SFT 156个task、P1 64-task roster及所有非训练角色
之间的task identity交集必须为0。冻结SFT adapter在每个task上采样K=4，使用与后续评测一致的
18 model turns/15 environment steps；总计128 task、512条轨迹。两分区均保留mixed、all-success
和all-failure组，禁止使用mixed-only过滤制造虚假的高质量数据观感。

预扫描后按任务组形成三路索引：

1. `mixed`：只证明该任务对冻结SFT策略存在探索支持，作为未来在线训练的**任务roster候选**；
   每次在线更新仍必须由当时的current policy重新采样，预扫描轨迹不能在策略更新后继续冒充
   on-policy batch。
2. `all-success`：进入能力保持与成功轨迹成本分析候选集，不承担strict二元奖励下的新能力学习。
3. `all-failure`：进入条件教师补充候选集。教师轨迹必须strict success且通过环境replay验证，
   只能用于恢复SFT、偏好学习或离线诊断；不得进入学生on-policy GRPO，也不得伪造为学生采样。

每条学生轨迹同时记录strict outcome、失败类别、public-only buy-readiness序列与failure quality；
禁止读取target ASIN或隐藏答案，也不把official dense task score直接当奖励。对mixed组生成
strict-over-failure对比，对同组差异至少0.1的失败生成failure-quality对比。完整性门为128 task、
512轨迹、全部哈希/adapter/split/seed闭合且零训练更新。未来在线roster的探索支持门为：总mixed
至少64组、每分区至少24组、strict/failure对至少128、strict/partial至少64、failure-quality
对至少32、至少32组存在可辨失败质量差，并有至少3类失败各不少于4例。门失败时不得直接扩大
在线训练，必须先处理覆盖不足或标签不可辨识。

教师补充不是默认步骤。只有all-failure task至少16个，或某个至少4-task的分层bucket中
all-failure占比达到50%，才触发教师候选生成。即使触发，也先在固定小探针上证明教师相对学生
具有更高strict pass率，再生成最小必要样本；无法证明更强时不调用教师。这样把“学生自己训练
自己”的闭环拆成两类证据：学生rollout负责真实on-policy边界，外部教师只补学生探索不到的
状态，并由环境verifier而非教师自评决定是否接受。

## 14. M6 Phase4 学生预扫描结果与条件教师探针

学生预扫描 Jobs 2302_0、2302_1 及依赖审计 Job 2303 均以 `COMPLETED 0:0` 结束。两分区严格
使用冻结SFT adapter、互斥且SFT-disjoint的train任务、K4和18/15 horizon，不执行optimizer
step。最终审计覆盖128个任务、512条轨迹，全部task/trajectory/group/ledger/invocation哈希、
分区seed、adapter和split血缘闭合。学生产生220条strict-success轨迹；任务组分布为51 mixed、
29 all-success、48 all-failure，其中A/B分区mixed分别为26/25。失败轨迹以partial purchase为主：
249条partial purchase、29条zero-match purchase、14条horizon exhaustion。

该数据同时形成166个strict-over-failure pair、146个strict-over-partial pair、86个可辨
failure-quality pair，34个组存在至少0.1的failure-quality spread。除探索覆盖外的完整性和对比
检查均通过；但预注册的在线roster门要求至少64个mixed task，实际只有51个。因此
`future_online_mixed_task_roster_ready=false`。这不是作业失败，也不是通过降低门槛可以修复的
统计瑕疵，而是直接证据：冻结SFT在这批新任务上确有强对比，但当前128-task预算尚不足以冻结
计划中的64-task在线roster。预扫描轨迹继续只能作为诊断索引；任意未来策略更新仍需current
policy重新采样，不能复用这些轨迹冒充on-policy batch。

48个all-failure任务触发了预注册的条件教师分支。为避免用大模型无界地制造数据，先冻结一个
16-task小探针：从student all-failure候选中按category、constraint count和dominant failure
class比例分层，使用独立seed `20260826`；教师固定为无adapter的
`/data/share/model/Qwen3.5-9B`，每任务K4、18/15 horizon，共64条轨迹，不执行任何训练。选择9B
而非直接使用35B，是为了先以单GPU、有限成本检验“外部策略能否扩张学生支持集”这一假设。

教师输出只有同时满足以下条件才允许形成补充语料：全部strict轨迹环境重放成功；至少4个任务
出现replay-verified strict success；至少8条语义不同的verified strict轨迹；每任务最多保留2条。
录取样本仅允许用于recovery SFT、preference learning或offline diagnostic，明确禁止进入
on-policy GRPO。若9B未过门，本轮停止并报告“未证明教师更强”，不会把失败教师数据带入RL，
也不会自动升级到35B刷结果。教师探针通过仍只证明数据支持扩张，不等于Raw/SFT/RL性能已经
提升；后续是否训练必须另行构建互补语料、验证数据质量并重新冻结小规模门控实验。

## 15. M6 Phase4 教师尺度上界：9B负结果与单次35B-FP8探针

Qwen3.5-9B探针 Jobs 2305–2307 均以`COMPLETED 0:0`结束，64条轨迹仅在2/16个学生全失败
任务上产生3条strict success；三条均通过环境回放，但明显低于4任务、8条不同轨迹的冻结门，
因此没有录取教师补充数据。该结果否定“9B教师足以扩张学生支持集”，不是基础设施失败。

为快速判定整个外部教师方向是否值得保留，下一步只做一次同任务、同K4、同18/15 horizon、
同seed的单GPU上界探针，模型改为`Qwen3.6-35B-A3B-FP8`。它是35GB FP8权重，成本远低于无界
扩大task或直接开展训练；门槛仍为至少4个replay-success任务和8条语义不同成功轨迹，不因模型
变大而降低。若该上界仍失败，就停止教师合成路线，转向学生mixed任务的更广覆盖和动作级对比；
若通过，只用每任务最多2条verified轨迹做一次小规模恢复训练，随后必须由更新后的学生策略重新
采集on-policy RL数据。该显式上界实验不覆盖9B负结果，也不构成按结果持续升级模型。

## 16. M6 Phase4 学生扩展覆盖与最小在线训练决策

35B-FP8上界链 Jobs 2308–2310 均完成，但同16个学生all-failure任务上只得到1个任务、2条
replay-verified strict轨迹，比9B的2任务、3轨迹更差，也未达到4任务/8轨迹的固定门。两种更大
模型均未证明能经济地扩张当前WebShop支持集，因此教师路线到此停止；不继续升级模型，也不把
这些稀疏成功混入SFT或on-policy RL。

学生侧改用相同总量级的小扩展：Job 2311_0在新增64个SFT-disjoint train任务上以冻结SFT、
K4、18/15采样，`COMPLETED 0:0`，耗时6分49秒。它得到26个mixed任务、40个至少一次strict
成功任务、110条strict轨迹。与原两个64-task分区的26和25个mixed合并后，共有77个互斥的
SFT-disjoint mixed任务，超过64任务覆盖门。该结果只用于选择任务身份；预扫描轨迹不进入
optimizer，更新后的策略必须重新采样。

下一轮不再增加教师、奖励arm或数据审计。固定从77个mixed候选按category×constraint比例选择
40个任务，运行唯一的binary strict trajectory-GRPO小实验：每任务K4、每次聚合4个任务组、
共10次optimizer step、18/15 horizon、RL dropout=0、LR=3e-6、单epoch、SFT reference KL
hard stop=0.01。相较旧链路，唯一成组修改是已经分别通过因果探针的训练配置修复：K8×1改为
K4×4、6/6改为18/15、dropout 0.05改为0，以及将已证实欠更新的1e-6提高到安全探针范围内的
3e-6；奖励仍保持strict binary，避免把前述冗余失败质量公式重新带回实验。

这是一轮快速可证伪验证，不以审计工作量为目标。除任务不泄漏、当前adapter身份、strict结果和
相同评测配置外不新增制度。训练完成后只在全新100–200 task tuning-dev2上比较Raw、SFT与RL；
若RL未超过SFT至少1 pp，或partial/schema明显恶化，则停止该配方并直接根据失败轨迹提出下一项
单变量假设，不靠增加seed或算力掩盖负结果。

## 17. M6 Phase4 在线训练与 fresh tuning-dev2 结果

Phase4 在线训练 Job 2313 以 `COMPLETED 0:0` 结束，耗时22分41秒。它按冻结配置完成10次
optimizer step，覆盖40个互异且与SFT语料任务身份不重叠的训练任务，其中26个current-policy
K4组为mixed并产生有效binary strict梯度。10个learner report的loss、gradient和KL全部有限，
每一步的adapter semantic hash均发生变化；最大SFT-reference KL为0.001069，低于0.01 hard
stop，最大initial replay clip fraction为0.674%，低于10%门。因此本轮是**工程上有效的在线
训练**，不能把后续性能负结果解释为“没有真正更新参数”。

训练后在全新、互斥且未参与SFT/RL训练的128-task tuning-dev2上，以相同任务、seed、K4和18/15
horizon完成Raw Job 2314、SFT Job 2315与RL Job 2316，共每个identity 512条严格配对轨迹。结果为：

| identity | strict success | strict rate |
|---|---:|---:|
| Raw | 155/512 | 30.273% |
| SFT | 183/512 | 35.742% |
| RL step-10 | 179/512 | 34.961% |

Raw→SFT保持明确正增益`+5.469 pp`，但SFT→RL为`-0.781 pp`；按task配对bootstrap的95% CI为
`[-2.734,+0.977] pp`。逐轨迹翻转中RL-only为8、SFT-only为12，净值-4，也未满足正向链路门。
partial purchase率两者完全相同（均58.008%），zero-match purchase也同为13条，schema/action
failure均为0；差异主要表现为RL相对SFT少4条strict success、同时多4条horizon exhaustion
（23对19）。因此当前证据不支持“在线RL改善末端购买判断”，反而提示10步更新可能把少量原有
成功轨迹推向未完成终止。`Raw < SFT < RL`门明确失败，本配方不得通过增加seed或训练规模晋级。

下一项只改变**训练长度/检查点**，不重新训练也不改变数据、奖励、学习率、horizon或评测配置：
在同一tuning-dev2上补评已保存的step-5 adapter，并与现有SFT和step-10结果形成0/5/10学习曲线。
该评测只用于诊断“10步过训练/更新抵消”假设，不把看过step-10结果后选择的checkpoint直接称为
独立最终结论。若step-5仍不高于SFT至少1 pp且RL-only flips不为正，立即否定训练长度假设；若
step-5通过，则冻结该checkpoint并只在另一份新鲜dev3上做一次确认，确认通过后才可声称发现
Raw<SFT<RL链路。该方案只新增一个RL checkpoint评测identity，不增加训练作业、算法或seed。

## 18. Phase4 step-5 检查点反事实：排除训练长度解释

Job 2320 只评测已保存的step-5 adapter，没有执行训练；两个64-task分区均`COMPLETED 0:0`，
并复用2314/2315完全相同的Raw/SFT任务、seed、K4与18/15配置。step-5得到178/512 strict
success（34.766%），相对SFT 183/512（35.742%）为`-0.977 pp`，task配对bootstrap 95% CI
为`[-2.734,+0.781] pp`；RL-only/SFT-only flips为6/11，净值-5。它不仅没有超过SFT，还把
partial purchase从SFT的297条增加到304条（58.008%→59.375%，`+1.367 pp`），超过0.5 pp
安全门；schema/action failure仍为0。

因此“10步训练过长，较早检查点可能形成正链路”的单变量假设被否定。step-5比step-10略差
（-0.977 pp对-0.781 pp），二者都落后SFT，不能通过早停选择解决。失败形态提供了更具体的下一
方向：step-5减少4条horizon exhaustion，却增加7条partial purchase和2条zero-match purchase，
说明binary strict的整轨迹梯度倾向于更快结束任务，但没有提高最终商品/option grounding；继续
改变步数或扩大相同数据只会重排失败类别。

本训练长度方向到此停止。若继续下一轮，只检验一个新的信用假设：保持同一strict binary奖励、
任务、K4×4、18/15、dropout、LR和训练步数不变，把policy-gradient loss从“整条轨迹所有动作
token”改为“商品页后的末端决策动作token”（option选择与Buy Now）掩码。先在冻结同batch上做
零更新gradient-direction探针；若与全轨迹梯度仍近似同向则零训练成本停止，只有方向明显不同才
允许5-step小训练。该变化直接针对已观察到的partial-purchase迁移，不重新引入已证实冗余的
failure-quality reward，也不靠更多seed或算力继续同一配方。

## 19. Phase5 预注册：tail-2 动作信用掩码零更新探针

Phase5不增加算法，而是在同一个single-epoch strict-binary trajectory group-normalized policy
gradient中只改变policy-loss的token支持集。`full` arm保持现有整轨迹动作token权重；`tail2` arm
对每条轨迹只保留最后两个有效agent action turn，并在这两个turn内重新归一化policy权重。二者
共享完全相同的trajectory-level advantage、reference-SFT KL、模型参数、轨迹和dropout=0；KL仍
覆盖完整轨迹，因此唯一因果变量是strict policy gradient落到哪些动作token上。该mask不读取隐藏
答案，不更改trajectory reward，也不把失败质量公式重新带入训练。

探针使用已冻结的step-00与step-01两批K4×4 collection，各自绑定采样时的input adapter；每个
panel在同一模型实例上先后计算`full`和`tail2`梯度，但optimizer steps恒为0。记录两梯度的参数
空间cosine、norm ratio、有限性以及tail active-token覆盖。任一tail梯度为零或非有限立即停止；
若两个panel的cosine都不低于0.98，判定该mask与整轨迹信用仍冗余，不消耗在线训练预算。只有
两个panel均有限非零、且至少一个panel cosine低于0.98，才把它视为有方向区分度，并另行冻结
一个5-step小训练；探针通过本身不能表述为性能提升。

## 20. Phase5 tail-2 零更新探针结果与最小在线验证

Job 2322 只计算梯度、没有建立optimizer或执行参数更新，最终报告明确记录
`optimizer_steps=0`与`training_performed=false`。两个冻结panel的full与tail-2梯度均有限且非零；
tail-2分别覆盖273/890（30.67%）和333/1191（27.96%）个动作token。参数梯度结果为：

| panel | full norm | tail-2 norm | tail/full norm | cosine |
|---:|---:|---:|---:|---:|
| step-00 / SFT input | 0.7441 | 1.0263 | 1.3793 | 0.7099 |
| step-01 / step-00 input | 0.6332 | 0.9955 | 1.5723 | 0.5862 |

两个cosine都远低于0.98冗余门，说明末两次动作信用不是对整轨迹梯度的微小缩放，而是明显改变
了参数更新方向。该结果只回答“是否值得训练”，不回答“性能是否提升”。因此下一步冻结为一个
5-step在线验证：从同一SFT起点、相同40-task roster、seed、K4×4、18/15、dropout=0、
LR=3e-6、strict binary reward与SFT-reference KL出发，唯一改变为policy loss只覆盖每条轨迹最后
两个有效agent action turn；KL继续覆盖整条轨迹。训练完成后复用同一fresh tuning-dev2做开发
诊断，并与SFT及历史full-credit step-5直接比较。只有tail-2超过SFT至少1 pp、RL-only flips净正、
partial/schema不恶化，才冻结候选并另建全新dev3确认；否则停止该信用方向，不扩大seed或步数。

首个在线提交Job 2323在任何采样或optimizer step前由既有兼容门拒绝：40-task roster由
`c0cfa32`生成，而consumer已更新为`f3db549`，作业没有显式传入已知producer SHA。该失败不含
算法证据，也没有adapter；修复仅把同一roster producer身份显式传给collector，不改变任务、
顺序、seed或训练参数，随后允许在同一输出根恢复。

## 21. Phase5 tail-2 五步训练与配对评测结果

同根恢复Job 2324以`COMPLETED 0:0`完成5次optimizer step，覆盖前20个冻结任务，其中16个
current-policy K4组为mixed。每步policy window均为tail-2，所有loss/gradient有限且adapter语义
哈希真实变化；最大SFT-reference KL为0.000398，最大initial replay clip为0.553%，均远低于硬门。
因此这是一次有效训练，不存在“参数未更新”或“KL压住更新”的工程解释。

评测Jobs 2325/2326在完全相同的128-task tuning-dev2、K4、18/15与seed上完成512条轨迹。
自哈希配对统计为：

| identity | strict success | strict rate | partial purchase |
|---|---:|---:|---:|
| Raw | 155/512 | 30.273% | 315/512 |
| SFT | 183/512 | 35.742% | 297/512 |
| full-credit step-5 | 178/512 | 34.766% | 304/512 |
| tail-2 step-5 | 181/512 | 35.352% | 300/512 |

tail-2相对full-credit step-5恢复3条strict success并减少4条partial purchase，说明末端信用聚焦有
方向性收益；但相对SFT仍为`-0.391 pp`，task bootstrap 95% CI为
`[-2.148,+1.367] pp`，RL-only/SFT-only flips为9/11，净值-2。partial purchase仍比SFT多3条，
即`+0.586 pp`，也略过0.5 pp安全门。故`Raw<SFT<RL`仍失败，tail-2方向不得靠增加步数或seed晋级。

失败形态进一步缩小了下一假设：tail-2同时把trajectory advantage分配给倒数第二步和最终
`Buy Now`，后者可能继续强化“尽快购买”而不是商品/option选择。下一轮仍先零训练验证，只把
policy credit从tail-2改为`preterminal-1`：若轨迹以public action `click[Buy Now]`结束，则只给
购买前一动作；非购买轨迹给最后动作。reward、group advantage、KL、任务和模型均不变。只有它与
tail-2在两个冻结panel上梯度有限非零且cosine明显低于0.98，才允许新的5-step训练。

## 22. Phase6 preterminal-1 零更新探针结果

Job 2327以`COMPLETED 0:0`结束，报告自哈希闭合并明确`optimizer_steps=0`、
`training_performed=false`。preterminal-1分别只覆盖153/890（17.19%）与173/1191（14.53%）
动作token，两个panel梯度均有限非零。相对tail-2的参数梯度cosine为0.8883与0.5374，norm ratio
为1.5908与0.9804；两者均低于0.98冗余门，说明排除最终`Buy Now`确实产生新的更新方向。

该探针仍不是性能证据。下一步只允许一个5-step在线验证，完全复用Phase5的SFT起点、前20任务、
seed、K4×4、18/15、dropout=0、LR=3e-6、strict binary reward和整轨迹SFT KL，唯一变量为
`policy_credit_window=preterminal1`。评测仍先使用同一tuning-dev2作开发诊断；若未超过SFT至少
1 pp或partial/schema恶化，就停止该信用方向，不增加训练步数或seed。

## 23. Phase6 preterminal-1 五步训练与配对评测结果

Job 2328以`COMPLETED 0:0`完成5次optimizer step，覆盖20个冻结任务，其中15个K4组为mixed。
每一步都使用`preterminal1`，所有loss/gradient有限且adapter语义哈希变化；最大SFT-reference KL为
0.000540，最大initial replay clip为0.439%，因此训练工程链有效。配对评测的两个64-task分区均
完整，统计报告SHA为`4e83188d...75a63d`：

| identity | strict success | strict rate | partial purchase |
|---|---:|---:|---:|
| Raw | 155/512 | 30.273% | 315/512 |
| SFT | 183/512 | 35.742% | 297/512 |
| full-credit step-5 | 178/512 | 34.766% | 304/512 |
| tail-2 step-5 | 181/512 | 35.352% | 300/512 |
| preterminal-1 step-5 | 182/512 | 35.547% | 299/512 |

preterminal-1相对full-credit恢复4条strict success，相对tail-2再恢复1条；partial purchase也分别
减少5条与1条，说明信用逐步聚焦到购买前决策具有一致方向性。但它相对SFT仍为`-0.195 pp`，
task bootstrap 95% CI为`[-1.562,+1.172] pp`，RL-only/SFT-only flips为6/7，仍未达到
`RL-SFT>=+1 pp`或flips净正门。partial purchase比SFT多2条（`+0.391 pp`），未越过0.5 pp
安全门；schema/action均为0。故该结果不能宣称RL提升，也不允许继续搜索动作窗口或增加步数。

三种动作信用从178→181→182呈小幅单调恢复，却始终未超过SFT，下一项可证伪假设转向数据而非
继续微调mask：当前mixed K4组是否真正包含“同任务、相同候选商品、只在option/购买时机上形成
成功—partial失败”的强对比。先对现有40-task训练collection及77-task prescan做零GPU离线诊断，
报告可配对终局对比的任务数、同商品不同option覆盖和每类失败占比。若高质量终局对比不足，就只
重建一个20-task强对比roster；若覆盖已足，说明数据选择假设不成立，停止该方向。诊断前不提交
新的训练，也不再使用已多次查看的tuning-dev2选择超参数。

## 24. Phase7 强终局对比数据诊断预注册

本阶段不训练，只检验Phase4在线数据是否真正提供了末端商品/option决策所需的组内反事实。输入
限定为40个既有在线K4任务和三份SFT-disjoint prescan（共77个mixed K4任务）；不得读取
promotion或holdout。诊断只从模型在`Buy Now`前看到的public observation读取current ASIN与
selected options，不读取target ASIN、隐藏答案或terminal server内部信息。

同一K4任务中，每条strict purchase与partial purchase构成候选对；current ASIN相同才算
`same-item strict-partial`强对比，且selected-options规范化后不同才算`option contrast`。报告同时
保留所有失败类型、类别/约束bucket、缺失public购买状态数与输入自哈希。门槛在运行前冻结：现有
online数据只有在至少50%的mixed任务含same-item对比、且其中至少50%含option contrast时才算
覆盖充足；否则确认数据质量缺口。只有prescan union至少有20个same-item任务且其中至少12个
option-contrast任务时，才允许重建20-task强对比roster；不足时必须先做针对性prescan，不能凑数
或直接训练。该诊断`optimizer_steps=0`，结果只决定数据路径，不构成性能证据。

首个CPU运行产生了实现无效报告：158/160个online购买轨迹、743/768个prescan轨迹被记为缺失
购买状态，所有对比均为0。原因不是数据缺少对比，而是诊断错误假设policy observation保留
`env_state.asin/selected_options`；实际漏泄防护会在写入轨迹前删除这些内部字段。该v1报告永久
保留但不得解释为数据结论。v2只使用真正public的信息：从`Buy Now`前的策略动作历史取最后一个
公开ASIN click，并从policy-visible text的`selected:`标记解析option。冻结门槛不变，修复后使用
新输出根重跑；不得为适配结果调整阈值。

## 25. Phase7 v2 强终局对比诊断结果

修正版CPU诊断报告SHA为`26257d77...3cebf59`，13个输入绑定、自哈希和public-only合同均
通过；`optimizer_steps=0`且缺失public购买状态数已降为0，证明v2解析有效。结果如下：

| source | mixed tasks | strict-partial tasks / pairs | same-item tasks / pairs | option tasks / pairs |
|---|---:|---:|---:|---:|
| 40-task online | 26 | 25 / 79 | 3 / 9 | 2 / 6 |
| 192-task prescan | 77 | 72 / 227 | 11 / 26 | 10 / 24 |

online中只有3/26（11.54%）mixed任务提供same-item strict-partial对比，远低于预注册50%；这3个
任务中2个含option contrast（66.67%），说明一旦命中同商品，对比通常有用，但覆盖严重不足。
因此数据质量缺口成立。现有prescan union也只有11个same-item任务、10个option任务，未达到
重建20-task roster所需的20/12门，不能从现有数据凑出下一训练集。

下一步先做一轮针对性prescan，不训练：从72个已有strict+partial但多数商品不一致的任务中冻结
32个候选，用同一SFT policy与独立seed再采一批K4，并与现有prescan按task合并诊断。目标是把
可用same-item任务增至至少20、option任务增至至少12；达不到就停止该数据合成方式，而不是增加
RL预算。即使达到门槛，后续RL仍须用当时current policy重新采样，prescan轨迹不能直接当作
on-policy更新批次。

## 26. Phase7 定向prescan设计冻结

下一轮仍不是训练，只改变“用于发现强终局对比的任务选择”。候选严格来自v2报告绑定的三批
prescan中72个strict+partial任务，并排除已经拥有same-item对比的11个任务。排序首先优先目标中
存在明确option约束的任务，其次按既有K4中strict×partial配对数降序，最后用冻结seed
`20260832`的哈希打破平局；取前32个。这样既不使用target ASIN/隐藏答案，也不通过已burn评测集
挑任务。

采样继续使用同一个冻结SFT adapter，K4、18/15、独立seed `20260832`，只运行一次GPU推理，
`optimizer_steps=0`。输出必须绑定split、SFT-disjoint train role、v2诊断与三个collection report；
promotion/holdout不得读取。完成后把新collection加入同一public-only诊断。只有合并后的
same-item任务不少于20且option-contrast任务不少于12，才允许构建20-task在线RL roster；否则
停止这种自采样数据路径。即便达到门槛，prescan轨迹也不能直接用于策略更新，正式RL必须以当时
current policy重新采样。

## 27. Phase7 定向prescan结果与停止决策

定向推理Job 2331以`COMPLETED 0:0`结束，32个任务、128条K4轨迹全部完整，25个group保持
mixed。collection与32个group自哈希、roster绑定、SFT adapter身份、seed `20260832`、18/15和
`training_updates_allowed=false`均通过；报告SHA为
`ff0401e5...dd409574`。这是一轮冻结SFT推理采样，没有optimizer step。

把新collection作为第四个prescan输入后，public-only诊断报告SHA为
`93438096...c4cc295b`。旧三批加新批共224个K4 group；由于定向任务来自旧任务集合，unique mixed
task仍为77。same-item强对比任务从11增加到15，option-contrast任务从10增加到13。option门
`>=12`已通过，但same-item仍低于预注册`>=20`，因此20-task强对比roster不可构建，不能为凑数
继续增加seed、K或GPU。

这项负结果说明“优先有option约束、既有strict×partial对多的任务再独立采样”提高了强对比命中率，
但同一个完整SFT策略从任务起点重新采样，仍主要在不同商品之间分叉，无法稳定生成同商品终局反事实。
该自采样数据方向停止。下一项若继续验证数据质量，只允许先做零GPU的共享公开前缀可行性诊断：检查
既有轨迹能否在同一公开商品页前缀上重放并分支option/购买决策。只有覆盖足够时才考虑prefix-reset
suffix rollouts；否则不实现新collector，也不提交训练。

## 28. Phase8 共享公开前缀可行性预注册

Phase7表明从任务起点独立重采样仍主要在不同商品之间分叉。Phase8不训练、不新增采样，先检查
能否把现有strict-success轨迹的公开成功action链重放到同一个商品页，再从该共同状态采样suffix。
前缀严格结束在`Buy Now`前最后一次公开ASIN click；只接受从起点到该click的每个
`action_result.success=true`。商品页状态从下一turn的policy-visible observation读取，只使用
`page_type`与`available_actions`，不读取post-action内部state、target ASIN、隐藏答案或完整正文。

目标必须含明确option约束；product页公开click actions排除搜索、Description/Features/Reviews、
Back to Search、Buy Now、分页导航和ASIN后，至少剩两个不同值才算branchable。四份prescan按task
去重并保留最短可重放前缀，只记录action sequence与option集合的hash。门槛在结果前冻结：至少20个
replayable任务且其中至少12个branchable，才允许设计共享前缀K4 suffix rollout；否则停止
prefix-reset方向。该诊断全程CPU，`optimizer_steps=0`，不会产生adapter或策略更新。

## 29. Phase8 共享公开前缀可行性结果

CPU报告SHA为`b5e07962...bd2ef34`，四份prescan输入、自哈希和public-only合同均通过。共有120个
unique strict-success task；其中85个存在“从起点到最后一次公开ASIN click的动作全部成功”的
可重放前缀，且这85个任务的下一policy-visible商品页都至少提供两个去除导航项后的公开option
actions。结果同时超过预注册的replayable `>=20`与branchable `>=12`门，decision为
`design_shared_prefix_k4_suffix_rollout`。诊断没有读取post-action内部state或隐藏答案，也没有采样、
optimizer step或adapter更新。

这说明Phase7失败不是环境缺少可控商品页状态，而是每条完整轨迹从任务起点独立搜索，导致K4主要
在不同商品上分叉。下一轮只允许一个8-task共享前缀suffix推理smoke：每个任务从同一条公开strict
前缀重放到同一商品页，再由冻结SFT独立采样K4 suffix；前缀token不参与任何未来policy loss。
在看到结果前冻结继续门：至少6/8任务精确重放前缀，至少4个任务形成strict mixed且same-item
strict-partial对比，至少3个任务形成option contrast。smoke仍不训练；未通过就停止，不能直接扩大到
20任务或RL更新。通过后才设计“共享前缀、只优化suffix token”的最小5-step trajectory-GRPO。

## 30. Phase9 共享前缀K4 suffix smoke实现冻结

本阶段只实现8-task推理smoke，不执行RL更新。确定性builder从Phase8的85个branchable task中以
seed `20260833`按类别/约束bucket分层选8个，并为每个task绑定一条strict source trajectory：
source collection/group自哈希、rollout index、截止最后公开ASIN click的prefix turn数、action序列
hash和token/logprob证据hash。builder重新核对四份prescan、SFT-disjoint train role、同一SFT
policy lineage和Phase8通过报告；不读取target ASIN、隐藏答案、post-action内部state、promotion或
holdout。

collector复用既有prefix replay backend。每个task的四条K4 rollout严格重放同一source prefix，
逐turn比较rendered prompt、prompt/generated token、action以及pre/post public-state hash；prefix完成
后才用rollout-specific seed从冻结SFT live-generate suffix。产物显式记录`shared_prefix_turns`、
`prefix_policy_loss_eligible=false`、prefix/suffix token数和`suffix_policy_turn_start`；smoke本身
`training_updates_allowed=false`且`optimizer_steps=0`。因此prefix token仍计入推理成本，但被排除在
任何未来policy loss合同之外。运行资源冻结为1 GPU、4 CPU、24 GiB、24 h上限，K4、18/15，
预计只需数分钟。

继续门保持Phase8结果前已冻结的三项：至少6/8任务精确重放；至少4个任务同时形成strict mixed和
same-item strict-partial；至少3个任务形成option contrast。未通过即停止，不扩到20任务、不训练；
通过后也只允许设计一个5-step、仅优化suffix token的trajectory-GRPO小试。

## 31. Phase9 共享前缀suffix smoke结果与停止决策

推理Job 2333以`COMPLETED 0:0`结束，用时1分41秒。8个任务、32条K4轨迹全部通过共享前缀
逐turn精确重放；collection、manifest、roster和8个group自哈希闭合，SFT lineage一致，
`training_updates_allowed=false`。每条轨迹都记录了非零`shared_prefix_turns`、
`prefix_policy_loss_eligible=false`和一致的prefix/suffix token边界。工程门完整通过，且没有
optimizer step或adapter更新。冻结门报告SHA为`f6cd579f...e588959`。

数据结果没有通过继续门。32条suffix中有29条strict success、2条partial purchase和1条horizon
failure；8个任务只有3个mixed。虽然exact replay为8/8，高于`>=6`门，但只有1个任务形成
same-item strict-partial（要求`>=4`），也只有1个任务形成option contrast（要求`>=3`）。换言之，
共享到同一商品页解决了“分叉到不同商品”的问题，却把suffix难度降得过低：冻结SFT在7/8任务上
几乎总能成功，K4缺少成功—失败方差，无法提供group-normalized policy gradient所需的对比信号。

因此按预注册停止shared-prefix方向：不扩20任务、不增加K/seed、不执行suffix-only RL训练。该负结果
把瓶颈进一步定位为“需要同商品页且处于真实决策边界的前缀”，而非任意strict成功前缀。若继续快速
探索，下一项单变量应先做零GPU前缀难度排序：利用既有prescan中同task的strict/partial行为，筛选
更早或更不确定的公开商品页状态；只有能预期产生足够mixed的8-task smoke才值得再次采样。

## 32. Phase10 exposure union 与六角色fresh split readiness结果

本阶段只验证新教师纠错计划的数据隔离是否可执行，不采样、不训练，也不打开promotion/holdout。
实现先冻结20类历史来源，包含完整Raw mini-train 256、SFT train/dev、mini/formal/tuning-dev2、
Phase1–Phase3诊断、Phase4学生/教师/online数据，以及Phase5–Phase9所有用于训练、评测或假设选择的
任务。每类来源均绑定明确路径、预期唯一任务数和文件内容hash；来源缺失、数量变化或hash漂移都会
fail closed。远端49项聚焦/相邻协议测试全部通过。

真实历史产物生成的`phase10_exposure_union.json`共含1,340个unique task，source spec SHA为
`c443128e...e4f6eec`，union SHA为`c270d7e2...b9be478`。旧train role扣除task identity与
normalized instruction exposure后仍有6,545个候选。随后按category × constraint-count ×
instruction-length分层并冻结六个互斥角色：teacher qualification 32、correction train 64、
monitor A 64、monitor B 64、RL train 40、dev3 500；split SHA为
`f05ded9e...2ffeca7`。

任务、goal-index与normalized instruction在六角色之间的交集均为0；相对历史exposure union的三类
交集也均为0。该结果只通过了Phase10第一道readiness门，`optimizer_steps=0`，不构成教师能力或性能
结论。下一步严格停在8个已暴露历史任务的端到端工程smoke：验证公开失败点、student/teacher同状态
suffix、fresh-session strict replay、query provenance与token mask。smoke和matched single-update
任一失败，都不得启用已冻结的32-task teacher qualification。

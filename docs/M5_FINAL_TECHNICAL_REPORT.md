# MiniWebWork-RL M5：WebShop 长程信用分配训练技术报告

> 报告状态：正式结果已冻结
>
> 研究标识：`m5_webshop_credit_assignment_v1`
>
> 报告日期：2026-08-12
>
> 最终分析代码提交：`a57a745937fb7e80e96570c6294335a0b9e01846`
>
> 基础模型：Qwen3.5-4B
>
> 正式产物根目录：
> `outputs/m5_webshop_credit_assignment_v1/formal`

## 摘要

本轮工作研究一个刻意收缩的问题：在同一个 verified SFT 起点、同一批 WebShop
训练任务、相同 K=4 在线采样、优化器和约 500,000 generated-action-token 预算下，
基于公开状态锚点的逐步信用分配，是否优于把终局组相对优势广播到整条轨迹的
multi-turn GRPO。

实验完成了 2 种在线方法 × 3 个训练 seed，并在所有策略冻结后一次性评测 8 个推理
身份：Raw base、SFT、3 个 GRPO 和 3 个 Anchor-GiGPO。冻结测试包含 500 个任务、
每任务 K=4，共 16,000 条真实环境轨迹。

核心结论如下：

1. **Raw base 仍是严格成功率最好的模型。** Raw 达到 33.50%，SFT 只有 0.65%，
   GRPO 为 9.42%，Anchor-GiGPO 为 9.27%。本轮不能声称后训练超过基础模型。
2. **SFT 发生了严重负迁移。** 它在 teacher-forced 指标上表现正常，却在闭环环境中
   退化为长标题搜索和连续翻页。主要根因不是数据量不足，而是专家搜索动作依赖
   policy prompt 不可见的精确商品标题，形成了 verified 但不可从公开观测稳定预测的
   privileged labels；同时，实际 SFT 准入代码没有把闭环开发评测和相对 Raw 的退化门禁
   设为硬条件。
3. **在线 RL 明显修复了 SFT 的闭环行为，但只恢复了部分基础能力。** 相对 SFT，
   GRPO 和 Anchor-GiGPO 分别提升 8.77 和 8.62 个严格成功率百分点；按 Raw–SFT 差距
   计算，只恢复约 26.7% 和 26.2%。在 dense score 上恢复约 55.4% 和 58.0%，说明在线
   奖励更擅长恢复状态推进、商品配置和完成购买，而不是精确选对商品。
4. **Anchor-GiGPO 没有在主指标上击败 GRPO。** Anchor−GRPO 的严格成功率差为
   −0.15 个百分点，crossed-bootstrap 95% CI 为 [−1.60, +1.20] 个百分点，3 对训练
   seed 的精确双侧检验 `p=1.0`。该结论是不显著差异，不是两者严格等价。
5. **逐步信用分配仍体现出可解释的过程收益。** Anchor-GiGPO 的 dense score 比 GRPO
   高 0.0160，搜索耗尽、格式失败和 schema-invalid 更少，评测 token 成本平均低约
   15.0%；但 partial-match purchase 更多，说明它更快学会“推进并购买”，没有同步解决
   “买对哪一个商品”的 grounding 问题。

因此，本项目最有价值的成果不是一个正向榜单数字，而是完整识别并验证了三层问题：
离线监督数据的可学习性、在线长程信用分配的有效边界，以及如何用冻结测试、配对统计、
失败轨迹和可审计工件避免把过程改善误报为最终决策提升。

## 1. 研究问题与范围

### 1.1 最终研究问题

本轮只回答：

> 在控制模型、SFT 起点、训练任务、采样、奖励、优化器和 generated-token 成本后，
> public-state step credit 能否比 terminal-advantage broadcast 带来更高的任务成功率，
> 或至少带来更好的过程质量和样本效率？

这一定义有意排除了 PPO、RLOO、RSFT、GSPO、critic 等额外方法。项目目标不是构造
“算法动物园”，而是深入展示多轮 Agent RL 中的轨迹采样、信用分配、on-policy parity、
恢复训练、数据隔离和统计推断。

### 1.2 本轮交付

- 公开 WebShop 环境的固定版本、数据锁和泄漏修复；
- 4,000 train / 400 dev 的 verified SFT 语料与一次正式 SFT；
- K=4 multi-turn GRPO 与 Anchor-GiGPO 两种 learner；
- 2 方法 × 3 seed 的正式在线训练；
- Raw、SFT 和 6 个 RL adapter 的冻结测试；
- 置信区间、配对差异、显著性检验和失败轨迹分类；
- 从数据问题、优化目标、信用分配和推理行为四层解释负结果。

## 2. 环境、数据与隔离

### 2.1 上游来源

| 组件 | 冻结版本 | 用途 |
|---|---|---|
| Princeton WebShop | `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` | benchmark 与任务语义 |
| Agent-R1 | `b124aa46534cbf2fb8bc8af11405774984c42ac7` | FastAPI/SQLite/Lucene WebShop 运行环境 |
| Agent-R1-data | `1e624211d47dc1d66758a056fa8ee1017d72de6f` | 1,181,430 个商品、12,087 个目标和搜索索引 |

适配层屏蔽 reset 返回的目标 ASIN，并拒绝当前公开 action list 中不存在的商品点击。
policy 只能看到 instruction、公开 observation、page type、历史公开动作结果和最多 256
个公开动作；oracle metadata 和 verifier target 不进入 prompt。

### 2.2 冻结切分

| 角色 | 原始目标索引 | 原始数量 | 精确 instruction 去重后可用 | 允许更新模型 |
|---|---:|---:|---:|---:|
| Test | `[0, 500)` | 500 | 500 | 否 |
| Dev | `[500, 1000)` | 500 | 499 | 否 |
| Train | `[1000, 12087)` | 11,087 | 10,885 | 是 |

上游 `goals.json` 存在后出现的重复 instruction。处理规则是保留更小的
`goal_index`，从 dev/train 排除重复项，保证三个 eligible roster 的精确 instruction
交集为零。测试集在 8 个推理身份和全部评测设置冻结前不用于训练、preflight、调参或
checkpoint 选择。

### 2.3 SFT 语料

SFT 专家使用 oracle metadata 构造可执行动作，但每一步必须出现在当时公开 action list
中，完整轨迹还必须由官方 verifier 验证 `reward=1`：

```text
search[sanitize(goal.name)[:200]]
→ 在公开 top-50 搜索结果中点击 goal.asin
→ 选择公开可见的 goal options
→ click[Buy Now]
→ 仅保留 task_score=1 且不超过 15 步的轨迹
```

正式语料统计：

| 项目 | 数值 |
|---|---:|
| Verified train tasks | 4,000 |
| Verified dev tasks | 400 |
| Train action-turn samples | 16,485 |
| Dev action-turn samples | 1,656 |
| Total completion turns | 18,141 |
| Zero-label samples | 0 |
| Test goals used | 0 |

Train 动作标签由 4,000 次 search、4,000 次 ASIN click、4,221 次 option/control、
4,000 次 Buy Now 和 264 次 Next 构成。数据在“环境可执行、最终奖励正确”层面有效，
但这不等于其每个教师动作都能从 policy 可见信息中学习，相关问题在第 7 节详述。

## 3. 模型与训练方法

### 3.1 Shared SFT

SFT 使用 Qwen3.5-4B，关闭 thinking，completion-only action loss。训练配置如下：

| 参数 | 值 |
|---|---:|
| Epoch | 1 |
| Completion-label token exposure | 339,925 |
| Optimizer updates | 1,031 |
| Micro batch / effective batch | 8 / 16 |
| Maximum sequence length | 8,192 |
| Observed maximum forward length | 4,576 |
| Truncated samples | 0 |
| Learning rate | `2e-4` |
| LoRA | `r=16`, `alpha=32` |
| LoRA targets | q/k/v/o/gate/up/down projections |

训练结果为 train NLL 1.16243、dev NLL 1.09514、teacher-forced exact action
68.66%、schema validity 92.21%。最终 adapter SHA-256 为：

```text
c97c9265fe0043a8cda59908713429eef9e13d9619261a144c13cfa5fb4d7334
```

这些 teacher-forced 指标只能说明给定专家状态时能够模仿标签，不能证明模型在自己的
状态分布上可以闭环完成任务。

### 3.2 Multi-turn GRPO

每个任务采样 K=4 条轨迹，使用官方 terminal dense task score `R_i∈[0,1]` 在组内做
population standardization，得到 trajectory-level macro advantage。该优势广播到轨迹
中的每个实际 action turn：

```text
A_turn(i,t) = A_macro(i)
```

训练只 replay 真实逐 turn prompt 和 action token，不把多轮环境交互伪装成单次
completion。

### 3.3 Anchor-GiGPO

Anchor-GiGPO 保留同一个 macro advantage，并按动作前的公开 MDP state 聚合不同轨迹
的 first visit。对轨迹 `i` 的 turn `t`：

```text
G(i,t) = 0.95^(T_i-t-1) * R_i
A_turn(i,t) = A_macro(i) + 1.0 * A_micro(public_state(i,t))
```

同一个公开 state anchor 至少被两条不同轨迹访问且 return 有方差时，才产生标准化的
`A_micro`；否则精确退化为 GRPO 的 macro credit。first-visit 去重防止循环点击重复增加
权重。anchor 只包含任务内公开状态，不包含 prompt hash、episode ID、step index、目标
ASIN 或 verifier 信息。

### 3.4 受控变量

两个在线方法共享：

- 同一个 SFT adapter、相同 seed 下的任务顺序与采样 seed；
- K=4、temperature=1、top-p=1、top-k=0；
- 最多 15 个环境 step、18 个模型 turn、每 turn 最多 128 新 token；
- 每 run 约 500,000 generated action tokens；
- 2 policy epochs、trajectory minibatch 4、AdamW、LR `5e-6`；
- clip epsilon 0.2、gradient clip 1.0、behavior-policy staleness 0；
- 相同官方 terminal reward、环境、prompt 和 on-policy parity 门禁。

因此，预期的唯一方法差异是 turn-level credit estimator。

## 4. 工程验证与正式训练

### 4.1 最小在线 preflight

Job 2162 在正式训练前完成 32 tasks × K=4 的真实 GPU 验证：

| 指标 | 结果 |
|---|---:|
| Infrastructure-valid trajectories | 128/128 |
| Mixed-reward K4 groups | 31.25% |
| Non-zero score trajectories | 8.594% |
| Mean official task score | 0.03984 |
| Binary success | 0.781% |
| Informative micro-credit turns | 168/1,994 = 8.425% |
| Shared non-initial-state groups | 29/32 = 90.625% |

两种 learner 均完成 10 次真实更新，loss/gradient 有限，adapter 参数确实改变。
该结果只证明 RL 管线、K4 混合信号和信用分配存在，不是 held-out 效果。

### 4.2 一次有价值的失败修复

首轮正式分配 Jobs 2143–2148 在第一次 optimizer update 前被 replay P99 门禁拒绝，
因此没有被纳入正式训练结果。后续诊断没有直接放宽全部门槛，而是先把 parity replay
改成与 optimizer 相同的 group 隔离和 forward-token 排序，再以独立的 mean、P95、
P99、P99.9、clip fraction 和 importance ratio 共同约束。

另一次 preflight 暴露浏览器 Agent 的 generation 是 GPU token burst 与 CPU/HTTP
环境步骤交替的负载。最终资源门禁采用 generation mean≥40%、P95≥60%、learner
median≥80%，避免用纯语言模型持续生成的利用率假设误判真实 Agent 训练。

### 4.3 六个正式在线 run

Jobs 2165–2170 均为 `COMPLETED/0:0`。每个 run 的 billed token 都位于冻结区间内，
并通过 adapter、optimizer、ledger、K4 group、telemetry、parity 和 self-hash 审计。

| 方法 | Seed | Billed action tokens | K4 groups | Iterations | Optimizer updates | Valid attempt | Train mean score | Binary success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 20260801 | 491,344 | 666 | 21 | 960 | 99.630% | 0.26857 | 7.995% |
| GRPO | 20260802 | 491,075 | 673 | 22 | 1,062 | 99.816% | 0.29428 | 7.021% |
| GRPO | 20260803 | 491,579 | 671 | 21 | 988 | 99.489% | 0.28031 | 8.383% |
| Anchor | 20260801 | 492,018 | 726 | 23 | 1,180 | 99.863% | 0.27225 | 6.887% |
| Anchor | 20260802 | 491,217 | 754 | 24 | 1,322 | 99.705% | 0.30860 | 7.626% |
| Anchor | 20260803 | 490,996 | 744 | 24 | 1,246 | 99.767% | 0.29694 | 8.905% |

训练期间 GRPO 的 mixed-reward group 比例为 72.1%–78.9%，Anchor 为 68.3%–75.9%；
两个方法都有充足的组内奖励方差。Anchor 的 informative micro-credit turn 比例为
32.9%–38.0%，shared non-initial-state group 比例为 89.1%–91.6%，说明其 step credit
不是只作用在第一步的名义特性。

### 4.4 最终 adapter 身份

| 身份 | 最终 iteration | Adapter SHA-256 |
|---|---:|---|
| GRPO seed 20260801 | i0020 | `813448245d7404da99bfeece36f943f7ad2305052b57a9ed7e0c6e80658efd38` |
| GRPO seed 20260802 | i0021 | `b2be0d3c66e890bad5c8c9b2d6a4e5ec481ef5e1c690ceb62910ef347058a1df` |
| GRPO seed 20260803 | i0020 | `25f00eb393659a6c9c2294be097064c7d3e2c879e48d5166cc84c28dc01ba3c1` |
| Anchor seed 20260801 | i0022 | `6945c784939b9245e0894d9173303d4af54c65eca1230c9ccd4bd3a9449fa0cd` |
| Anchor seed 20260802 | i0023 | `0f5f0e2af4e94ba5bfbc0f0e529d97c8c2d91d9d0542b8b413e006fb3ec4b4ec` |
| Anchor seed 20260803 | i0023 | `e329ca3edd6be7913f052256bce3d8a38852948bc2594d611615413c112ca25f` |

### 4.5 资源与计算成本

| 阶段 | 单作业资源 | 逻辑工作量 | 实际计费/生成规模 |
|---|---|---:|---:|
| Shared WebShop service | 0 GPU / 24 CPU / 96 GiB | 1 个共享服务 | 16 workers |
| Online training | 1 GPU / 8 CPU / 32 GiB | 6 runs | 2,948,229 action tokens |
| Frozen evaluation | 1 GPU / 6 CPU / 24 GiB | 8 identities | 16,000 trajectories / 2,482,380 tokens |

六个训练 run 可在资源允许时并行，服务资源不在每个 run 中重复申请。每个 Slurm
allocation 上限 24 小时；只有 Slurm 的超时预警允许 same-root successor。generated-turn
ledger 每 turn fsync，K4 组和 learner checkpoint 原子提交，因此恢复不会重复使用不完整组，
也不会把逻辑实验条件扩展成新 job。

训练和评测的 token 不能直接互换成相同 FLOPs：online 阶段还包含逐 turn HF replay、
两轮 policy update 和 optimizer 状态，评测只有 rollout。这里报告 token 是为了形成透明、
可比较的 Agent 交互成本，不把它伪装成完整算力账单。

## 5. 冻结评测与统计方法

### 5.1 评测矩阵

正式测试只在全部模型冻结后打开一次：

```text
8 identities × 500 test tasks × K=4 = 16,000 trajectories
```

主指标是 `task_score >= 0.999` 的严格成功率；dense task score 是次指标。每个 identity
必须精确产生 500 个任务和 2,000 条 infrastructure-valid 轨迹，评测阶段禁止 optimizer
和任何模型选择。

### 5.2 推断口径

- Raw/SFT 的区间：按 task cluster 做 percentile bootstrap；
- 三 seed 方法的区间：同时重采样配对训练 seed 和共同冻结任务的 crossed bootstrap；
- Anchor−GRPO：3 对训练 seed 的精确双侧 sign randomization；
- 跨 500 个任务的 paired test：只作为这些冻结 checkpoint 条件下的诊断；
- 5 个解释性基线/恢复比较统一做 Holm 校正。

只有 3 对训练 seed，因此算法级双侧检验的最小可达 `p` 值是 0.25。最终精确分析代码
是在测试结果产生后固化，主比较虽在最终报告中预先指定，但不能回溯称为严格预注册；
所有显著性结果均应视为探索性证据。

## 6. 最终结果

### 6.1 总体表现

| 模型 | 严格成功率 | 95% CI | Dense score | 评测 tokens/trajectory |
|---|---:|---:|---:|---:|
| Raw base | **33.50%** | [30.05%, 37.20%] | **0.6294** | **75.6** |
| Shared SFT | 0.65% | [0.25%, 1.20%] | 0.0256 | 202.7 |
| Multi-turn GRPO | 9.42% | [7.68%, 11.32%] | 0.3599 | 173.5 |
| Anchor-GiGPO | 9.27% | [7.63%, 11.05%] | 0.3759 | 147.5 |

Raw 的 K4 至少一次成功率为 49.8%，四次全部成功率为 20.4%；SFT 分别只有 2.0% 和
0.2%。在线 RL 的均值在数值上明显高于 SFT，但仍明显低于 Raw；算法级显著性受 3 个
训练 seed 的检验分辨率限制，见第 6.4 节。

### 6.2 各训练 seed

| 方法 | Seed 20260801 | Seed 20260802 | Seed 20260803 | 三 seed 均值 |
|---|---:|---:|---:|---:|
| GRPO | 9.45% | 8.70% | 10.10% | 9.42% |
| Anchor | 9.50% | 9.45% | 8.85% | 9.27% |
| Anchor−GRPO | +0.05 pp | +0.75 pp | −1.25 pp | −0.15 pp |

三个配对差异方向不一致，没有稳定的主指标优势。

### 6.3 主比较：Anchor-GiGPO 对 GRPO

| 指标 | Anchor−GRPO | 95% CI | 算法级精确 p | 解释 |
|---|---:|---:|---:|---|
| Strict success | −0.0015 | [−0.0160, 0.0120] | 1.0 | 无显著差异 |
| Dense score | +0.0160 | [−0.0223, 0.0463] | 0.5 | 算法级不显著 |

在 500 个任务聚合后的诊断中，Anchor 有 95 个任务改善、320 个持平、85 个退化；
task-conditioned strict-success `p=0.740`。dense score 的 task-conditioned 区间为
[0.0070, 0.0251]、`p≈0.00074`，但该检验把已冻结的 3 个 checkpoint 当作条件，不能
替代训练 seed 级算法推断。

6,000 对 rollout 的严格成功 discordance 为：两者都成功 266、仅 GRPO 成功 299、
仅 Anchor 成功 290、两者都失败 5,145。这再次说明主指标差异很小。

### 6.4 与 Raw/SFT 的解释性比较

| 比较 | Strict-success 差值 | 检验口径 | 原始 p | Holm p | 结论 |
|---|---:|---|---:|---:|---|
| SFT − Raw | −32.85 pp | paired task | `2.0e-5` | `1.0e-4` | 显著负迁移 |
| GRPO − SFT | +8.77 pp | 3 training seeds | 0.25 | 1.0 | 有恢复，算法级证据不足 |
| Anchor − SFT | +8.62 pp | 3 training seeds | 0.25 | 1.0 | 有恢复，算法级证据不足 |
| GRPO − Raw | −24.08 pp | 3 training seeds | 0.25 | 1.0 | 未恢复到 Raw |
| Anchor − Raw | −24.23 pp | 3 training seeds | 0.25 | 1.0 | 未恢复到 Raw |

“p 不显著”不能解释为没有真实恢复；这里的限制主要来自只有 3 个训练 seed。效应大小、
轨迹和过程指标仍清楚显示 RL 相对 SFT 的行为修复，但不能据此宣称算法总体显著优于 SFT。

## 7. 为什么 SFT 比基础模型更差

### 7.1 根因一：verified 不等于 learnable

专家搜索标签由隐藏的精确 `goal.name` 生成，而 policy prompt 只含公开 instruction 和环境
状态。4,000 个搜索 query 的统计显示：

| 诊断 | 结果 |
|---|---:|
| Unique search queries | 3,695 |
| Query 平均字符数 | 130.9 |
| Query P50 / P90 字符数 | 136 / 194 |
| Query 是 instruction 原文子串 | 0.175% |
| Query 与 instruction 平均 word Jaccard | 0.125 |
| Query token 可由 instruction 覆盖 | 24.6% |

可以把问题写成：训练标签是 `a*=f(x,z)`，其中 `x` 是 policy 可见状态，`z` 是隐藏的
精确商品标题；推理时模型只有 `x`。即使每条轨迹都能由 oracle 完成，模型也无法稳定从
`x` 还原 `z`。这类标签通过了可执行性和 reward 验证，却没有通过条件可预测性验证。

### 7.2 根因二：SFT 把强基础策略压窄成脆弱模板

Raw base 原本具有通用的关键词提取、短查询、页面阅读和纠错能力。一次 339,925 label-token、
1,031 update 的 LoRA SFT 强化了单一模式：生成接近商品标题的长查询，若搜索未命中则连续
`Next`。adapter 没有修改 base 权重，但在推理时足以覆盖 base 的行动分布，形成行为层面的
负迁移。

闭环数据印证了这一点：SFT 有 91.35% 的轨迹因 search/navigation exhaustion 失败，
平均使用 202.7 tokens 和 14.6 个环境 step；Raw 仅 2.35% 搜索耗尽，平均 75.6 tokens
和 5.2 step。

### 7.3 根因三：teacher forcing 掩盖 exposure bias

SFT 的 exact action 68.66% 和 schema validity 92.21% 都是在专家状态分布上计算。闭环时，
第一步搜索稍有偏差就进入训练语料较少覆盖的页面；模型随后用高频 `Next` 模板继续偏离，
错误状态不断累积。teacher-forced NLL 无法检测这种自生成状态分布上的退化。

### 7.4 根因四：准入门禁没有实现研究合同的闭环保护

冻结方案曾要求第一轮后运行 closed-loop dev 和 base-signal gate，但实际 SFT preflight
硬门禁只检查 epoch、label-token 下限、有限 train/dev NLL、zero-label 和 truncation。
teacher-forced 指标被记录但没有成为硬门禁，闭环相对 Raw 的表现也没有阻止 adapter 进入
在线阶段。

这是本轮最重要的实验治理经验：**训练损失、格式正确率和 oracle reward 都不能替代
policy-on-own-state-distribution 的闭环准入。**

### 7.5 次要放大因素

- 语料只保留 oracle 能在 top-50 内完成的任务，偏向教师搜索策略容易解决的子分布；
- completion-only 目标优化“像教师一样输出动作”，没有直接优化最终商品匹配；
- 单一确定性专家缺乏多种合法搜索和恢复轨迹，模型把一种实现方式误学成唯一策略；
- 没有显式保持 Raw 行为的 KL/reference 或混合通用轨迹目标，无法保护已有浏览能力。

## 8. 在线 RL 学到了什么

### 8.1 最大提升是恢复闭环推进，不是最终精确决策

| 过程指标 | Raw | SFT | GRPO | Anchor |
|---|---:|---:|---:|---:|
| Purchase termination | 95.05% | 7.50% | 91.17% | **95.62%** |
| Search exhaustion | **2.35%** | 91.35% | 8.00% | 3.57% |
| Output-format failure | **0.00%** | 0.20% | 0.27% | 0.07% |
| Schema-invalid trajectory | 7.50% | 17.45% | 13.10% | **7.37%** |
| Action-error trajectory | **7.85%** | 99.45% | 67.25% | 82.77% |
| Mean environment steps | **5.20** | 14.63 | 8.70 | 8.96 |

在线 RL 几乎完全修复了 SFT 的连续翻页和“不购买”问题。相对 Raw–SFT 的差距：

- GRPO 恢复约 26.7% strict-success gap、55.4% dense-score gap；
- Anchor 恢复约 26.2% strict-success gap、58.0% dense-score gap；
- purchase termination 的差距分别恢复约 95.6% 和 100.6%。

这说明 reward 确实把策略从崩溃的 SFT 分布拉回可执行闭环，但恢复的是“走完流程”的能力，
不是全部的商品识别和约束满足能力。

### 8.2 为什么 dense score 恢复多于 strict success

WebShop dense score 会奖励部分属性、选项和购买进度。K4 组内只要这些分数有方差，GRPO
就能产生宏观优势；Anchor 还能把终奖折扣回传到共享公开状态。相比之下，strict success
要求最终商品和约束全部正确。训练因此更容易先学会搜索、进入商品页、配置选项和购买，
却仍可能买到 partial match。

### 8.3 Anchor-GiGPO 相对 GRPO 的真实收益与代价

Anchor 的主要正向证据是：

- dense score +0.0160；
- 搜索耗尽从 8.00% 降到 3.57%；
- schema-invalid trajectory 从 13.10% 降到 7.37%；
- 评测 token 从平均 173.5 降到 147.5，节省约 15.0%；
- output-format failure 从 0.27% 降到 0.07%。

但 Anchor 的 action-error trajectory 更高，partial-match purchase 也更多。它利用共享状态
对比更快强化“有回报的推进动作”，却无法为只出现于单条轨迹的细粒度商品 grounding
提供同样密集的对照。这解释了为什么过程效率改善，而严格成功率没有改善。

## 9. 相对 Raw 仍然有价值的局部能力

虽然聚合结果低于 Raw，RL 不是在每个任务上都更差：

| 方法 | 比 Raw 更好的任务 | 持平 | 更差 | 解决 Raw 零成功任务 |
|---|---:|---:|---:|---:|
| SFT | 0 | 252 | 248 | 0/251 |
| GRPO 三 seed 均值 | 51 | 212 | 237 | 44/251 |
| Anchor 三 seed 均值 | 39 | 225 | 236 | 32/251 |

所有 3 个 GRPO seed 都稳定胜过 Raw 的任务有：
`00064, 00107, 00184, 00273, 00385, 00445, 00498`。

所有 3 个 Anchor seed 都稳定胜过 Raw 的任务有：
`00015, 00064, 00138, 00184, 00334, 00498`。

这说明在线训练形成了局部互补能力，但没有任何类别或约束复杂度分层在聚合成功率上超过
Raw。因此目前最合理的应用结论不是替换 Raw，而是进一步研究路由、策略融合或更合理的
SFT 起点；冻结 test 已经打开，不能再用本轮测试结果选择模型或补训并声称同一次正式结论。

## 10. 失败轨迹分类

每条失败轨迹只分配一个互斥主类；格式、公开动作错误、终止原因和最后页面作为正交诊断。

| 身份 | 失败总数 | Partial-match purchase | Zero-match purchase | Search exhaustion | Item configuration | Output format |
|---|---:|---:|---:|---:|---:|---:|
| Raw（2,000） | 1,330 | 1,139 | 92 | 47 | 52 | 0 |
| SFT（2,000） | 1,987 | 108 | 29 | 1,827 | 19 | 4 |
| GRPO（6,000） | 5,435 | 4,098 | 807 | 480 | 34 | 16 |
| Anchor（6,000） | 5,444 | 4,344 | 837 | 214 | 45 | 4 |

失败分布展示了清晰的阶段转换：SFT 主要卡在搜索；RL 解决搜索和终止后，主要错误迁移到
“已经买了，但商品只部分匹配”。这不是简单的失败减少，而是瓶颈从 navigation 转移到
grounding 和 constraint verification。

失败 taxonomy 是确定性诊断规则，不是人工标注的因果真值。它适合定位工程和策略瓶颈，
不应被解释为对模型内部原因的直接测量。

## 11. 代表性轨迹

### 11.1 Task 00064：RL 修复 SFT 的翻页循环

任务要求 yoga sweatpants。Raw 使用简短查询、点击商品、选择 medium 并购买，dense reward
为 0.75；SFT 生成记忆式长标题，然后连续 15 步 `Next`，reward=0。GRPO seed 3 在 3 次
翻页后找到正确商品并购买，Anchor seed 2 在 5 次翻页后完成，二者 reward=1。

该案例说明 RL 能在 SFT 崩溃的任务上重新学习从环境反馈恢复，并可能超过 Raw 的单次采样。

### 11.2 Task 00452：成功不代表策略更简洁

Peet's 商品任务中，Raw 以 3 步、45 tokens 成功；SFT 使用错误的长标题后连续翻页失败；
GRPO 和 Anchor 都在 3 步内成功，但分别使用约 82 和 81 tokens，并产生更冗长、带幻觉式
标题的查询。

这说明终局 success 相同也可能具有不同成本和风险。报告 token、step、action error 和
轨迹文本是必要的，不能只汇报成功率。

## 12. 可审计性与产物

### 12.1 关键仓库文件

- 冻结研究合同：[`M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md`](M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md)
- 训练与恢复手册：[`M5_EXECUTION_READINESS.md`](M5_EXECUTION_READINESS.md)
- 机器训练计划：[`../data/m5_webshop_formal_plan_v1.json`](../data/m5_webshop_formal_plan_v1.json)
- 冻结评测计划：[`../data/m5_webshop_frozen_eval_plan_v1.json`](../data/m5_webshop_frozen_eval_plan_v1.json)
- 上游数据锁：[`../data/m5_webshop_upstream_lock_v1.json`](../data/m5_webshop_upstream_lock_v1.json)
- Split 排除锁：[`../data/m5_webshop_split_exclusions_v1.json`](../data/m5_webshop_split_exclusions_v1.json)
- 正式 online runner：[`../scripts/m5_webshop_formal_online.py`](../scripts/m5_webshop_formal_online.py)
- 冻结评测 runner：[`../scripts/m5_webshop_frozen_eval.py`](../scripts/m5_webshop_frozen_eval.py)
- 最终分析 runner：[`../scripts/m5_webshop_analyze_final.py`](../scripts/m5_webshop_analyze_final.py)

### 12.2 远端正式产物

所有路径相对远端仓库 `/home/wushaohua/data/MiniWebWork-RL`：

```text
outputs/m5_webshop_credit_assignment_v1/formal/online/{method}/seed_{seed}
outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/{identity}
outputs/m5_webshop_credit_assignment_v1/formal/analysis
```

最终分析目录包含：

| 文件 | 行数/记录数 | SHA-256 |
|---|---:|---|
| `FINAL_STATISTICAL_REPORT.md` | Markdown | `fb929d20d37567a9d4d51bd800d22ea6024988b55ad082e46c6c643ae103db8f` |
| `final_statistical_report.json` | 机器可读报告 | `04b25a3e4b2f85fe0ae26f4b26b7009645ba9b5c2ecf177008ac6b75e07f8c10` |
| `paired_task_differences.csv` | 1,500 rows | `d709ae47b079cd40151d24bf5bb337326cbafb3f4da9224610fe564ca218b48b` |
| `failure_trajectory_classification.jsonl` | 14,196 rows | `1497ebf5c785209a2fe729587776d547f0b0b16fc5abd8132c1fe39d9db9a4c4` |

Job 2184 生成最终统计工件，状态为 `COMPLETED/0:0`。共享 WebShop 服务 Job 2172
在全部评测与分析结束后主动停止，不再占用集群资源。

## 13. 有效性边界

1. 只有 3 个训练 seed，算法级显著性分辨率很低；不能用 task 数量替代训练重复数。
2. 最终精确分析实现晚于结果生成，统计结论是探索性的，不是预注册验证。
3. 研究只覆盖一个环境、一个 4B base model、一个 prompt 和一套 SFT 起点。
4. RL 与 Raw/SFT 的比较条件于固定 baseline checkpoint，不能外推到全部基础模型或 SFT。
5. 商品会在 WebShop 任务间复用；task-cluster 处理了 K4 相关性，但不能消除 benchmark
   生成机制本身的所有依赖。
6. SFT 专家使用隐藏 metadata 构造动作；这正是本轮识别出的数据设计缺陷，不能把该 SFT
   结果推广为“监督微调普遍有害”。
7. 冻结 test 已使用，后续任何基于这些结果的训练只能作为新版本、新假设和新测试协议，
   不能回写本轮正式矩阵。

## 14. 经验与下一步

### 14.1 可以直接复用的经验

1. **数据验收必须增加 conditional learnability。** 除了 action 可执行和终奖正确，还要
   检查教师动作是否由 policy 可见信息决定；可用 query–instruction overlap、多个合法专家、
   反事实状态和小规模闭环 probe 检测。
2. **SFT 准入必须以闭环行为为主。** 至少要求相对 Raw 的 dev success、search exhaustion、
   action error 和成本不发生超阈值退化；teacher-forced NLL 只作辅助指标。
3. **在线 RL 的 reward 要与最终目标分层报告。** dense reward 适合提供训练信号，但必须和
   strict success、partial match 及成本共同呈现。
4. **信用分配的收益可能先出现在过程指标。** Anchor 的 dense、搜索恢复和 token 成本改善
   是真实证据，但在 strict success 无提升时不能声称最终决策更好。
5. **失败迁移比单一失败率更有信息。** 本轮瓶颈从 SFT 的 search exhaustion 转移到 RL 的
   partial-match purchase，直接指出下一步应做商品 grounding 和约束核验，而不是继续增加
   导航奖励。

### 14.2 若启动新一轮研究

推荐把新工作定义为独立 M6，而不是在已经打开的 M5 test 上继续调参：

- 重建 SFT 数据：只用公开 instruction 可推导的短查询，或收集多样化合法搜索/恢复轨迹；
- 在新的 dev 集上以 Raw 为下限做闭环 acceptance gate；
- 保留 Raw 能力：评估 reference KL、SFT/online mixture 或 adapter routing；
- 对 partial-match purchase 增加公开约束核验状态或辅助 verifier，但保持最终 reward 合同透明；
- 在资源允许时增加训练 seed，提高算法级推断能力；
- 预先冻结分析代码和新的 untouched test/holdout，避免事后选择统计口径。

## 15. 面试表达建议

本项目最准确的一句话总结是：

> 我没有把项目做成多算法刷榜，而是在 WebShop 上控制变量比较终局 GRPO 与公开状态逐步
> 信用分配；实验发现 verified SFT 因 privileged search labels 发生闭环负迁移，在线 RL
> 能恢复大部分状态推进和购买能力，但只恢复约四分之一严格成功率差距。Anchor-GiGPO
> 没有显著提高最终成功率，却以约 15% 更低 token 成本获得更高 dense score，并把失败瓶颈
> 从搜索耗尽推进到商品 grounding。整个结论由 16,000 条冻结轨迹、seed-aware 统计、失败
> 分类和 adapter/optimizer/ledger 血缘支持。

这段经历体现的不是“每次训练都赢”，而是能发现错误的数据假设、修复训练系统、设计受控
对比、诚实处理负结果，并把模型效果、过程效率和统计证据分开解释。

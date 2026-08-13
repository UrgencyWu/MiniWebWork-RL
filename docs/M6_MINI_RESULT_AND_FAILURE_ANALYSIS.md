# M6-mini Raw → SFT → RL 结果与失败分析

> 状态：M6.1 development-only 链已结束；正式扩展未获准
>
> 冻结结果提交：`d2560c31f5caa1855adf0ce799343810551939ea`
>
> 统计单位：200 个隔离 mini-dev task，每个 task `K=4`，共 800 条轨迹/策略

## 1. 结论先行

M6 修复了 M5 中 SFT 严重退化的问题，但没有实现预先要求的
`Raw < SFT < RL` 单调提升：

| 策略 | strict success | 相对 Raw | 相对 SFT |
|---|---:|---:|---:|
| Raw | 286/800 = 35.750% | — | -5.875 pp |
| SFT | 333/800 = 41.625% | +5.875 pp | — |
| multi-turn GRPO | 331/800 = 41.375% | +5.625 pp | -0.250 pp |
| Anchor-GiGPO | 331/800 = 41.375% | +5.625 pp | -0.250 pp |

这次 SFT 的闭环效果真实超过 Raw，说明 success-replay、公开可见标签和 Raw retention
方向有效。两个 RL adapter 也都真实更新并保持高于 Raw，但都没有超过共同 SFT。

“RL 相比 SFT 下降 0.25 pp”只是点估计，不是统计显著退化。两个方法均为：

- task-cluster bootstrap 95% CI：`[-1.25, +0.75] pp`；
- SFT-only 成功轨迹 10，RL-only 成功轨迹 8；
- exact McNemar `p=0.8145`。

因此本轮支持的结论是：**有限预算下没有检测到 RL 的净提升，观察到的 -2/800
差异与采样波动相容。** 这不支持“RL 已经系统性损害 SFT”，也不允许声称 RL 提升。
预注册晋级门要求 RL-SFT 至少 `+3 pp` 且 bootstrap 正方向比例至少 `0.8`；两方法的
正方向比例分别只有 `0.2776` 和 `0.2645`，故最终决策为
`STOP_AND_BURN_MINI_DEV`。

## 2. 训练确实发生了什么

两个 learner 都从同一个 M6 SFT adapter 开始，使用相同任务 curriculum、K=8、
seed 方案和动作 token 预算：

| 项目 | multi-turn GRPO | Anchor-GiGPO |
|---|---:|---:|
| 训练 task/group | 5 | 5 |
| 训练轨迹 | 40 | 40 |
| optimizer updates | 5 | 5 |
| action tokens | 2,338 | 2,315 |
| mixed-reward updates | 5/5 | 5/5 |
| credit-covered turns | 163 | 163 |
| nonzero optimizer-turn fraction | 1.0 | 1.0 |
| finite loss/gradient | PASS | PASS |
| adapter parameter change | PASS | PASS |

共同的 5 个训练 task 为 `04381, 05896, 08156, 11256, 01759`。GRPO 每组
strict successes 为 `3, 3, 4, 1, 6`，Anchor-GiGPO 为 `4, 3, 4, 2, 6`。

multi-turn GRPO 把同一轨迹的 strict terminal advantage 分配给全部 action turn；
Anchor-GiGPO 在同一 macro advantage 上加入 detached、逐轨迹零和的 verifier-TD
turn redistribution。信用分配实现和更新证据均通过，但这只证明训练链正确工作，
不等价于证明最终策略更好。

## 3. 成对行为变化

### 3.1 multi-turn GRPO 对 SFT

| 成对结果 | 轨迹数 |
|---|---:|
| 两者都失败 | 459 |
| 两者都成功 | 323 |
| 仅 SFT 成功 | 10 |
| 仅 GRPO 成功 | 8 |

200 个 task 中，182 个 strict success 数完全相同，10 个回退 1/4，8 个提升
1/4；没有 task 变化超过一条成功轨迹。688/800 条轨迹动作序列完全相同，发生变化
时首次分歧 turn 的中位数为 4。

GRPO dense score 从 SFT 的 `0.651152` 微升到 `0.651800`，差值
`+0.000648`，95% CI `[-0.007417, +0.008804]`。它修复了少量 action/schema/
search failure，却把更多原严格成功变成 partial purchase。主要失败分类变化如下：

| 分类 | SFT | GRPO | 变化 |
|---|---:|---:|---:|
| success | 333 | 331 | -2 |
| partial purchase | 360 | 368 | +8 |
| zero-score purchase | 45 | 41 | -4 |
| action error | 35 | 31 | -4 |
| schema error | 13 | 11 | -2 |
| max environment steps | 5 | 8 | +3 |
| search exhaustion | 9 | 10 | +1 |

回退 task：`01872, 02022, 02777, 03689, 03794, 05387, 06006, 06990,
09440, 10542`。提升 task：`02861, 04358, 05505, 06605, 06817, 07089,
08544, 10269`。

典型现象不是动作格式崩坏，而是末端决策边界漂移：选择相近但不完全匹配的商品、
在已找到正确商品后重新搜索、或在正确商品上选择错误 option。也就是说，GRPO
改善了一些“能否继续行动”的错误，但在“是否应该停止并提交准确购买”上损失了更多
严格成功。

### 3.2 Anchor-GiGPO 对 SFT

Anchor-GiGPO 的 strict 配对表与 GRPO 相同：459 个共同失败、323 个共同成功、
10 个 SFT-only、8 个 RL-only。task 层有 184 个持平、9 个回退、7 个提升；
682/800 条动作序列完全相同，首次分歧 turn 的中位数同样为 4。

其 dense score 为 `0.646721`，相对 SFT `-0.004431`，95% CI
`[-0.012769, +0.003729]`。分类为 331 success、358 partial purchase、45
zero-score purchase、35 action error、11 schema error、8 max-step 和 12
search exhaustion。

回退 task：`01296, 01872, 02777, 03794, 05387, 06990, 07628, 07934,
10542`。提升 task：`02861, 04783, 06605, 06817, 07089, 08541, 10269`。

### 3.3 两种信用分配方法之间

两方法最终 strict success 完全相同。直接配对时，324 条轨迹共同成功、462 条共同
失败、各自独占 7 条成功。Anchor-GiGPO 相对 GRPO 的 dense score 低
`0.005079`。因此本轮没有证据证明 turn-level anchor credit 优于 trajectory-level
GRPO；更合理的解释是两者都受相同的数据量和更新预算限制，信用分配差异尚未成为
主导变量。

## 4. 为什么 RL 没有超过 SFT

按证据强度排序，主要原因是：

1. **RL 暴露量不足。** 每种方法只看 5 个 task、40 条轨迹并做 5 次更新，覆盖远低于
   200-task 评测分布。单个 task 的偶然更新就足以抵消数条严格成功。
2. **真实更新很小。** 学习率为 `1e-6`，观测 KL 约 `1.08e-4–4.71e-4`，远低于
   控制器的 `0.005` 下界目标。adapter 确实变化，但 5 步不足以产生稳定的闭环行为迁移。
3. **strict reward 稀疏且边界尖锐。** 部分匹配购买与完全失败在 macro reward 上同为
   0。GRPO dense score 略升但 strict success 略降，正好说明“进度变好”未必跨过
   最终全匹配门槛。
4. **SFT 已经是强基线。** SFT 在同一 mini-dev 上比 Raw 高 5.875 pp，RL 要在很小
   预算下继续提高，比从受损 SFT 恢复能力更难。
5. **训练覆盖略低于原数据门。** corpus 有 493 条轨迹和 31,362 个有效 label token，
   但只有 156 个唯一 success task，低于冻结门槛 160。waiver 只允许开发验证，不会
   把覆盖不足变成正式合格数据。
6. **方法隔离并非完美。** 两个 learner 各自做 on-policy 采样；即使初始 policy、task
   和 seed 相同，第一组也只有 7/8 条动作轨迹完全相同。之后 policy 本应分岔，但初始
   随机差异增加了小样本方法比较的噪声。
7. **只有最终 checkpoint，没有独立 tuning-dev 上的学习曲线选择。** 当前 mini-dev 已
   用于晋级判断，不能再用于挑选最佳 step。5 个更新中间是否短暂优于 SFT，现有证据
   无法回答。

## 5. 数据与门禁解释

Corpus audit 的唯一失败项是 `mini_success_task_count=156 < 160`。以下项目均通过：

- 493 条 replay-verified trajectory，超过 320 下限；
- 31,362 completion-label tokens，处于 20k–80k 区间；
- strict success/replay success 均为 1；
- hidden-field 与 target-ASIN label 均为 0；
- public-query token fraction `0.9461`；
- recovery fraction `0.2860`；
- 最大 action-family fraction `0.3036`。

因此 corpus 不是“完全无效”，但它没有满足正式实验的预注册覆盖门。SFT pilot 通过
development-only waiver 是合理的快速验证；最终 chain 仍把
`corpus_audit_passed=false` 记为失败，也是正确的治理行为。

## 6. 可从本轮学习的经验

1. **先修数据语义，再谈 RL 算法。** 从 M5 的 0.65% SFT 到 M6 的 41.625%，最大提升
   来自去除 privileged labels、使用真实成功/恢复轨迹，而不是换一个 policy-gradient
   公式。
2. **训练正确不代表效果有效。** finite loss、非零梯度、参数变化和信用覆盖是必要条件，
   但最终必须由冻结闭环评测决定。
3. **保留 paired flips 比只报均值更重要。** 两方法都是“修复 8、破坏 10”；这比
   “下降 0.25 pp”更清楚地说明策略发生了真实但不稳定的局部重排。
4. **信用分配不是当前第一瓶颈。** 两种方法得到相同 strict 结果，说明下一轮优先增加
   独立 task 覆盖、seed 和更新曲线证据，而不是继续堆叠新的 RL 算法。
5. **失败门禁应保留。** 本轮没有把略高于 Raw 包装成 Raw→SFT→RL 成功；停止正式扩展
   避免了把 development-only 偶然波动放大为高成本训练。

## 7. 下一轮最小改进边界

本报告不授权新训练。若开启下一轮，应先冻结新的独立 tuning-dev 与 promotion slice，
不得继续复用本轮已经打开的 mini-dev。最小改进应聚焦：

1. 先使 corpus 真正达到唯一成功 task 门槛，不再依赖 waiver；
2. 将 RL curriculum 扩到至少 20 个独立 task，并用多个 seed 判断符号稳定性；
3. 保存并在专用 tuning-dev 上查看逐 checkpoint 曲线，预先固定 early-stop 规则；
4. 同时报告 strict success、paired flips、partial-purchase 转移、KL 和训练成本；
5. 只保留 GRPO 与一个 anchor credit 变体，不新增算法；
6. 只有新 promotion slice 上 RL 显著超过 SFT，才申请正式全量训练。

## 8. 权威产物

远端根目录：`/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1`

```text
mini/corpus_v2/corpus_audit_report.json
mini/pilot_sft_gate_v3.json
mini/rl_v7/grpo/learner/online_update_report.json
mini/rl_v7/grpo/evaluation/identity_report.json
mini/rl_v7/grpo/chain_gate_report.json
mini/rl_v7/anchor_gigpo/learner/online_update_report.json
mini/rl_v7/anchor_gigpo/evaluation/identity_report.json
mini/rl_v7/anchor_gigpo/chain_gate_report.json
mini/rl_v7/comparison_report.json
```

关键自哈希：corpus audit `21f0677a…`、SFT gate `3dacbe2b…`、GRPO audit
`2f7e99c4…`、Anchor audit `6888dce0…`、GRPO chain `20af3eb5…`、Anchor chain
`33448964…`、comparison `fa678363…`。完整作业失败与修复见
[`TRAINING_FAILURE_LEDGER.md`](TRAINING_FAILURE_LEDGER.md)。

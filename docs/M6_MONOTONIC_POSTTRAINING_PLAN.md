# M6 WebShop：Raw → SFT → RL 单调提升改进计划

> 状态：`planning_only`
>
> 制定日期：2026-08-12
>
> 起点提交：`f5d6d4fa5e1cda7348f056bba72972c9bb157c25`
>
> 前序证据：[`M5_FINAL_TECHNICAL_REPORT.md`](M5_FINAL_TECHNICAL_REPORT.md)
>
> 当前权限：仅设计、实现与 preflight；本文件不授权正式 SFT、RL 或最终测试

## 1. 目标与不可伪造的成功定义

M6 的目标是得到一条真实、可复现的性能链：

```text
Raw base < SFT < RL
```

这里的“性能”首先指 WebShop **strict task success**（`task_score >= 0.999`），不是训练
loss、teacher-forced exact action 或 dense score。最终研究成功必须同时满足：

1. 在一次性打开的 M6 untouched holdout 上，`SFT − Raw >= 3.0 pp`；
2. 在同一 holdout 上，`RL − SFT >= 3.0 pp`；
3. 两个差值的 task-cluster 95% CI 下界均大于 0；
4. RL 的 tokens/trajectory ≤SFT×1.15、steps/trajectory ≤SFT×1.10，schema/action error
   不高于 SFT +2 pp，partial-match purchase 不高于 SFT；
5. 3 个独立训练 seed 均满足 `SFT > Raw`，且 3 对 seed 均满足 `RL > SFT`，不能只报告
   最好 seed；
6. 所有模型、checkpoint、prompt、阈值和分析代码在首次打开 promotion gate 前冻结。

这不是承诺模型一定会提升。我们能保证的是：**只有满足上述证据才会把 M6 标记为成功；
任何失败都会在 tuning-dev 门禁或正式报告中保留，不能靠测试泄漏、事后挑 seed 或改指标
制造成功。**

为避免重复使用 M5 test，M5 的 `[0,500)` 只保留为历史审计证据，不再参与 M6 的模型选择
或正式结论。M6 必须先汇总 M5 SFT、preflight、online 和人工轨迹诊断读取过的全部
`goal_index`，形成历史暴露登记；promotion gate 与最终 holdout 只能从原 eligible train 中
**从未被 M5 artifact 或报告按任务读取过**的目标抽取。新角色冻结后只按权限读取。

“按任务读取”指某个 goal 的 observation、动作、reward、成功/失败或诊断结果曾进入模型更新、
checkpoint/阈值选择、人工分析或正式报告。仅对完整 `goals.json` 做字节哈希、计算无内容的
roster identity 或服务启动时加载数据库，不把所有 12,087 条都机械标为暴露；registry 必须
记录每个暴露判断的具体证据路径和原因。

## 2. M5 失败如何映射到 M6 修复

| M5 证据 | 根因 | M6 直接修复 | 硬门禁 |
|---|---|---|---|
| SFT strict success 33.50% → 0.65% | 教师 search 使用 prompt 不可见的精确 `goal.name` | 禁止 hidden-title teacher；只收集 policy-visible、环境真实执行的成功轨迹 | label conditional-learnability audit + closed-loop SFT > Raw |
| SFT 91.35% search exhaustion | 单一长标题模板和 exposure bias | 短查询、多样重写、成功/恢复轨迹、状态分布混合 | search exhaustion 不高于 Raw +2 pp |
| Teacher-forced 指标好但闭环崩溃 | SFT preflight 未实现闭环门禁 | 先冻结 Raw tuning-dev baseline，每个候选 checkpoint 都做闭环 paired eval | 不过门不允许进入 RL |
| RL 主要恢复 purchase，strict 仍低 | dense terminal reward 奖励 partial match | macro reward 改为 strict-success 主导；dense 只作诊断和有限 potential | partial-match 不得恶化；strict success 是选点依据 |
| Anchor dense 较好但 strict 不升 | 共享状态 credit 无法解决商品 grounding | 使用 verifier progress 的 TD 差分给 turn credit，并保留 terminal strict outcome | TD credit telescoping audit + grounding slice gate |
| RL 仍远低于 Raw | 坏 SFT 起点、无能力保持约束 | 先证明 SFT 胜 Raw；RL 加 frozen-SFT reference KL 和 tuning-dev early stop | RL checkpoint 必须同时胜 SFT 和 Raw |

## 3. 技术路线：一条主线，不扩算法清单

M6 只训练一个 SFT recipe 和一个在线 RL recipe：

```text
公开成功轨迹蒸馏
→ 闭环保真 SFT
→ strict-success GRPO + verifier-TD turn credit
```

不再比较 GRPO、GiGPO、PPO、RLOO、DAPO 等多个名字。M5 已经展示了 macro-vs-anchor
对比；M6 的问题是修复数据和目标，使后训练真正超过 Raw。GRPO 仍是稳定的 group-relative
optimizer 骨架，新贡献集中在：

- policy-visible success data；
- Raw 能力保持；
- strict outcome 与 verifier progress 一致的 turn credit；
- 闭环逐阶段晋级。

### 3.1 与公开工作的关系

- 原始 [WebShop](https://arxiv.org/abs/2207.01206) 发布了 1,600+ 人类示范，说明优质、
  公开观察下的行为克隆是合理数据源，
  但任何外部示范必须先完成许可证、split 和 action-schema 审计；
- [WebAgent-R1](https://arxiv.org/abs/2505.16421) 强调 web agent 的 end-to-end multi-turn
  RL 和 warm-up policy，同时用最终 binary success 约束在线学习；
- [GiGPO](https://arxiv.org/abs/2505.10978) 证明 multi-turn agent 需要 step-level credit，
  但 M5 表明仅按重复状态分组不足以修复精确商品 grounding；
- 2026 年 [TRACE](https://arxiv.org/abs/2607.13988) 使用状态价值的 TD 变化提供逐 turn
  信号，核心启发是让中间 credit telescoping 到可靠 outcome，而不是任意叠加 shaping reward。

M6 不声称逐行复现这些算法。它采用与 WebShop 确定性 verifier 更匹配、无需额外 critic 的
`verifier-TD`：从当前公开状态计算可审计 progress potential，并用相邻状态差分重新分配
既有终局目标。

## 4. 数据重建：从“oracle 能执行”改为“policy 能学习”

### 4.1 数据来源优先级

正式 SFT 数据按以下优先级收集：

1. **Raw-policy success distillation（主来源）。** 只在 M6 train tasks 上用冻结 Raw
   Qwen3.5-4B 采样 K=8；只保留 official strict success=1 的完整轨迹。动作天然由 policy
   可见 observation 生成，不存在 hidden-title label。
2. **Raw-policy recovery trajectories。** 保留包含 query reformulation、Back/Prev、换商品
   后最终严格成功的轨迹，专门覆盖 M5 的 search exhaustion 和 exposure bias。
3. **官方 WebShop human demonstrations（可选补充）。** 先验证许可证、下载内容哈希、
   goal split、动作可在冻结环境 replay，以及是否包含 M6 holdout。任一项不清楚就排除，
   不把外部数据接入作为项目阻塞项。
4. **公开状态下的 deterministic query expert（最多 20%）。** 若成功轨迹覆盖不足，只允许
   从 instruction 中抽取类别、属性、材质、尺寸等公开词构造 2–8 token 查询；严禁读取
   `goal.name`、target ASIN 或其他 prompt 不可见字段。

不再使用 M5 的 hidden-title oracle corpus，也不把 M5 test 上的成功轨迹蒸馏进 M6。
Raw rejection sampling 的意义不是创造强于 Raw 的新知识，而是把 Raw 偶尔采样成功的行为变成
更稳定的 pass@1/K4 策略；对 Raw 从未成功的困难任务，增量覆盖只能来自 K16 成功、已审计
human demonstration 或 public-only expert，不能使用 hidden metadata 补答案。

### 4.2 语料结构

每个样本保留完整 trajectory identity，并以 action-turn completion 训练。数据至少覆盖：

- first search、query reformulation、候选点击；
- 读取公开商品属性后返回或继续；
- option configuration、Buy Now；
- 合法 recovery，而不仅是最短成功路径。

为防止 `Buy Now`、`Next` 等高频动作压过 grounding，训练 sampler 以 trajectory 为单位，
并对 action family 设上限：任一 family 不超过 completion action turns 的 35%。同一 task
最多保留 4 条语义不同轨迹，按 command sequence 去重。

### 4.3 Conditional-learnability 硬门禁

SFT corpus 发布前必须满足：

| 门禁 | 要求 |
|---|---:|
| Hidden goal fields in prompt/completion provenance | 0 |
| Search label 使用 `goal.name` 或 target ASIN | 0 |
| Strict-success verified trajectories | 100% |
| Environment replay success | 100% |
| Zero-label / truncation | 0 / 0 |
| Train/tuning-dev/promotion/holdout exact instruction overlap | 0 |
| M5-exposed goals in promotion/holdout | 0 |
| Search query token 来自 instruction 或先前公开 observation | ≥90% |
| Search query mean length | ≤48 characters |
| Search query P95 length | ≤96 characters |
| Unique command-sequence fraction | ≥60% |
| Recovery trajectories | ≥20% |
| Action-family maximum share | ≤35% |

剩余最多 10% query token 可以是基础模型从公开语义生成的同义改写；必须记录来源，不能来自
hidden metadata。所有 corpus 统计和逐样本 provenance 写入 self-hashed manifest。

### 4.4 规模和采集停止规则

先采集，不先拍脑袋规定固定 4,000 条。目标是：

- 至少 2,000 个不同 strict-success train tasks；
- 4,000–8,000 条去重成功轨迹；
- 至少 250,000、最多 600,000 completion-label tokens；
- 每个任务类别和 constraint-count stratum 至少有 100 条轨迹，否则记录 coverage gap。

若 Raw K=8 不能得到 2,000 个成功任务，先提高 train-only test-time sampling 到 K=16 或加入
已审计 human demonstration；不允许回退到 hidden-title oracle 伪造覆盖。

## 5. SFT：闭环保真而不是追求最低 NLL

### 5.1 训练设置

固定 Qwen3.5-4B、相同 compact prompt 和 action schema，LoRA 仍采用 `r=16/alpha=32`。
为降低 M5 的过强覆盖：

- 初始 LR 候选只设 `2e-5` 与 `5e-5`，不再使用 `2e-4`；
- 训练最多 1 epoch，每 10% token exposure 保存 checkpoint；
- 从 M6 train 的 Raw rollout 中另冻结 20,000 个 retention prompt states，按
  home/search-results/item/options/recovery 分层，成功与失败轨迹都覆盖；它们不提供 oracle label；
- 每个 optimizer batch 中 90% 为 strict-success action imitation，10% 为 retention state；
- reference 是冻结 Raw policy；在 retention state 的 action-token distribution 上优化
  `L_SFT + β_KL KL(π_SFT || π_Raw)`；
- `β_KL` 只允许 `{0.01, 0.03}` 两个 dev 候选；
- checkpoint 只按闭环 tuning-dev strict success 选择，NLL 只做诊断。

这是一个 2×2 的小型开发网格，不是正式实验矩阵。候选都只读取 tuning-dev；确定 recipe
后重新用冻结 recipe 训练 3 个正式 SFT seed。

### 5.2 Raw baseline 先冻结

任何 SFT 训练前，先在 M6 tuning-dev roster 上运行 Raw：

```text
500 tuning-dev tasks × K=4
```

冻结 strict success、dense score、partial-match purchase、search exhaustion、action/schema
error、tokens 和 steps。Raw 结果及 bootstrap 代码在查看任何 SFT 闭环结果前自哈希冻结。
promotion gate 和 holdout 的 Raw rollout 此时不运行。

### 5.3 SFT 晋级门禁

候选 recipe 确定后，用它训练 3 个正式 SFT seed。只有在 tuning dev 同时满足以下条件，
才允许继续 RL；这里是开发门禁，不是最终效果结论：

1. paired tuning-dev strict success 相对 Raw **至少 +3.0 pp**；
2. task-cluster bootstrap 95% CI 下界 **> 0**；
3. 3 个正式 SFT seed 均为正差，且 seed 均值满足门槛；
4. search exhaustion ≤ Raw +2.0 pp；
5. partial-match purchase ≤ Raw；
6. schema-invalid 和 action-error trajectory 均不高于 Raw +2.0 pp；
7. tokens/trajectory ≤ Raw ×1.25；
8. 主要 category/constraint strata 中不允许出现超过 5 pp 的系统性退化。

未通过时，按失败类别回到数据层：长查询失败修 query corpus，partial match 修候选比较轨迹，
action error 修公开动作/recovery 覆盖。**禁止为了进入 RL 而降低闭环门槛。**

## 6. RL：Strict-success GRPO + Verifier-TD credit

### 6.1 为什么不继续以 dense task score 做 macro reward

M5 的 RL purchase termination 恢复到 91%–96%，但失败集中为 partial-match purchase；说明
官方 dense score 对“买了但没完全买对”仍给出可优化信号。M6 的 terminal macro reward
因此固定为：

```text
R_terminal = 1[official task_score >= 0.999]
```

dense task score继续记录，但不进入 trajectory-level group ranking，不用于 checkpoint 选择。

### 6.2 Outcome-preserving verifier potential

WebShop verifier 能在不向 policy 泄漏 oracle 的离线 learner 侧判断进展。对动作后的状态
先定义受限的非终局 progress score `ψ(s_t)`：

| 公开阶段 | Potential 组成 | 上限 |
|---|---|---:|
| Search results | 当前公开候选 ASIN 的最高 verifier attribute-match | 0.15 |
| Item page | 当前公开商品相对 goal 的 verifier attribute-match | 0.55 |
| Options | 正确公开 option 已选择比例 | 0.20 |
| Non-terminal maximum | 上述组成封顶 | 0.90 |

只有当前 observation 中已经公开的商品、属性、价格和 option 才能参与 `ψ`；search index 内
存在但页面未展示的商品或属性贡献必须为 0。该值只在 learner/reward service 内计算，不进入
prompt。为了不改变 strict-success 优化目标，
边界被强制覆盖：

```text
Φ(s_0) = 0
Φ(s_t) = ψ(s_t)                    for non-terminal t > 0
Φ(s_T) = R_terminal                for every terminal trajectory
r_t^TD = Φ(s_{t+1}) - Φ(s_t)
```

不做会破坏守恒的 clipping。每条完整轨迹必须逐浮点精度满足：

```text
sum_t r_t^TD = R_terminal
```

因此 strict-success 轨迹的 TD credit 总和为 1，所有失败轨迹——包括先到达高匹配商品页、
最后 partial-match purchase 的轨迹——总和严格为 0。失败轨迹早期的正 progress 会在错误
购买或终止时被负差分抵消，中间 credit 只是重新分配既有 outcome，不会凭空奖励多走几步。

### 6.3 Advantage

每个 task 采样 K=8，提高 strict binary reward 的组内方差。宏观优势为 strict terminal reward
的 group-relative advantage。对长度为 `T_i` 的轨迹，把 TD increment 转成零和的轨迹内
重分配项：

```text
D_i,t^TD = r_i,t^TD - R_i / T_i
A_turn(i,t) = A_strict_macro(i) + λ_TD * D_i,t^TD
sum_t D_i,t^TD = 0
mean_t A_turn(i,t) = A_strict_macro(i)
```

`λ_TD` 在 RL preflight 前固定为 `0.5`。loss 继续按 token mean → turn mean → trajectory
mean → K8 group mean 聚合，因此每条轨迹的 macro 权重不随长度变化，TD 项只在轨迹内部
重分配 credit。对成功轨迹，均匀基线是每 turn `1/T_i`；比均匀进度更有效的转移得到正修正，
冗余或回退得到负修正。对失败轨迹 `R_i=0`，早期正 progress 会被最终错误的负差分完全抵消。
若 strict reward 全同，macro 项为 0，TD 项仍可学习可验证的动作顺序，但任一轨迹都不会获得
净正/负 macro 权重。

正式训练前必须证明：100% 完整轨迹的 telescoping error ≤`1e-8`；成功/失败累计 TD credit
分别精确为 1/0；每条轨迹 `mean(D^TD)` 的绝对值 ≤`1e-8`；进入高 potential 后错误购买
的 final transition 为负 credit；至少 30% turns 得到非零 TD credit。任一项失败即停止。

### 6.4 能力保持与优化稳定性

- reference policy 固定为通过门禁的 SFT；
- PPO-style clipped objective 保留，并加入 adaptive KL 到 SFT reference；目标 observed KL
  在 `[0.005, 0.03]`，越界自动调整 coefficient；
- LR 从 M5 的 `5e-6` 降为 `1e-6`，每 iteration 只做 1 policy epoch；
- 每 50,000 generated action tokens 做一次 tuning-dev eval；
- checkpoint 选择只按 tuning-dev strict success，同时受成本/错误 guardrail 约束；
- patience=2 次 eval；连续两次没有刷新 best 或任一 guardrail 失败即停止；
- behavior/sampling/HF replay parity、真实参数变化和 24h same-root recovery 沿用 M5。

### 6.5 RL 晋级门禁

正式 RL 每 seed 最多 500,000 generated action tokens，但达到 early stop 可以更早结束。最终
checkpoint 必须：

1. paired tuning-dev strict success 相对对应 SFT seed **至少 +3.0 pp**；
2. task-cluster bootstrap 95% CI 下界 >0；
3. 同时高于 Raw tuning-dev baseline；
4. partial-match purchase 不高于 SFT；
5. search exhaustion、schema/action error 不高于 SFT +2 pp；
6. tokens/trajectory 不高于 SFT ×1.15；
7. steps/trajectory 不高于 SFT ×1.10；
8. 3 个 RL seed 方向一致；
9. 至少 20% K8 groups 有 strict mixed reward，至少 30% turns 有非零 TD credit；
10. 每个 iteration loss/gradient 有限、参数真实改变、parity 全通过。

如果 SFT 已经很强导致 strict mixed groups <20%，采用难度 curriculum：从 Raw/SFT pass@8
介于 0.125 和 0.75 的 train tasks 采样，而不是改变奖励或故意降低 SFT。

## 7. 数据切分与双层未见评测

### 7.1 新的 M6 角色

M6 不复用 `[0,500)` 作为正式结论。先生成 `m5_goal_exposure_registry_v1.json`，扫描 M5
corpus、preflight、online、frozen-eval 和报告中的 task/goal identity。设原 eligible train
`[1000,12087)` 的 canonical-instruction groups 为 `U`，其中没有 M5 task-specific exposure
的子集为 `F`。分配顺序必须固定为：先从 `F` 选 promotion 和 holdout，再从剩余 `U` 选
tuning dev，最后其余全部归 train。

| 角色 | 目标数量 | 用途 | 可更新模型 |
|---|---:|---|---:|
| Train | `U − 500 − 2×N_eval`，且至少 2,000 | 轨迹采集、SFT、RL | 是 |
| Tuning dev | 500 | 小型 recipe、checkpoint 和 early stop | 否 |
| Promotion gate | `N_eval` | 7 个身份冻结后的一次性晋级检查 | 否 |
| M6 untouched holdout | `N_eval` | 最终只打开一次 | 否 |

Promotion gate 和 holdout 必须来自 registry 中 `ever_read=false` 的 `F`。历史 M5 task
可进入 M6 train/tuning-dev，因为它们只承担优化和选择角色，不再承担效果证据；其 exposure
flag 仍完整保留。`N_eval` 在任何 M6 训练前
由第 7.4 节的前瞻功效分析冻结，且在 `[1,000, 2,000]` 内。若
`F < 2×N_eval`，或 `|U| − 500 − 2×N_eval < 2,000`，M6 停止并寻找新的合规任务源，
不能缩小既定未见评测或把已暴露任务伪装成新测试。
精确数量以排除重复后的 group 分配为准，最终 roster 和 SHA-256 必须写入机器合同。M5 test、
M6 train、tuning dev、promotion gate 和 holdout 的规范化 instruction 交集必须为零。
由于 promotion/holdout 是从 WebShop 上游 train 区域重切的内部未见集，M6 只声称这条冻结
分布上的 Raw→SFT→RL 因果对比，不把其绝对分数包装成官方 WebShop `[0,500)` test 排名。

### 7.2 Promotion gate 与最终矩阵

SFT recipe、3 个 SFT checkpoint、RL 算法、3 个 RL checkpoint、prompt、采样和分析代码全部
冻结后，先一次性运行 promotion gate。只评测必要身份：

```text
Raw + 3 SFT seeds + 3 RL seeds = 7 identities
7 × N_eval promotion tasks × K=4
```

只有 promotion gate 同时满足本计划的 Raw→SFT 和 SFT→RL 门槛，才打开 untouched holdout，
再运行同样的 `7 × N_eval × K4`。当 `N_eval=1,000` 时每层为 28,000 trajectories；当
`N_eval=2,000` 时为 56,000。promotion 结果不能触发模型、阈值或
分析代码修改；失败则 M6 以失败结束，holdout 保持未见。

不再评测多个 RL 算法。SFT 与 RL 以训练 seed 配对，任务和 rollout seed 共同。两层均报告：

- strict success 主指标；
- dense score 诊断；
- Raw→SFT、SFT→RL、Raw→RL 的 paired task differences；
- training-seed crossed bootstrap 与 exact paired-seed sign test；
- category/constraint slices；
- token、step、wall time、action/schema error；
- partial/zero-match purchase、search exhaustion 和恢复轨迹分类。

### 7.3 成功判定与统计现实

最终“单调提升成功”必须同时满足：

```text
mean(SFT) > Raw
mean(RL) > mean(SFT)
mean(SFT) - Raw >= 3.0 pp
mean(RL) - mean(SFT) >= 3.0 pp
CI_low(SFT - Raw) > 0
CI_low(RL - SFT) > 0
all 3 paired seeds have RL - SFT > 0
all 3 SFT seeds have SFT - Raw > 0
```

3 seed 的精确双侧算法检验最小 `p=0.25`，所以不能同时要求 seed-level `p<0.05`。本轮把
task-cluster CI 作为固定训练 recipe 和 checkpoint 下的性能证据，把 seed 方向一致性作为算法
稳定性要求，并诚实注明训练 seed 数限制。若资源允许，把正式 seed 增至 5；此时精确双侧
检验最小 `p=0.0625`，仍不足 0.05，至少需要 6 个完全同方向 seed 才可能达到 0.03125。

### 7.4 前瞻功效与 `N_eval`

M5 的 Raw 单模型 500-task CI 为 [30.05%, 37.20%]，说明固定 500 个 task 对 3 pp 效应可能
分辨率不足。M6 在生成 split 前，使用 M5 的 `paired_task_differences.csv` 做纯设计仿真：

1. 只读取 task-level paired-difference 分布，不用它选择任何 M6 模型；
2. 对 `N ∈ {1000,1500,2000}`，注入 +3 pp 的最小目标效应，做至少 20,000 次 cluster
   resampling；
3. 选择使 `CI_low>0` 概率至少 80% 的最小 `N`；
4. 分别用 M5 Raw↔SFT、SFT↔RL、Raw↔RL 的方差估计，取最大的 `N`；
5. 写入 self-hashed power report 和 split lock，之后不得根据 M6 结果改变。

若 `N=2,000` 仍不能达到 80% 设计功效，或者 fresh goals 不足，则在训练前停止。不能先训练，
再通过换置信区间算法或降低 3 pp 门槛补救。

## 8. 分阶段执行与停止点

### Phase A：数据与基线（CPU + inference GPU）

1. 生成 M5 goal-exposure registry；
2. 用 M5 paired differences 完成前瞻功效分析，冻结 `N_eval`；
3. 生成 M6 split lock，证明五角色无 instruction overlap；
4. 冻结 Raw tuning-dev baseline；
5. 收集 Raw K8/K16 train success 和 recovery trajectories；
6. 可选审计官方 human demonstrations；
7. 生成 corpus manifest 与 conditional-learnability report。

**停止点 A：** corpus 任一硬门禁失败，不提交 SFT。

### Phase B：小型 SFT recipe search

1. 训练 4 个候选：2 LR × 2 KL；
2. 每 10% exposure 做闭环 tuning-dev；
3. 选择第一个满足门禁、且成本最低的 recipe；
4. 用冻结 recipe 训练 3 个正式 SFT seed。

**停止点 B：** 正式 SFT 未在 tuning dev 形成一致 `SFT > Raw`，不实现/提交正式 RL。

### Phase C：RL preflight

1. K8 strict mixed-signal probe；
2. verifier-TD telescoping、相关性和 anti-partial-match audit；
3. 至少 2 次真实 update、finite loss/gradient、adapter change；
4. 50-task tuning-dev 小评测验证方向，不作为正式效果结论。

**停止点 C：** credit 不满足 outcome-preserving 守恒，或 RL 未在小 tuning-dev probe 上胜
SFT，不提交正式 RL。

### Phase D：正式 RL

3 个 seed，可并行；每 run 单 GPU、8 CPU、32 GiB、24h allocation、same-root resume。
每 50k token 独立 tuning-dev eval，按 strict-success early stop。正式逻辑 run 为 3，不增加
算法矩阵。

**停止点 D：** 3 seed 未在 tuning dev 形成一致 `RL > SFT`，不打开 promotion gate；回到
新的版本化研究，不能继续在同一 dev 上无限调参。

### Phase E：一次性 promotion gate

冻结 7 个身份、prompt、sampling、checkpoint 和分析代码后，一次性打开 `N_eval`-task promotion
gate。通过才进入 Phase F；失败则生成失败报告，不修改模型，holdout 继续封存。

### Phase F：最终 holdout 与报告

一次性打开 `N_eval`-task untouched holdout。无论是否成功都生成最终报告；测试结果不得触发
M6 补训。

## 9. 预计作业与资源

| 阶段 | 逻辑 GPU jobs | 最大并行 | 估计 wall time |
|---|---:|---:|---:|
| Raw train rollout data collection | 2–4 | 4 | 6–18 h |
| Raw tuning-dev baseline | 1 | 1 | 2–4 h |
| SFT recipe candidates | 4 | 4 | 3–8 h |
| Formal SFT seeds | 3 | 3 | 3–8 h |
| RL preflight | 1 | 1 | <1 h |
| Formal RL seeds | 3 | 3 | 12–30 h |
| Promotion-gate identities | 7 | 4 | 12–36 h |
| Frozen holdout identities | 7 | 4 | 12–36 h |

如果阶段门禁一次通过，端到端约 5–8 天；包含一次数据修订约 7–11 天。CPU 数据准备、共享
WebShop service 和分析作业另计，但不应请求过量资源。任何时刻每波最多 4 个 GPU job；
训练 job 默认单 GPU、8 CPU、32 GiB，纯推理评测默认单 GPU、4 CPU、16 GiB，并在提交前
根据实际利用率下调而不是上调。共享 service 单独申请，避免每个 run 重复占用 CPU/内存。

正式作业提交仍遵循：提交前检查参数、数据身份和资源；提交成功后只确认一次 job ID，不自动
轮询。只有用户再次明确要求查看进展时，才执行一次只读状态检查。

## 10. 实现清单

### 10.1 必须新增或修改

- `data/m5_goal_exposure_registry_v1.json`：M5 曾读取 goal 的完整登记；
- `data/m6_webshop_split_v1.json`：新的 train/tuning-dev/promotion/holdout lock；
- `scripts/m6_plan_statistical_power.py`：在训练前冻结 `N_eval`；
- `scripts/m6_collect_policy_success.py`：Raw 成功/恢复轨迹收集；
- `scripts/m6_build_sft_corpus.py`：公开可学习 corpus、去重和 action balance；
- `scripts/m6_audit_conditional_learnability.py`：hidden-field、query provenance 和 replay 审计；
- `scripts/m6_closed_loop_eval.py`：共享 Raw/SFT/RL tuning-dev gate；
- `scripts/m6_sft_train.py`：low-LR、Raw-reference KL、频繁 checkpoint；
- `src/miniwebwork/webshop_rl/verifier_td.py`：potential 与 telescoping credit；
- `scripts/m6_online_rl.py`：K8 strict macro + TD turn advantage + adaptive KL；
- `scripts/m6_analyze_final.py`：单调链统计与失败分类；
- 对应 CPU tests、GPU preflight 和 24h Slurm entrypoints。

### 10.2 直接复用

- M5 WebShop runtime/data locks 和 prompt/action schema；
- public-action whitelist 与 oracle-blind adapter；
- K-group 原子采集、generated-token ledger；
- vLLM behavior / sampling / HF replay parity；
- LoRA/optimizer 原子 checkpoint、same-root recovery；
- 冻结评测的 isolated worktree 和统计框架。

### 10.3 明确删除的非必要工作

- 不建设新算法动物园；
- 不重跑或包装 M5 hidden-title SFT；
- 不把 M5 test 当 M6 tuning-dev；
- 不训练额外 critic 或 LLM reward model；
- 不用 LLM-as-judge 替代官方 verifier；
- 不以 dense score 单独选择 checkpoint；
- 不为了“保证成功”挑最好 seed、隐藏失败 run 或继续窥视 holdout。

## 11. 分阶段准入审查清单

### 11.1 开始数据采集与 SFT 前

- [ ] 用户批准按本计划进入 Phase A/B；
- [ ] M5 exposure registry 完整，promotion/holdout 中 M5-exposed goal 为 0；
- [ ] 前瞻功效报告已冻结，3 pp 目标的预期 `CI_low>0` 概率 ≥80%；
- [ ] M6 split 已冻结，M5 test 与 M6 train/tuning-dev/promotion/holdout 无 instruction overlap；
- [ ] Raw tuning-dev baseline 与分析代码冻结；
- [ ] SFT corpus 100% strict-success replay、0 hidden-title、0 target-ASIN label leak；
- [ ] query provenance、长度、多样性、recovery 和 action balance 全部过门。

### 11.2 开始正式 RL 前

- [ ] 3 个正式 SFT seed 在闭环 tuning-dev 上一致胜 Raw；
- [ ] verifier-TD 对成功/失败严格求和为 1/0，partial-match 失败无净正 credit；
- [ ] RL preflight 有 strict mixed groups、有限更新和真实参数变化；
- [ ] RL 小型闭环 probe 胜对应 SFT，且 partial-match 没有恶化；
- [ ] 正式 RL 的 budget、early stop、KL、资源和 24h recovery 冻结；
- [ ] 用户批准进入 Phase D。

### 11.3 打开 promotion 与最终 holdout 前

- [ ] 3 个正式 RL seed 在 tuning-dev 上一致胜对应 SFT；
- [ ] 7 身份、prompt、checkpoint、sampling、评测与统计代码全部冻结；
- [ ] 用户批准一次性打开 promotion gate；
- [ ] promotion 全部门槛通过后，用户批准一次性打开最终 holdout。

任何阶段在对应条件完成前都不能越级提交；最终 holdout 通过前，不能宣称 Raw → SFT → RL
已经实现。

## 12. 决策摘要

M6 不通过“更多 GPU、更多算法或更长训练”修复 M5。核心改动只有三个：

1. **把 SFT 标签改成 policy-visible 的真实成功行为；**
2. **把每一阶段的晋级条件改成闭环 strict success 必须胜前一阶段；**
3. **把 RL 的宏观目标改成 strict success，并用可 telescoping 的 verifier-TD 只重分配进展。**

这条方案最可能得到真实的 Raw → SFT → RL 单调提升，也最能解释为什么成功；如果仍失败，
门禁会精确指出是数据覆盖、SFT 保真、strict 信号稀疏还是 turn credit 不一致，而不是再得到
一批无法归因的训练结果。

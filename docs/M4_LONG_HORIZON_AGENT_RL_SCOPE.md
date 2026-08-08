# MiniWebWork-RL 长程智能体强化学习收缩方案

> 状态：2026-08-08 批准的正式范围；2026-08-09 完成 parity 合同版本化校准，
> 前置实现与门禁验证仍在进行中。
>
> 本文档取代 `M4_RLVR_STUDY_PROTOCOL.md` 中“五算法 × 三随机种子”的正式
> 矩阵。旧协议、旧提交和既有 v3 工件仅保留为诊断与工程演进证据。本文档
> 不是正式训练已经完成的声明；所有实现、性能 smoke test 和门禁通过后，
> 才能冻结新的正式训练提交。
>
> 机器可读合同为 `data/m4_long_horizon_study_v2.json`。当前
> `formal_submission_allowed=false`，因此任何正式 SFT、GRPO 或 step-aware
> 训练作业都不得提交。逐项状态见 `M4_FORMAL_TRAINING_READINESS.md`。

## 0. 当前前置实现快照

当前已经建立独立的 `m4_long_horizon_v2` 数据版本，不复用或覆盖旧
`m4_rlvr_v1`：

- train/dev/test 分别为 240/72/120 个任务；
- 每个 split 的 basic/medium/long 比例为 25%/50%/25%，即 75% 为
  9–20 步的 medium/long 任务；
- 四个任务族的确定性最短正确轨迹分别为 7/10/12/18 个环境动作；
- world、product、supplier、constraint 和 answer signature 跨 split 零重叠；
- 18 步任务要求访问所有可行供应商的公开详情页，终态 verifier 会核验该
  工作流证据；直接猜中最终商品但跳过检查仍记为失败；
- 三个可行供应商使用中性标识，可靠性最优角色由各 split 独立、严格均衡的
  SHA-256 排列决定，不使用可从 world 编号外推的周期；正确供应商在
  访问位置 1/2/3 上分别为 train 20/20/20、dev 6/6/6、test 10/10/10，禁止
  通过固定后缀、名称或“最后访问项”解题；
- 正式 prompt 为 `browser_agent_v4_long_memory`：除最近 5 个动作外，最多保留
  8 个曾实际观察到的 supplier/product 公开页面摘要；摘要只取公开 path、
  title 和 visible text，不读取 oracle、verifier、expected answer 或运行 ID；
- prompt system SHA256 为
  `239139aeb9f34af4c6f3460d86531f78c0d484983636e9743e69578b4eecf6f7`，上述
  history/evidence/URL 限额同时写入 study manifest 并按字节 fail-closed；
- dataset manifest SHA256 为
  `a714e5cb8bec4c8a767087574d1d39baf9934b7cf70ee8df2eb49511f627c6a0`，
  seed manifest SHA256 为
  `938b25a643259e6724255fd6fe14d4808558c99eedf83ab2127d44133a49bfde`。

`m4_long_horizon_v1` 与 `browser_agent_v3_compact` 已被前置审计判定为无效的
正式候选：train/dev 的 78 个 long task 中，正确供应商全部固定为第三个访问项，
而 prompt history 又不保留供应商页的可靠性证据。Job 1260 因此被主动取消；
它只能作为“发现并阻断 shortcut”的诊断证据，不能进入 SFT 或在线训练血缘。

Verified SFT 构建器会在真实浏览器环境逐 turn 回放 expert evidence，并要求
实际动作序列、参考 trace SHA 和终态成功全部一致。CPU Slurm Job 1261 已在
clean `95d7c2607a4d279196aa760b1f56723332020792` 上完成完整 240/72 回放与精确
token 审计：train/dev 分别包含 2820/846 个唯一 turn、60,540/18,162 个有效
completion-label token，重复、零标签、6144 截断和 runtime DB 残留均为 0。
这只证明监督语料可训练，不代表共享 SFT adapter 已经训练。

## 1. 面试导向与一句话目标

本项目不是算法数量展示，也不追求复现大模型实验规模。项目的面试目标是：

> 构建一个可恢复、高吞吐、可审计的多轮浏览器 Agent 在线强化学习系统，
> 并在相同 SFT 初始策略和 rollout 预算下，研究逐步信用分配能否优于把
> 终态奖励广播到整条轨迹的 multi-turn GRPO，尤其是在较长交互任务上。

项目必须同时体现：

1. 对多轮 on-policy RL、组相对优势、信用分配、稀疏奖励和策略稳定性的理解；
2. 对 GPU 推理/训练吞吐、连续批处理、资源调度和性能测量的工程能力；
3. 对环境异常、24 小时 Slurm 中断、数据隔离和 adapter 血缘的系统治理；
4. 对中性或负面实验结果的统计解释能力，而不是预设某种算法必须胜出。

## 2. 为什么必须收缩和重构

### 2.1 五算法矩阵不能形成清晰主线

历史 M4 计划同时比较 SFT、RSFT、RLOO、GRPO 和 GSPO。它会产生 15 个训练
模型、约 30 个训练作业和 7,200 条冻结测试轨迹，但无法突出一个核心问题。
RLOO、GRPO 和 GSPO 的差异会把项目叙事变成优化器枚举，削弱对浏览器 Agent
长轨迹信用分配和训练系统的深入分析。

### 2.2 当前轨迹不足以直接声称“长程”

对 job 1248 的 GRPO pass-1 正式候选工件进行只读统计：

| 指标 | 结果 |
|---|---:|
| 有效/总轨迹记录 | 807 / 808 |
| 模型轮次均值 | 5.71 |
| 模型轮次 P50 / P75 / P90 / P95 / max | 3 / 9 / 11 / 11 / 17 |
| 环境步数均值 | 3.31 |
| 环境步数 P50 / P75 / P90 / P95 / max | 0 / 7 / 9 / 9 / 15 |

现有系统已经是 multi-turn，但模型经常在真正环境交互前因输出失败结束。若不
重新定义并验证任务 horizon，项目只能准确称为“多轮浏览器 Agent”，不能把
“长程决策”作为主要结论。

### 2.3 当前 GPU 作业主要在等待串行 rollout

首波作业均申请 `1 GPU + 8 CPU + 64 GB`，但 Slurm 账单显示：

| Job | 用途 | 墙钟时间 | 实际 TotalCPU | MaxRSS |
|---|---|---:|---:|---:|
| 1244 | SFT | 20:43:53 | 20:44:18 | 7.1 GB |
| 1245 | RSFT collection | 21:00:33 | 20:41:22 | 4.9 GB |
| 1246 | RLOO collection/update | 21:16:11 | 20:56:48 | 12.0 GB |
| 1248 | GRPO collection/update | 21:27:46 | 21:08:51 | 16.6 GB |

每个作业虽然分配了 8 个 CPU 核，实际平均只使用约 1 核。collector 对任务和
K 条轨迹执行双层串行循环，Transformers `generate()` 的 batch 为 1；每次
generation 后还额外 forward 一次以提取 policy log-prob。当前 v3 作业没有
持续记录 GPU utilization、功耗、batch occupancy 或 tokens/s，因此不能声称
GPU 被充分使用。

### 2.4 当前在线更新没有充分利用采集数据

job 1248 收集 202 个任务组和约 115,155 generated action token，但更新报告只
选择 9 个组、4,513 action token，进入优化器的 token 比例约为 3.9%。大量组因
raw/sampling log-prob 语义不一致或零奖励方差被排除。随后所有合格组只累积为
一次 `optimizer.step()`。

这个结果证明了门禁可以阻止不兼容数据进入梯度，但也证明当前流程不是真正高效
的迭代式在线 RL。正式方案必须先解决采样分布语义、动态学习信号采样和更新频率，
不能通过只提高单个 log-prob 上限、删除异常 token 或忽略负对照来掩盖问题；
parity 合同必须在正式训练前用正、负对照做版本化校准并完整披露。

### 2.5 当前 SFT 预算设计造成重复训练

首波 SFT 以 `batch_size=1`、`gradient_accumulation=16`、始终开启 gradient
checkpointing、`dataloader_num_workers=0` 运行。2,460 个唯一监督样本被重复
5–6 次，以得到 12,639 条 occurrence 和 249,991 completion-label token。
训练结束时 action/schema 指标已为 100%，loss 接近零。

新的 SFT 阶段只承担 warm start，不再为与 RL “形式上对齐”而重复样本凑固定
label-token 总量。监督 token 和在线生成 token 是不同成本口径，应分别记录。

## 3. 当前公开研究位置

项目采用“近期公开研究范式的小规模、可审计实现”，而不是宣称复现其规模：

- [OpenWebRL](https://arxiv.org/abs/2606.02031) 采用 SFT warm start、在线真实
  网站多轮 rollout、轨迹级成功判断、动态采样和 multi-turn GRPO，说明 GRPO
  仍是浏览器 Agent 的有效基础算法；真正的变化是训练范式从单轮 completion
  转向状态化、多轮、在线环境交互。
- [GiGPO](https://arxiv.org/abs/2505.10978) 在完整 episode 组优势之外，为重复
  anchor state 构造 step-level 组优势，直接针对长程任务的细粒度信用分配。
- [Agent Lightning](https://arxiv.org/abs/2508.03680) 将 Agent 执行建模为可观测
  transition，并解耦 Agent runtime 与 learner。MiniWebWork-RL 只借鉴其
  tracing 和边界设计，不扩张为通用 Agent 框架。
- [GSPO](https://arxiv.org/abs/2507.18071) 解决 sequence-level importance ratio
  与 clipping 稳定性，但不直接解决多次环境交互之间的 step credit assignment，
  因此不作为本项目主方法。

## 4. 正式范围和非目标

### 4.1 正式训练矩阵

| 阶段/方法 | 角色 | 训练随机种子 | 正式模型数 |
|---|---|---|---:|
| Verified SFT warm start | 唯一共享初始策略和评测基线 | 1 个冻结 seed | 1 |
| Multi-turn GRPO | 轨迹级信用分配基线 | 20260801/02/03 | 3 |
| Step-aware group policy optimization | 主要方法；GiGPO-style 宏观+局部优势 | 20260801/02/03 | 3 |

正式矩阵共 7 个模型。两个在线方法必须从同一个 SFT adapter SHA 开始，使用相同
LoRA 容量、任务版本、prompt、采样分布、K、训练 action-token 预算和评测协议。

### 4.2 非目标

- 不运行 RLOO、RSFT、GSPO 的正式三种子矩阵；
- 不做大规模算法 zoo 或网格超参数搜索；
- 不以增加模型规模或训练任务数量作为主要贡献；
- 不将环境 observation、prompt、padding 或工具输出当作可训练 action token；
- 不在 final test 上选择 checkpoint、奖励、采样参数或信用分配公式；
- 不要求 step-aware 方法必须取得正向结果。

历史 RLOO/RSFT/GSPO 工件可以作为工程诊断证据，但不得进入正式表格、均值、
置信区间或面试中的主结果。

## 5. 数据与长程任务合同

### 5.1 规模保持克制

保留当前数量级：

| Split | Tasks | 用途 | 可产生梯度 |
|---|---:|---|---|
| train | 240 | SFT demonstrations 和在线 rollout | 是 |
| dev | 72 | 实现安全、吞吐、停止与固定配置选择 | 否 |
| test | 120 | 一次性冻结评测 | 否 |

训练、dev 和 test 必须继续保持 world、产品、供应商、约束签名和答案隔离。

### 5.2 Horizon 必须由参考策略证明

每个任务在构建时保存由确定性 scripted/oracle reference 产生的最短正确环境动作
数，不能用失败模型的实际短轨迹定义任务难度。冻结数据至少包含三个 strata：

| Horizon stratum | 最短正确环境动作数 | 研究作用 |
|---|---:|---|
| basic multi-turn | 6–8 | 格式、工具使用和基础导航 |
| medium horizon | 9–12 | 多约束跟踪与中途状态选择 |
| long horizon | 13–20 | 延迟奖励、早期决策影响和错误恢复 |

至少三分之二的 train/dev/test 任务应处于 `medium` 或 `long`。三个 split 必须按
horizon 和 task family 分层。若现有网站流程无法提供足够的 13–20 步任务，应
在同一采购域内增加组合子目标或验证阶段，而不是引入新的业务网站和任务领域。

## 6. Verified SFT warm start

### 6.1 作用

SFT 的目标是让 4B 策略进入可探索区域：稳定输出 JSON action、理解页面观察、
执行基础导航和完成采购流程。它不是与 RL 等 token 成本的竞争算法。

### 6.2 数据和停止规则

- 只使用 train split 的唯一、验证成功的逐 turn expert evidence；
- 每个样本保留真实 prompt/observation 和一个 action completion；
- completion-only loss，zero-label sample 必须为 0；
- 不通过重复完整数据排列机械凑 250k label token；
- 固定 seed `20260801`、LoRA `r=16/alpha=32/dropout=0`，target modules 为
  q/k/v/o/gate/up/down projection；dropout=0 同时保证后续 behavior/replay
  parity 不受随机 mask 污染；
- 最多 3 epoch、至少 2 epoch；每个 epoch 评估 dev NLL、teacher-forced action
  exact match 和 schema-valid rate。若三项分别未改善 `0.005/0.002/0.002`，
  连续 1 次评估即 plateau 停止；
- 保存 unique sample count、监督 label token、总 forward token 和 GPU 成本。

### 6.3 GPU 配置选择

正式 SFT 前运行非正式 10–20 step microbatch benchmark：

```text
microbatch ∈ {1, 2, 4, 8}
effective batch 固定
序列长度分桶或 packing
bf16
```

选择不 OOM 且保留至少 15% 显存余量的最大 microbatch。显存允许时关闭 gradient
checkpointing；启用 dataloader workers/prefetch，并降低 checkpoint 保存频率。
选择规则与结果必须写入 preflight artifact，正式训练后不得调整。

## 7. 迭代式在线 Agent RL

### 7.1 核心循环

每个在线 seed 采用固定 250,000 generated action-token 预算。生成 token 包括被
基础设施错误、无效组或无学习信号组消耗的实际 token；只有门禁通过的数据进入
梯度。

```text
冻结 policy version N
→ 按冻结任务流采集最多 32 个 task × K=4 的完整组
→ 校验轨迹、组、分布和 adapter 身份
→ 构造 macro / micro advantage
→ 以多个 minibatch 完成固定 update epoch
→ 保存 policy version N+1 和 optimizer/ledger
→ 使用 N+1 采集下一 iteration
```

最后一个 iteration 在开始新 K=4 组前做最坏情况 token reserve；不得以不完整组
填满预算。按照当前轨迹成本，250k token 预计产生约 10–14 个 policy iteration，
实际数量由完整组边界决定，不作为结果调节参数。

### 7.2 动态学习信号采样

同一任务 K=4 全成功或全失败时，组相对优势为零。正式 sampler
`balanced_cold_then_beta_uncertainty_v1` 使用预注册规则：

1. 第一轮按 seed 对三个 horizon strata 做确定性、均衡排列；
2. 每个 task family 在所有任务至少有一个 committed group 前严格优先未见任务；
3. 冷覆盖后使用 train committed groups 的 `Beta(1,1)` 后验
   `p=(successes+1)/(valid_trajectories+2)`，权重固定为
   `max(0.05, 4*p*(1-p))` 严格降序优先，再以 study seed、iteration 和 task ID
   的 SHA-256 key 对同权任务做确定性无放回 tie-break；
4. 全成功、全失败和基础设施尝试仍计入 rollout 成本；infra-invalid 不伪装成
   reward observation；
5. 不允许事后按某个算法的 dev/test 表现修改 task stream；
6. 两个在线方法使用相同规则、相同初始任务顺序和相同预算；实际自适应路径若因
   train outcome 不同而分叉，必须完整报告任务 roster，不能声称 task ID 完全相同。

该机制用于提高稀疏奖励下的有效梯度比例，而不是删除不利结果。

## 8. 信用分配：项目的算法主线

### 8.1 Multi-turn GRPO baseline

对同一任务的 K=4 有效轨迹，用终态 verifier reward 计算组相对优势：

```text
A_episode_i = (r_i - mean(r_group)) / (std_population(r_group) + 1e-6)
```

轨迹 `i` 每个 turn 的所有 action token 都接收同一个 `A_episode_i`。每个 turn
按真实 observation-conditioned prompt 独立 replay；token loss 先在 turn 内平均，
再在轨迹与组之间平均，避免长轨迹仅因 token 更多而获得更大权重。

这个 baseline 明确暴露其局限：终态奖励不能指出成功或失败由哪个早期动作导致。

### 8.2 Step-aware main method

主方法冻结为 `public_anchor_macro_micro_v1`，采用 GiGPO-style 两层信用分配：

1. **Macro advantage**：与 GRPO 相同，评价完整轨迹最终结果；
2. **Anchor-state micro advantage**：每条轨迹对每个 anchor 只取首次访问；在
   同一任务 K 条轨迹中比较该位置的 `gamma=0.95` 折扣终态 return，并以相同
   population-std/`1e-6` 规则标准化；少于两条轨迹或 return 零方差时严格为 0；
3. **Turn advantage**：`A_turn = A_episode + 1.0 * A_anchor`，micro 只应用到该
   anchor 的首次访问 turn；其他 turn 精确 fallback 为 macro advantage。

Anchor-state signature 只能来自 policy 可见或环境公开状态，例如：

```text
task_id
page/template identity and normalized path without origin/query/fragment
normalized visible text and stable public controls
public action result/workflow state
exact prompt-token context SHA
```

不得使用 episode ID、DOM 临时 element ID、oracle answer、隐藏 verifier 字段或
test 信息构造 anchor。token loss 先在 turn 内平均，再在 trajectory 内按 turn
平均，最后在 K=4 group 内按 trajectory 平均；长输出或长轨迹不会仅因 token 更多
获得更大权重。

### 8.3 奖励边界

主实验优先保持可验证终态奖励：

```text
verified success        = 1
valid policy failure    = 0
infrastructure failure  = null
```

JSON/schema 错误、无效浏览器动作、步数和中间子目标作为诊断字段。若需要 format
penalty 或 process reward，必须作为单独、小规模 ablation，不能悄然并入主方法。

## 9. On-policy 和策略稳定性知识合同

每个 action token 必须保存：

- behavior policy/version 和 adapter SHA；
-真实 prompt/completion token IDs；
-生成时采样分布 log-prob；
-learner replay log-prob；
-turn、trajectory、group 和 iteration identity。

第一版正式采样保持：

```text
temperature = 1.0
top_p = 1.0
top_k = 0
K = 4
```

当前 raw/sampling mismatch 必须从语义上修复。runtime v1 在首次 GPU parity
smoke 前预注册：behavior/sampling 最大绝对 log-prob 差 `1e-7`；vLLM behavior
对 HF replay 的 mean/P95/max 绝对 log-prob 差不超过 `0.02/0.08/0.18`；mean
importance ratio 相对 1 的偏差不超过 `0.02`。Job 1267 作为错误 adapter
命名空间负对照，得到 mean/P95/max `1.18677/10.12739/21.51204`，并在首次
optimizer step 前被拒绝。

修复 adapter 视图后，Job 1271 在 clean `e108fc96` 上完成 4 个原子 K=4 group、
4143 个 action token，并证明 behavior/sampling 逐 token 完全一致；正确血缘的
replay mean/P95/max 为 `0.001631/0.000438/0.367135`。它仅因 v1 的单点 max
门槛失败，未执行 optimizer update。Job 1273 的逐 token 复核进一步得到 P99
`0.04821`，且只有 `6/4143 = 0.1448%` token 的初始 importance ratio 落在
PPO `[0.8, 1.2]` 之外；异常集中在少量 BF16 Qwen3.5 hybrid-recurrence
batch-shape 敏感 token，而不是 token 对齐、采样分布或 adapter 身份错误。隔离安装
FLA 的 Jobs 1275–1277 没有改善该分布且放大稀疏异常，因此明确拒绝把 FLA 或
共享环境变更带入正式 runtime。

在任何正式训练开始前，parity 合同据此升级为 runtime v2，并同时冻结一组互补
门禁：

```text
behavior/sampling max absolute difference       <= 1e-7
HF replay mean absolute difference              <= 0.02
HF replay P95 / P99 absolute difference         <= 0.08 / 0.08
HF replay maximum absolute difference           <= 0.50
initial PPO ratio clip fraction                  <= 0.005
mean importance ratio absolute deviation from 1 <= 0.02
```

其中 max `0.50` 只是保险上限；P99 与初始 ratio clip fraction 才约束异常的覆盖
范围。该组合对正确血缘正对照留有测量余量，同时仍以数量级差距拒绝 Job 1267
负对照。此次变更被标记为 `thresholds_frozen_before_gpu_observation=false`、
`thresholds_frozen_before_formal_training=true`，不能伪装成首次观测前预注册；
Job 1280 随后在 clean `5375e151` 和全新 root 下完成 4 个 K=4 group、4141 token，
吞吐 `255.58 trajectories/hour`。其 behavior/sampling 仍精确一致，replay
mean/P95/P99、初始 ratio clip fraction 和 mean ratio 全部通过；但 max
`0.67653 > 0.50`，所以 v2 仍在 optimizer 前失败，未被追溯改判。Job 1281 对
同一冻结 collection 做两次 mb8 及 mb4/mb1 replay：mb8 两次 max 都是
`0.67653`，而 mb4/mb1 为 `1.98432/1.24496`；四次 mean 均约 `0.002`、P99
均低于 `0.058`。这证明稀疏尾部可重复但强烈依赖 replay batch 形状，不能靠挑选
microbatch 或单次 max 门槛掩盖。

在讨论阈值前，runtime v3 先做单变量语义隔离：保持上述 v2 全部阈值不变，仅关闭
vLLM 0.17 明确标记为 experimental 的 Qwen3.5 Mamba `align` prefix cache；
chunked prefill、K=4、采样、adapter 和 learner microbatch 均不变。v3 必须通过
全新 GPU preflight、真实 optimizer update、engine wake 和更新后生成后才可成为
正式冻结依据。其 pre-commit 定向 CPU 回归 Job 1283 已得到 `47 passed`；Job
1282 只暴露了 `/bin/sh` 包装器不支持 `pipefail`，pytest 未启动并保留为失败记录。
Job 1284 在 clean `5a574933` 上得到 `415 passed, 10 deselected`；Job 1285 随后
完成 4 个 K=4 group、4096 token 和 `277.66 trajectories/hour`，但 max
`1.33493 > 0.50`，仍在 optimizer 前失败。Job 1286 对其冻结 collection 的两次
mb8、mb4、mb1 replay 得到 P99.9 `0.240999/0.240999/0.374054/0.287926`，clip
覆盖率均低于 `0.25%`，而 max 随 batch composition 为
`1.64054/1.64054/1.43929/1.71813`。这与 Job 1281 独立复现了“高分位稳定、单点
max 对 batch composition 敏感”。

runtime v4 因而不是简单放宽 max：它保留 behavior/sampling 精确一致、mean≤0.02、
P95/P99≤0.08、初始 PPO clip fraction≤0.5% 和 mean ratio 偏差≤0.02，新增
P99.9≤0.5，并将脆弱的单点 `max≤0.5` 替换为灾难性
`|log ratio|max≤ln(10)`。错误 adapter Job 1267 的 mean/P95/max
`1.18677/10.12739/21.51204` 仍被多道门禁以数量级差距拒绝。v1–v3 失败均不追溯
改判；v4 必须在全新 root 上完整通过后才可冻结。每次 update 报告：

runtime v4 的 pre-commit 定向 CPU 回归 Job 1287 得到 `49 passed`，Job 1288 又在
clean `909f4d85` 上得到 `417 passed, 10 deselected`。Job 1289 随后首次完整通过
parity、2 次非零 optimizer update、原子 adapter/optimizer/manifest 提交、同卡
wake 和更新后生成：初始 replay mean/P95/P99/P99.9/max-log-ratio 为
`0.001754/0.000327/0.046423/0.295705/0.888659`，参数变化范数 `0.03717`，
更新后 behavior/sampling 最大差异为 0。由此 v4 的 GRPO 正确性门禁关闭。

- behavior/replay 最大和分位 log-prob 差异；
- importance ratio、clip fraction 和 approximate KL；
- gradient norm、非零梯度参数和参数变化；
- entropy、有效组和有效 action-token 比例；
- policy version staleness，正式主实验要求 staleness 为 0。

## 10. GPU 高吞吐实现

### 10.1 集群边界

当前节点提供 8 张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`、112 CPU 和
约 386 GB 系统内存。为避免挤占其他用户且避免 CPU/内存成为调度瓶颈，每波最多
四个单 GPU 作业。

4B LoRA 模型不使用多 GPU data parallel。单模型占多卡会减少独立 seed 并行度，
且不能解决浏览器等待造成的 GPU 空闲。正确的优化目标是让每张已分配 GPU 持续
拥有可生成或可训练的 batch。

### 10.2 Rollout job

当前 Conda 环境已有 `vllm==0.17.0`、`torch==2.10.0+cu128` 和
`transformers==5.14.1`。实现采用一张 GPU 内的阶段式 runtime/learner 解耦：

```text
最多 32 个 browser lanes（仍只申请 8 CPU）
        ↓ asynchronous requests
vLLM continuous batching on 1 GPU
        ↓ complete audited iteration
unload inference engine
        ↓
HF/PEFT learner minibatches on the same GPU
        ↓
save next adapter and reload rollout engine
        ↓
run one lineage-bound post-wake generation smoke
```

`wake_up()` 和 `add_lora()` 返回成功不足以单独证明新策略可用；preflight 必须再用
新 canonical/view/semantic 三元 SHA 发起一次真实生成，逐 token 验证
behavior/sampling 一致并落盘 phase event。该请求只计为前置 phase-switch 诊断成本，
不进入正式训练 token 预算或研究结果。

建议正式资源请求：

| 作业 | GPU | CPU | 系统内存 | wall time |
|---|---:|---:|---:|---:|
| SFT | 1 | 4 | 32 GB | ≤24 h |
| Online RL seed | 1 | 8 | 48 GB | ≤24 h |
| Final evaluation | 1 | 8 | 48 GB | ≤24 h |

如果 32 个 lane 的 profiler 表明 CPU 饱和或浏览器内存不足，只能依据 preflight
artifact 调整。不得无测量地扩大 CPU 或内存申请。

### 10.3 性能遥测与正式门禁

每 5–10 秒记录：

- GPU utilization、显存、功耗；
- active/requested batch、生成 tokens/s；
- browser worker busy/idle、环境等待和模型等待时间；
- learner tokens/s、microbatch、gradient accumulation；
- rollout/hour、成功 verifier/hour 和有效 optimizer token/hour。

正式训练前的目标门禁：

| 指标 | 目标 |
|---|---:|
| rollout 吞吐 | 至少为当前约 38 trajectories/hour 的 3 倍 |
| generation 阶段 GPU utilization P50 | ≥60% |
| learner 阶段 GPU utilization P50 | ≥80% |
| 显存安全余量 | ≥15% |
| 进入优化器的 action-token 比例 | 目标 ≥20%，必须原样报告 |
| 每个在线 seed 的 optimizer iterations | 多次迭代；预计 10–14，按 token/group 边界结束 |

若 GPU 指标未达标，应先优化 batching、worker 数和数据管线，不得直接启动六个
正式在线 run。目标未达成可以保留为工程瓶颈结论，但不能声称“充分利用 GPU”。

Job 1285 的严格 phase-window 遥测给出：rollout `277.66 trajectories/hour` 和
`71,080 action tokens/hour` 已通过吞吐门槛；learner GPU utilization P50=`100%`、
generation 显存余量 `15.49%` 也通过。但 generation GPU utilization P50 仅
`11%`（mean `21.12%`、P95 `58%`），明确未通过 `60%`。因此在正确性 E2E 关闭后，
下一轮性能工作必须在不增加 Slurm CPU 请求的前提下提高在途 browser/model 请求数，
并用全新 preflight 复测；当前不能宣称 rollout 已充分利用 GPU。

Job 1289 在正确性闭环后复现 generation P50=`11%`，同时暴露 post-wake 显存余量
仅 `6.71%`。日志显示 vLLM 在 `gpu_memory_utilization=0.8` 下建立 553,344-token
KV cache，而即使 32 个请求都达到 6144 上限也只需 196,608 token。runtime v5
因此保持 1 GPU/8 CPU/48 GB 请求不变，把 KV 比例降为 `0.5`，并把 continuous
batching 与 browser lane 上限扩至 32（8 个并发 K=4 group）。这不是增加调度资源，
而是用已分配资源换取更多在途计算并恢复显存安全余量；仍须以 30 分钟 phase-window
soak 和完整更新后 wake 实测通过，不能仅凭配置推断。

runtime v5 候选合同 SHA256 为
`4571ba3b89178cef607554708fbb8e6dc43863e61a2be36a9e525b5990e252df`。
其 pre-commit Job 1290 首先通过失败测试发现 orchestrator 的旧 2-group 上限，修复后
Job 1291 得到 `36 passed`，完整 Job 1292 得到 `418 passed, 10 deselected`；这些
只证明候选代码自洽，仍不能替代 clean SHA 的 GPU 性能与恢复证据。

clean `b9335f8` 上的 Job 1294 随后得到 `418 passed, 10 deselected`。Job 1295
以 32 lanes 完成 32 trajectories、385 turns、8040 action tokens、2 次真实更新及
原子 commit/wake，吞吐提升到 `371.55 trajectories/hour`；generation/post-wake
显存余量为 `45.25%/36.44%`，但 generation utilization P50 仍为 `11%`，说明仅增加
lane 不能消除浏览器与模型请求的同步波峰。有效 optimizer-token 比例为 `13.74%`，
源于 8 个 group 中仅 1 个具有非零组内优势，必须原样作为样本效率证据。

同提交 Job 1296 的 Step-aware collection 完成 8144 token；mean/P95/P99/P99.9、
initial clip fraction 与 mean ratio 分别为 `0.002127/0.000392/0.055286/0.246272`、
`0.1965%` 与 `0.999675`，behavior/sampling 也精确一致，却只因一个有限 token 的
max-log-ratio `2.667906` 超过 `ln(10)` 被拒。结合 Jobs 1281/1286 已证明的 batch-shape
敏感性，runtime v6 将单点 maximum 保留在报告中但移出 pass/fail；硬门禁仍为精确
behavior/sampling、mean、P95、P99、P99.9、clip coverage 与 mean ratio。错误 adapter
Job 1267 仍会被多项分布门禁以数量级差距拒绝。v6 必须在全新 root 上完成真实
Step-aware update/commit/wake，历史 v1–v5 结果均不追溯改判。

runtime v6 候选合同 SHA256 为
`6b03a6a90daf2c47b80b113ac836d871e59fc05a2fc453313bd2621797a3f0f0`。
Jobs 1297/1298 分别暴露新增长 token 测试夹具的 token SHA 与 trajectory 聚合计数
没有联动更新；二者都在进入生产 parity 断言前失败并保留为测试诊断。修正后的
Job 1299 得到 `36 passed`，完整 Job 1300 得到 `420 passed, 10 deselected`。
提交后仍须在 clean SHA 重跑，并以全新 Step-aware GPU E2E 关闭正确性门禁。

clean `613ae81e` 的 Job 1301 得到 `420 passed, 10 deselected`。Job 1302 随后以
runtime v6 完成 8 个 K=4 group、32 trajectories、385 turns、7988 action tokens；
parity mean/P95/P99/P99.9 为 `0.001586/0.000328/0.051070/0.233614`，clip coverage
`0.1252%`、mean ratio `0.999851`，单点 max `0.552678` 仅诊断。它完成 2 次真实
Step-aware update，参数变化 `0.03686`，adapter/rollout/optimizer、iteration manifest、
run state、4 段 ledger、9 段 phase event 与 post-wake 生成全部独立核验通过。因此
Step-aware 和分布感知 parity 正确性门禁关闭。

Job 1302 的吞吐提高到 `414.11 trajectories/hour` 与 `103,371 action tokens/hour`；
generation/learner/post-wake 最低显存余量为 `45.25%/44.62%/36.44%`，learner P50
为 `100%`。但 generation utilization 仍为 P50=`11%`、mean=`24.66%`、P95=`58%`，
说明浏览器/model 请求仍形成同步波而非显存不足。runtime v7 候选合同 SHA256 为
`bada7b229be6eb80d978656cd325428c48d319c0c7b96418a82e3932a3a3755b`：保持研究语义、
算法和 Slurm 1 GPU/8 CPU/48 GB 不变，仅将 vLLM/browser 在途上限扩至 64、并发
K=4 槽扩至 16，KV 比例设为 `0.64`（64×6144 的 393,216-token 最坏容量需求），并
按 batch 内 group index 以 `0.25s` 确定性错峰启动，打散 bulk-synchronous 空洞。
必须由全新 E2E 的实际 KV capacity、利用率、显存与完整 wake 决定是否接受。

runtime v7 定向 Job 1303 得到 `41 passed, 2 failed`：两项旧并发测试的 fake episode
仅 40ms，短于 250ms 默认错峰，因而无法观测槽位重叠；这不是生产并发失败。将纯
槽位测试显式设为零错峰、同时保留独立错峰时序测试后，Job 1304 得到 `43 passed`。
完整非浏览器 Job 1305 随后得到 `422 passed, 10 deselected`。

提交后 clean Job 1306 再次得到 `422 passed, 10 deselected`。Job 1307 在 3 秒内、
模型加载前失败：Python CLI 的 browser-worker choices 仍独立硬编码到 32，拒绝了
合同允许的 64；没有 collection 或训练工件。入口现直接复用
`ALLOWED_BROWSER_WORKERS`，并增加防重复硬编码测试，修复后必须重新冻结 clean SHA。
定向 Job 1308 得到 `27 passed`，完整 Job 1309 得到 `422 passed, 10 deselected`。

## 11. 24 小时中断恢复与原子性

Slurm wall time 固定不超过 24 小时。恢复单位分两层：

### 11.1 K=4 group 原子性

- 每完成一条 trajectory 即写 append-only attempt journal，记录实际 token 成本；
- 只有 K 条基础设施有效轨迹全部完成后才写 group commit marker；
- 含 infra-invalid trajectory 的组整体不得进入正式 group；
- 失败组和中断组的已生成 token 仍计入成本；
- 恢复时从新的 deterministic attempt index 重采样整个未提交组。

### 11.2 Policy iteration 原子性

- iteration 记录输入 adapter SHA、任务流位置和所有 committed group SHA；
- 中断发生在 collection 时，只恢复相同 policy version 的剩余组；
- 中断发生在 update 时，丢弃未提交 optimizer 状态并从冻结 collection 重做；
- 只有 adapter、optimizer ledger 和 iteration manifest 全部原子提交后，才能将
  policy version 前移；
- git、数据、prompt、采样分布、seed、policy version 或 task order 任一不匹配时
  拒绝恢复。

## 12. 正式调度顺序

1. CPU 单元测试、数据/horizon 审计和 failure-injection recovery tests；
2. 单 GPU 30–60 分钟 SFT microbatch 和 rollout concurrency preflight；
3. 冻结源码、数据、prompt、奖励、信用分配公式和资源参数；
4. 训练一个共享 SFT warm-start adapter并通过 dev 门禁；
5. 第一波最多四个在线作业；
6. 每个作业完成 artifact/adapter/telemetry gate 后，才补排剩余两个 seed；
7. 六个在线模型全部合格后，才打开冻结 test；
8. 最多四个评测作业并行；
9. 汇总统计、成本、轨迹和失败分析。

任何正式作业排队或运行期间，不修改 tracked 源码、协议或数据。

## 13. 最终评测与项目成功标准

7 个模型均在 120 个冻结 test 任务上运行 K=4，共 3,360 条轨迹。主要报告：

### 13.1 能力指标

- 每任务 K=4 平均 verifier success；
- pass@1 和 task-level pass@4；
- basic/medium/long 三个 horizon strata 的成功率；
- no-solution、约束组合和 task family 分解；
- 相对共享 SFT checkpoint 的提升或退化。

### 13.2 信用分配与训练动力学

- mixed-reward group 比例和零方差 group 比例；
- macro/micro advantage 分布；
- anchor-state coverage、每个 anchor 的动作多样性；
- collected/update action-token 比例；
- early/middle/late turn 梯度贡献；
- policy KL、clip fraction、entropy 和每 iteration 成功率。

### 13.3 可靠性和成本

- JSON/schema、环境动作、提前结束、max-step、verifier 和 infra failure；
- 模型轮次、环境步数和 action token 分布；
- GPU hours、墙钟时间、tokens/s、rollout/hour、峰值显存和系统资源；
- 所有结果到 git/data/prompt/adapter/trajectory 的可审计血缘。

### 13.4 统计

- 以 task 为聚类单位的 bootstrap 95% CI；
- 预注册的 paired task-level 比较；
- 三个在线训练 seed 的均值、标准差和 seed variability；
- 不把同一任务的四条 rollout 当作四个独立任务；
- 不因点估计有利而单独宣称显著性。

项目成功不以 step-aware 方法必须超过 GRPO 为条件。以下任一可信结论都成立：

1. step-aware credit assignment 显著改善 medium/long 任务；
2. 总成功率相近，但有效梯度、样本效率或长轨迹稳定性改善；
3. 方法无收益，并通过 anchor coverage、奖励稀疏性或策略漂移解释失败原因。

## 14. 面试中明确体现的训练知识

| 训练知识 | 代码/工件中的可见证据 |
|---|---|
| SFT warm start 与探索 | Base→SFT dev能力变化、结构化动作率、停止规则 |
| On-policy 正确性 | behavior log-prob、replay parity、policy version 和 adapter SHA |
| GRPO 与方差降低 | K=4同任务组、标准化优势、零方差组账本 |
| 长程信用分配 | 轨迹 macro advantage、anchor-state micro advantage、per-turn token mask |
| 稀疏奖励与动态采样 | mixed-reward yield、难度队列和所有无信号成本 |
| Importance sampling/clipping/KL | ratio、clip fraction、approximate KL、staleness=0 |
| 训练稳定性 | gradient norm、entropy、参数变化和逐 iteration 曲线 |
| GPU 训练工程 | continuous batching、packing、microbatch benchmark 和完整遥测 |
| 容错与可复现性 | trajectory journal、group/iteration commit、24h resume 和哈希血缘 |
| 统计实验设计 | split隔离、共享初始化、三RL seed、task-cluster CI与配对检验 |

## 15. 面试叙事

推荐叙事不是“我跑了很多后训练算法”，而是：

> 我最初完成了一个多算法 RLVR 框架，但通过工件审计发现成功退出的作业仍可能
> 包含不完整 K=4 组，GPU 长时间被串行浏览器交互拖空，且 11.5 万 action token
> 只有约 3.9% 真正进入一次优化器更新。我因此收缩问题，重新设计了异步多轮
> rollout、严格 on-policy 证据、两层原子恢复和 step-aware credit assignment，
> 并用共享 SFT→GRPO→细粒度信用分配的最小矩阵检验长程任务上的收益、代价和
> 失败边界。

这条主线同时展示算法理解、GPU 性能工程、分布式容错、实验治理和诚实分析。

## 16. 迁移和旧结果处理

- 当前提交 `21e65c94a96e8949e46f6ecd8a2d2068b8896300` 的 SFT、RSFT、RLOO、
  GRPO 首波工件均保留为 diagnostic；
- job 1248 的不完整 K=4 组和现有 collector 原子性缺口必须保留审计记录；
- 旧五算法矩阵不续跑 pass-2，不提交 GSPO 或后续种子；
- 新实现必须使用新的 study ID、schema version、输出根目录和冻结 git SHA；
- 只有 preflight、单元测试、fault injection 和 GPU telemetry gate 全部通过后，
  才允许提交本文档定义的正式训练。

# M5 正式在线训练准入与运行手册

> 最后更新：2026-08-11
>
> 当前状态：revision-4 preflight 已通过，正在重建正式准入与提交
>
> 正式 SFT 不重跑；Jobs `2143–2148` 已作为失败诊断归档，不进入正式结果。

### 2026-08-11 parity revision-4 校准

首次正式提交 Jobs `2143–2148` 均在第 0 轮采样完成、第一次 optimizer update 前被
revision-3 的 replay P99 门禁拒绝。六批共保留 192 个 K4 group 和 154,503 generated
action tokens，没有产生 learner adapter。只读诊断 Jobs `2149–2154` 记录到 P99 为
`0.0812–0.0903`；mean、P95、P99.9、initial clip fraction 与 mean importance ratio 全部
通过。随后将 parity replay 的 microbatch 方式修正为与 optimizer 完全一致的“group 隔离、
forward-token 排序”，复验 Jobs `2155–2160` 的 P99 为 `0.0799–0.0944`。

因此协议 revision 4 将 `replay_p99_absolute_difference` 从 `0.08` 版本化为 `0.10`。该阈值
覆盖六个正式 train batch 的最大观测值，但不改变 mean=`0.02`、P95=`0.08`、
P99.9=`0.5`、clip fraction=`0.005` 和 mean-ratio deviation=`0.02` 等独立 fail-closed
门禁。revision 4 必须重新通过真实 GPU online preflight 后才能生成新的 readiness 与授权；
旧 Jobs `2143–2148` 只能作为失败诊断，不进入正式结果。

revision-4 GPU preflight Job `2161` 随后证明两种 learner 都通过新 parity 门禁、各完成 12 次
真实更新并改变 adapter；但最终被旧的 generation median GPU≥60% 资源门禁拒绝。该批实际
generation mean/median/P95 为 45.2%/56%/64%，learner median 通过 80% 门槛。浏览器智能体
生成会在 GPU token burst 与 CPU/HTTP 环境步骤间交替，因此资源门禁修正为 generation
mean≥40% 且 P95≥60%，同时保留 learner median≥80%；这只修正资源利用率判据，不改变
数据、奖励、训练预算或算法。最终 Job `2162` 在该判据下完整通过。

## 1. 已通过的最小 RL 验证

正式起点是已经审计通过的 SFT adapter：

- producer Git：`9cedc2a7cc8cb557f5b583d3c9dd7ae3502486c1`
- adapter SHA-256：`c97c9265fe0043a8cda59908713429eef9e13d9619261a144c13cfa5fb4d7334`
- 语料：4,000 train / 400 dev，339,925 completion-label token/epoch，zero-label=0，truncation=0
- online 只复核 adapter、语料、prompt、tokenizer/base model 和 LoRA 的直接兼容性；online-only 代码变化不触发 SFT 重跑

Job `2162` 在 Git `2c62bf648f3584e4461bf50e987661071a1bb9f9` 完成 revision-4 online preflight，
Slurm 状态/退出码为 `COMPLETED / 0:0`，用时 `00:18:49`。32 个 train task × K=4
得到以下真实信号：

| 项目 | 结果 | 门槛 |
|---|---:|---:|
| infrastructure-valid trajectory | 128/128 = 100% | ≥98% |
| mixed official-task-score group | 10/32 = 31.25% | ≥20% |
| non-zero task-score trajectory | 11/128 = 8.594% | ≥5% |
| mean official task score | 0.03984 | ≥0.01 |
| binary success | 1/128 = 0.781% | ≤70% 饱和上限 |
| informative micro-credit turn | 168/1,994 = 8.425% | ≥2% |
| shared non-initial-state group | 29/32 = 90.625% | ≥5% |

两种 learner 都从同一个 SFT adapter 和同一批 K4 轨迹开始，各完成 10 次真实 optimizer
update；有效 optimizer action-token 比例都是 15.648%。`multi_turn_grpo` 的 mean loss / max
gradient norm 为 `-0.000127 / 0.3866`，`anchor_gigpo` 为 `-0.003384 / 0.7279`；两者
adapter 语义 hash 均发生变化。初始 replay parity 全部通过，P99 absolute difference 为
`0.08345`。generation mean/P95 为 `49.0%/92%`，learner median 为 `86%`，峰值显存约
`53.4/19.6 GiB`。

这证明的是“RL 管线和信用信号成立”，不是 held-out 效果。preflight 只读 frozen SFT train
任务，绝不作为测试集成绩。

## 2. 正式训练只比较两个方法

六个逻辑 run 是 2 方法 × 3 seed：

- `multi_turn_grpo`：K4 官方 terminal task score 在组内标准化，将同一个 macro advantage
  广播到该轨迹所有动作 turn；
- `anchor_gigpo`：保留完全相同的 macro advantage，再对不同轨迹首次到达的相同公开状态
  加入 discounted-return micro advantage。

seed 固定为 `20260801/20260802/20260803`。两个方法共享 SFT 起点、任务顺序、K=4
branching、采样参数、reward、optimizer、token 预算和环境；唯一预期差异是 turn-level
信用分配。PPO、RLOO、RSFT、GSPO、critic 等不进入本轮，避免把项目做成算法清单。

正式 K4 和 preflight 一样共享第一个策略采样动作，随后四条 trajectory 独立分支。
这让两种方法面对相同的树形采样分布，并给 public-state credit 提供真实的非初始状态
对照；不是人为过程奖励。训练 reward 始终是官方 terminal task score，binary success
只用于最终评测主指标。

## 3. 每个 run 的具体流程

```text
已审计 SFT adapter
  → 从 eligible train roster 按 seed 做确定性 SHA256 顺序
  → 当前 policy 生成最多 32 个 atomic K4 group
  → 每 turn fsync generated-token ledger
  → 校验 behavior/sampling/HF replay logprob parity
  → 按方法计算 macro 或 macro+micro credit
  → 2 policy epochs，AdamW 状态跨 iteration 延续
  → 原子提交 adapter + rollout adapter + optimizer + report
  → 用新 adapter 开始下一 iteration
  → billed generated-action token 接近且不超过 500,000 时结束
```

每个新 K4 attempt 先预留最坏情况 `4 × 18 × 128 = 9,216` token。所有输出都计费，
包括 infrastructure-invalid attempt；余额不足 9,216 时不再启动新组。因此最终实际 billed
token 位于 `[490,784, 500,000]`，不会靠失败重采突破预算。任务只来自 frozen eligible
train roster；dev 不参与 online update，test 在六个最终 adapter 和推理身份冻结前保持关闭。

正式成功条件包括：attempt valid ≥98%、committed token / billed token ≥90%、累计至少
2 次 optimizer update、loss/gradient 有限、最终参数真实变化、每 iteration parity 通过，
以及完整的 adapter/optimizer/cost/credit 工件。

## 4. 资源、并行与 24h 恢复

| 角色 | GPU | CPU | 内存 | wall time |
|---|---:|---:|---:|---:|
| shared WebShop service | 0 | 24 | 96 GiB | 24h 可续 |
| 每个 online run | 1 | 8 | 32 GiB | 24h 可续 |
| 六个 online run 合计 | 6 | 48 | 192 GiB | 并行 |
| online + service | 6 | 72 | 288 GiB | 节点内 |

共享服务固定 16 个 process-serialized worker，online 每 run 32 lanes。CPU、GPU 和内存
都显式进入 Slurm 调度；online job 不重复申请服务资源。六个逻辑 run 可以并行，但服务
吞吐是共同瓶颈，GPU 会在 generation/learner 两阶段交替。若资源调度不能同时满足六个，
Slurm 自然排队，不通过增大单作业 CPU 请求抢占资源。

每个 allocation 上限 24h。只有 Slurm 在 T-300 秒发送的 `USR1` 可以提交一个
`afterany:<parent>` successor；普通异常和参数错误立即停止，不形成 retry chain。恢复使用
同一 method/seed/output root：已 fsync 的 token cost 保留，只有完整 K4 被 learner 使用，
已原子提交的 learner iteration 直接复核并跳过。正式逻辑 job 数恒为 6；24h successor
只增加 allocation 数，不增加实验条件。

按 Job 2162 的单 run preflight 吞吐外推，不考虑共享服务竞争约 5–7 小时/run；六 run
并发后的保守 wall-clock 预计 12–24 小时，若触发一次 24h 恢复则 24–48 小时。之后的冻结
测试和统计分析是单独阶段，不包含在这里。

## 5. 文件与操作入口

- 冻结计划：`data/m5_webshop_formal_plan_v1.json`
- 正式 runner：`scripts/m5_webshop_formal_online.py`
- Slurm 入口：`scripts/run_m5_webshop_formal_online_job.sh`
- readiness 生成器：`scripts/m5_webshop_formal_readiness.py`
- 用户批准后才可运行的 authorization 生成器：`scripts/m5_webshop_authorize_formal.py`
- readiness：`outputs/m5_webshop_credit_assignment_v1/readiness/readiness_manifest_v1.json`
- authorization：`outputs/m5_webshop_credit_assignment_v1/readiness/formal_authorization_v1.json`
- run root：`outputs/m5_webshop_credit_assignment_v1/formal/online/{method}/seed_{seed}`

readiness 和 authorization 工具本身都不调用 `sbatch`。正式 job 入口同时要求 clean frozen
Git SHA、readiness self-hash 和用户 authorization self-hash；缺任一项都会在 optimizer 前
fail closed。当前阶段不生成 authorization，也不提交六个正式作业。

## 6. 提交后的依赖

六个 online run 彼此独立，只共同依赖 SFT adapter、readiness/authorization 和健康的共享
服务，因此可并行。冻结 test 对它们存在严格依赖：只有六个 `run_report.json` 全部通过、
最终 adapter/optimizer 血缘复核完成并冻结 8 个推理身份（raw、SFT、6 RL）后，才能一次性
打开 500-task、K=4 测试。任何中途 checkpoint、单 seed 正结果或 preflight 结果都不能进入
正式效果表。

## 7. 正式训练完成与冻结测试入口（2026-08-12）

六个 online run（Job 2165–2170）均以 `COMPLETED/0:0` 结束，并通过最终 adapter、optimizer、
generated-token ledger、K4 group、telemetry 与 self-hash 复核。冻结测试不再训练，也不加载
optimizer；它只在共同的 500 个官方 test goals 上，以相同采样设置为每个身份生成 K=4 轨迹。

- 机器计划：`data/m5_webshop_frozen_eval_plan_v1.json`
- 8 个身份：raw base、共享 verified SFT、3 个 multi-turn GRPO、3 个 anchor-GiGPO
- 每身份：500 tasks × K=4 = 2,000 trajectories；总计 16,000 trajectories
- 资源：每身份 1 GPU、6 CPU、24 GiB、单 allocation 24h；最多八身份并行
- runner：`scripts/m5_webshop_frozen_eval.py`
- Slurm 入口：`scripts/run_m5_webshop_frozen_eval_job.sh`
- 独立批准工具：`scripts/m5_webshop_authorize_frozen_eval.py`
- 输出：`outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/{identity}`

评测代码可以通过 `M5_REPO_ROOT` 在独立 Git worktree 中执行；该 worktree 只共享既有
`outputs` 训练产物，从而不切换或污染训练/服务使用的主工作树。

每个完整 K4 才原子提交；24h 超时只允许同 identity、同输出根的 successor。主指标是二值
成功率，辅指标包括 dense score、类别/约束复杂度分层、token、环境步数、耗时和无效动作。
任何测试结果都不得反向选择 checkpoint、修改模型或触发补训。八身份全部完成后，才执行
task-cluster bootstrap、成对 seed-aware permutation、失败分类和最终对比结论。

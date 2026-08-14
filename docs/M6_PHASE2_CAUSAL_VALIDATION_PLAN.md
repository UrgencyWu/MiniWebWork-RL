# M6 Phase2：SFT → RL 单调提升因果验证计划

> 状态：冻结前置实验设计，development-only
>
> 日期：2026-08-14
>
> 目标：在不打开 promotion/holdout、不重跑 SFT、不增加第三种 RL 算法的条件下，定位并修复
> 当前在线 RL 无法稳定超过 SFT 的主因。

## 1. 决策背景

M6.2 的 500-task formal-dev 已证明：三 seed trajectory-GRPO 平均比 SFT 低 0.383 pp，
Anchor-GiGPO 与 SFT 持平。Phase1 又证明两者在同一 batch 上的参数梯度 cosine 为
0.990806、相对 norm 差为 0.097956，达到冗余停止门。因此 Phase2 不再比较算法名称，
只保留一个 trajectory group-normalized policy-gradient learner，并把实验资源用于数据、奖励、
horizon、batch 结构和训练时 policy parity 的因果验证。

M6.2 formal-dev、M6.1 mini-dev 和 Phase1 seen24 均已使用，不能继续承担调参或晋级判断。
Phase2 必须创建与 SFT corpus/dev、既往 RL curriculum 和所有已使用开发集身份互斥的新 roster。

## 2. 总体实验图

```text
P0a dropout parity ─┐
P0b buy-readiness ──┼─> P2 reward same-batch counterfactual
P0c batch structure ┘                 │
                                      ├─ fail -> stop / record
P1 horizon single-variable ───────────┘
                                      │ pass
                                      v
                         P3 two-roster online mini A/B
                                      │ pass
                                      v
                         request approval for scale-up
```

每个阶段必须先审计产物和停止门，再进入下一阶段。正常排队或长训练不构成失败；确定性失败只允许
最小修复，不允许通过放宽门槛推进。

## 3. P0a：训练时 dropout parity

### 假设

RL loss 前调用 `model.train()` 会重新启用 SFT LoRA dropout=0.05，而 vLLM behavior policy 与
预更新 parity 检查都在无 dropout 推理状态。该差异可能给 importance ratio 和参数梯度增加未被
门禁覆盖的随机噪声。

### 设计

- 使用同一冻结 SFT adapter、同一预采 mixed group、相同 microbatch 和 reference adapter；
- arm A：保持 LoRA dropout=0.05；
- arm B：policy 保持 trainable，但把所有 dropout module 固定为 eval，即有效 dropout=0；
- 两个 arm 各使用 8 个相同 RNG seed；
- 不做 optimizer step，不重新采样，不写入可复用 checkpoint；
- 记录 training-forward ratio、clip fraction、loss、梯度 norm、梯度方向方差和 finite 状态。

### 门禁

dropout=0 必须满足：

- 所有 loss/gradient/ratio 有限；
- training-time clip fraction 不高于 10%；
- reference KL 不高于 0.01；
- 相对 dropout=0.05，跨 seed 梯度方向离散度或 norm 变异系数下降；
- 不破坏既有 eval-mode replay parity。

若不降低方差，则只把 dropout=0 作为 on-policy 一致性修复，不能宣称它解释了历史性能下降。

预计资源：1 GPU、4 CPU、24 GiB，30–60 分钟。

## 4. P0b：buy-readiness 与失败质量校准

### 假设

历史 public Phi 更像“页面阶段/候选出现”信号，而不是严格购买就绪信号。它对 strict success
有一定信息，但 50.21% 的失败曾达到高 Phi；同时 84.18% 的失败是 partial purchase。因此必须
先证明新的失败质量分数能够区分 strict、partial purchase 和完全失败，才能用于训练。

### 设计

- 只读取冻结轨迹中 policy-visible 的当前 item、selected options、price 和 instruction constraint
  evidence；
- search-results 中“任一候选最大匹配”不得作为 buy-readiness 主分量；
- 分别报告 item、option、price 分量及组合分数；
- 主要比较 strict vs partial purchase，不把 strict vs 全部失败的高 AUC 当充分证据；
- 本阶段只离线计算，不训练模型。

### 门禁

- final buy-readiness strict-vs-all-failure AUC >= 0.80；
- strict-vs-partial-purchase AUC >= 0.70；
- buy-readiness >= 0.8 的失败比例 <= 30%；
- 不读取 target ASIN、hidden exact title 或其他 policy-invisible 字段。

任何一项失败，均停止 process reward/Anchor 类扩展；下一轮只能使用 strict binary 或重新设计
可验证的 item/option grounding verifier。

预计资源：CPU-only，5–15 分钟。

## 5. P0c：K8×1 与 K4×4 的固定预算比较

### 假设

当前每个 optimizer step 只包含一个 K8 task-group，任务特异梯度方差高；在相同 16 trajectories
预算下，K4×4 tasks 能覆盖更多任务并通过跨任务平均减少梯度抵消和单任务偏差。

### 设计

- 从同一冻结 SFT、同一分层 roster 和相同 rollout snapshot 构造 16 条轨迹；
- arm A：2 个 K8 group，按历史 one-task/group 方式分别观察；
- arm B：4 个 K4 group，在一次 optimizer objective 中按 task 等权平均；
- 固定 generated-action-token 上限和 attempted-task 数报告；
- homogeneous group 计入预算，不通过动态跳过改变任务分布；
- 比较 unique-task coverage、有效优势组比例、梯度 norm/方向方差和 fixed-state KL。

若只能利用旧 K8 产物，允许先把每个 K8 确定性拆成两个 K4 做同 batch 反事实，但该结果只能
用于工程预检，不能替代新 task roster 的在线比较。

预计资源：1 GPU、4 CPU、24 GiB，30–60 分钟。

## 6. P1：6/6 与 18/15 horizon 单变量实验

### 假设

训练使用 6 model turns / 6 environment steps，而开发评测使用 18/15，导致策略没有机会学习
第 7 步后的重新搜索、option 修正和长程恢复。

### 设计

- 冻结同一 SFT adapter、task roster、K、sampling seed、prompt、tokenizer 和服务版本；
- arm A：6 model turns / 6 environment steps；
- arm B：18 model turns / 15 environment steps；
- 使用 64 个全新、SFT-disjoint、按类别/约束/预估难度分层的 tuning task，K4；
- paired rollout key 除 horizon 外完全一致；
- 报告 strict、partial purchase、horizon exhaustion、恢复成功、tokens 和 steps。

### 决策

- 若 full horizon strict 至少 +1 pp，或至少 20% 的 short-horizon exhaustion 转为 strict，
  后续训练必须采用 18/15；
- 若 strict 差异 <0.5 pp 且转化 <10%，horizon 降为次要因素；
- 中间区间保留 full-horizon arm，但不得把收益归因于长程信用。

预计资源：两个并行推理作业，各 1 GPU、4 CPU、24 GiB，2–4 小时。

## 7. P2：严格支配的奖励反事实

### 方法范围

优化器固定为 single-epoch trajectory group-normalized policy gradient + SFT-reference KL。实验只比较：

1. `binary_strict`：现有 strict success=1、failure=0；
2. `strict_dominant_quality`：strict 为 1，failure 位于 [-0.1, 0]。

失败质量先定义为 `q in [-1, 0]`：

```text
partial/zero purchase: q = -clip((1 - z_before_buy) + premature_penalty, 0, 1)
horizon/no purchase:   q = -clip((1 - max_t z_t) + horizon_penalty, 0, 1)
schema/action collapse:q = -1
R = strict + (1 - strict) * 0.1 * q
```

因此所有 strict 恒为 1，所有 failure <= 0，失败永远不能超过成功。不得直接把 official dense
task score 当宏观 reward。

### 同 batch 门禁

- 每个 group 中 strict reward/advantage 均严格高于 failure；
- secondary gradient norm 为 primary 的 10%–25%；
- 不翻转任何 strict-vs-failure macro sign；
- binary 与新 reward 的梯度 cosine 应位于 0.90–0.98；
- cosine >=0.98 说明新奖励实质冗余，<0.80 或 secondary norm >25% 说明过强，均停止在线 A/B。

预计资源：1 GPU、4 CPU、24 GiB，30–60 分钟。

## 8. P3：两个独立 roster 的在线小试

只有 P0–P2 全部通过才允许提交。

```text
algorithm                       trajectory group-normalized policy gradient
reward arms                     binary_strict / strict_dominant_quality
task-roster seeds               2 个独立、分层等价 roster
SFT overlap                     0 task
K                               4 / task
task groups / optimizer step    4
optimizer steps                 10
target unique tasks / run       40
policy epochs                   1
RL dropout                      0
learning rate                   3e-6
checkpoints                     0 / 5 / 10
model/env horizon               由 P1 决定
KL monitor band                 0.001–0.005
KL hard stop                    0.01
```

每个 reward arm × roster seed 为一个独立训练作业，共 4 个 GPU 作业；同一 seed 的两 arm 使用
相同 attempted-task roster、顺序、预算和 SFT 起点。评测使用全新 100–200 task tuning-dev2 K4，
不得读取 promotion/holdout。

### 晋级门

- mean RL-SFT >= +1 pp；
- 两个独立 roster seed 都为正；
- RL-only flips > SFT-only flips；
- partial purchase 不增加超过 0.5 pp；
- schema/action error 各不增加超过 0.5 pp；
- seen-unseen gap <= 1 pp；
- reference KL < 0.01；
- replay parity rejection < 2%；
- training-time clip fraction < 10%。

不通过即停止扩量并写入技术报告；通过后也不能自动打开 promotion/holdout，只能请求用户批准
进入更大规模验证。

预计资源：4 个并行作业，各 1 GPU、4 CPU、24 GiB、24h 上限；训练与中间评测预计 4–8 小时。

## 9. 调度与恢复

- 单作业最长 24h；只有 wall-time 超时允许同输出根恢复；
- 确定性代码、数据或门禁失败不允许无限 successor；
- 同时最多 4 个 GPU 作业，CPU/内存按真实需要申请；
- 正式提交只确认 job ID 一次，不在同一执行回合持续轮询；
- 定时检查按阶段预计耗时进行：P0 每 15 分钟，P1 每 30–60 分钟，P3 每 1–2 小时；
- 正常 PENDING/RUNNING 不通知；完成、确定性失败、停止门触发或需要批准时才报告。

## 10. 记录规则

每个阶段完成后，必须向 `RAW_SFT_RL_ITERATIVE_TRAINING_TECHNICAL_REPORT.md` 追加：研究问题、
唯一变化、冻结配置、训练/评测证据、被排除解释、失败分类和下一决策。具体 Slurm Job、退出码、
基础设施故障、修复和后继关系只追加到 `TRAINING_FAILURE_LEDGER.md`。

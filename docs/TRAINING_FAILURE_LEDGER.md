# 训练与评测失败账本

> 用途：按时间记录每次未成功的训练链事件、根因、修复、替代作业和研究处置。
>
> 规则：未知根因明确记为未知；用户主动取消不伪装成算法失败；失败工件不进入正式结论。

## 1. 记录口径

每条记录至少包含：阶段、Job、现象、可证根因、处理、后继作业、是否可用于研究。
`FAILED`、确定性门禁拒绝、被替换的 `CANCELLED`、以及最终性能门禁失败均记录。
共享服务的正常关闭单列为操作事件，不计作训练算法失败。

M5 的历史训练、恢复和最终失败分析已记录在
[`M5_EXECUTION_READINESS.md`](M5_EXECUTION_READINESS.md) 与
[`M5_FINAL_TECHNICAL_REPORT.md`](M5_FINAL_TECHNICAL_REPORT.md)。本账本从 M6
开始作为连续入口，后续训练必须追加而不是覆盖。

跨 M5、M6.1、M6.2 与 Phase1 的训练—分析—优化技术演进统一记录在
[`RAW_SFT_RL_ITERATIVE_TRAINING_TECHNICAL_REPORT.md`](RAW_SFT_RL_ITERATIVE_TRAINING_TECHNICAL_REPORT.md)；
本账本继续只承担逐作业失败、修复和后继关系记录。

## 2. M6 失败与修复

| Job | 阶段 | 状态/现象 | 根因或证据 | 修复与后继 | 研究处置 |
|---:|---|---|---|---|---|
| 2288 | M6 Phase2 P0a dropout parity | Slurm `COMPLETED 0:0`，但报告给出 dropout-off mean cosine=1.0056 | float32 直接归约超长梯度向量，数值越过 cosine 的数学范围；属于报告实现缺陷，不是训练失败 | 保留 v1 产物；改为分块 float64 累积并建立 `[-1,1]` 硬门，只重跑 P0a | v1 不纳入技术结论 |
| 2289 | M6 Phase2 P0b buy-readiness | Slurm `COMPLETED 0:0`；strict-vs-all AUC=0.7862，未过0.80 | scorer 对 partial purchase 有信息，但不能可靠统一排序全部 failure | 不降门槛、不重跑同配方；停止现有 process-reward arm | 有效负结果，纳入研究结论 |
| 2290 | M6 Phase2 P0a dropout parity v2 | `COMPLETED 0:0`；dropout-on/off mean gradient cosine=0.7765/0.99993，off clip=0、KL=0 | 分块 float64 cosine 与 `[-1,1]` 硬门通过，自哈希和 adapter 血缘完整 | 后续 RL 固定 dropout=0；不再重跑该探针 | P0a 有效通过结果 |
| P1 horizon roster preflight r1 | M6 Phase2 P1 horizon | 提交前退出，未创建 Slurm Job | 冻结 split 的 `train` 角色包含256个 `mini_train` 任务；隔离门正确拒绝。`train` 与 formal-dev、promotion、holdout、mini-dev 均无重叠 | 保留硬隔离门；显式用 `train - union(non-train roles)` 得到6929个候选，再分层选择64个任务并审计零重叠 | 预提交数据合同失败；无采样、训练或研究结果 |
| 2293 | M6 Phase2 P1 horizon paired audit | `FAILED 1:0`；两个采样 arm 2292_0/1 均 `COMPLETED 0:0` | 两个独立 vLLM 并发运行即使 sampling seed 全部相同，仍在42个同prompt同seed turn上生成不同token，继而造成89个prompt分叉；不能把差异归因于horizon | 保留两份完整采样；以2292_0短程轨迹为不可变动作前缀，只重跑18/15续写并重新配对审计 | 2292独立-arm结果不进入horizon因果结论；2293为有效门禁拒绝 |
| 2295/2296 | M6 Phase2 P1 exact-prefix successor | 两项均 `COMPLETED 0:0`；共享前缀 seed/token mismatch=0 | 18/15 相对6/6 strict `+5.078 pp`，51条短程horizon failure中13条转strict（25.49%） | 后续训练冻结18/15；继续单独修末端partial purchase奖励 | 2295/2296为2293的有效成功替代链；不含训练 |
| 2297 | M6 Phase2 P0b-r2 buy-readiness | `COMPLETED 0:0`；开发/外部复核AUC均过原门 | 固定item/options语义块50/50修复旧80/20低估错误option的问题；不使用隐藏答案 | 允许进入P2同batch梯度反事实；2289负结果继续保留 | 有效校准通过；无训练，不代表策略增益 |
| 2298 | M6 Phase2 P2 reward counterfactual | `COMPLETED 0:0`；安全合同通过但研究门失败 | 两panel gradient cosine=0.999861/0.999903，secondary/primary norm=1.674%/1.407%；strict-dominant质量项经K4组内标准化后与binary实质冗余 | 不调epsilon/penalty刷门；停止P3，不提交四个在线A/B作业 | 有效算法负结果；optimizer steps=0，无模型参数更新 |
| 2300 | M6 Phase3 strict-first residual credit probe | `COMPLETED 0:0`；残差幅度门通过但方向门失败 | 标准化后残差使secondary/primary norm达到11.99%/15.59%，但两panel gradient cosine仍为0.992890/0.990219，高于0.98冗余线 | 不调整冻结0.2 scale、不重排group；停止Phase3在线A/B并关闭定时推进 | 有效算法负结果；optimizer steps=0，无模型参数更新 |
| 2313–2316 | M6 Phase4 full-horizon online RL + fresh tuning-dev2 | 训练工程门全部通过，但性能链路STOP：Raw 30.273%、SFT 35.742%、RL 34.961% | SFT→RL `-0.781 pp`，task-bootstrap 95% CI `[-2.734,+0.977] pp`，RL-only/SFT-only flips=8/12；RL相对SFT多4条horizon exhaustion，partial与schema未改善 | 不扩大seed或训练规模；下一步只补评已保存step-5 checkpoint，诊断10步更新是否过长 | Job 2313 adapter为有效训练产物；step-10不晋级，2314–2316为本轮权威开发负结果 |
| 2320 | M6 Phase4 step-5 checkpoint evaluation | 两分区均`COMPLETED 0:0`，但step-5仍低于SFT：34.766% vs 35.742% | SFT→RL `-0.977 pp`，95% CI `[-2.734,+0.781] pp`，flips=6/11；partial purchase恶化`+1.367 pp` | 否定“10步过训练/早停可修复”假设；停止训练长度方向，下一轮先做末端决策token信用掩码的同batch零更新探针 | 有效单变量负结果；无新增optimizer step，不晋级 |
| 2190 | corpus gate | FAILED 1:0 | 唯一 strict-success task 156，低于冻结门槛 160；其余 corpus 检查通过 | 保留失败报告；以版本化 development-only waiver 运行 2192，正式门槛不下调 | 仅开发诊断，不算正式合格 corpus |
| 2193 | mini SFT | FAILED 1:0 | stored/runtime SFT input-file drift | 修复输入合同绑定并重提 2200 | 失败 adapter 不使用 |
| 2202 | SFT promotion gate | FAILED 1:0 | paired evaluation contract drift | 修复 Raw/SFT 配对评测身份与合同，2206 重跑 gate | 原 gate 不使用 |
| 2207 | GRPO v3 | FAILED 1:0 | replay parity P99 门禁失败 | 暂停正式 learner，建立小批量 parity 校准链 | 无 adapter 纳入评测 |
| 2208 | Anchor-GiGPO v3 | FAILED 1:0 | replay parity P99 门禁失败 | 同 2207，共同校准 | 无 adapter 纳入评测 |
| 2209 | parity microbatch-1 | FAILED 1:0 | 作业 wrapper 被 `sh` 执行，`set -o pipefail` 非法 | 改为正确 shell 入口，重提 2210 | 基础设施失败 |
| 2210 | parity microbatch-1 | FAILED 1:0 | P99 与 initial ratio clip fraction 超过冻结阈值 | 拆分 precision probe 与经验校准，后继 2211/2212 | 校准诊断，不是算法结果 |
| 2213 | precision probe | FAILED 1:0 | 保留日志为空，无法可靠恢复根因 | 明确记为 unknown/insufficient log；重提 2214 成功 | 不猜测、不纳入结果 |
| 2215 | parity K4 calibration | FAILED 1:0 | parity probe collection drift | 绑定冻结 probe collection，重提 2216 | 失败校准不使用 |
| 2216 | parity K4 calibration | FAILED 1:0 | parity probe group binding drift | 修复 group binding，2217/2218 完成 | 失败校准不使用 |
| 2219 | parity full microbatch-4 | FAILED 1:0 | wrapper 仍由 `sh` 执行，`pipefail` 非法 | 修复提交入口并继续 2220/2221 | 基础设施失败 |
| 2220 | parity microbatch-4 r2 | FAILED 1:0 | 保留日志不足，根因无法证实 | 记为 unknown/insufficient log；2221 成功 | 不纳入结果 |
| 2223 | GRPO v4 | FAILED 1:0 | SFT gate 的 Raw evaluation binding drift | 修复冻结 Raw binding，重提 v5 | 无 adapter 纳入评测 |
| 2224 | Anchor-GiGPO v4 | FAILED 1:0 | 同 2223 | 同步修复，重提 v5 | 无 adapter 纳入评测 |
| 2225 | GRPO v5 | CANCELLED | 用户主动取消被替代版本；只留下阶段性诊断工件 | 由修正版 v6/v7 取代 | 非算法失败，部分工件不纳入结果 |
| 2226 | Anchor-GiGPO v5 | FAILED 1:0 | replay parity P99 `0.157 > 0.125`，ratio clip fraction `0.006315 > 0.005`；固定经验尾阈值不适合当前有限样本 | 改用有统计含义的有限样本尾部检验，重提 v6 | 无 adapter 纳入评测 |
| 2228 | GRPO v6 | FAILED 1:0 | P99/clip 二项检验已通过，但实现仍以单个 P99.9 极值 `0.616 > 0.5` 拒绝小样本 | 提交 `9ca3078` 修复小样本 P99.9 语义，v7 为 2231 | 无 adapter 纳入评测 |
| 2229 | Anchor-GiGPO v6 | CANCELLED | 用户取消并以 v7 替换；日志同时出现一个无关 prompt-matrix 调用缺参，无法证明是本 learner 的确定性失败 | 不复用部分工件；v7 为 2232 | 非正式结果 |
| M6 final chain | RL promotion | STOP | 两方法均为 331/800，比 SFT 少 2/800；CI 跨 0，bootstrap 正方向比例低于 0.8，且 corpus 正式门仍未通过 | 停止扩展并 burn 当前 mini-dev；不得在该 slice 继续调参 | 权威负结果，见 M6 结果报告 |
| 2236_0 | M6.2 GRPO seed 20260812 | FAILED 1:0；16/20 更新后停止 | 第 17 个 collection 的 HF replay P99 `0.1672 > 0.125`，且有限样本尾检验 `p=0.0007247`；其余 parity、loss、gradient 与资源检查通过 | 保留 collection 和 staging；修复为只跳过该不可信组、计入成本、沿冻结 curriculum 同根恢复 | 已完成的 16 个更新保留；被拒组不产生梯度 |
| 2236_2 | M6.2 GRPO seed 20260813 | FAILED 1:0；15/20 更新后停止 | 第 17 个 collection 的 HF replay P95 `0.08525 > 0.08`；其余检查通过 | 同上，结构化记录 parity rejection 后同根恢复 | 已完成的 15 个更新保留；被拒组不产生梯度 |
| 2236_3 | M6.2 Anchor-GiGPO seed 20260813 | FAILED 1:0；15/20 更新后停止 | 第 17 个 collection 的 HF replay P95 `0.08356 > 0.08`；其余检查通过 | 同上 | 已完成的 15 个更新保留；被拒组不产生梯度 |
| 2236_5 | M6.2 Anchor-GiGPO seed 20260814 | FAILED 1:0；16/20 更新后停止 | 第 17 个 collection 的 HF replay P95 `0.08182 > 0.08`；其余检查通过 | 同上 | 已完成的 16 个更新保留；被拒组不产生梯度 |
| 2242_[0,2,3,5] | M6.2 recovery submission | FAILED 1:0；均在 0–1 秒退出 | 重提时未显式传入版本化 `pilot_sft_gate_v3.json` 与 `pilot_sft_eval_v2/identity_report.json`，默认路径不存在；日志为空且 learner 未启动 | 保留原检查点；补齐两项环境绑定后重提同根恢复作业 | 基础设施失败；无新增采样、梯度或参数更新，不进入算法结果 |
| 2246_[0,2,3,5] | M6.2 recovery submission r2 | FAILED 1:0；均在 1–2 秒退出 | 虽补齐文件路径，但 v3 gate 绑定的是 Job 2206 同协议重建报告，而重提错误引用了后来分别生成的 Raw/SFT identity；入口正确拒绝 `Raw evaluation binding drift` | 从不可变的原 Raw/SFT K4 collections 在同一当前代码上重建一对 identity/binding，生成版本化 v4 gate 后再同根续跑 | 门禁基础设施失败；无新增采样、梯度或参数更新，不进入算法结果 |
| 2251_[0,2,3,5] | M6.2 recovery submission r3 | FAILED 1:0；均在约 30 秒退出 | 修复逻辑已将原 parity 失败组结构化记录为零更新 skip，随后采集下一 curriculum task 时，提交参数错误地把上游 Raw K8 group producer `acd23e5…` 作为 curriculum producer；curriculum 文件自身的 producer 是 `e0c7bc6…`，显式兼容门正确拒绝 | 保留新 parity rejection/skip 证据与原 learner checkpoints；以 curriculum `git_sha=e0c7bc6…` 作为 roster producer bridge 同根重提 | 恢复配置失败；拒绝组不产生梯度，新 collection 未开始，不进入算法结果 |
| M6.2 final gate | formal-dev performance | STOP | 六个 RL audit 全通过，但 GRPO 三 seed 平均相对 SFT `-0.383 pp`、Anchor `+0.000 pp`；crossed 95% CI 分别为 `[-1.033,+0.283]` 与 `[-0.633,+0.633] pp`，均未达到预注册 `+3 pp` 与 bootstrap 门 | 保留全部训练和 16,000 条 formal-dev 评测轨迹；停止中等规模 RL，不打开 promotion/holdout | 权威性能负结果；工程成功不改写为算法晋级 |

## 3. M6 成功替代链

以下成功作业用于证明失败已经被替换，而不是删除失败历史：

| Job | 作用 | 结果 |
|---:|---|---|
| 2192 | development-only corpus waiver chain | COMPLETED；唯一任务数例外仍显式保留 |
| 2200 | mini SFT | COMPLETED；SFT adapter 进入后续配对评测 |
| 2206 | SFT-vs-Raw gate v3 | COMPLETED；SFT-Raw +5.875 pp，允许 mini RL |
| 2211/2212/2214 | parity/precision probes | COMPLETED |
| 2217/2218/2221 | K4/full microbatch parity calibration | COMPLETED |
| 2231 | GRPO v7 learner | COMPLETED；5 次真实更新 |
| 2232 | Anchor-GiGPO v7 learner | COMPLETED；5 次真实更新 |
| 2233/2234 | 两个冻结 mini-dev 评测 | COMPLETED；各 800 条轨迹 |
| 2235 | 最终配对统计 | COMPLETED；结论 STOP |
| 2250 | Raw/SFT identity 与 v4 gate 重建 | COMPLETED；SFT-Raw +5.875 pp，允许原分支同根恢复 |
| 2255_[0–3] | 四个中断分支最终恢复 | COMPLETED；六个 method×seed 分支均达到 20/20 更新，全部 RL audit PASS |
| 2259 matrix（Jobs 2259, 2261–2267） | M6.2 formal-dev 冻结评测 | COMPLETED；8 identities × 500 tasks × K4，共 16,000 条轨迹 |
| 2260 | M6.2 配对统计 | COMPLETED；报告 SHA `dd600451…eb1637`，决策 `STOP_MEDIUM_RL` |

关键修复提交包括：`1543a20`（校准测试）、`d6b48f6`（冻结 K4 校准）、
`2266743`（绑定冻结 prefix）、`7bc3c4f`（replay parity 校准）、`ed99ea3`
（有限样本尾部检验）、`9ca3078`（小样本 P99.9 语义）、`924b4b3`
（安全 parity rejection 与恢复）、`acf0f35`（中等规模评测链）、`c7b6ec6`
（按审计更新链解析最终 adapter）和 `d2560c3`（M6.1 最终配对评测收口）。

## 4. 操作事件

Jobs `2186, 2191, 2199, 2205, 2222, 2227` 为被后继服务替代后的用户取消/正常
关闭，不计作模型训练失败。服务 Job `2230` 在 2026-08-14 本次文档审计时仍运行；
除非明确要求，本报告不改变其状态。

## 5. 后续追加模板

M6.2 已结束，完整结果见
[`M6_MEDIUM_TRAINING_AND_EVALUATION_REPORT.md`](M6_MEDIUM_TRAINING_AND_EVALUATION_REPORT.md)。
后续若启动新 study，仍按下方模板追加，不得覆盖 M6.1/M6.2 记录。

```text
日期 / commit:
阶段 / Job:
状态与退出码:
直接现象:
可证根因（未知则写 unknown）:
失败前是否产生有效 adapter/checkpoint:
修复提交与聚焦验证:
后继 Job:
是否进入研究结果:
新增经验:
```

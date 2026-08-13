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

## 2. M6 失败与修复

| Job | 阶段 | 状态/现象 | 根因或证据 | 修复与后继 | 研究处置 |
|---:|---|---|---|---|---|
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

关键修复提交包括：`1543a20`（校准测试）、`d6b48f6`（冻结 K4 校准）、
`2266743`（绑定冻结 prefix）、`7bc3c4f`（replay parity 校准）、`ed99ea3`
（有限样本尾部检验）、`9ca3078`（小样本 P99.9 语义）和 `d2560c3`
（最终配对评测收口）。

## 4. 操作事件

Jobs `2186, 2191, 2199, 2205, 2222, 2227` 为被后继服务替代后的用户取消/正常
关闭，不计作模型训练失败。当前服务 Job `2230` 在结果审计时仍运行；除非明确要求，
本报告不改变其状态。

## 5. 后续追加模板

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

# MiniWebWork-RL：实习项目交付摘要

## 一句话介绍

MiniWebWork-RL 是一个端到端、可审计的文本浏览器 Agent 强化学习项目：
Qwen3.5-4B 阅读 Playwright 提供的页面文本，输出受约束 JSON 动作操作确定性
采购网站，并由独立于 LLM 的 Verifier 判定终态奖励。

## 解决的问题

普通 LLM 训练示例往往把多轮网页交互压缩成一次 completion，或只报告正向
分数。本项目把真实浏览器轨迹保留为逐 turn 的 prompt/action token 证据，并
解决了“生成采样分布与训练时 replay logprob 数值不一致”的关键问题：严格
更新路径固定为 `T=1, top_p=1, top_k=0, use_cache=false`，同时审计 generation
raw logits 与后处理 sampling scores。

## 已交付的工程链路

| 能力 | 交付与证据 |
|---|---|
| 浏览器环境 | FastAPI/SQLite 采购站点、Playwright reset/step/close、文本 observation 和固定 JSON action schema |
| 可信奖励 | 私有 Oracle 与确定性 Verifier；Agent prompt 不携带 Oracle |
| 多轮 RL | 每个 turn 按真实 prompt replay，trajectory 层聚合 terminal group-relative advantage；不伪装为单次 completion |
| 严格 on-policy | job 1080：8/8 有效、0 infra、raw/sampling 最大差异 `0.006612 <= 0.05`、存在混合奖励与学习信号 |
| 正式更新 | job 1081：1,146 action token、256/256 LoRA 张量有非零梯度并变化；checkpoint 保存、哈希、重载 forward 均通过 |
| 可信评测 | jobs 1082/1083 在同一冻结 no-gradient regression gate 上，以同 seed 和采样设置完成 96 个成对 rollout |

## 实验结论：中性结果也保留

正式单 batch GRPO update 没有在冻结 `rollout_dev_feasible_v2` 回归门禁上显示
可检测提升：M2.2R 与更新策略均为 **14/96（14.58%）** 成功、0 基础设施错误。
成对成功率差为 **0.00%**，task-bootstrap 95% CI 为 **[-3.13%, +3.13%]**，
exact McNemar **p=1.0**。成对表为 12 个共同成功、2 个基线独有成功、2 个更新
策略独有成功和 80 个共同失败。

这不是“训练无效”的泛化结论，而是关于这一次预注册单 batch、单个
no-solution 更新组的可复现中性结论。评测失败以
`model_output_failure_limit` 为主（每个策略 74/96），说明下一轮优先方向应是
格式/动作恢复能力与更多样的开发训练信号，而不是在该冻结门禁上继续调参。

## 可向导师或招聘方说明的技术难点

1. **概率合同而非只看 loss。** 发现 cache generation 与 no-cache teacher-forced
   replay 的数值路径会产生超阈值差异后，没有放宽阈值；改为严格 no-cache
   采集并独立审计 raw logits、sampling scores 和 replay。
2. **多轮 credit assignment。** 训练只覆盖实际 action token，按真实逐 turn
   prompt 重放，再在轨迹层使用终态 group-relative advantage。
3. **可审计的 GPU 训练。** 更新前检查 old/current logprob，训练时启用
   checkpointing 并固定 dropout 行为，训练后检查非零梯度、参数变化、adapter
   哈希和重载 forward。
4. **实验治理。** no-solution 开发训练源、冻结 feasible-v2 regression gate 和
   未来 final test 分离；基础设施异常永不进入奖励或梯度；中性结果不删除。

## 10 分钟演示路径

按 [`DEMO_GUIDE.md`](DEMO_GUIDE.md) 进行：先运行 CPU 质量门，再用 rule agent
演示 `TASK-001` 与 `TASK-004` 的浏览器 observation → JSON action → verifier
reward；随后打开正式 update 报告和
[`../reports/M3_0_DELIVERY_REPORT.md`](../reports/M3_0_DELIVERY_REPORT.md)，
从 strict artifact、checkpoint hash 到 paired frozen comparison 追溯整个训练
链路。

## 局限性与下一步

- 当前结论来自单次严格 one-batch update，不能声称通用策略提升。
- `rollout_dev_feasible_v2` 是冻结的 no-gradient regression gate；最终
  `final_test_v2` 仍应在训练和超参数完全冻结后一次性打开。
- 下一轮应预注册新的开发任务族、多 seed / 多 batch 的停止规则，并针对主要的
  JSON 输出失败开展诊断；不得将当前冻结集用于梯度或调参。

完整可复现实验步骤见 [`REPRODUCIBILITY_RUNBOOK.md`](REPRODUCIBILITY_RUNBOOK.md)，
完整机器可读和 Markdown 证据见
[`../reports/m3_0_delivery_report.json`](../reports/m3_0_delivery_report.json) 与
[`../reports/M3_0_DELIVERY_REPORT.md`](../reports/M3_0_DELIVERY_REPORT.md)。

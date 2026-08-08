# M4 正式训练前置就绪清单

> Study：`m4_long_horizon_credit_v1`
>
> 当前阶段：`preflight`
>
> 正式提交开关：**关闭**（`formal_submission_allowed=false`）
>
> 本文档记录“可以开始正式训练之前”的必要证据。通过一项测试不等于完成
> 训练，更不等于形成正式研究结果。

## 1. 完成定义

只有以下所有门禁都有可审计工件、冻结哈希和通过结论，才算完成本轮目标：

1. 聚焦协议、数据、prompt、奖励、信用分配和输出目录合同冻结；
2. 长程数据集与 Verified SFT 语料完成构建、隔离、回放和 token 审计；
3. multi-turn GRPO 与 step-aware GPO 共用同一 on-policy rollout/learner；
4. K=4 group、policy iteration、250k action-token 账本和 24 小时恢复语义通过；
5. behavior/replay log-prob parity、staleness=0 和 adapter 血缘 fail-closed；
6. 单 GPU preflight 达到或如实记录吞吐、GPU 利用率、显存和有效 token 门禁；
7. CPU、真实 Slurm 恢复和端到端单 GPU smoke 全部通过；
8. 工作区干净，代码、数据、语料、prompt、配置与运行 manifest 都绑定到同一
   冻结提交；
9. 就绪审计明确给出 `READY`，且没有正式训练作业已经被提前提交。

完成本清单后，正式训练仍需一次独立的开启决策。前置 goal 不会自行提交正式
SFT、GRPO 或 step-aware 作业。

## 2. 当前状态（2026-08-08）

| 门禁 | 状态 | 当前证据 / 缺口 |
|---|---|---|
| 聚焦研究问题与 7 模型矩阵 | PASS | `M4_LONG_HORIZON_AGENT_RL_SCOPE.md`；只保留共享 SFT、GRPO、step-aware GPO |
| 机器可读 study/output 合同 | PASS | `data/m4_long_horizon_study_v2.json`；正式开关关闭，旧输出根目录被排除 |
| 独立长程数据集 | PASS | v2 为 240/72/120；7/10/12/18 步；75% medium/long；正确供应商在访问位置 1/2/3 按 split 严格均衡；最佳角色采用 split 独立 SHA-256 排列，world-index mod-3 一致率仅 26.7%–28.3% |
| Split 隔离 | PASS | world/product/supplier/constraint/answer signature 跨 split 零重叠 |
| 真实浏览器与 verifier 工作流 | PASS | 18 步成功；跳过供应商检查即使答案正确也失败；旧 M4 流程兼容 |
| Prompt / 公开证据合同 | PASS | `browser_agent_v4_long_memory` 只保留有界、模型可见的 supplier/product 页面摘要；移除 origin/query/episode ID；oracle 属性泄漏测试与三供应商真实页面记忆回放通过 |
| Verified SFT 构建器 | PASS | Job 1261 在 clean `95d7c26` 上完成 240/72 全 roster 真实浏览器回放；train/dev 为 2820/846 个唯一 turn，任务零重叠，全部参考轨迹/verifier 通过，临时数据库零残留 |
| SFT 精确 token/零标签审计 | PASS | Qwen3.5 tokenizer 精确审计：train/dev completion-label token 为 60,540/18,162；forward token 为 10,492,517/3,148,254；重复、零标签和截断均为 0；最大序列 5494/5495 < 6144 |
| SFT trainer 与 dev 停止规则 | PENDING | 需实现 completion-only loss、microbatch benchmark 和 plateau 工件 |
| 异步 vLLM rollout | PENDING | 需实现多 browser worker、连续批处理和完整采样 log-prob |
| 迭代 learner / GRPO | PENDING | 需实现多 iteration/minibatch、有效组账本和 250k token cap |
| Step-aware 信用分配 | PENDING | 需冻结公共 anchor signature、macro+micro 公式与无 anchor fallback |
| On-policy / parity 门禁 | PENDING | 需验证 behavior/replay 分布语义、staleness=0、adapter SHA |
| 24 小时原子恢复 | PENDING | 需完成 group/iteration fault injection 和一次真实 Slurm resume smoke |
| GPU 性能门禁 | PENDING | 需完成 SFT microbatch 与 rollout concurrency 单卡 preflight |
| 最终冻结 manifest | PENDING | 待所有实现和工件完成后绑定最终 clean Git SHA |
| 正式训练就绪总审计 | PENDING | 只有所有上项通过后才能生成 `READY` 结论 |

`PARTIAL` 和 `PENDING` 均禁止打开正式训练提交开关。

本次 v2/v4 修复在提交前的验证快照为：完整非浏览器 CPU 回归
`297 passed, 10 deselected`；全量真实浏览器回归
`10 passed, 297 deselected`，其中包含 v2 SFT train/dev 小回放，并真实保留
三个供应商的 88%/93%/99% 公开证据。
最终冻结仍必须在提交后的 clean SHA 上重跑，不能只引用此工作区快照。

## 3. 已冻结的数据合同

```text
study_id                  m4_long_horizon_credit_v1
dataset_id                m4_long_horizon_v2
prompt                    browser_agent_v4_long_memory
train/dev/test            240 / 72 / 120
horizon actions           basic=7, medium=10/12, long=18
online K                  4
online seeds              20260801 / 20260802 / 20260803
action-token cap          250000 per online method/seed
sampling                  temperature=1.0, top_p=1.0, top_k=0
reward                    success=1, policy failure=0, infra failure=null
maximum concurrent jobs   4 single-GPU jobs
wall time                 <=24h per Slurm job
```

正式输出只能写入 `outputs/m4_long_horizon_credit_v1/formal`。当前前置工件只写入
`preflight`、`readiness` 或版本化的数据目录；`outputs/m4_invalidated`、
`outputs/m4_v2_runs` 和 `outputs/m4_v3_runs` 永不进入正式结果。

## 4. 前置执行顺序

1. 冻结并提交协议、长程数据和 Verified SFT 构建器；
2. 使用 2 CPU、8 GB、4 小时上限的 Slurm 作业构建完整 SFT 语料；
3. 审计唯一样本、train/dev 边界、真实 trace、6144 上限、completion-label token
   和零标签比例；
4. 实现并测试统一在线 runtime、GRPO learner 和 step-aware credit assigner；
5. 完成 K=4 / iteration 原子性、恢复、血缘、on-policy 和 token 账本测试；
6. 提交短时单 GPU preflight，选择 SFT microbatch 和 rollout worker 数；
7. 汇总所有工件到版本化 readiness manifest，并在干净冻结提交上重跑总审计。

队列中存在本研究的任何作业时，不修改 tracked 源码、协议或数据。CPU 和 GPU
均属于调度资源：请求以测量为依据，避免用过多 CPU/内存阻塞并行作业。

## 5. 就绪审计必须输出

最终 readiness manifest 至少包含：

- Git SHA、工作区 clean 状态和所有 tracked 文件哈希；
- study、dataset、seed、SFT corpus、token audit 和 prompt SHA；
- 每个 CPU/GPU/恢复 preflight 的 Slurm JobID、状态、ExitCode 和日志 SHA；
- GPU 型号、软件版本、选中的 microbatch/workers 及其基准指标；
- rollout 吞吐、generation/learner GPU utilization P50、VRAM headroom；
- behavior/replay log-prob 差异、有效 group/token 比例和 staleness；
- fault-injection 与真实 24 小时边界恢复结论；
- `formal_submission_allowed` 仍为 false，以及明确的 `READY` / `NOT_READY`
  判定和未通过原因。

任何缺失、哈希漂移、旧产物混入或只能人工推断的门禁均按 `NOT_READY` 处理。

## 6. Preflight 作业账本

| JobID | 提交 SHA | 用途 | 资源 | 状态 / 结论 |
|---:|---|---|---|---|
| 1259 | `b3d96dc4b83ceeae5e232698d03a8bd4f4461bd8` | 完整 Verified SFT 语料与 token 审计 | 2 CPU / 8 GB / 4 h | `FAILED 127:0`，1 秒内退出；批处理 PATH 中找不到裸 `srun`，第一个构建命令未执行，输出目录不存在；不得重用为研究工件 |
| 1260 | `e9dd16441c5974332e289a1fb02620cf31eab2ef` | 完整 Verified SFT v1 语料 | 2 CPU / 8 GB / 4 h | `CANCELLED`，运行 35:48 后主动停止；train/dev 的 78 个 long task 中正确供应商 78/78 固定为第三个访问项，且 v3 prompt 不保留前三个供应商页的可靠性证据；该设计会教授位置捷径，所有部分产物均禁止进入正式血缘 |
| 1261 | `95d7c2607a4d279196aa760b1f56723332020792` | 完整 Verified SFT v2 语料、token 审计与最终验证 | 2 CPU / 8 GB / 4 h | `COMPLETED 0:0`，58:37；240/72 任务、2820/846 唯一 turn；zero-label/duplicate/truncation 均为 0；runtime DB 零残留；PASS |

job 1259 的 stdout/stderr SHA256 分别为
`6f8ea5b13cf84336a7f46352970fb7babb875d0201e306a94d5a971fc0e39e70` 和
`6e4598b08632afb44f184d6ac8ff3e67ed49d394d66cd5c171ebe6b9381aa345`。
job 1260 的 stdout/stderr SHA256 分别为
`3b9ff739a45a1403805e34c3d0e94f5d930c52536a15e03143f2e6e4e3e7bb38` 和
`e0695093f5cb8ed2594212fa01ed3be3c8c704ad5e5403cc08b0d7e2528b0e46`；
取消时尚未写出 corpus 文件，不能将 runtime DB 或日志误认为可续跑语料。
1259 的 `srun` 路径问题已在 1260 前做最小修复；1260 进一步暴露的是数据与
prompt 的语义门禁，而非基础设施故障。Job 1261 已按要求绑定 v2 数据、v4
公开证据记忆、全新 clean Git SHA、全新 JobID 和空的 v2 输出目录；它没有续接
1260 的进度或复用其样本。

Job 1261 的 stdout/stderr SHA256 分别为
`c96e04904b5e0ddaabbea39ae2b477d971846908ea13fe3b8b959f3bd8c5313c` 和
空文件哈希 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
正式语料目录 `data/sft/m4_long_horizon_verified_v2` 的关键 SHA256 为：

```text
manifest.json        5f5b2e253594dc7fa58c9ab1e8f653558f69a2a98885f0f0487179b185bb4df0
token_audit.json     61e38e0b3feca7b0f72b602354ea7fd2a6919d56c3b8a6c55c24dda94501c001
train.jsonl          f962e84fd4bd7a58b73c3ed30ee21f6b05ea867b68718498182fea2de097a94c
valid.jsonl          5e9d43aab308fd8720f0f6df849e6f2f299d8c1a0595bc1b7cd0dc5d2074f2b3
train_manifest.json  2b75b7bb6a8d37044dba8b7247866acbd7c1fd848d76de2c6c026feee93ed595
dev_manifest.json    6a315c3524a1ae240d362e9748468a58a52d501612544a510221d1209399543a
```

独立复核确认 train/dev sample ID 各自唯一、task ID 零重叠、split 标签和 assistant
completion 角色无错误；dataset、seed 和 prompt 哈希仍分别为 `a714e5cb...`、
`938b25a...` 与 `239139ae...`。该 PASS 只关闭数据与监督 token 门禁，不代表已
训练共享 SFT adapter。

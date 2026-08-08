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
| 基础模型血缘 | PASS | `/data/share/model/Qwen3.5-4B` 共 16 个实体文件、9,342,906,882 bytes；逐文件 SHA-256 manifest 为 `data/m4_long_horizon_base_model_manifest_v1.json`，manifest SHA `290ecd9ec4eaa1f5ac6927b10e9cb4c600d22aec78a6d743baa8a01d72c1b7a3`，二次全量复核一致 |
| Verified SFT 构建器 | PASS | Job 1261 在 clean `95d7c26` 上完成 240/72 全 roster 真实浏览器回放；train/dev 为 2820/846 个唯一 turn，任务零重叠，全部参考轨迹/verifier 通过，临时数据库零残留 |
| SFT 精确 token/零标签审计 | PASS | Qwen3.5 tokenizer 精确审计：train/dev completion-label token 为 60,540/18,162；forward token 为 10,492,517/3,148,254；重复、零标签和截断均为 0；最大序列 5494/5495 < 6144 |
| SFT trainer 与 dev 停止规则 | PASS | Job 1264 在 clean `3a0acb4` 上完成全部 microbatch 候选与精确 20-update disposable smoke；选择 microbatch=8、grad accumulation=2，reserved-VRAM 余量 51.02%，完整 846-turn dev NLL/action/schema 为 0.03441/0.84634/0.93972；adapter 全部 256 tensors 非零且有限，机器选择合同见 `data/m4_long_horizon_sft_preflight_selection_v1.json` |
| 异步 vLLM rollout | PARTIAL | `data/m4_long_horizon_runtime_v1.json` 已冻结单 GPU AsyncLLM、1/2/4/8 browser workers、最多 2 个并发 K=4 group、同卡 sleep/wake adapter swap、CuMem allocator 环境与完整 raw/sampling log-prob 合同；vLLM 0.17 接口、同步 browser bridge、exact-K collector、双组预算预留及“一组失败不归档在途 peer”已通过 CPU 回归，待真实浏览器并发和 GPU smoke |
| 迭代 learner / GRPO | PARTIAL | 已实现真实 tensor replay、分层 PPO clipped loss、2 policy epochs、零信号跳过、PEFT/AdamW artifact 保存，以及 adapter/optimizer/token/sampler 原子 iteration 提交；目录提交后状态推进前可向前对账，partial learner stage 从同一 frozen collection 重做；待同卡 GPU 更新 smoke |
| Step-aware 信用分配 | PARTIAL | 已冻结 `public_anchor_macro_micro_v1`：公共 observation+prompt-token context、gamma=0.95、omega=1、first-visit、macro fallback 与三层长度归一，并已接入共享 tensor learner；CPU 单测通过，待真实 collection 的同卡 GPU smoke |
| On-policy / parity 门禁 | PARTIAL | v2 run identity 绑定 base-model/runtime hash、iteration/policy；turn 绑定 attempt/request/sampling seed、adapter、prompt/completion IDs 与 behavior/sampling log-prob；首次 GPU 观测前已预注册 behavior/sampling `1e-7` 及 replay mean/P95/max `0.02/0.08/0.18`、mean ratio 偏差 `0.02`，待 vLLM↔HF GPU smoke 实测 |
| 24 小时原子恢复 | PARTIAL | v2 journal 使用 cached hash-chain append+flush+fsync，并用 stat 加有界头尾内容哨兵适配远端粗粒度时间戳；先持久化最小 token charge、再原子落 full turn/trajectory/group；不完整 attempt 保留成本并定点归档；iteration 原子提交、run_state 向前对账与 partial-stage fault injection 已通过；若更新已提交但缺少完整报告，则保留 commit 但明确判同卡 wake 门禁失败，绝不补写 PASS；待真实 Slurm 进程中断/续跑 smoke |
| GPU 性能门禁 | PARTIAL | SFT 单卡门禁已通过：microbatch=8 为 1122.89 forward tok/s，外部遥测 GPU util P50=100%、显存余量 50.58%；仍需 rollout concurrency、同卡 learner 与 phase-specific 门禁 |
| 最终冻结 manifest | PENDING | 待所有实现和工件完成后绑定最终 clean Git SHA |
| 正式训练就绪总审计 | PENDING | 只有所有上项通过后才能生成 `READY` 结论 |

`PARTIAL` 和 `PENDING` 均禁止打开正式训练提交开关。

本次 v2/v4 修复在提交前的验证快照为：完整非浏览器 CPU 回归
`297 passed, 10 deselected`；全量真实浏览器回归
`10 passed, 297 deselected`，其中包含 v2 SFT train/dev 小回放，并真实保留
三个供应商的 88%/93%/99% 公开证据。
最终冻结仍必须在提交后的 clean SHA 上重跑，不能只引用此工作区快照。

核心在线合同的当前 CPU 回归覆盖：隐藏 oracle/verifier/episode/query/DOM ID 不进入
anchor；GRPO macro、step-aware micro、零方差 fallback；task-family 冷覆盖与
Beta(1,1) uncertainty priority；并发 journal hash chain、无效组成本保留、exact-K
commit、路径逃逸和 collection freeze 幂等。加入专用 SFT trainer、协议漂移和
Slurm 资源门禁后，完整非浏览器/非 GPU/非 Slurm 回归为
`325 passed, 10 deselected`。GPU/runtime 相关项未通过前，这些 `PARTIAL` 仍禁止正式提交。

加入 runtime v1、证据 schema v2、两阶段 turn/trajectory/group 落盘、真实 tensor
learner、双 K4 collector 与原子 iteration state 后，完整非 GPU 回归更新为
`388 passed`。聚焦的 runtime/credit/journal/iteration 回归分别覆盖：
生成 token 在完整 turn artifact 前 fsync、重复 charge 拒绝、incomplete attempt 恢复归档、
冻结 collection 禁止存在未终止 attempt、adapter/optimizer 与全局 token 账本原子推进，
同长度外部 journal 篡改拒绝、并发失败 attempt 的定点归档、在途 worst-case token
预留，以及“iteration 目录已 rename、run_state 尚未写入”故障后的只前进对账。

在线 runtime 机器合同 SHA256 为
`0f3cac765032cc369f510c40fb0efb14ad54bb0176bff4da49077ad4ec93174a`；
它固定 Python 3.11.14、PyTorch 2.10.0+cu128、Transformers 5.14.1、PEFT
0.19.1、vLLM 0.17.0 与 Playwright 1.61.0，并继续保持
`formal_submission_allowed=false`。该合同只冻结实现边界，不代表 GPU 门禁通过。

SFT 训练参数现由机器合同强制校验：learning rate `2e-4`、effective batch
`16`、候选 microbatch `1/2/4/8`、reserved-VRAM headroom 至少 `15%`。GPU
preflight 只能执行 20 次 disposable optimizer update，入口没有 formal mode，
并以 5 秒间隔记录 GPU 利用率、显存与功耗；通过 preflight 仍不会产生正式 SFT。

首次 SFT GPU preflight Job 1262 在 frozen `d603807` 上按门禁失败：microbatch=1
对最长 5494-token 样本反向时显存达到 `97,250 / 97,887 MiB`，随后更大候选
OOM，未进入 20-update smoke。日志确认 Qwen3.5 的线性注意力缺少 FLA 与
causal-conv 快路径并回退 PyTorch；因此 v2.1 固定启用 non-reentrant gradient
checkpointing，而不是放宽 15% 显存门槛。1262 只保留为失败诊断，不可作为
通过工件。

修复后的 Job 1263 在 frozen `eb96af3` 上将最长序列候选 1 的显存降至约
13.8GB，证明 checkpointing 修复了资源门禁；但候选结束落盘时，局部模型输出
变量遮蔽了 benchmark 路径并触发 `TypeError`。failure artifact 正确落盘，20 次
update 仍未开始。v2.2 将 artifact 参数改为不可混淆的 `output_path` 并加入源码
级防回归测试；1263 同样只作失败诊断。

Job 1264 在 frozen clean `3a0acb44f69b46f05efb4d7650ee88aee1931af1` 上完整
通过 SFT GPU 前置门禁。四个候选 microbatch 1/2/4/8 均满足 15% reserved-VRAM
余量，吞吐分别为 807.65/1011.42/1087.24/1122.89 forward tok/s；按冻结规则选择
microbatch=8、gradient accumulation=2。随后精确执行 20 次 disposable update，
覆盖 320 个训练样本和 6,710 completion-label tokens，并对全部 846 个 dev turn、
18,162 labels 评估。最终 dev NLL/action exact/schema valid 为
0.03441/0.84634/0.93972。该 adapter 只作运行正确性证明，已明确标记 disposable，
绝不可作为正式共享 SFT adapter；正式 SFT 仅复用冻结的 microbatch 选择。

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
4. 实现专用 SFT trainer，并提交 disposable 单 GPU preflight 选择 microbatch；
5. 实现并测试统一在线 runtime、GRPO learner 和 step-aware credit assigner；
6. 完成 K=4 / iteration 原子性、恢复、血缘、on-policy 和 token 账本测试；
7. 提交单 GPU online preflight，选择 rollout worker 数并验证同卡 learner；
8. 汇总所有工件到版本化 readiness manifest，并在干净冻结提交上重跑总审计。

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
| 1262 | `d60380714613d6f6d51fb9e6532601e84dd0d728` | SFT microbatch 与 20-update disposable GPU preflight | 1 GPU / 4 CPU / 32 GB / 24 h 上限 | `FAILED 1:0`，1:04；microbatch=1 最长序列反向占用 97,250/97,887 MiB，候选 2 OOM；未训练 adapter；促成冻结 gradient checkpointing 与失败工件落盘修复；不得记为 PASS |
| 1263 | `eb96af377eef0774c90a28318f8fcce4fd53dcfd` | checkpointing 后 SFT GPU preflight 重试 | 1 GPU / 4 CPU / 32 GB / 24 h 上限 | `FAILED 1:0`，2:59；候选 1 显存约 13.8GB，但候选明细落盘时路径被 forward 输出变量遮蔽；failure artifact 完整，未进入 20-update smoke；不得记为 PASS |
| 1264 | `3a0acb44f69b46f05efb4d7650ee88aee1931af1` | 完整 SFT microbatch 选择与 20-update disposable GPU preflight | 1 GPU / 4 CPU / 32 GB / 24 h 上限 | `COMPLETED 0:0`，41:23；四候选均通过，选择 microbatch=8、grad accumulation=2；reserved-VRAM 余量 51.02%，外部遥测余量 50.58%，GPU util P50=100%；完整 dev 与 adapter 审计通过；PASS |
| 1265 | `2501b9980547cd7caaeefa345c2df24a342227ec` | 首次 8-worker online E2E preflight | 1 GPU / 8 CPU / 48 GB / 24 h 上限 | `FAILED 1:0`，0:38；vLLM CuMem sleep pool 在模型载入前拒绝 `expandable_segments:True`；仅创建 journal identity，generated token、trajectory、learner update 与 PASS report 均为 0；旧 run root 冻结为失败诊断且不得续用 |

Job 1265 绑定的 runtime SHA256 为
`0d2a77e44a11e27e7c94112c1084c0e6ac28caf6cbc51e547263597e7d47ca1d`。
stdout/stderr/GPU telemetry/run-config/engine-starting event SHA256 分别为
`3d2241fb05bb120d4e9b845149395afc9acc825a60d57ded634d533b8df8cc6e`、
`9af59b7de32d8b4f40e66bdcb7e4dcd738ffdbe6923f95f4ab1c6c36d9d8cb13`、
`da707163b0f51c1ad9896728b8098b2c67aeef9e66c5f774337112ed5295a624`、
`0504250d82b4312d37935de661c1e5aeabd34eb8d46ec00f23d48052f5782506` 和
`765be65f260b4dec62bf32a31ad4e36c69a4afa2bde08f4d0b54b5364966c96e`。
修复只将两个 PyTorch allocator 环境别名显式清空，以满足 vLLM sleep-mode 的
CuMemAllocator 前置条件；算法、数据、采样和首次 GPU 观测前冻结的 parity
阈值均未改变。

Job 1262 的 stdout/stderr/GPU telemetry SHA256 分别为
`1f55c71aa82601589236a49341f47c6a12d386e5b3c33209eccbb4a879365e9f`、
`a4a14e8dfbfe529d5ae719fafd6d8a10c20fd824f2d054f9b4d9fe648f73466a` 和
`b28c302e76fdf507479de54a5e7bcf6ed989c843fe7ef60dfd0158ff42a7466c`。
旧实现是在最终选择后才写 benchmark，因此 1262 只有 invocation 与遥测，没有
候选明细 JSON；修复后每个候选结束即原子落盘，最终门禁失败也生成显式 failure
artifact。此审计缺口本身也是 1262 不可提升为 PASS 的原因。

Job 1263 的 stdout/stderr/GPU telemetry/invocation/failure SHA256 分别为
`0e2f75a710cac96ae7097c68a4a169eec466ca0aa9d0c715d9c0e5f1d7b44891`、
`1fb02bd89360bdaa9d7fefec66e36e2aebc69711c335180bf02fc9c9253ea23e`、
`ceb51091f3783adc08b85bb4303aea450f3ad65d356b7d613a1feb1501c0044c`、
`d0047be40be638157cdcfc962edc1d4df8d998427669e61330588adfb2683a30` 和
`581eb2732e9762bfca54aaa7a96df166a633638ba0d12b0777bd0b1492942219`。

Job 1264 的 stdout/stderr/GPU telemetry/training-progress SHA256 分别为
`2dc022171727332b2e201b377177d89ce7cb4e7a2478a30fa9adee6f9e766670`、
`76754c702f40dcd9904d224d422913bc7a97cc9bf321554ec57f456659801543`、
`394c8fb8e622f0106f7087a3352a82bd87160f2f493ebffad8713ef9149be0f5` 和
`0ab0dfa99b6aff4d5dcf3073e9cf5055bf04aad829b75577ece9c21ce386a4e8`；
invocation/benchmark/training-report/preflight-report SHA256 分别为
`77cb661d317a51f59b4ef4a719a3a21d22e20d0ee016f732d4bd7b0e794ee0b8`、
`9f0bc1ced4dfb15ee40e79bdce5a0d9144ae96fcc0d2096fec2d4d273c8afd50`、
`f9941b1844f1aa3ce84ec8fd3ede7eae26194446be1f1ae0fee43e723a492c90` 和
`1a99f96636e6dea570659e84fff3e412aad57c9ca29c49ef55ee923a8b1b4e38`。
checkpoint 与 final adapter 的目录 SHA256 均为
`fcf1b04eb462c50b70381f250a36f15bb2348b2420fac4a8b2e74959385753b2`；
256/256 tensors 非零且全部有限，共 21,233,664 个可训练参数。完整结构化证据已
冻结在 `data/m4_long_horizon_sft_preflight_selection_v1.json`。

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

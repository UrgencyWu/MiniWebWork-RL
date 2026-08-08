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

## 2. 当前状态（2026-08-09）

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
| 异步 vLLM rollout | PARTIAL | Job 1285 已在 clean `5a574933`、正确三元 adapter 血缘和 runtime v3 上关闭 experimental prefix cache 后完成 8-worker、4 个原子 K=4 group、16 trajectories、197 turns 与 4096 action tokens；207.45 秒内达到 277.66 trajectories/hour、71,080 action tokens/hour，behavior/sampling 精确一致。它在 learner 稀疏 max gate 失败；runtime v4 已版本化稳健尾部门禁，待全新 run 验证 |
| 迭代 learner / GRPO | PARTIAL | 已实现真实 tensor replay、分层 PPO clipped loss、2 policy epochs、零信号跳过、PEFT/AdamW artifact 保存，以及 canonical adapter、派生 rollout view、optimizer、semantic hash、token、sampler 的原子 iteration 提交；目录提交后状态推进前可向前对账，partial learner stage 从同一 frozen collection 重做。Jobs 1271/1280/1285 均按各自合同在首次 optimizer step 前失败且不追溯改判；仍需 runtime v4 同卡真实非零更新、三工件身份变化、engine wake 与后续生成 smoke |
| Step-aware 信用分配 | PARTIAL | 已冻结 `public_anchor_macro_micro_v1`：公共 observation+prompt-token context、gamma=0.95、omega=1、first-visit、macro fallback 与三层长度归一，并已接入共享 tensor learner；CPU 单测通过，待真实 collection 的同卡 GPU smoke |
| On-policy / parity 门禁 | PARTIAL | 错误命名空间负对照 Job 1267 的 mean/P95/max 为 `1.18677/10.12739/21.51204`。正确血缘 Job 1285 在 prefix cache 关闭后 mean/P95/P99/max 为 `0.001755/0.000329/0.033360/1.334930`，初始 ratio clip `4/4096=0.0977%`，仍只因 v3 单点 max=0.5 失败。Job 1286 的 mb8 重复、mb4、mb1 得到 P99.9 `0.240999/0.240999/0.374054/0.287926`，而 max 随 batch composition 为 `1.64054/1.64054/1.43929/1.71813`。runtime v4 因此保留 mean/P95/P99/clip/mean-ratio，新增 P99.9≤0.5，并以 `|log ratio|max≤ln(10)` 只拦截灾难性单点；负对照仍被多道门禁以数量级差距拒绝。待全新 v4 run 验证，既有失败不改判 |
| 24 小时原子恢复 | PARTIAL | v3 journal 使用 cached hash-chain append+flush+fsync，并用 stat 加有界头尾内容哨兵适配远端粗粒度时间戳；先持久化最小 token charge、再原子落 full turn/trajectory/group；不完整 attempt 保留成本并定点归档；iteration v2 原子提交、run_state 向前对账与 partial-stage fault injection 已通过；若更新已提交但缺少完整报告，则保留 commit 但明确判同卡 wake 门禁失败，绝不补写 PASS；旧 v2 root 因 schema/runtime/git 身份不一致 fail closed，待全新 root 的真实 Slurm 进程中断/续跑 smoke |
| GPU 性能门禁 | PARTIAL | SFT 单卡门禁已通过：microbatch=8 为 1122.89 forward tok/s，外部遥测 GPU util P50=100%、显存余量 50.58%。Job 1285 rollout 达到 277.66 trajectories/hour、71,080 action tokens/hour，吞吐门槛通过；严格 phase 窗口内 generation GPU util P50 仅 11%（未通过 60%），learner P50=100%（通过 80%），generation 显存余量 15.49%（通过 15%）。仍需在正确性 E2E 通过后增加在途 browser/model 请求并复测，且补齐 optimizer throughput、有效 optimizer-token 比例、三工件变化与 wake 门禁 |
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

加入 runtime v1、证据 schema v3、两阶段 turn/trajectory/group 落盘、真实 tensor
learner、双 K4 collector 与原子 iteration state 后，完整非 GPU 回归曾更新为
`388 passed`。在三元 adapter 血缘和 fail-closed 断言全部提交后，Job 1270 又在
clean `e108fc96dda88d800468c21c8d811b4219a37ef3` 上完成 `420 passed`。聚焦的
runtime/credit/journal/iteration 回归分别覆盖：
生成 token 在完整 turn artifact 前 fsync、重复 charge 拒绝、incomplete attempt 恢复归档、
冻结 collection 禁止存在未终止 attempt、adapter/optimizer 与全局 token 账本原子推进，
同长度外部 journal 篡改拒绝、并发失败 attempt 的定点归档、在途 worst-case token
预留，以及“iteration 目录已 rename、run_state 尚未写入”故障后的只前进对账。

runtime v1 历史机器合同 SHA256 为
`4e89447a49b0d353fb676817e391e2aad383851b19d2441aecd14ffe5dd7d8c8`；Job 1271
仍严格按该版本执行并失败，不能用 v2 追溯改判。runtime v2 历史合同 SHA256
为 `e3a93463c9d84badef89b5efc66b5a2f396a5afaf3383bc191d0e040bcd21b9a`；它在保持
Python 3.11.14、PyTorch 2.10.0+cu128、Transformers 5.14.1、PEFT 0.19.1、
vLLM 0.17.0、Playwright 1.61.0 与 `formal_submission_allowed=false` 不变的同时，
加入 P99、初始 ratio clip fraction、正负对照校准血缘和更新后真实生成 smoke；
Job 1280 严格按它执行并因 max `0.67653 > 0.50` 失败，不能由后续版本追溯改判。

runtime v3 候选合同 SHA256 为
`a0a54069dc8a1ee1fbfb8ca9323bc942a88c0e61ef9d27638aca864b77aaab34`。它完整保留
v2 阈值，仅将 `enable_prefix_caching` 从 `true` 改为 `false`，隔离 vLLM 0.17
日志明确标记为 experimental 的 Qwen3.5 Mamba `align` prefix cache；chunked
prefill、adapter、采样、K=4 和 learner microbatch 均未改变。全新 v3 GPU
preflight 未通过前仍为 `NOT_READY`。pre-commit CPU 定向回归 Job 1283 已在
该候选脏树与上述精确 runtime SHA 下得到 `47 passed`；Job 1282 仅因 Slurm
`--wrap` 的 `/bin/sh` 不支持 `pipefail` 而在测试启动前退出，两者均不能替代新
clean commit 的完整回归。

Job 1284 随后在 clean `5a574933126f60832dabab32aa5132d145bba9f1` 上完成
`415 passed, 10 deselected`。Job 1285 严格绑定该提交与 runtime v3，prefix cache
关闭后完成全新 collection，但仍因 max `1.33493 > 0.50` 在 optimizer 前失败；
因此 v3 也保持 `NOT_READY`。Job 1286 对这份冻结 collection 做两次 mb8 及 mb4/mb1
回放，证明 P99.9 和 clip 覆盖率稳定，而单点 max 随 batch composition 变化。

runtime v4 候选合同 SHA256 为
`a7d189f0ce59041c167076b8ecbdbb395fcc516bea27da3a20b8ce7b0e59cb55`。它继续关闭
experimental prefix cache，并保留 behavior/sampling、mean、P95、P99、初始 PPO
clip fraction 与 mean ratio 门禁；新增 P99.9 `0.50`，将 batch-composition-sensitive
的 `max≤0.50` 替换为灾难性 `|log ratio|max≤ln(10)=2.302585...`。这一版本在
Job 1285/1286 之后、正式训练之前明确校准，不能声称首次观测前预注册；只有全新
v4 run 完整通过 optimizer、iteration commit、wake 和更新后真实生成才可接受。

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
| 1266 | `9191b2e71b5ab114a2feb06458279e250c679b6c` | allocator 修复后的 8-worker online E2E preflight | 1 GPU / 8 CPU / 48 GB / 24 h 上限 | `FAILED 1:0`，1:37；allocator 门禁已通过、8.06 GiB 模型载入和 66.48 GiB KV cache 建立成功，但 vLLM 0.17 dummy-LoRA CUDA graph warm-up 将 Qwen3.5 四个 `in_proj_qkvz` slice 与两个逻辑 checkpoint module 错配并 `IndexError`；仍未进入 collection，旧 root 不得续用 |
| 1267 | `e7f397e5b81d5c36db5a62f3b1cc1c07c386f7e2` | eager Qwen3.5 LoRA 的 8-worker online E2E preflight | 1 GPU / 8 CPU / 48 GB / 24 h 上限 | `FAILED 1:0`，4:02；真实 collection、K=4 原子提交、吞吐和 level-2 sleep 均成功；首次 optimizer step 前 replay parity fail-closed，未产生 update/iteration commit/wake/PASS。定位为 vLLM conditional wrapper 与 canonical text-only PEFT adapter 的模块前缀错配；旧 root 冻结为失败诊断且不得续用 |
| 1268 | pre-commit working tree（基于 `e7f397e`） | adapter-view 实现后的完整非 GPU 回归 | 2 CPU / 8 GB / 1 h 上限 | `COMPLETED 0:0`，3:42；`412 passed, 2 warnings`，无 GPU。用于开发回归；因提交时 worktree 非 clean，不能替代最终冻结 SHA 的 clean-room 回归 |
| 1269 | pre-commit working tree（基于 `13030e8`） | adapter-view 三元血缘接入后的完整非 GPU 回归 | 2 CPU / 8 GB / 1 h 上限 | `COMPLETED 0:0`，3:42；`416 passed, 2 warnings`，峰值约 1.17 GB、无 GPU。用于开发回归；后续仍有 fail-closed 断言增强，不能替代最终冻结 SHA 的 clean-room 回归 |
| 1270 | `e108fc96dda88d800468c21c8d811b4219a37ef3` | adapter 三元血缘提交后的 clean CPU 回归 | 2 CPU / 8 GB / 1 h 上限 | `COMPLETED 0:0`，3:42；`420 passed`；作为 1271 的 clean-code 基线，PASS |
| 1271 | `e108fc96dda88d800468c21c8d811b4219a37ef3` | 正确 adapter view 的 8-worker online E2E preflight | 1 GPU / 8 CPU / 48 GB / 24 h 上限 | `FAILED 1:0`，7:18；完成 4 个 K=4 group、16 trajectories、4143 action tokens，behavior/sampling 完全一致；v1 replay mean/P95 通过、稀疏 max `0.367135>0.18`，按合同在 optimizer 前失败；不得追溯改判 PASS |
| 1272 | `e108fc96` + output-only diagnostic | token-level parity 诊断首次包装 | 1 GPU / 4 CPU / 32 GB | `FAILED 127:0`，0:01；批处理内使用相对 `srun`，诊断正文未执行；无可用结果 |
| 1273 | `e108fc96` + output-only diagnostic | 正确血缘逐 token parity 与 batch-shape 复核 | 1 GPU / 4 CPU / 32 GB | `COMPLETED 0:0`，2:23；P99=`0.04821`，初始 ratio clip `6/4143=0.1448%`，behavior/sampling 完全一致；仅用于校准诊断 |
| 1274 | `e108fc96` + output-only diagnostic | right-padding 与 selected-position 复核 | 1 GPU / 4 CPU / 32 GB | `COMPLETED 0:0`，4:25；right-padding 未消除 hybrid-recurrence batch-shape 敏感异常，方案拒绝；仅用于诊断 |
| 1275 | `e108fc96` + output-only diagnostic | 隔离 FLA 包安装 | 2 CPU / 4 GB | `COMPLETED 0:0`，0:07；只安装到 preflight diagnostics，未修改共享 Conda、tracked 源码或 runtime |
| 1276 | `e108fc96` + output-only diagnostic | Transformers 5.14.1 与隔离 FLA 探测复核 | 2 CPU / 4 GB | `COMPLETED 0:0`，0:18；确认自动探测与发行包名不一致，转入显式隔离绑定测试；不改变正式环境 |
| 1277 | `e108fc96` + output-only diagnostic | 显式隔离 FLA backend parity 对照 | 1 GPU / 4 CPU / 32 GB | `COMPLETED 0:0`，5:36；未改善正确血缘分布且放大稀疏 batch-shape 异常，正式拒绝 FLA 方案；不作为研究结果 |
| 1278 | pre-commit working tree（基于 `e108fc96`） | runtime v2 parity、post-wake 与 runner 定向回归 | 2 CPU / 8 GB / 30 min 上限 | `COMPLETED 0:0`，0:06；远端 Python 3.11 Conda 中 `36 passed`，MaxRSS 约 443 MB；只作开发回归，不能替代新 clean SHA 的完整回归 |
| 1279 | `5375e151534e2a69db1b94b8c03ca9d71df6d8d1` | runtime v2 clean 非浏览器回归 | 2 CPU / 8 GB / 1 h 上限 | `COMPLETED 0:0`，0:18；`414 passed, 10 deselected`，MaxRSS 约 789 MB；PASS |
| 1280 | `5375e151534e2a69db1b94b8c03ca9d71df6d8d1` | runtime v2 全新 8-worker online E2E preflight | 1 GPU / 8 CPU / 48 GB / 24 h 上限 | `FAILED 1:0`，7:06；4 个 K=4 group、16 trajectories、198 turns、4141 token，255.58 trajectories/hour；behavior/sampling、mean/P95/P99/clip/mean-ratio 通过，max `0.67653>0.50`，optimizer 前失败；不得追溯改判 |
| 1281 | `5375e151` + output-only diagnostic | Job 1280 冻结 collection 的 replay batch-shape 矩阵 | 1 GPU / 4 CPU / 32 GB / 1 h 上限 | `COMPLETED 0:0`，9:32；mb8 两次 max 均 `0.67653`，mb4/mb1 为 `1.98432/1.24496`；四次 mean 约 `0.002`、P99 < `0.058`。证明稀疏尾部对 batch 形状敏感；不作为研究结果 |
| 1282 | pre-commit working tree（基于 `5375e151`） | runtime v3 定向 CPU 回归首次包装 | 2 CPU / 8 GB / 30 min 上限 | `FAILED 2:0`，0:01；Slurm `--wrap` 使用 `/bin/sh`，不支持 `set -o pipefail`，pytest 未启动；保留为包装器失败诊断，不是代码/测试失败 |
| 1283 | pre-commit working tree（基于 `5375e151`） | runtime v3 prefix-cache 单变量变更定向回归 | 2 CPU / 8 GB / 30 min 上限 | `COMPLETED 0:0`，0:05；绑定 runtime v3 SHA `a0a54069...`，`47 passed`，MaxRSS 约 382 MB；PASS，但不替代新 clean SHA 的完整回归 |
| 1284 | `5a574933126f60832dabab32aa5132d145bba9f1` | runtime v3 clean 非浏览器回归 | 2 CPU / 8 GB / 1 h 上限 | `COMPLETED 0:0`，0:18；`415 passed, 10 deselected`，MaxRSS 约 770 MB；PASS |
| 1285 | `5a574933126f60832dabab32aa5132d145bba9f1` | runtime v3 关闭 prefix cache 的全新 8-worker online E2E | 1 GPU / 8 CPU / 48 GB / 24 h 上限 | `FAILED 1:0`，6:43；4 个 K=4 group、16 trajectories、197 turns、4096 token，277.66 trajectories/hour；behavior/sampling、mean/P95/P99/clip/mean-ratio 通过，max `1.33493>0.50`，optimizer 前失败；不得追溯改判 |
| 1286 | `5a574933` + output-only diagnostic | Job 1285 冻结 collection 的 replay batch-shape 矩阵 | 1 GPU / 4 CPU / 32 GB / 1 h 上限 | `COMPLETED 0:0`，9:41；mb8 两次完全一致；四次 P99.9 均 ≤`0.374054`、clip-proxy ≤`0.2441%`，max 为 `1.43929`–`1.71813`；支持稳健覆盖率门禁，不作为研究结果 |
| 1287 | pre-commit working tree（基于 `5a574933`） | runtime v4 P99.9 + catastrophic cap 定向回归 | 2 CPU / 8 GB / 30 min 上限 | `COMPLETED 0:0`，0:05；绑定 runtime v4 SHA `a7d189f0...`，`49 passed`，MaxRSS 约 382 MB；PASS，但不替代新 clean SHA 的完整回归 |

Jobs 1272–1277、1281 与 1286 的脚本和安装均位于已失败 run 的 `diagnostics/` 输出空间，没有写入
tracked 路径、共享 Conda 或正式输出根目录。这些作业只回答 backend parity 的
工程问题；不会进入算法成功率、成本、显著性或正式模型血缘。runtime v4 也不把
Jobs 1271/1280/1285 改写为成功：它要求全新 run root 在新合同下重新收集、通过全部 parity
门禁、产生可验证的非零 optimizer update、原子推进 adapter/optimizer 三元血缘，
并成功 wake vLLM、加载新 view 且完成更新后真实生成，才能关闭该门禁。

Job 1267 绑定 runtime SHA256
`45158a21efaa6976dc4e0994d27e1fbf342ec60db2204e6f8047f43b7ae988b6`。
它完成 2 个 K=4 group、8 条 trajectory、41 个 model turn、1667 个 action
token；collection 用时 147.2167 秒，吞吐为 195.6299 trajectories/hour、
1002.603 model turns/hour 和 40,764.38 action tokens/hour。attempt journal 共
96 个事件，token charge 与 turn artifact 一一配对；vLLM level-2 sleep 释放
76.03 GiB，仅保留 4.56 GiB 后，同卡 HF/PEFT learner 成功加载。

1267 在首次 optimizer step 前得到 behavior/sampling 最大差异 `0`，但初始
vLLM/HF replay 的 mean/P95/max 绝对 log-prob 差异为
`1.1867697 / 10.1273904 / 21.5120354`；mean importance ratio 为
`0.9912768`。因此预注册 replay 门禁正确拒绝该作业；optimizer update、adapter
变化、iteration commit、vLLM wake 和 PASS report 均为 0，不能作为成功工件。
stdout/stderr/GPU telemetry/run-config SHA256 分别为
`b9ff4e20ef5f1d0e43f11a5e8a562224a3094a9ddb611d7c976b21810f5e861a`、
`cdc8c32ba91a7a1511e4e25e34ede24497d4bcee932266c8f7a9864707fc5ea3`、
`50919b8c8b49f6409fcb40068583572113ae7e9798076091a46973424be975dc` 和
`4b0cf42b9b895bf4416e35e051fbbb461d6b200e33d6a3bc0db582b52e7455b3`。
journal/collection/group-0/group-1/run-identity SHA256 分别为
`9dabeb270692ecdd245612868d163078c5ea3d679b1d97b30a30fc156053a1d2`、
`50902b2758febf004f99f24cc5cbbda849b6fefe3c7f8de1b3f912cee5298659`、
`287e0a0b7d2e39589fee4732c3c3a7b508e1cbdb5cfd0bee6e3df692846944d1`、
`3fcc2f0160bb76a3e19b43c9ea12bfaaf0a32ef8086a474243bcb37a667821af` 和
`a57872529c85e8dccf8bed4146af92e7c8b559c4f43acca993081947a52edbe7`。

CPU 根因审计证明 canonical adapter 的 256 个 tensor 被 vLLM 0.17 解析为
128 个 `model.layers.*` 模块，而 Qwen3.5 conditional wrapper 的实际语言层为
`language_model.model.layers.*`。诊断视图只插入该命名空间，不改变 tensor；它
覆盖 32 层 MLP 与第 3/7/11/15/19/23/27/31 层 full-attention，共 128 模块、
256 tensor。源 adapter 与视图的规范化语义 tensor SHA256 均为
`0ebd0dbc5013ba2c687e52ec106622749112f5177c3a5d86f2f6c5e5cf654528`；
诊断视图目录 SHA256 为
`cc93243cec3a048344fc61d59ef68c370f9d38edb6b7d05021fe6184bc1c805f`，
manifest 内容 SHA256 为
`ee96d8cdeb9aabcf8387acfefe1af93b7198002eb5c274ae1ade6ae13a503f74`，
manifest 文件 SHA256 为
`0899adc43d8312cbfabe0b3b52ad3c3090fd32c04b5bb258aef49317440631b4`。
该 CPU 证据只证明结构/字节语义等价，不能替代修复后的 GPU parity smoke。

Job 1268 的 stdout/stderr SHA256 分别为
`fe39404d731b0ccc8dd99ab4d697d0d0b81415e848f879e6d33f6648c80f7ea4` 和
`ad8eec64037d25f853723e6dbb6f005db017ab18a003d41c48296ef3df33cc41`。
它验证新增 adapter-view 测试在内的 412 项非 GPU 回归，但明确保留
`pre-commit working tree` 标记；最终 readiness 只能引用 clean 冻结提交上的重跑。

Job 1269 的 stdout/stderr SHA256 分别为
`39e96193f223cb54217ab618930531300bccef85f0fbee5d07b648ad19922049` 和
`ad8eec64037d25f853723e6dbb6f005db017ab18a003d41c48296ef3df33cc41`。
它验证 canonical adapter、vLLM rollout view 与 normalized semantic SHA 在
run identity、journal、轨迹、iteration state、optimizer 和同卡 engine swap 的
完整接线；该作业仍是 dirty working-tree 开发证据，不能提升为最终 PASS。

Job 1266 绑定的 runtime SHA256 为
`0f3cac765032cc369f510c40fb0efb14ad54bb0176bff4da49077ad4ec93174a`；
stdout/stderr/GPU telemetry/run-config/engine-starting event SHA256 分别为
`952865ad391ac3085a9294e2e744b03edd9f61a3c58d4a633daf4c88b75b04ec`、
`bf463439da3d230a7e7c3eb1da9ce117ad52dcb50eb2aedecc5b23e47fe62c3f`、
`5a6afc0a1e1d7e47202d6114ea9c1a15a1ae711e1e5426ff6fe318ced50a5039`、
`90fa7c58eae217615375742566e6292a9d4eef140f9f2aaa72780ceed113e271` 和
`8e2dcd32b66869fbd78ae290f447b1bad6a3247ae6b91b15e7d9cd084b8918d0`。
修复采用 vLLM 官方 `enforce_eager` 配置，绕过有缺陷的 dummy CUDA graph capture；
没有修改第三方 site-packages，也没有改变 adapter、算法、数据或预注册阈值。
若 eager 模式无法达到冻结 GPU 利用率/吞吐门禁，则该 runtime 必须保持
`NOT_READY`，不能以“兼容”为由放宽性能标准。

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

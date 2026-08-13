# M6 最小 Raw → SFT → RL 执行手册

> 状态：M6.1 Phase A/B 已执行并在 SFT→RL 晋级门停止；正式 Phase C–F 未授权
>
> 所有 GPU 作业为 24 小时 allocation；只有 Slurm `USR1` 超时边界可提交同根 successor。
> 确定性失败不自动重试。正式作业提交后只确认一次 job ID，不自动轮询。
>
> 最终结果与失败记录：[`M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md`](M6_MINI_RESULT_AND_FAILURE_ANALYSIS.md)、
> [`TRAINING_FAILURE_LEDGER.md`](TRAINING_FAILURE_LEDGER.md)。当前 mini-dev 已 burn，本文保留为
> 冻结执行记录，不得据此在原 slice 继续调参。

## M6.1：批准继续的 156-task 开发验证

原始 M6 门槛仍是 160 个 replay-verified task，不回写、不降低。两个冻结 Raw K8
采样一共得到 156 个唯一成功任务、493 条严格回放成功轨迹和 31,362 个
completion-label token；只有任务数一项未达到 160，回放、泄漏、轨迹多样性、
恢复行为、动作均衡、轨迹数和 token 量检查均通过。因此
`data/m6_mini_pilot_waiver_v1.json` 只批准一个 **development-only** 的 M6.1
pilot；原失败工件保留，结果不得作为 formal claim。

pilot 在完全相同的 SFT adapter、Raw 排序 K8 curriculum、学习率、seed、动作
token 预算、冻结 SFT reference 和单 epoch 下比较两个真实 learner：

- `multi_turn_grpo`：同任务 K8 严格终奖组内标准化，同一轨迹全部 turn 使用相同
  macro advantage（`verifier_td_lambda=0`）。
- `anchor_gigpo`：保留相同 macro advantage，再加入 detached、逐轨迹零和的
  verifier-TD turn redistribution（`verifier_td_lambda=0.5`）。

依赖图固定为：

```text
corpus_v2 -> {pilot_sft, paired_raw_K4}
pilot_sft -> pilot_sft_K4
{paired_raw_K4, pilot_sft_K4} -> SFT promotion gate
SFT promotion gate -> {multi_turn_grpo, anchor_gigpo}
```

Raw/SFT K4 必须使用同一个冻结 200-task mini-dev roster、K=4、seed、prompt、
tokenizer、base model 和 evaluation contract。仅当 SFT-Raw 至少 +3 pp、paired
bootstrap 正方向比例至少 0.8 且错误 guardrail 全部通过，两个 RL 作业才启动。
每个 Slurm allocation 为 24 小时，只允许超时后的同根恢复；确定性失败不能自动
生成 successor。

## 1. 固定流程与停止点

```text
Phase A0  M5 exposure + historical power -> freeze M6 split
Phase A1  Raw mini-train K8 + Raw mini-dev K4 (可并行)
Phase A2  replay successes -> corpus/token audit -> RL curriculum
STOP A    corpus 任何检查失败（M6.1 仅允许版本化 waiver 中明确的 156-task 单项例外）
Phase B1  mini-SFT (1 GPU, auto microbatch benchmark, recoverable)
Phase B2  mini-SFT mini-dev K4 -> SFT-vs-Raw gate
STOP B1   SFT-Raw < 3 pp 或 paired bootstrap/行为门禁失败
Phase B3  K8 collect -> verifier-TD strict-GRPO update loop (1 GPU sequential reuse)
Phase B4  mini-RL mini-dev K4 -> complete Raw/SFT/RL chain gate
STOP B2   RL-SFT < 3 pp 或成本/credit/update 门禁失败
REQUEST   只有完整 mini gate 通过后，向用户申请 Phase C 正式训练批准
```

Mini checkpoint 始终为 `development_only`，通过后正式 SFT 仍从 Raw base 独立开始。

## 2. Phase A0：无 GPU 的冻结前置

从 M5 结果工件构建逐任务暴露登记。只扫描 SFT corpus、online/frozen evaluation、轨迹诊断和
技术报告，不扫描 `goals.json`、模型、adapter、telemetry 或 server runtime：

```bash
python scripts/m6_build_exposure_registry.py \
  --evidence sft_corpus=outputs/m5_webshop_credit_assignment_v1/preflight/sft_corpus \
  --evidence online_training=outputs/m5_webshop_credit_assignment_v1/formal/online \
  --evidence frozen_evaluation=outputs/m5_webshop_credit_assignment_v1/formal/frozen_test \
  --evidence technical_report=docs/M5_FINAL_TECHNICAL_REPORT.md \
  --output outputs/m6_monotonic_posttraining_v1/locks/m5_goal_exposure_registry_v1.json
```

从 M5 Raw/SFT/三 seed RL 的 K4 groups 提取 paired task differences，再做至少 20,000 次
前瞻仿真。`selected_n_eval` 必须是 1000/1500/2000 且 `passed=true`：

```bash
python scripts/m6_extract_m5_power_inputs.py \
  --raw-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/raw_base_model/groups \
  --sft-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/shared_verified_sft/groups \
  --rl-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/anchor_gigpo_seed_20260801/groups \
  --rl-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/anchor_gigpo_seed_20260802/groups \
  --rl-groups outputs/m5_webshop_credit_assignment_v1/formal/frozen_test/anchor_gigpo_seed_20260803/groups \
  --output-dir outputs/m6_monotonic_posttraining_v1/power_inputs

python scripts/m6_plan_statistical_power.py \
  --input raw_sft=outputs/m6_monotonic_posttraining_v1/power_inputs/raw_sft.csv \
  --input sft_rl=outputs/m6_monotonic_posttraining_v1/power_inputs/sft_rl.csv \
  --input raw_rl=outputs/m6_monotonic_posttraining_v1/power_inputs/raw_rl.csv \
  --output outputs/m6_monotonic_posttraining_v1/power_report.json
```

以 report 选出的 `N` 构建 `outputs/m6_monotonic_posttraining_v1/locks/m6_webshop_split_v1.json`。该文件必须证明 train、mini-dev、
formal-dev、promotion、holdout 主角色无交叉，promotion/holdout 的 M5 exposure 为 0。

## 3. Phase A1/A2：Raw 数据与共同基线

先提交一份 CPU service（8 CPU、48 GiB、0 GPU）。Service 默认监听 allocation 节点地址并把
`http://<service-host>:44151` 原子写入 `outputs/m6_monotonic_posttraining_v1/service_base_url`；后续
作业默认读取该地址，也可用 `M6_SERVICE_BASE_URL` 显式覆盖。每个消费者在启动训练/采样前均
执行跨节点 health/reset 检查。若集群策略禁止计算节点互访，就把 service 与 GPU 阶段放进同一
allocation，不能用 `127.0.0.1` 指向另一节点。

Raw collection 与 Raw mini-dev eval 均是 1 GPU、4 CPU、24 GiB，可并行：

- collection：256 mini-train × K8；
- baseline：200 mini-dev × K4，写出 `mini/raw_eval/identity_report.json`；
- 每个完整 K group 原子落盘，基础设施失败最多重试 4 次；算法失败不重试；
- 所有生成 token 在执行浏览器动作前 fsync 到 ledger。

collection 完成后必须对每条拟录取成功轨迹重新 reset/replay，生成 corpus。不存在
`skip-replay` 开关。Corpus 硬门包括 ≥160 success tasks、≥320 success trajectories、20k–80k
exact train completion labels、100% strict/replay success、0 hidden field/target-ASIN label、query
provenance、≥20% recovery 和 action-family ≤35%。任一项失败即 STOP A；同一 roster 只允许再做
一轮独立 K8 补采，不能使用 hidden-title teacher。

如且仅如首轮规模不足，补采必须写入独立目录并使用独立 seed（例如 `20260813`）；corpus
入口通过重复 `--collection-root`（Slurm 包装器用冒号分隔的 `M6_RAW_COLLECTION_ROOTS`）合并。
它会强制两轮 task order、split、Git、protocol 和 Raw policy lineage 完全相同，并拒绝重复 seed。

用 Raw K8 groups 生成 `mini/rl_curriculum.json`，只纳入 Raw K8 成功数 1–6 的任务。

## 4. Phase B1/B2：mini-SFT 与独立门禁

M6.1 mini-SFT 请求 1 GPU、4 CPU、24 GiB。启动时按候选 microbatch `1/2/5` 对最长序列各测 10 个
microsteps，选择仍保留 ≥15% reserved-VRAM headroom 的最大值。训练固定：

- LoRA r16/alpha32/dropout0.05；LR 2e-5；最多一遍 unique imitation data；
- 每次 optimizer update 恰好 9 条成功 imitation + 1 个 Raw retention state；
- retention 不含监督标签，优化 `KL(Raw || SFT)` 的 Raw-action sampled k3 estimator，beta=0.03；
- 每个 optimizer boundary 原子保存 adapter、optimizer 和 RNG；24h successor 从此恢复；
- token/NLL 只作训练审计，晋级只看闭环 K4。

训练后在同一个 200-task mini-dev、同一 K4 rollout keys 上评测 mini-SFT，运行
`scripts/m6_gate_mini_sft.py`。只有 `decision=ALLOW_MINI_RL` 才能提交 RL loop；否则立即 STOP B1，
并将该 mini-dev roster 标为已烧毁。

## 5. Phase B3/B4：mini-RL 与完整门禁

`run_m6_mini_rl_loop_job.sh` 每个方法请求 1 GPU、4 CPU、24 GiB，并在一张卡上顺序执行 vLLM K8 采样
和 HF learner，避免 collection 与 learner 同时占两张卡。它还要求 passing `mini/pilot_sft_gate.json`。

- frozen curriculum 最多读取 12 个 task，每个 iteration 1 个 K8 group；
- 每条 rollout 最多 6 model turns / 6 environment steps；
- 每次 collection 最多 12,288 action tokens；所有尝试累计 ≤50,000；
- strict reward 同质组记录成本但不更新；得到 5 个 mixed group 后停止；
- `multi_turn_grpo` 使用 strict GRPO macro + lambda=0 的 uniform turn credit；
- `anchor_gigpo` 使用相同 macro + lambda=0.5 verifier-TD 零和 turn redistribution；
- frozen mini-SFT reference、LR 1e-6、1 policy epoch、adaptive KL；
- 每个 learner step 验证 vLLM behavior / sampling / HF replay parity、finite loss/gradient、真实参数
  hash 变化；最终逐段验证 adapter、optimizer、reference 与 update counter 血缘。

RL loop 通过后，对最后 adapter 跑同一个 200-task mini-dev K4，随后用
`scripts/m6_run_mini_chain.py` 生成最终报告。通过条件为 Raw < SFT < RL、相邻各 ≥3 pp、两个
paired bootstrap 正向率各 ≥80%，行为 guardrails 与完整 credit/update audit 通过。

## 6. 作业数、依赖与成功判据

| 逻辑作业 | GPU | CPU | 依赖 | 预计 |
|---|---:|---:|---|---:|
| WebShop service | 0 | 8 | Phase A0 | 持续，24h 可续 |
| Raw K8 collection | 1 | 4 | service + split | 1–3 h |
| Raw K4 eval | 1 | 4 | service + split | 1–2 h |
| Corpus/curriculum | 0 | ≤8 | Raw collection | <1 h |
| Mini-SFT | 1 | 4 | STOP A passed | 0.5–2 h |
| Mini-SFT K4 eval | 1 | 4 | mini-SFT | 1–2 h |
| Multi-turn GRPO loop | 1 | 4 | STOP B1 passed | 0.5–3 h |
| Anchor-GiGPO loop | 1 | 4 | STOP B1 passed | 0.5–3 h |
| Mini-RL K4 eval | 1 | 4 | RL audit | 1–2 h |
| Final mini analysis | 0 | ≤4 | all K4 identities | minutes |

GPU 逻辑作业共 6 个；Raw collection/Raw eval 可并行，其余因门禁顺序执行。正常资源下预计
6–13 小时；排队、一次 K16 补采或 24h successor 不计入。成功不是“job COMPLETED”，而是
`mini_chain_report.json` 的 `passed=true`。失败产物必须保留，不能把 development checkpoint
转为正式 checkpoint。

## 7. 主要产物

- `outputs/m6_monotonic_posttraining_v1/locks/m5_goal_exposure_registry_v1.json`
- `outputs/m6_monotonic_posttraining_v1/power_report.json`
- `outputs/m6_monotonic_posttraining_v1/locks/m6_webshop_split_v1.json`
- `outputs/m6_monotonic_posttraining_v1/mini/raw_collection/`
- `outputs/m6_monotonic_posttraining_v1/mini/corpus/`
- `outputs/m6_monotonic_posttraining_v1/mini/raw_eval/identity_report.json`
- `outputs/m6_monotonic_posttraining_v1/mini/sft/`
- `outputs/m6_monotonic_posttraining_v1/mini/sft_eval/identity_report.json`
- `outputs/m6_monotonic_posttraining_v1/mini/sft_gate.json`
- `outputs/m6_monotonic_posttraining_v1/mini/rl/rl_audit.json`
- `outputs/m6_monotonic_posttraining_v1/mini/rl_eval/identity_report.json`
- `outputs/m6_monotonic_posttraining_v1/mini/mini_chain_report.json`

M6.1 的隔离产物使用 `corpus_v2/`、`pilot_sft/`、`pilot_raw_eval/`、
`pilot_sft_eval/`、`pilot_sft_gate.json` 和 `pilot_rl/<method>/`；不得覆盖上面原始
M6 路径或失败的 `mini/corpus/`。

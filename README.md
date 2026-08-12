# MiniWebWork-RL

MiniWebWork-RL 是一个面向确定性采购调研流程的轻量浏览器 Agent 算法项目。Qwen3.5-4B 读取文本化网页状态，通过固定 JSON 动作空间操作本地采购网站，终态由非 LLM Verifier 给出确定性奖励。

项目目标是形成一条可运行、可训练、可审计的最小多轮 Agentic RL 链路，而不是复现通用 WebArena、构建视觉浏览器 Agent，或套用单次 completion 的现成 GRPO 示例。

## 当前阶段

**M6 Raw → SFT → RL 单调提升研究已完成 Phase A/B 实现，尚未运行最小链。** M6 针对 M5 的两个主要错误重建
后训练链路：SFT 不再使用 prompt 不可见的精确商品标题，而从 Raw policy 的真实成功与
恢复轨迹蒸馏；RL 不再用 dense score 做宏观排序，而以 strict success 为终局目标，用
verifier potential 的 TD 差分分配逐 turn credit。每一阶段只有在独立闭环 dev 上胜过前一
阶段才允许晋级，最终使用新的 untouched holdout，不复用 M5 test 调参。

全量训练前先运行 development-only 的 M6-mini：256 个 mini-train task、200 个隔离
mini-dev task，依次验证 `Raw → mini-SFT → mini-RL`。只有两个相邻阶段都至少提升 3 pp
且行为门禁通过，才扩展到全量 SFT/RL；mini checkpoint 不会续训为正式模型。

当前已实现数据暴露/切分锁、前瞻功效、Raw K8/K4 采样、success-replay corpus、90/10
Raw-retention SFT、K8 strict-GRPO verifier-TD、闭环晋级门禁和 24h same-root recovery。
协议仍明确禁止正式训练；必须先在集群跑通 development-only mini chain。方案与执行入口见
[`docs/M6_MONOTONIC_POSTTRAINING_PLAN.md`](docs/M6_MONOTONIC_POSTTRAINING_PLAN.md) 和
[`docs/M6_EXECUTION_RUNBOOK.md`](docs/M6_EXECUTION_RUNBOOK.md)。

**M5 WebShop 长程信用分配研究已经完成。** 项目使用公开的 1.18M 商品/12,087 goal
WebShop full benchmark，训练一个 shared verified SFT，然后在相同约 500k
generated-action-token 预算下比较 multi-turn GRPO 与 public-anchor GiGPO-style
credit（各 3 seeds）。所有策略冻结后一次性评测 raw base、SFT 和六个 online adapter，
共 8 identities × 500 test tasks × K4 = 16,000 条轨迹。

最终严格成功率为 Raw 33.50%、SFT 0.65%、GRPO 9.42%、Anchor-GiGPO 9.27%。
结论不是后训练超过基础模型：verified SFT 因 policy 不可见的 privileged search labels
发生严重闭环负迁移；在线 RL 恢复了状态推进和购买行为，但只恢复约四分之一严格成功率
差距。Anchor-GiGPO 没有显著提高最终成功率，不过获得更高 dense score，并将评测 token
平均降低约 15%。

完整结果见
[`docs/M5_FINAL_TECHNICAL_REPORT.md`](docs/M5_FINAL_TECHNICAL_REPORT.md)，事前范围见
[`docs/M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md`](docs/M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md)，
执行与恢复记录见 [`docs/M5_EXECUTION_READINESS.md`](docs/M5_EXECUTION_READINESS.md)。

下列 M1–M3 状态是已经完成的历史基线：

```text
M1.0–M1.2  Environment and Agent Runtime        PASS
M2.0       Canonical Base Agent                 PASS
M2.1F      Expert trajectories and SFT data     PASS
M2.2R      Canonical SFT and Frozen E2E         PASS
M3.0A      Rollout readiness audit              PASS → Route B
M2.3-mini  No-solution/recovery SFT patch       PASS
M2.3 Probe Historical readiness GPU evidence   PASS
M3.0B-0A   Frozen feasible-v2 regression gate   COMPLETE / neutral result
M3.0B-0C   Strict update collection             PASS
M3.0B-1    One-batch optimizer smoke            PASS
M3.0B-2    Formal one-batch GRPO update         PASS
M3.0C      Frozen paired comparison             COMPLETE / no improvement supported
```

正式状态见 [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)：

```text
M5_VERIFIED_SFT_COMPLETE=true
M5_FORMAL_ONLINE_RUNS_COMPLETE=6/6
M5_FROZEN_IDENTITIES_COMPLETE=8/8
M5_FROZEN_TRAJECTORIES_COMPLETE=16000/16000
M5_FINAL_STATISTICAL_REPORT_COMPLETE=true
M2_3_MINI_CANONICAL_PROBE_PASS=true
ROLLOUT_DEV_FEASIBLE_V2_FROZEN=true
READY_FOR_STRICT_ON_POLICY_COLLECTION=true
READY_FOR_GRPO_UPDATE=true
M3_0_FORMAL_GRPO_UPDATE_PASS=true
M3_0_FROZEN_REGRESSION_COMPLETE=true
M3_0_DELIVERY_REPORT_COMPLETE=true
```

## M3.0 交付结果

严格采集、正式单 batch GRPO 更新和冻结回归评测均已完成。正式更新使用
8/8 有效且具有奖励方差的 no-solution 轨迹；raw policy 与采样 logprob
最大差异为 `0.006612`，低于固定阈值 `0.05`。更新后 checkpoint 已保存、
重载并审计。

在不进入梯度的 `rollout_dev_feasible_v2` 冻结回归集上，M2.2R 与更新
策略均为 **14/96（14.58%）** 成功、0 基础设施错误。成对差异为 **0.00%**，
task-bootstrap 95% CI 为 **[-3.13%, +3.13%]**，exact McNemar `p=1.0`。
因此本项目明确报告“没有证据支持提升”，而不选择性地声明效果改进。

完整证据见 [`reports/M3_0_DELIVERY_REPORT.md`](reports/M3_0_DELIVERY_REPORT.md)；
面向导师或招聘方的摘要见
[`docs/INTERNSHIP_PROJECT_SUMMARY.md`](docs/INTERNSHIP_PROJECT_SUMMARY.md)。

## 架构

```text
Public Task + Private Oracle + SQLite
                 ↓
FastAPI/Jinja2 Procurement Site
                 ↓
Playwright Browser Environment
(reset / step / close / text observation / JSON action)
                 ↓
QwenBrowserAgent
(Canonical Prompt v2 → Qwen3.5-4B → Parser → Action)
                 ↓
Deterministic Verifier
                 ↓
Expert SFT / Grouped Multi-turn Rollout / GRPO-style Update
```

每个浏览器 turn 都会基于新 Observation 重新构造 Prompt，因此正式 RL 路径按 turn 重放真实条件生成，再在 trajectory 层聚合 terminal advantage；不能把整条浏览器轨迹伪装为一次普通 completion。

## 权威文档

- [M6 Raw → SFT → RL 单调提升计划](docs/M6_MONOTONIC_POSTTRAINING_PLAN.md)
- [M6 Phase A/B 最小链执行手册](docs/M6_EXECUTION_RUNBOOK.md)
- [M5 最终训练技术报告](docs/M5_FINAL_TECHNICAL_REPORT.md)
- [M5 WebShop 信用分配研究合同](docs/M5_WEBSHOP_CREDIT_ASSIGNMENT_STUDY.md)
- [M5 执行、恢复与冻结评测记录](docs/M5_EXECUTION_READINESS.md)
- [当前实现状态](docs/CURRENT_STATUS.md)
- [架构与运行合同](docs/ARCHITECTURE_AND_CONTRACTS.md)
- [实验与数据治理](docs/EXPERIMENT_GOVERNANCE.md)
- [M3.0 多轮 Agentic RL 计划](docs/M3_0_AGENTIC_RL_PLAN.md)
- [可复现运行手册](docs/REPRODUCIBILITY_RUNBOOK.md)
- [十分钟项目演示](docs/DEMO_GUIDE.md)
- [M3.0 正式交付报告](reports/M3_0_DELIVERY_REPORT.md)
- [实习项目摘要](docs/INTERNSHIP_PROJECT_SUMMARY.md)
- [Slurm 入口](scripts/slurm/README.md)

历史 `docs/M1_*`、`docs/M2_*` 只作为阶段证据；发生冲突时，以上权威文档优先。

## 安装

Python 3.11 是当前支持环境。先安装与节点 CUDA/驱动匹配的 PyTorch，再安装项目：

```bash
conda env create -f environment.yml
conda activate miniwebwork
pip install -e ".[test,training]"
python -m playwright install chromium
```

## 质量门

```bash
bash scripts/run_quality_checks.sh
```

等价步骤：

```bash
python -m compileall -q src scripts tests
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-seed
python -m miniwebwork.cli validate-tasks
python -m pytest -q -m "not gpu and not slurm"
```

GitHub Actions 运行 CPU 语法、数据合同与测试；GPU、Playwright 长程稳定性和 Slurm 训练链仍需在集群验证。

## 数据治理

| 数据源 | 用途 | 允许进入梯度？ |
|---|---|---:|
| `data/tasks/rollout_dev_no_solution_v1/` | no-solution rollout 与 RL 开发 | 是，需版本化严格采集 |
| `data/tasks/rollout_dev_feasible_v2/` | 起始策略选择、false-no-solution 与通用能力回归 | 否 |
| historical frozen test v1 | 历史连续性 | 否 |
| future `final_test_v2` | 最终一次性评价 | 否 |

`rollout_dev_feasible_v2` 是当前唯一 feasible canonical slice。它由冻结规范和统一约束合同确定性生成：

```bash
python scripts/build_rollout_dev_feasible_v2.py \
  --output-dir data/tasks/rollout_dev_feasible_v2
```

生成器必须逐字节复现：

```text
valid_public.jsonl
valid_oracle.jsonl
dataset_manifest.json
```

其中 manifest 冻结规范 SHA-256、Public/Oracle SHA-256，以及 products/suppliers 的 Git blob identity。该集合 `may_update_model=false`。

## Canonical Rollout

正式入口：

```text
scripts/m2_3_mini_single_probe.py
scripts/slurm/m2_3_mini_single_probe.sbatch
scripts/analyze_probe_ab.py
```

Slurm 参数：

```text
POLICY TEMPERATURE MASTER_SEED K [MAX_TASKS] [TOP_P] [TOP_K] [TASK_SOURCE]
```

`TASK_SOURCE` 支持：

```text
no_solution
feasible
绝对任务目录
仓库相对任务目录
```

`feasible` 已固定映射到 `data/tasks/rollout_dev_feasible_v2`。

诊断采集：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 0.2 20260731 8 "" 0.9 0 no_solution
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 8 "" 0.9 0 no_solution

sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  A 0.2 20260731 8 "" 0.9 0 feasible
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 0.2 20260731 8 "" 0.9 0 feasible
```

严格更新分布：

```bash
sbatch scripts/slurm/m2_3_mini_single_probe.sbatch \
  B 1.0 20260731 8 "" 1.0 0 no_solution
```

首版严格行为分布固定为：

```text
temperature = 1.0
top_p = 1.0
top_k = 0
```

只有 raw-policy 与 sampling-distribution token log-prob 对齐、组内存在 mixed reward、且基础设施错误为 0 的 group，才能标记为 `valid_for_grpo_update=true`。

在升级 Transformers、修改生成实现或排查严格门禁时，先运行保存 prompt
证据的 logprob 审计：

```bash
sbatch scripts/slurm/m3_0b_logprob_audit.sbatch \
  outputs/m2_3_mini/runs/<STRICT_RUN>/<ARTIFACT>.json A 16
```

严格模式明确使用 `use_cache=false`，并验证 generation raw logits 与
post-processor sampling scores 没有行为分布差异；不能通过放宽 `0.05`
阈值来绕过不一致。

A/B 配对分析：

```bash
python scripts/analyze_probe_ab.py \
  --a outputs/m2_3_mini/runs/<A_RUN>/<A_ARTIFACT>.json \
  --b outputs/m2_3_mini/runs/<B_RUN>/<B_ARTIFACT>.json \
  --output outputs/m2_3_mini/paired_ab.json
```

分析按 `(task_id, rollout_index)` 配对，并报告 feasible success、false no-solution、no-solution success、McNemar 检验、task bootstrap 区间和终止原因。

## 单 batch 优化器 smoke

正式入口：

```text
scripts/m3_0_single_batch_smoke.py
scripts/slurm/m3_0_single_batch_smoke.sbatch
```

示例：

```bash
sbatch scripts/slurm/m3_0_single_batch_smoke.sbatch \
  <STRICT_ARTIFACT_JSON> \
  outputs/m2_3_mini/seed_1234/final_adapter
```

该 smoke 只执行一次 LoRA 更新，使用 `AdamW(weight_decay=0.0)`，并检查：

- artifact 与 Adapter hash 绑定；
- old/current log-prob 预更新一致性；
- LoRA-only finite/non-zero gradients；
- 一次 optimizer step 后参数发生变化；
- Adapter 保存、重载与 finite forward。

它只证明训练合同可运行，不代表策略性能提升。在 GPU smoke 通过前，`READY_FOR_GRPO_UPDATE` 保持 `false`。

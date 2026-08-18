# M6 Phase10-D：模型规模 vs 数据来源归因技术报告

> 状态：**归因闭环（规模screen负结果定案 + 数据来源效应方向性成立）**
> 更新日期：2026-08-18
> 关联文档：[CURRENT_EXPERIMENT_STATE.md](./CURRENT_EXPERIMENT_STATE.md)、
> [M6_MODEL_SCALE_SELF_CORRECTION_OPD_HYPOTHESIS.md](./M6_MODEL_SCALE_SELF_CORRECTION_OPD_HYPOTHESIS.md)、
> [RAW_SFT_RL_ITERATIVE_TRAINING_TECHNICAL_REPORT.md](./RAW_SFT_RL_ITERATIVE_TRAINING_TECHNICAL_REPORT.md)

## 1. 问题与动机

主链路 `Raw < SFT < corrected/OPD < RL` 的SFT环节存在一个未分离的混杂：
`SFT4(D4)`（4B，Raw4自我采样数据）与 `SFT35(D35)`（35B，Raw35自我探索数据）在资格评测上统计持平
（`-0.781 pp`，CI `[-6.510,+5.208] pp`），但两者的**模型规模**与**训练数据来源策略**同时不同，
无法判定是"规模无用"还是"数据来源决定SFT效果"。Phase10-D以单变量实验链完成该归因。

## 2. 方法与单变量纪律

每轮只改变一个变量，其余全部冻结（数据、LR、batch、LoRA覆盖、保留KL参考、评测集合）：

| 轮次 | 唯一变化 | 产物 |
|---|---|---|
| A | 评测集合：新建fresh 96-task×K4 same-corpus roster（排除历史exposure/SFT语料/Phase10-C全角色/已查看任务，category/constraint分层） | Jobs 2401/2402 |
| B | 无（只读瓶颈检查） | `d4_bottleneck_check_v1.json` |
| C | SFT35(D4)训练预算：+245 updates（第2 epoch，同schedule，resume参数SHA恒等） | Job 2408 |
| D | 训练数据来源：SFT4(D35)（4B基座+D35语料，与SFT35(D35)完全同协议） | Jobs 2409/2410 |
| E | 评测集合：fresh 96-task×K4 roster（额外排除A轮已查看96任务） | Jobs 2415/2416 |

冻结约定：`K=4`、`18/15` horizon、同task order、每stage固定rollout seed（A轮20260867、
E轮20260868）；两模型tokenizer完全相同（sha `5f9e4d49...`，vocab 248,044），NLL可直接比较；
不读取promotion/holdout；不重复相同训练轮次刷结果。

## 3. 证据链

### 3.1 A轮：same-corpus环境配对（D4语料）

`SFT4(D4) = 37.240%`（143/384）vs `SFT35(D4) = 36.979%`（142/384），
`Δ = -0.260 pp`，task-bootstrap 95% CI `[-4.948,+3.906] pp`，seed错配0。
**统计持平**——固定D4时35B无环境优势。统计报告`content_sha256 dcb4cfb0...`。

### 3.2 B轮：D4信息瓶颈检查（只读）

同dev split（`1de4a6f2`）下：`SFT4(D4)` dev NLL `0.0957` vs `SFT35(D4)` dev NLL
`0.1572 → 0.1169`（高22.2%）；两模型均远离NLL地板；SFT35(D4)逐batch训练NLL窗口均值
0.13-0.18持续波动。**瓶颈不成立**——35B不是"语料饱和"而是欠拟合/未收敛；D4语料覆盖健康
（156互异任务/指令、493条互异命令序列、5类均衡、无截断）。报告`content_sha256 8d9b3366...`。

### 3.3 C轮：预算扩展（单变量）

从formal_v1 adapter精确恢复（参数SHA恒等`a3661493...`、dev NLL连续性1e-4内），同一schedule
第2 epoch（+245 updates，共490）。结果：dev NLL `0.116896 → 0.119291`（**不降反升**），
retention KL均值27.86（epoch 1为~0.004）、raw梯度范数均值117（epoch 1为~2）——进入记忆/漂移区。
**Plateau定案：更多epoch不能使35B追上4B。** 报告`content_sha256 82575cde...`。

### 3.4 D轮：SFT4(D35)训练（2×2第四格）

4B基座+D35冻结语料（975行/93任务+109 dev行，SHA绑定`76491b41/42e49891`），与SFT35(D35)
完全同协议（108 updates、capability重加权0.20/0.40/0.40精确到1e-12、LR 2e-5、r16/a32、
保留参考改为adapter-disabled Raw4）。9/9工程门通过；dev weighted_nll `0.0915 → 0.0792`。
报告`content_sha256 649b1a9d...`。

### 3.5 E轮：D35侧环境配对

`SFT4(D35) = 33.333%`（128/384）vs `SFT35(D35) = 38.021%`（146/384），
`Δ = +4.688 pp`，95% CI `[-0.521,+9.896] pp`（贴0），方向门全过（任务flips 23正/17负、
轨迹flips 46/28、partial_purchase 220→200）。统计报告`content_sha256 c31d4010...`。

### 3.6 四格全图

| | D4语料（Raw4行为） | D35语料（Raw35行为） |
|---|---|---|
| **4B** | env 37.2% / dev NLL **0.0957** | env 33.3% / dev NLL 0.0792 |
| **35B** | env 37.0% / dev NLL 0.1169 | env **38.0%** / dev NLL **0.0612** |

环境率与NLL双向对称：每个模型都在**自己的行为数据**上占优。

## 4. 归因结论

1. **模型规模效应（4B→35B，固定D4协议）：无独立正贡献，负结果定案。** 证据：A轮环境持平、
   B轮NLL 35B更高、C轮预算扩展plateau。
2. **数据来源效应：方向性成立。** D35侧35B名义+4.69pp（CI贴0，未达严格95%显著），与NLL对称、
   方向门全过共同支撑；D4侧4B更优同理。
3. **推论**：此前`Raw35 < SFT35(D35)`的`+5.208 pp`提升本质是**自我模仿数据效应**而非规模红利。
   任何模型用"自己行为的strict-success数据"做SFT都能获得类似幅度的提升（4B侧由mini阶段
   Raw4<SFT4(D4)门佐证）。

**统计警示**：两轮环境配对均为96 task×K4（384轨迹/arm），95% CI宽度约±5pp；E轮的点估计
+4.69pp未达严格显著。归因结论依赖"四格方向一致 + NLL对称 + 方向门全过"的证据链联合，
而非单一配对显著性。若需严格统计定案，需扩大任务数（如256 task）——成本翻倍，建议仅在
该结论被用于不可逆决策前再补。

## 5. 对主链路的含义与路线建议

- 主链路 `Raw < SFT < corrected/OPD < RL` 的SFT环节在两个模型上都可复现
  （4B：mini阶段已验证；35B：dev +5.21pp已验证），且收益来源已归因为数据而非规模；
- **低成本原则指向4B学生**：4B训练/评测成本约为35B的1/10-1/2，而规模无独立贡献；
- 建议：以4B为学生推进下一环节（专项自纠错/OPD，Phase10已有基础设施），35B作为可选的高成本
  平行路线不再追问；若最终链路在4B上停滞且怀疑容量，可基于本报告的2×2框架用D35侧模型复验。

## 6. 产物索引（全部冻结SHA）

| 产物 | 路径 | content_sha256 |
|---|---|---|
| A轮roster | `phase10d_same_corpus_scale_v1/rosters/same_corpus.json` | `038780b8...` |
| A轮统计 | `.../same_corpus_eval/sft4_vs_sft35_d4_stats.json` | `dcb4cfb0...` |
| B轮分析 | `.../bottleneck_analysis/d4_bottleneck_check_v1.json` | `8d9b3366...` |
| C轮报告 | `.../s35_d4/formal_v2/training_report.json` | `82575cde...` |
| D轮报告 | `.../s4_d35/formal_v1/training_report.json` | `649b1a9d...` |
| E轮roster | `.../rosters/same_corpus_d35.json` | `22116d3c...` |
| E轮统计 | `.../same_corpus_eval_d35/sft4_d35_vs_sft35_d35_stats.json` | `c31d4010...` |

## 7. 未决问题

- E轮+4.69pp是否值得扩大到256 task严格定案（见第4节警示）；
- 4B自纠错/OPD路线的具体切入点（树上既有分支：4B专项自纠错，Phase10基础设施可复用）；
- 若走35B自我模仿路线，其+5.21pp的dev提升能否在fresh roster上复现（本报告A/E两轮的fresh
  roster均未直接对Raw35 vs SFT35(D35)重测——dev/qualification roster已burn不可复用）。

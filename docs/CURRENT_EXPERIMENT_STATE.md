# MiniWebWork-RL 当前实验状态

> 更新日期：2026-08-16  
> 用途：额度、上下文或会话中断后的唯一快速恢复入口。只记录当前有效结论、运行身份和下一步，不替代完整技术报告。

## 1. 当前目标

以低成本、可证伪实验建立稳定的：

```text
Raw < SFT < corrected/OPD < Student-only on-policy RL
```

当前优先级不是直接运行OPD或GRPO，而是先消除 `SFT4` 与 `SFT35` 使用不同训练数据造成的混杂。

## 2. 当前已验证事实

- `Raw35 < SFT35(D35)` 已成立：冻结96-task×K4评测为 `41.406% -> 46.615%`，净增
  `+5.208 pp`，task-bootstrap 95% CI `[+1.302,+9.635] pp`。
- 在另一份独立96-task×K4资格评测上，`SFT4(D4)=45.573%`，`SFT35(D35)=44.792%`，
  `SFT35-SFT4=-0.781 pp`，95% CI `[-6.510,+5.208] pp`；统计持平，SFT35未获得教师资格。
- 上述 `SFT4≈SFT35` 不是模型规模受控比较，因为两者使用不同任务、不同source policy、不同采样预算和
  不同语料组成。
- 当前不能严格声称 `SFT4>Raw35`；仍缺同任务、同K、同seed、同horizon的直接配对。
- 历史4B在线数据与当前35B strict-success语料都缺少高覆盖的same-state、same-item、option和购买边界强对比。

## 3. 当前数据身份

### `D4`：当前SFT4语料

- 来源：固定256-task mini-train roster上的Raw4自主采样；
- 采样：两个K8 seed，共4096条轨迹；
- 录取：156个strict-success task、493条strict/fresh-replay-success轨迹；
- 当前冻结文件：`train.jsonl` 2,209行/141任务，`dev.jsonl` 232行/15任务；
- 当前4B tokenizer audit：train/dev completion-label token分别为32,845/3,399，共36,244；历史技术报告中的
  31,362是早期语料统计口径，`S35(D4)`以当前冻结JSONL文件哈希和重新tokenize结果为准，不据旧数裁剪；
- train/dev SHA256分别为
  `9f8dcd1ffb036a67eb2589bbfc78c3e661b4b5e95bbbcea601ad8e2bc876df1e`、
  `1de4a6f23c4f02dc22c235e210cfb3bec6e86e3ce36b9ebc9d20eaef17fa067a`；
- 当前模型：`SFT4(D4)`，adapter为
  `outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter`。

### `D35`：当前SFT35语料

- 来源：另一组160个fresh teacher-train任务上的Raw35自主探索；
- 采样：首批32-task K4 + 扩展128-task K8，共1152条轨迹；
- 录取：399条首次strict均fresh replay成功；路径上限后为93任务、238路径；
- 正式训练：975条train action row、109条dev row；
- 能力质量：nav/match/finish=`0.20/0.40/0.40`；
- 当前模型：`SFT35(D35)`，adapter为
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/teacher_self_sft_v2_4b_paradigm/formal_v1/final_adapter`；
- 合并模型：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_self_sft_v1/teacher_self_sft_v2_4b_paradigm/merged_model_v1/model`。

## 4. 当前唯一假设

固定训练信息为 `D4` 时，35B是否比4B更好地吸收并泛化相同的public-only、strict/replay-success轨迹：

```text
SFT4(D4)  vs  SFT35(D4)
```

这是当前最低成本的模型规模screen。它不回答Raw35是否能生成更好的数据；完整归因后续需要补
`SFT4(D35)`形成现有语料2×2，出现明确信号后才考虑全新共同任务上的严格2×2。

## 5. 本轮唯一变化

- 保留 `SFT4(D4)` 不变；
- 从Raw Qwen3.5-35B-A3B重新训练 `SFT35(D4)`；
- 训练数据固定为现有D4，不增加任务、不重新采样、不加入纠错数据；
- 优先复用已经证明可使35B闭环提升的4B范式35B trainer；
- 在实现前冻结D4到35B trainer的字段映射、task/path权重、train/dev身份及训练预算；
- 本轮不得同时改变数据、LR、epoch、LoRA覆盖或评测集合。

## 6. 代码与Git状态

- 本地工作树：`/Users/wsh/Documents/MiniWebWork-RL/m5_worktree`；
- 本地分支：`codex/m6-monotonic-posttraining`；
- 本轮开始时的已提交基线HEAD：`9fa210662516875c401513dbcd1adf5f3e5bc778`；
- `S35(D4)`实现与计划提交：`f7d506faa1c9ca1f1703054a92f5d9b98c3f749d`；
- 本地实现已通过19项聚焦测试、Python语法、wrapper语法和diff检查；
- 本地/远端训练代码提交：`9c06d74b1b55bf0b2ac61dab37ed6636f4d3641d`；远端由`d1e7175`安全快进，
  12项远端聚焦测试与D4数据身份检查通过；
- 远端仓库：`/home/wushaohua/data/MiniWebWork-RL`；
- 远端访问：aTrust隧道，SSH别名`610.160.22.96`；
- 当前远端训练HEAD：`9c06d74b1b55bf0b2ac61dab37ed6636f4d3641d`；远端仅保留未跟踪
  `.m5_patch_staging/`。

## 7. 已完成Job与关键产物

- Job2377：`SFT35(D35)` 108-update正式训练，`COMPLETED 0:0`；
- Job2378：SFT35合并模型，`COMPLETED 0:0`；
- Job2379：Raw35 vs SFT35(D35)配对评测，`COMPLETED 0:0`；
- Jobs2380/2381：SFT35(D35) vs SFT4(D4)资格评测，均`COMPLETED 0:0`；
- Job2382：`S35(D4)` 2-update探针，已于2026-08-16提交并确认`PENDING`；输出
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/s35_d4/probe_v1`；
- Raw35 vs SFT35统计：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_stage_eval_v2_4b_paradigm/dev/raw35_vs_sft35_stats.json`；
- SFT4 vs SFT35统计：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_stage_eval_v2_4b_paradigm/qualification/sft4_vs_sft35_stats.json`。

## 8. 下一步唯一动作

按预计20-30分钟查收Job2382一次，不轮询。审计`training_report.json`中的D4文件身份、35B tokenizer
无截断、2次有限非零梯度、真实参数更新、dev NLL安全、Raw35 retention KL和两卡余量；全部通过才提交
同一代码/数据/LR/LoRA配置的245-update正式训练，输出新目录`.../s35_d4/formal_v1`。

## 9. 后续决策

- 若 `SFT35(D4)>SFT4(D4)`：模型规模在固定D4下有正信号，补 `SFT4(D35)`形成现有语料2×2；
- 若统计持平：不能直接否定35B，先检查D4是否对35B形成信息瓶颈；但不重复相同训练轮次刷结果；
- 若 `SFT35(D4)<SFT4(D4)`：当前D4下4B吸收/泛化更优，优先转向4B专项自纠错；
- 只有现有语料交叉训练出现明确模型或数据来源效应，才建设全新同任务、同协议2×2；
- 任何OPD或GRPO均在上述归因完成且教师/纠错门通过后再执行。

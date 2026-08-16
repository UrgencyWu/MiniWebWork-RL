# MiniWebWork-RL 当前实验状态

> 更新日期：2026-08-17  
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
- `SFT35(D4)` 245-update正式训练（Job2383）`COMPLETED 0:0`：245次优化器更新、2205条互异
  imitation row、dev token-NLL `0.157152 -> 0.116896`、relative parameter displacement `0.042515`、
  全部10项预注册工程门通过；报告`content_sha256`为
  `36fd4d54b5fc67788c3f388bcf433ea8171269b78ca894a2f1de946acc85d37d`；
  最终adapter目录SHA256为`c48f9faa761d330470c4c0aa79b63b90199c16d337e5b9f314c7f1795cb8b6d5`。
- `SFT4(D4)` 与 `SFT35(D4)` 尚未在冻结配对评测上比较；这正是本轮（Phase10-D same-corpus）要做的事。
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

- `SFT35(D4)` 正式训练已完成且工程门全部通过（见第2、7节），模型不再变化；
- 本轮唯一变化是**评测集合**：构建全新96-task same-corpus paired环境评测
  `SFT4(D4) vs SFT35(D4)`；
- 确定性roster：从base train role排除版本化历史exposure union（M5 registry + Phase10/10B/10C
  exposure union）、SFT D4/D35语料任务、Phase10-C全split角色及全部已查看评测任务后，按公开
  category/constraint proxy分层选取96个fresh development-only任务（当前fresh池1088个）；
- 冻结 `K=4`、`18/15` horizon、相同task order与rollout seed `20260867`；
- `SFT4` 保持既有adapter路径；`SFT35(D4)` 用formal adapter按Job2378同款合并门合并到Raw35
  （r=16、alpha=32、310 target），合并在评测arm内执行；
- 不得读取promotion/holdout；不改数据、LR、epoch、LoRA覆盖或既有Phase10-C评测结论。

## 6. 代码与Git状态

- 训练服务器仓库：`/home/wushaohua/data/MiniWebWork-RL`（本状态文件所在仓库）；
- 分支：`codex/m6-monotonic-posttraining`；
- 远端同步基线：`e18c1af9d67fd8826c97eb554c1bba208501f19b`（`docs: record S35 D4 formal training`，
  本轮开始前已由`c862671`安全快进）；
- Phase10-D same-corpus评测实现提交：`b8872b8`（roster builder、collector/wrapper/stats扩展、
  4项新聚焦测试；与既有12项相关聚焦测试合计16项全部通过，py_compile与bash -n通过）；
- macOS侧工作树（`/Users/wsh/Documents/MiniWebWork-RL/m5_worktree`）持有Job2383完成审计的本地提交
  `d37b67d`，因Codex外部操作额度限制尚未推送；服务器侧状态以本文件为准；
- 远端仅保留未跟踪 `.m5_patch_staging/`，不得覆盖或删除。

## 7. 已完成Job与关键产物

- Job2377：`SFT35(D35)` 108-update正式训练，`COMPLETED 0:0`；
- Job2378：SFT35合并模型，`COMPLETED 0:0`；
- Job2379：Raw35 vs SFT35(D35)配对评测，`COMPLETED 0:0`；
- Jobs2380/2381：SFT35(D35) vs SFT4(D4)资格评测，均`COMPLETED 0:0`；
- Job2382：`S35(D4)` 2-update探针，`COMPLETED 0:0`；18条唯一imitation row、两次有限非零梯度，
  adapter 620个语义张量发生变化；dev token-NLL由`0.199705`降至`0.197499`，Raw35 retention KL为
  `0/0.003897`，两卡reserved headroom均约`64%`，全部预注册工程门通过；报告SHA256为
  `209d64fc2fd5832dbe41f8d621fb0fcfa3e534b31c63bb63562e92e50d79b155`；
- Job2383：`S35(D4)` 245-update正式训练，`COMPLETED 0:0`；245次更新、2205条互异D4 row、
  dev NLL `0.157152 -> 0.116896`、relative displacement `0.042515`、10/10工程门通过；运行Git为
  `c8626710907356017c8ac773e11575e5f0c5bd74`，输出
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/s35_d4/formal_v1`；
- Phase10-D same-corpus评测基础设施（本仓库`b8872b8`）：
  roster `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/rosters/same_corpus.json`
  （构建后冻结）；两arm输出
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/same_corpus_eval/{sft4,sft35_d4}`；
  SFT35(D4)合并模型
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/s35_d4/formal_v1/merged_model_v1/`；
- Raw35 vs SFT35统计：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_stage_eval_v2_4b_paradigm/dev/raw35_vs_sft35_stats.json`；
- SFT4 vs SFT35统计：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_stage_eval_v2_4b_paradigm/qualification/sft4_vs_sft35_stats.json`。

## 8. 下一步唯一动作

依次执行：推送`b8872b8`到origin并安全快进远端 → 服务器侧聚焦测试通过 → 用
`scripts/m6_phase10d_build_same_corpus_eval_roster.py`构建并冻结96-task roster（输出与既有
Phase10-C/D评测无路径冲突）→ 确认WebShop服务health → 并行提交两个推理arm
（`run_m6_phase10d_same_corpus_eval_job.sh`，identity分别为`sft4`与`sft35_d4`，仅确认Job ID并
确认进入RUNNING后按预计耗时设置一次查收）。两arm完成后用
`m6_phase10c_teacher_stage_eval_stats.py --stage same_corpus`出`SFT4 vs SFT35(D4)`配对统计。
若推送因外部额度限制失败：保留本地进展并NOTIFY一次，不使用替代网络路径绕过。

## 9. 后续决策

- 若 `SFT35(D4)>SFT4(D4)`（本same-corpus配对评测）：模型规模在固定D4下有正信号，补
  `SFT4(D35)`形成现有语料2×2；
- 若统计持平：不能直接否定35B，先检查D4是否对35B形成信息瓶颈；但不重复相同训练轮次刷结果；
- 若 `SFT35(D4)<SFT4(D4)`：当前D4下4B吸收/泛化更优，优先转向4B专项自纠错；
- 只有现有语料交叉训练出现明确模型或数据来源效应，才建设全新同任务、同协议2×2；
- 任何OPD或GRPO均在上述归因完成且教师/纠错门通过后再执行。

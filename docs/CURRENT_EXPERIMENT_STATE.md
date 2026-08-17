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
- `SFT4(D4)` 与 `SFT35(D4)` 已完成全新96-task×K4 same-corpus配对评测（Phase10-D，seed 20260867，
  18/15，配对rollout seed错配0）：`SFT4(D4)=37.240%`（143/384），`SFT35(D4)=36.979%`（142/384），
  净差 `-0.260 pp`，task-bootstrap 95% CI `[-4.948,+3.906] pp`，任务级flips 19正/14负。
  **统计持平**：在固定D4监督下，35B没有比4B更好地吸收/泛化，模型规模screen为负。
  失败类迁移：SFT35(D4)的horizon_exhaustion 18→30（+12），partial_purchase 205→195（-10），
  zero_score_purchase 17/18≈持平；无schema/action失败类。
- 按冻结决策树（见第9节），统计持平后不直接否定35B、不重刷相同训练：下一步先检查D4是否对35B形成
  信息瓶颈。
- **瓶颈检查已完成（只读分析，报告`content_sha256 8d9b3366...`）：瓶颈不成立。** 同一冻结dev split
  （1de4a6f2）与完全相同的tokenizer（sha 5f9e4d49，vocab 248,044）下：
  `SFT4(D4)` dev NLL `0.095675`，`SFT35(D4)` dev NLL `0.157152 -> 0.116896`——35B比4B高22.2%，
  且两模型都远未接近0；SFT35(D4)逐batch训练NLL窗口均值0.13-0.18持续波动、末update仍0.1544，
  无收敛证据。D4语料覆盖本身不差（156互异任务/指令、493条互异命令序列、5类均衡、无截断）。
  更一致的解读：D4是Raw4自身行为的采样（SFT4近乎自蒸馏），而SFT35(D4)是跨策略模仿且只被给了
  1个epoch（245×9=2205行恰好一遍）——same-corpus持平带**协议混杂**，不能单独归因于"35B无法从
  D4获益"。
- **规模screen已以负结果定案（用户授权后的单变量预算实验完成）**：Job2408（extend，第2 epoch，
  +245 updates，resume参数SHA恒等、dev NLL连续性门全过）后dev NLL
  `0.116896 -> 0.119291`——不降反升；batch NLL窗口均值仍0.11-0.16波动，retention KL均值
  27.86（epoch 1为~0）、raw梯度范数均值117（epoch 1为~2），证明35B在冻结D4协议下已进入
  记忆/漂移区而非欠拟合。**结论：D4固定时，4B吸收与泛化均不弱于35B；35B无法通过更多epoch
  追上4B的dev NLL 0.0957。模型规模screen负结果成立，带源策略混杂（D4为Raw4自身行为）。**
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

该规模screen已完整走完（环境配对持平 + 瓶颈检查否定 + 预算扩展plateau，见第2节），**负结果定案**：
D4固定下35B不优于4B。遗留的唯一归因缺口是**模型规模效应 vs 数据来源效应**尚未分离——D4是Raw4
自身行为（SFT4近乎自蒸馏），所以仍需 `SFT4(D35)` 完成现有语料2×2才能断言"规模无益"而非
"数据来源决定吸收"。当前唯一待检验假设转为：**D35语料上4B是否同样不弱于35B（即每个模型都在
自己的行为数据上更优，规模效应无独立贡献）**。

## 5. 本轮唯一变化

- `SFT4(D4)`、`SFT35(D4)` 1-epoch正式模型及same-corpus评测结果均已冻结（见第2、7节）；
- 瓶颈检查判定后（用户已选择`extend_budget_single_variable`），本轮唯一变化是**SFT35(D4)训练预算**：
  从formal_v1 adapter精确恢复（参数SHA与dev NLL连续性门），用与epoch 1完全相同的245-update
  schedule再训练一遍（共2 epochs / 490 updates），输出新目录`.../s35_d4/formal_v2`；
- 数据、LR、batch、LoRA覆盖、保留KL、seed全部不变；epoch 2复用的schedule与epoch 1逐位一致；
- `formal_v1`产物保持不动；判定标准已冻结：若formal_v2 dev NLL逼近或低于`SFT4(D4)`的`0.0957`，
  则重跑原冻结96-task roster的SFT35(D4) arm重评规模效应；若plateau，则规模screen以负结果定案；
- 不得读取promotion/holdout；不重复相同训练轮次刷结果（本轮是预算变量，不是重跑）。

## 6. 代码与Git状态

- 训练服务器仓库：`/home/wushaohua/data/MiniWebWork-RL`（本状态文件所在仓库）；
- 分支：`codex/m6-monotonic-posttraining`；
- 远端同步基线：`e18c1af9d67fd8826c97eb554c1bba208501f19b`（`docs: record S35 D4 formal training`，
  本轮开始前已由`c862671`安全快进）；
- Phase10-D same-corpus评测实现提交：`b8872b8`（roster builder、collector/wrapper/stats扩展、
  4项新聚焦测试；与既有12项相关聚焦测试合计16项全部通过，py_compile与bash -n通过）；
- 后续修复与加固提交（均为本地+origin）：`3b375e7`（物理显存预检门）、`0095b04`（sbatch export
  分隔符规范化）、`77a90a9`（CUDA_VISIBLE_DEVICES外查询物理设备）、`0aa13d4`（改为门控Slurm分配的
  job-local设备——探测证实本集群按设备追踪分配并在job内重映射，主机ID固定不可行）；
- `0aa13d4`推送origin时两次超时（网络/额度），origin暂停在`77a90a9`；后续提交时重试一次推送；
- 初版两arm（Jobs2394/2395）因GPU门物理ID查询失败fail-fast（exit 6:0），修正后重投Jobs2401/2402成功；
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
- Job2393：WebShop环境服务（CPU-only），`RUNNING`；
- Jobs2394/2395：初版两arm，`FAILED 6:0`（GPU门按主机ID查询在job内不可见设备，已修复，不重投）；
- Job2401：`SFT4(D4)` same-corpus评测arm，`COMPLETED 0:0`（33:41）；96 task×K4=384轨迹，
  143 strict success；collection_report `content_sha256`
  `a44f4931f9a5d309354c0384784865710f82b8ad4e1fa84d7674b8e77e665829`；
- Jobs2404/2406/2407：extend训练初版尝试，均`FAILED 1:0`（peft.load_adapter WeightConverter
  TypeError；load_state_dict被PeftModel覆写误报；lora_A/B key与live参数`.default`后缀不一致）；
  修复提交`6b71ab5`/`94a9d23`/`4d3a1d6`后重投成功；
- Job2409：`S4(D35)` 2-update探针，`COMPLETED 0:0`（01:01）；9/9门通过（含exact capability
  loss mass、single GPU placement）、128个LoRA target（q/k/v/o+gate/up/down）、256张量真实更新；
- Job2410：`S4(D35)` 108-update正式训练（2×2第四格），已确认`RUNNING`；输出
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/s4_d35/formal_v1`；
- Job2408：`SFT35(D4)` extend训练（第2 epoch，+245 updates，输出`.../s35_d4/formal_v2`），
  `COMPLETED 0:0`（02:42:44）；12/12门通过（含resume参数SHA恒等、dev NLL连续性）；dev NLL
  `0.116896 -> 0.119291`（无下降），retention KL均值27.86、raw梯度范数均值117；报告
  `content_sha256 82575cde451f8a6c7c9e476891d42b90569952b4cd879f3e28d93567a65fbe64`；
- Job2402：`SFT35(D4)` same-corpus评测arm（含formal adapter合并Raw35），`COMPLETED 0:0`（50:31）；
  384轨迹，142 strict success；collection_report `content_sha256`
  `be03c5c0d88ca42fd2b542f8a4cc5375d01f44f2dd24ed54a2209b33ab43c760`；
- 配对统计（seed错配0，task order一致）：
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/same_corpus_eval/sft4_vs_sft35_d4_stats.json`
  （`content_sha256 dcb4cfb04b94f5cb9a4b04f09e187a5adb42a66d16629f9145c272e19af37e3a`）；
- D4信息瓶颈检查（只读分析）：
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/bottleneck_analysis/d4_bottleneck_check_v1.json`
  （`content_sha256 8d9b3366fb43ef5792d3c924558dce9e1e7d9d0bcdc9e56d0f9df2d57e707a64`）；
  SFT35(D4)合并模型
  `outputs/m6_monotonic_posttraining_v1/phase10d_same_corpus_scale_v1/s35_d4/formal_v1/merged_model_v1/`；
- Raw35 vs SFT35统计：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_stage_eval_v2_4b_paradigm/dev/raw35_vs_sft35_stats.json`；
- SFT4 vs SFT35统计：
  `outputs/m6_monotonic_posttraining_v1/phase10c_qwen35_sft_specialist_opd_v1/teacher_stage_eval_v2_4b_paradigm/qualification/sft4_vs_sft35_stats.json`。

## 8. 下一步唯一动作

用户已选择`complete_2x2_sft4_d35`。Job2409探针已过、Job2410正式训练已提交并确认`RUNNING`
（预计约30-60分钟，按+50分钟查收一次）。查收时审计formal_v1报告（self-hash、108次更新、
capability mass、dev NLL、passed）；通过后进入2×2配对评测轮：用same-corpus同款排除规则**新建**
fresh roster（额外排除已查看的Phase10-D 96-task roster与S4(D35)训练任务），对
`SFT4(D35)`（需合并adapter到Raw4）vs `SFT35(D35)`（已有merged model）做同task/K/seed配对；
该评测提交需在Job2410审计通过后再申请授权。若Job2410 FAILED：读日志定位，如实汇报，不自动重投。

## 9. 后续决策

- ~~若 `SFT35(D4)>SFT4(D4)`~~（未发生，same-corpus评测为统计持平，见第2节）；
- **当前路径（已触发）**：统计持平 → 不能直接否定35B，先检查D4是否对35B形成信息瓶颈（第8节）；
  不重复相同训练轮次刷结果；
- 若瓶颈检查确认D4覆盖/多样性不足：优先低成本扩大或增强D4类语料，或按明确信号补`SFT4(D35)`
  形成现有语料2×2；
- 若 `SFT35(D4)<SFT4(D4)` 且差距显著：当前D4下4B吸收/泛化更优，优先转向4B专项自纠错；
- 只有现有语料交叉训练出现明确模型或数据来源效应，才建设全新同任务、同协议2×2；
- 任何OPD或GRPO均在上述归因完成且教师/纠错门通过后再执行。

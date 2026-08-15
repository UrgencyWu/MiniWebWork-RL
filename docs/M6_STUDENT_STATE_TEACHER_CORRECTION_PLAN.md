# M6 Phase10：学生状态教师纠错蒸馏与学生在线 GRPO 计划

> 状态：原“单教师纠错 SFT -> 学生 GRPO”路径在 equal-source-loss-mass single-update readiness 中因
> rehearsal fixed-state KL=`0.03016 > 0.01` 停止，未进入 qualification 或正式训练；本文件新增
> Phase10-B 多 Specialist On-policy Distillation（OPD）后续计划，目前只完成模型可用性/tokenizer
> 兼容性盘点，尚未提交 OPD 采样或训练作业
>
> 日期：2026-08-15
>
> 目标：保留并解释单教师纠错路径的 readiness 负结果；下一轮验证“学生保持 behavior policy、多个
> 强模型 Specialist 只在学生实际访问的 token prefix 上提供分布监督，通过 OPD 汇总回统一 4B
> Student，随后再由统一 Student 执行可选 on-policy GRPO”能否建立稳定的 Raw < SFT < OPD < RL
> 正向链路。

## 1. 执行结论

下一轮不再继续搜索 credit window、增加 K、追加相同起点 seed 或放大既有 binary-GRPO 训练。
唯一新增变量是**学生状态上的教师纠错数据**。训练链冻结为：

```text
现有 Self-SFT 学生 pi_0
  -> 学生在全新 train 任务上采样
  -> 从 partial / wrong-option / horizon 轨迹提取公开失败状态
  -> 教师从完全相同的学生状态生成短纠错 suffix
  -> WebShop 严格执行与重放验证
  -> prefix masked、suffix-only 纠错蒸馏，得到 pi_1
  -> pi_1 在独立 RL 任务上自行 on-policy K4 采样
  -> strict-binary trajectory-GRPO，得到 pi_2
  -> 全新 dev3 同条件评测 Raw / SFT / pi_1 / pi_2
```

教师只参与纠错数据生成。教师轨迹不得进入学生 GRPO policy-gradient batch，不得伪造为学生
behavior policy。最终推理只运行学生模型。

## 2. 背景与新假设

### 2.1 已排除的解释

既有 M6 实验已经表明：

- Raw -> SFT 在 fresh tuning-dev2 上为 `+5.469 pp`，说明严格成功、环境重放验证的监督数据有效；
- Phase4 full-credit GRPO 相对 SFT 为 `-0.781 pp`，step-5 为 `-0.977 pp`；
- tail-2 与 preterminal-1 分别恢复到 `-0.391 pp` 和 `-0.195 pp`，但仍未超过 SFT；
- Phase7 中，40-task online 数据的 26 个 mixed task 只有 3 个 same-item strict-partial task、
  2 个 option-contrast task；
- Phase9 虽然实现 8/8 shared-prefix exact replay，但 32 条 suffix 有 29 条 strict，只有 1 个
  same-item strict-partial task，说明 strict source prefix 位于过于容易的一侧。

因此，当前瓶颈不是“optimizer 没有真实更新”，也不是“继续压缩 policy loss 到更少动作”即可解决，
而是当前学生自采样数据没有稳定命中同商品、option/购买决策仍不确定的边界状态。

### 2.2 与旧教师路线的区别

Phase4 曾让 Qwen3.5-9B 和 Qwen3.6-35B-A3B-FP8 从任务起点独立解决 16 个学生 all-failure
任务；两者分别只在 2/16 和 1/16 个任务上产生 replay-verified strict success，未通过资格门。
该负结果永久保留，但它检验的是：

```text
teacher 从任务起点独立搜索并完成整个任务
```

本计划检验的是不同假设：

```text
state 由学生真实策略访问
teacher 只从学生失败前的公开状态提供纠正 suffix
```

该方法减少教师的搜索负担，使监督集中在学生已经证明存在错误的状态，并降低完整教师轨迹与
学生部署分布之间的偏移。旧教师负结果不构成本计划自动通过的证据，也不能用来免除新的教师资格门。

### 2.3 形式化目标

学生访问状态：

```text
s ~ d_student
```

教师只在该状态上给出纠正行为：

```text
y_teacher ~ pi_teacher(. | s)
```

教师数据用于 suffix-only 监督目标，而不是直接用于 policy gradient：

```text
L_corr = - mean_task mean_{t in teacher_suffix} log pi_student(a_t | student_prefix, a_<t)
```

纠错后学生 `pi_1` 再自行产生 RL 轨迹：

```text
tau ~ pi_1
R(tau) = 1[official task_score >= 0.999]
```

这样把“外部能力注入”和“学生 on-policy 稳定化”分成两个可独立归因的阶段。

## 3. 模型角色与禁止混用

| 符号 | 角色 | 数据来源 | 允许的训练用途 |
|---|---|---|---|
| `pi_raw` | 冻结 Raw 基线 | 历史冻结 checkpoint | 只评测，不重训 |
| `pi_0` | 当前 Self-SFT 学生 | 历史 replay-verified strict 语料 | 本计划起点、student rollout policy |
| `pi_T` | 候选教师 | 待资格测试的外部策略 | 只生成学生状态纠错 suffix |
| `pi_1` | 教师纠错后的学生 | `pi_0` + corrective distillation | 新 reference、RL behavior 起点 |
| `pi_2` | 最终 RL 学生 | `pi_1` 自己的 on-policy 轨迹 | 最终候选 |

禁止事项：

- 不把 `pi_T` 轨迹当作 `pi_1` 的 on-policy GRPO 轨迹；
- 不用教师自评分替代 WebShop strict reward；
- 不用旧 SFT checkpoint 作为 `pi_2` 的 KL reference，reference 必须是 `pi_1`；
- 不在同一首轮同时改变 reward、credit window、K、horizon、LoRA rank 或 policy epochs；
- 不在最终推理阶段调用教师。

### 3.1 执行根与运行时路径

本计划的权威执行环境在远端；本地工作树只用于代码和文档修改：

```text
REMOTE_REPO_ROOT = /home/wushaohua/data/MiniWebWork-RL
LOCAL_WORKTREE    = /Users/wsh/Documents/MiniWebWork-RL/m5_worktree
PYTHON            = /home/wushaohua/miniconda3/envs/miniwebwork/bin/python
SLURM_BIN         = /opt/slurm/slurm.25.05/bin
```

任何 Slurm wrapper 默认 `M6_REPO_ROOT=$REMOTE_REPO_ROOT`，并在执行前绑定 expected Git SHA。

### 3.2 冻结学生模型

| identity | 模型/adapter | 权威远端路径 | 说明 |
|---|---|---|---|
| `pi_raw` | Qwen3.5-4B，无 adapter | `/data/share/model/Qwen3.5-4B` | Raw 评测基线，不训练 |
| `pi_0` | Qwen3.5-4B + M6 SFT LoRA | `/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter` | 本计划学生起点 |

学生 base model 与 tokenizer 均冻结为：

```text
/data/share/model/Qwen3.5-4B
```

base model manifest 冻结为：

```text
/home/wushaohua/data/MiniWebWork-RL/data/m4_long_horizon_base_model_manifest_v1.json
```

`pi_0` adapter 必须与既有 M6 SFT identity/semantic hash 完全一致；不得使用任何 Phase4–6 RL
adapter、step-5/step-10 checkpoint 或旧 M5 adapter 作为本计划起点。纠错 SFT 与 rehearsal 均从
相同 `pi_0` 权重复制；LoRA 继续采用既有 `r=16, alpha=32` target-module 合同，除本计划明确新增的
multi-source/task-normalized loss 外不改模型容量。

### 3.3 冻结教师模型

本次只预注册一个 teacher：

| identity | 模型 | 权威远端路径 | adapter |
|---|---|---|---|
| `pi_T` | Qwen3.6-35B-A3B-FP8 | `/data/share/model/Qwen3.6-35B-A3B-FP8` | 无 adapter |

教师 tokenizer 从同一路径加载。教师使用与学生相同的 WebShop compact prompt 内容和 canonical action
schema，但由教师自己的 tokenizer 推理；录取后的 canonical action label 必须重新用学生
`/data/share/model/Qwen3.5-4B` tokenizer 编码后进入 corrective SFT。

teacher qualification 前必须生成并验证：

```text
/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/phase10_student_state_teacher_correction_v1/readiness/teacher_base_model_manifest.json
```

选择该 35B 模型只表示把它作为单一 state-local correction 候选，不表示它已证明更强。它此前从
任务起点独立解题的上界探针只在 1/16 个学生 all-failure task 上成功；本计划的新假设是它在学生
已到达的失败边界状态上可能更有效。因此 32-task teacher qualification 仍是硬门。资格失败时：

- 不自动回退到 `/data/share/model/Qwen3.5-9B`；
- 不继续寻找更大模型；
- 不更换 teacher prompt/temperature 后复用同一 qualification roster；
- 只记录 state-local teacher 假设未过门并停止。

`/data/share/model/Qwen3.5-9B` 只作为历史 Phase4 负对照身份保留，不参与 Phase10 训练或数据生成。

### 3.4 Prompt、环境与已有数据路径

```text
PROMPT_CONTRACT = webshop_agent_v1_compact
PROMPT_PATH     = /home/wushaohua/data/MiniWebWork-RL/prompts/webshop_agent_v1_compact.txt
SPLIT_LOCK      = /home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/locks/m6_webshop_split_v1.json
GOALS           = /home/wushaohua/data/MiniWebWork-RL/outputs/m5_webshop_credit_assignment_v1/upstream/webshop_full/goals.json
SFT_TRAIN       = /home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/corpus_v2/train.jsonl
SFT_DEV         = /home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/corpus_v2/dev.jsonl
```

WebShop 服务地址不复用历史主机名，运行时由 `M6_SERVICE_BASE_URL` 显式传入并写入 invocation；
服务必须由当前仓库版本启动并通过 `/health`。不同阶段允许重启服务，但 student、teacher、fresh
replay 和 evaluation 必须绑定相同的环境/catalog identity 与版本 hash；同一候选的首次执行和
fresh replay 不得混用不同 catalog identity。

### 3.5 Phase10 输出与派生模型路径

权威输出根冻结为：

```text
PHASE10_ROOT = /home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/phase10_student_state_teacher_correction_v1
```

派生模型不通过手写 checkpoint 路径推断，而由对应 `run_report.json::final_adapter` 唯一解析：

| identity | report 路径 | 解析字段 |
|---|---|---|
| `pi_rehearsal` | `$PHASE10_ROOT/rehearsal_sft/run_report.json` | `final_adapter` |
| `pi_state_self`（可选） | `$PHASE10_ROOT/state_self_sft/run_report.json` | `final_adapter` |
| `pi_1` | `$PHASE10_ROOT/corrective_sft/run_report.json` | `final_adapter` |
| `pi_rehearsal_RL` | `$PHASE10_ROOT/rehearsal_rl/run_report.json` | `final_adapter` |
| `pi_2` | `$PHASE10_ROOT/corrective_rl/run_report.json` | `final_adapter` |

每个 report 必须同时记录 base model、input adapter、output adapter、tokenizer、prompt、teacher（若适用）、
Git、数据 manifest 和 semantic hash。禁止把路径存在本身当作模型身份；字段和 hash 不闭合时 fail closed。

## 4. 数据切分

实现前必须先生成版本化 `phase10_exposure_union.json`，逐来源记录 `task_id`、`goal_index`、
normalized instruction hash、暴露原因和来源产物 hash。该并集至少包含：

- Raw 两 seed 覆盖的完整 mini_train 256，而不只是进入 SFT 的 156 task；
- SFT corpus train/dev 与历史 mini-dev、formal-dev、tuning-dev2；
- M6 Phase1–Phase3 的所有 probe/diagnostic roster；
- Phase4 三批 student prescan、16-task 9B/35B teacher probe、40-task online roster；
- Phase5 tail-credit 与 Phase6 preterminal-credit 的探针、在线训练任务和对应已看评测任务；
- Phase7 targeted prescan、Phase8 branchable source 和 Phase9 8-task smoke；
- 所有虽未进入 optimizer、但已用于假设选择或数据筛选的任务。

暴露并集必须 fail-closed：来源缺失、hash 漂移或角色未知时不得构建新 split。随后从仍未暴露的
WebShop `train` role 中创建以下互斥集合：

| split | 建议规模 | 用途 | 是否训练 |
|---|---:|---|---|
| `teacher_qualification` | 32 tasks | 证明教师能从学生失败状态恢复 | 否 |
| `correction_train` | 64 tasks | 构造教师纠错蒸馏语料 | 是，仅纠错蒸馏 |
| `correction_monitor_a` | 64 tasks | 低成本方向 screening | 否 |
| `correction_monitor_b` | 64 tasks | 独立 confirmation；仅 A 过门后启用 | 否 |
| `rl_train` | 40 tasks | `pi_1` 学生 on-policy GRPO | 是，仅 RL |
| `dev3` | 500 tasks x K4 | 一次性 paired 最终开发评测 | 否 |

所有集合必须满足：

- normalized instruction / task identity 彼此互斥；
- 相对 `phase10_exposure_union` 的 task、goal-index、normalized-instruction 交集全部为 0；
- promotion/holdout 不读取；
- 按 category x constraint-count x 预估长度分层；
- 六个新角色之间 task、goal-index、normalized-instruction 两两零交集；
- 同一任务只承担一个角色；
- 任务顺序、split producer、selection seed 和自哈希在采样前冻结。

formal-dev 与 tuning-dev2 已 burn，不承担本计划的选择或晋级判断。`dev3` 只在所有前置门通过后
运行一次；结果一旦查看即 burn。

## 5. 学生失败状态生成

### 5.1 冻结采样配置

`pi_0` 在 `teacher_qualification` 和 `correction_train` 上使用：

```text
K                               4 / task
model / environment horizon     18 / 15
policy                          frozen pi_0
sampling seed                   split-specific, frozen
optimizer steps                 0
public observations only        true
```

必须保留 all-success、mixed 和 all-failure，不使用“采到 mixed 为止”的动态过滤。

### 5.2 失败分类与纠错位置必须分离

终局 scalar `task_score` 可以由离线 verifier 用于把轨迹分类为 strict、partial 或 zero-score，
但它不是 policy-visible 字段，不能参与纠错位置索引，也不能进入 teacher prompt。纠错点必须只由
学生公开动作、policy-visible observation、available actions、终止标志和已消耗预算决定。

增加 fail-closed 泄漏测试：删除 goal dict 中的 target ASIN、structured goal options、内部商品名、
score component 和 post-action internal state 后，correction-point index 必须逐样本完全不变。教师
collector 的输入类型只接受 safe public instruction text、safe observation/history 和 remaining budget。

### 5.3 失败类型优先级

按下列优先级选择纠错候选：

1. `purchase_failure`：所有 non-strict purchase 使用同一 public selector；`task_score` 只在 selector
   完成后用于离线分层为 partial/zero-score；
2. `public_option_conflict / premature_buy`：仅当 instruction 明示约束且当前公开 selected option
   与之冲突时成立；否则统一使用 pre-Buy 状态，不用 oracle 判断“错误 option”；
3. `horizon`：在非终局、仍有合法公开动作的商品页或搜索结果页耗尽预算；
4. `schema/action`：仅用于教师动作格式诊断，不作为首轮主要能力数据。

当前历史失败以 partial purchase 为主，因此首轮至少一半录取纠错任务必须来自 partial 路径。

### 5.4 纠错点选择

纠错点由 public-only 规则产生，不依赖隐藏答案：

- 所有 non-strict purchase：统一取最终 `Buy Now` 前的公开状态；只有 public instruction text 明示
  目标 option 且公开页面显示 selected option 冲突时，才允许前移到该 option click 前状态；
- horizon：取最后一个非终局、仍有合法公开动作且可精确重放的商品页/搜索结果状态，不使用
  “隐藏 oracle 判断可恢复性”；
- 每个候选都必须从任务起点精确重放学生 prefix，并匹配 public-state hash；
- terminal `Buy Now` 之后不能纠错。

传给教师的上下文只包括：任务指令、学生公开动作历史、当前 policy-visible observation、
available actions 和剩余预算。

剩余预算按学生 prefix 的真实消耗扣除：

```text
suffix_model_cap = 18 - prefix_model_turns
suffix_env_cap   = 15 - prefix_env_steps
```

教师不得从纠错状态重新获得完整 18/15 预算。

## 6. 教师资格与纠错生成

### 6.1 教师资格探针

首轮只预注册一个教师模型、版本、prompt 和 sampling 配置。在 `teacher_qualification` 的 32 个
任务上运行，不训练学生。每个 task 最多按公开失败优先级选择一个纠错状态；出现并列时用冻结 hash
确定性打破。没有 eligible failure state 的 task 计作 qualification fail/infeasible，不调用教师，
也不从分母删除。每个有状态的 task 总计允许 `pi_0` 自恢复最多 2 次、教师恢复最多 2 次，双方使用
完全相同的 prefix、remaining budget 和 rollout seed schedule。分别报告 student/teacher pass@1 与
pass@2，不能把 any-of-many 表述为单次成功率。

教师候选只有全部满足以下门槛才可进入正式纠错语料构建：

- 至少 20/32 个任务在最多 2 次尝试内获得 replay-verified strict correction；
- 所有正式录取 suffix 均满足首次执行 strict 且 fresh-session full replay strict；
- replay stability >= 95%，分母定义为“首次执行 strict 的全部 teacher suffix 候选”，不是录取集；
- 至少 12 个任务形成 same-state、same-item 的 option/购买纠错；
- 所有动作均来自当时公开 available actions；
- target ASIN、隐藏答案、post-action 内部 state 使用数为 0；
- 教师严格成功必须发生在从 prefix 消耗后扣除的剩余预算内；
- 每个任务最多录取 2 条语义不同的 strict suffix。

同时报告 teacher 相对 state-matched student 的新增可恢复任务数。若 student 同状态自恢复已经与教师
相当，则教师没有证明额外支持扩张，不进入正式 corpus。

若教师未过门，本轮停止，不通过继续扩大教师模型、K、seed 或任务数刷结果。更换教师属于新的
显式假设，必须使用新的 qualification roster；原 32 task 一旦查看结果即 burn。

### 6.2 教师纠错执行

教师必须真实地与 WebShop 交互，不能离线编写一串未执行命令：

1. 精确重放学生 prefix 到候选状态；
2. 将同一公开状态提供给教师；
3. 教师从公开 available actions 中选择动作；
4. 环境执行动作并返回下一公开状态；
5. 教师继续生成，直到 strict success、失败或预算耗尽；
6. 在独立新 session 中从任务起点重新重放“学生 prefix + 教师 suffix”，逐 turn 核对 prefix
   public-state hash；
7. 只有两次执行均 `task_score >= 0.999` 的 suffix 才能录取。

教师 query/action provenance 必须额外满足：

- search query token 只能来自 public instruction text、当前/历史 public text；
- 任何 ASIN-like token 只有先在公开页面出现后才允许进入 search query/action；
- target ASIN、隐藏精确标题和内部 goal 字段访问计数为 0；
- query 长度不超过冻结上限；
- 每一步 teacher action 都属于当时公开 available actions。

### 6.3 纠错样本结构

学生失败 suffix `y_minus` 也必须从同一 prefix 在独立 session 重放，并确认仍为 non-strict；若失败
类别不稳定，记录转移而不把它当稳定的强负例。

每条样本同时保留：

```text
x       = task + student public prefix + correction-state observation
y_minus = original student failure suffix
y_plus  = replay-verified teacher strict-success suffix
```

`y_plus` 与 `y_minus` 只有在首个动作分歧处是真正 same-state。builder 必须寻找双方最长共同公开
action-state prefix，把训练状态移动到首次分歧点，并要求正负首动作不同。后续 observation 已分叉，
不能把整个多步交互误写成普通同-prompt DPO pair。

并记录：

- task/split/student/teacher/environment/prompt/tokenizer identity；
- source trajectory、prefix turn、prefix action/token/public-state hash；
- student suffix outcome 与 failure class；
- teacher suffix actions、strict reward、replay result、token/step cost；
- correction type：public-option-conflict、premature-buy、zero-score-purchase、horizon-recovery；
- leakage flags；
- student/teacher attempted states、pass@1、pass@2、first-strict 与 replay-strict 分母；
- content hash 与 producer Git。

## 7. 纠错语料录取与配额

录取要求：

- strict success 与 replay strict success；
- student prefix exact replay；
- teacher suffix 全部 schema-valid、public-action-valid；
- `correction_train` 每任务最多 1 个冻结 state、每 state 最多 2 条纠错 suffix；
- 对 action sequence 与语义等价 suffix 去重；
- 任务级采样，不能让长 suffix 通过更多 action rows 获得更大总权重；
- 同一学生失败状态的多个教师成功只保留最短严格成功和至多一个语义不同恢复路径。

same-item option/购买纠错要求 teacher strict suffix 从公开 ASIN 状态开始后不切换到另一 ASIN；
否则只能标记为“same-start-state recovery”，不能计入 same-item 门。选择最短 strict suffix 是冻结的
selection rule，必须同时报告全部 strict 候选及未录取原因。

首轮 `correction_train` 语料门：

- 至少 40 个 unique task 有 strict correction；
- 至少 24 个为 same-state、same-item option/购买纠错；
- partial 路径任务占比 >= 50%；
- replay pass >= 95%；
- category/constraint bucket 相对 `correction_train` 任务池偏差不超过 10 pp；
- 无 task/split/leakage 冲突。

训练前必须生成 `control_feasibility_report.json`：对每个冻结 correction state，统计 state-matched
student 两次 suffix 是否能 strict、teacher 是否能 strict，并报告 task/category/constraint/failure
type、prefix/horizon、suffix action rows 和 labeled token。另报告 attempted -> first-strict ->
replay-strict 的 bucket 转化率，不能只报告被录取后的高质量子集。

未过门不提交纠错训练。

## 8. 纠错蒸馏训练

### 8.0 训练前工程阻断门

在读取正式 qualification 结果或提交纠错训练前，先使用已暴露、不会进入任何性能统计的 8 个
historical train task 完成端到端 smoke：exact student prefix replay、state-matched student suffix、
teacher live suffix、first strict、fresh-session full replay strict、query provenance 和 tokenizer mask。
该 smoke 只证明工程链可运行，不产生教师能力结论。

在查看该 smoke 结果前冻结工程门：student/teacher 均须 8/8 exact prefix replay，8/8 task 的
correction prompt hash 跨身份一致；teacher 至少产生 1 条 first-strict suffix，且所有 first-strict
suffix 的 fresh-session full replay strict 率必须为 100%；student non-strict suffix 的 fresh replay
failure class 必须稳定；所有 suffix search 均通过 public-query provenance；所有进入候选 corpus 的
assistant action row 在学生 tokenizer 下 label token 非空，system/instruction/public observation/prefix token
全部为 `-100`。任何一项失败即修实现错误或停止，不得放宽该工程门来进入 qualification。

随后用已暴露样本完成 teacher/rehearsal matched single-update probe，验证 source weighting、prefix
gradient=0、action gradient>0、`pi_0` reference/retention、sampler recovery 和参数更新。两项任一失败，
不得启动 32-task teacher qualification 或正式训练。

### 8.1 首轮只做 suffix-only SFT

首轮不同时增加 DPO、offline RL 或新的 process reward。学生 prefix 只作为条件上下文，所有 prefix
token 的 label mask 为 0；只有教师 suffix token 进入交叉熵：

```text
system / public instruction / student prefix       labels = -100
environment observations after every teacher turn labels = -100
teacher canonical action tokens                    labels = token ids
```

多步 suffix 不能被拼成没有环境 observation 的动作串。推荐每个 teacher turn 构造一条 completion row：
prompt 包含 exact student prefix、此前 teacher action 和逐步环境返回，只监督当前 canonical action；
或者在完整 multi-turn chat 中只 label correction boundary 之后的 assistant action。mask 边界必须在
学生 tokenizer/chat template 应用后计算，BOS/system/user/observation 全部 mask。

损失按 source、task、state、path 和 action token 分层等权：

```text
L_source = mean_task mean_state mean_path mean_suffix_action_token CE
L_total  = 0.60 L_teacher + 0.25 L_old_SFT + 0.15 L_student
```

同 task 两条 teacher path 必须先在 task 内平均，不能获得双倍任务权重。每个 source 报告 task draw、
state/path、action row、labeled token 与实际 loss contribution。

### 8.2 数据混合

首轮冻结数据采样比例：

| 数据 | 比例 | 作用 |
|---|---:|---|
| teacher correction suffix | 60% | 注入学生原本无法稳定产生的纠正行为 |
| original SFT replay-success retention | 25% | 保持 Raw -> SFT 已获得能力 |
| current student replay-success trajectories | 15% | 保持当前策略状态分布 |

现有 M6 SFT trainer 不能直接复用：它按 completion token 总量归一、使用固定 imitation/retention
schedule，且 disable-adapter reference 指向 Raw。Phase10 必须先实现可恢复的 multi-source task sampler
和上述显式 source-weighted loss。纠错训练从 `pi_0` 起步；首轮唯一冻结 retention 机制是表中 25%/15%
的 replay-success CE，不加入 reference KL，避免把 Raw 或另一 reference 选择引入为额外算法变量。
冻结 `pi_0` 只用于 fixed-state KL/NLL 无梯度诊断。teacher/rehearsal 两 arm 的 retention 机制完全相同。

“1 epoch”改为 readiness report 中预先冻结的 optimizer updates、各 source task draws、effective labeled
tokens 与每 task 最大重复数；该预算只能由录取 corpus 规模决定，不能依据 monitor/dev3 结果调整。
teacher/rehearsal 两 arm 使用完全相同的 update、RNG、sampler schedule 和起点。

正式训练前先做一个 matched single-update probe，必须证明：

- prefix/system/observation token 的梯度严格为 0；
- teacher canonical action label 数 > 0、梯度有限非零；
- teacher suffix NLL 按预期下降；
- `pi_0` fixed-state KL、retention 和参数位移在安全范围；
- sampler/recovery 保存并恢复 source cursor、task/state/path order、optimizer、adapter 和全部 RNG。

该探针在读取结果前冻结为：从同一 `pi_0` 起点各运行 teacher/rehearsal 一个 optimizer update，
学习率沿用 M6 mini-SFT 的 `2e-5`、weight decay 0、gradient clip 1.0；`new/old-SFT/current-student`
loss mass 精确为 `0.60/0.25/0.15`。两臂均要求 new-source NLL 严格下降、fixed-state sampled k3 KL
`<=0.01`、old-SFT 与 current-student NLL 各自增加 `<=0.10 nat`、LoRA 相对 L2 位移位于
`(0,0.01]`，且 optimizer-boundary recovery/RNG round-trip 完整通过。该一次更新的 adapter 标记为
`formal_checkpoint_reusable=false`，只用于 readiness，不进入 qualification 或任何性能评测。

### 8.3 同状态自恢复探针与训练对照

每个 correction state 都先运行 state-matched `pi_0` 自恢复 probe：与教师使用相同 prefix、remaining
budget、最多 2 次尝试和 seed schedule。这直接测量 teacher 相对 student 的恢复增量，但不自动产生
足量训练对照。

训练时至少保留一个 compute-matched rehearsal control，用于回答“教师纠错包是否优于多做一次
SFT”：

- 60% 新数据槽由 `pi_0` replay-verified strict/self-success 或冻结 `pi_0` retention 填充；
- optimizer updates、source task mass、category/constraint bucket、labeled action token、suffix length
  和 seed 与教师 arm 硬匹配，容差在 readiness report 中冻结，不能只写“尽量匹配”；
- retention 比例、LoRA 起点、optimizer 和 seed 与教师 arm 相同；
- 不使用教师 suffix。

若 state-matched student strict suffix 足够覆盖冻结门，则额外构建 `pi_state_self`，在 exact matched
state 子集上进行主因果分析；若不足，必须显式报告 control infeasible。此时 rehearsal control 只能
控制“额外训练预算/额外 self-success 数据”，`pi_1 - control` 只能称为**teacher corrective package
净效应**，不能声称纯 teacher action 效应。

模型身份冻结为：

| identity | 含义 |
|---|---|
| `pi_0` | 当前 SFT 基线 |
| `pi_rehearsal` | 等预算继续 SFT / self-success 控制 |
| `pi_state_self` | 可选：exact state-matched student-success 控制 |
| `pi_1` | 教师纠错 suffix 数据 |

主报告必须区分 `pi_1 - pi_rehearsal` 的 package effect 与可选 exact-state subset effect。

#### 8.3.1 精确路径匹配不可行后的批准修订

CPU feasibility 已证明现有441条original-SFT完整成功路径中没有路径能与选定teacher suffix同时匹配
2个action row和29个student-token label。经用户批准，single-update及后续rehearsal主对照改为
**等optimizer/update与等source loss mass的continued-SFT control**：两臂仍从同一`pi_0`起步，
`new/old-SFT/current-student` loss mass精确为60%/25%/15%，同一optimizer、学习率、seed与retention
输入；rehearsal new-source从与teacher task相同Phase10 category/constraint/length bucket的完整
replay-success self path中按冻结seed hash选择。原始action row、label token和prompt长度只报告、不再作为
相等门，因为分层mean-token loss已将每个source的总梯度权重固定。

该修订只允许估计“teacher corrective package相对等loss-mass continued-SFT”的净效应；
`pure_teacher_action_effect_claim_allowed=false`。旧的精确匹配负报告永久保留，不能回写为通过。

### 8.4 可选偏好阶段

只有 suffix-only corrective SFT 在未见 monitor 上通过门禁后，才允许另立 DPO 实验。若以后使用
DPO，只能在最长共同公开前缀后的首次分歧状态比较单步 action，不能把带有不同后续 environment
observation 的完整 suffix 当普通 DPO pair。DPO 不属于首轮计划。

## 9. 纠错蒸馏监控门

显式评测三个面板：

1. `correction_train`：测量状态/任务拟合；
2. `correction_monitor_a/b`：按 category、constraint、failure difficulty 匹配的未见迁移；
3. `dev3`：只在所有前置门通过后测总体未见迁移。

先在 `correction_monitor_a` 上从任务起点运行 `pi_0`、`pi_rehearsal` 和 `pi_1`，相同 task、K4、
18/15、seed，教师不参与推理。A 只作低成本 screening：

- `pi_1 - pi_rehearsal` point strict >= +1 pp；
- `pi_1 - pi_0` point strict > 0、paired net flips > 0；
- paired task bootstrap `P(delta > 0) >= 0.8`；
- paired net flips > 0；
- 正向 task 至少覆盖 2 个 category x constraint bucket；
- partial、zero-score purchase、horizon、premature-buy、schema/action 均报告 paired transition table；
- key failure class（partial、zero-score purchase、horizon、premature-buy）各自 point rate 不增加
  超过 1 pp，四类合计 point rate 不增加超过 0.5 pp；schema/action 不增加超过 0.5 pp。

A 过门后才打开冻结的 `correction_monitor_b`，要求同方向 point delta 与净 flips；A/B 合并后报告
paired task bootstrap/permutation 或 exact discordant test。64-task A 或 B 单独不能证明 +1 pp，
只能作为 screening/confirmation。

`pi_1` 进入 RL 的最低门：

- A 与 B 的 `pi_1 - pi_rehearsal` 均为正；
- A 与 B 的 `pi_1 - pi_0` 均为正且 paired net flips 均为正；
- A/B 合并 point strict >= +1 pp、paired net flips > 0、`P(delta > 0) >= 0.8`；
- correction_train gain 与合并 monitor gain 的差距 <= 2 pp；
- 正向迁移覆盖至少 2 个 category、2 个 constraint bucket；
- 合并 monitor 的 key failure class 各自 point delta <= +1 pp、四类合计 <= +0.5 pp、
  schema/action <= +0.5 pp，并报告 paired 90% bootstrap upper bound；
- 原 SFT retention task strict 下降不超过 0.5 pp；
- inference tokens/steps 不超过 `pi_0` 的 1.10 倍。

若只在纠错训练任务上提升而 monitor 不提升，判定为状态/任务记忆，不进入 RL。若
`pi_1 <= pi_rehearsal`，判定教师纠错包未提供超过等预算 continued-SFT 的信息增益。

## 10. 学生 on-policy GRPO

只有 `pi_1` 通过 monitor 门才执行。RL 使用与纠错数据完全独立的 `rl_train` 40-task roster。为检验
“teacher correction 是否使 RL 更有效”，必须同时从 `pi_rehearsal` 和 `pi_1` 启动同配置 RL，
各自以自己的起点作为 frozen reference：

```text
pi_rehearsal -> pi_rehearsal_RL
pi_1         -> pi_2
```

```text
behavior/reference checkpoint     each arm's own frozen start
rollout policy                     current student only
reward                             strict binary
K                                  4 / task
task groups / optimizer step       4
curriculum batches                 10 frozen batches
optimizer updates                  at most 10
target unique tasks                40
model / environment horizon        18 / 15
RL dropout                         0
learning rate                      3e-6
policy epochs                      1
policy credit window               full
KL hard stop                       0.01
post-update KL warning             0.005
pre-update replay clip stop        0.01
```

选择 `full` credit window 是为了保持首轮唯一变量为教师纠错数据。tail-2 与 preterminal-1 已产生
有效负结果，不在本轮重新引入。homogeneous group 计入 attempted-task 和 token 预算。若某个冻结
4-task batch 全 homogeneous，则记录成本并执行零更新，继续下一个固定 batch；不得替换任务或追到
10个有效 update。每个 batch 的 policy loss 始终除以 4 个 attempted task-group；homogeneous group
贡献精确 0，不能改为只除 mixed group 数，否则有效学习率会随 mixed 数变化。报告 attempted/mixed/
effective groups、effective policy tokens、nonzero update 数。

K4x4 的 parity rejection 语义必须在实现前冻结：任一 group parity fail 时，首轮采用整 snapshot
batch 零更新并停止定位，不能仅跳过该 group 改变任务权重。single-epoch 下 initial clip 主要是
behavior/HF replay parity，而不是 PPO trust region，因此方法名记录为
`single-epoch trajectory group-normalized policy gradient + reference KL`。

GRPO policy loss 只使用学生当前策略产生的轨迹。教师纠错 corpus 在首轮 RL 中不得产生任何
auxiliary gradient；只能做无梯度 NLL/retention 诊断。两个 RL arm 使用相同 task order、rollout
seed、attempted budget 和配置，但每个 arm 的轨迹都由自己的 current policy 在线产生。

RL 恢复必须以原子 optimizer-step 为边界保存 batch index、40-task roster/order、rollout seeds/RNG、
policy/reference/optimizer SHA、attempted/mixed/effective group/token ledger。中断的部分 snapshot batch
必须隔离，并从最后已提交 policy 对同一 batch 整批重采/重放；不得续接半批或把前后 policy 轨迹
混为一个 on-policy batch。

## 11. 最终对照矩阵与归因

在全新 500-task `dev3` 上以相同 task、K4、18/15、prompt、tokenizer、环境和 rollout seed 评测：

| identity | 目的 |
|---|---|
| Raw | 原始基线 |
| `pi_0` | 当前 Self-SFT 基线 |
| `pi_rehearsal` | 控制“额外 SFT 预算/额外 self-success 数据” |
| `pi_rehearsal_RL` | 测量无教师纠错时相同 GRPO 的增量 |
| `pi_1` | 测量教师纠错蒸馏增益 |
| `pi_2` | 测量学生 on-policy GRPO 的额外增益 |

必须分别报告：

```text
Self-SFT gain             = pi_0 - Raw
extra rehearsal gain      = pi_rehearsal - pi_0
teacher package gain      = pi_1 - pi_rehearsal
self-start RL gain        = pi_rehearsal_RL - pi_rehearsal
post-correction RL gain   = pi_2 - pi_1
teacher x RL interaction  = (pi_2 - pi_1) - (pi_rehearsal_RL - pi_rehearsal)
end-to-end gain           = pi_2 - pi_0
```

不能只比较 Raw 与 `pi_2`，否则无法判断收益来自教师蒸馏还是 RL。若 exact state-matched
`pi_state_self` 可行，作为补充因果面板报告，不替代上述完整矩阵。

## 12. 性能晋级门

### 12.1 教师纠错成立

在 fresh `dev3` 上：

- `pi_1 - pi_rehearsal >= +1 pp`；
- paired teacher-only flips > rehearsal-only flips；
- 500-task paired bootstrap/permutation 的 95% CI 下界 > 0；
- partial、zero-score purchase、horizon、premature-buy 各自 point delta <= +0.5 pp、四类合计
  point delta <= 0；schema/action 各 <= +0.5 pp；所有指标报告 paired 95% upper bound，upper bound
  > +1 pp 时按安全性 inconclusive/失败处理；
- retention strict 的 paired 95% CI 下界 > -0.5 pp；
- 收益不局限于 `correction_train` 的 task/category bucket。

通过只证明教师纠错蒸馏有效，不证明 RL 有效。

### 12.2 RL 额外成立

- `pi_2 - pi_1 >= +1 pp`；
- RL-only flips > `pi_1`-only flips；
- 500-task paired 95% CI 下界 > 0；
- `pi_2 - pi_1` 高于或至少不弱于 `pi_rehearsal_RL - pi_rehearsal`，并报告 interaction；
- partial、zero-score purchase、horizon、premature-buy 各自 point delta <= +0.5 pp、四类合计
  point delta <= 0；schema/action 各 <= +0.5 pp；
- 对预注册安全指标报告 paired 95% non-inferiority CI；upper bound > +1 pp 或 CI 太宽时结论为
  inconclusive，不按通过处理；
- reference KL < 0.01；
- seen-unseen gap <= 1 pp；
- steps/tokens <= `pi_1` x 1.10。

只有同时满足：

```text
Raw < pi_0 < pi_1 < pi_2
```

且两个增量均满足各自门禁，才能宣称建立了“教师扩张支持集 + 学生 RL 继续提升”的链路。

若 `pi_1 > pi_0` 但 `pi_2 <= pi_1`，结论应为“教师纠错蒸馏有效，当前 GRPO 无额外价值”；不能把
教师收益归入 RL。若 `pi_1 <= pi_rehearsal`，停止教师方向。若两条 RL 分支均同幅提升，则只能
说明冻结 GRPO 配置本身有效，不能声称教师使 RL 成为可能。

## 13. 最小资源路径

计划本身不授权立即提交作业。实现完成并通过聚焦测试后，按门顺序执行：

| 阶段 | 主要作业 | 预计资源 |
|---|---|---|
| exposure union + split + match feasibility | CPU | 10–20 min |
| 8-task prefix/teacher/replay/mask smoke | 1 GPU | 10–30 min |
| matched single-update probe | 1 GPU | 20–40 min |
| student qualification rollouts | 1 GPU | 10–20 min |
| teacher correction qualification | 1 GPU，单一预注册 teacher | 20–60 min |
| correction_train student rollouts | 1 GPU | 15–30 min |
| teacher correction generation | 最多 2 GPU 并行 | 30–120 min |
| rehearsal / corrective SFT | 2 x 1 GPU | 各 1–4 h |
| monitor A/B | 每片最多 3 GPU | 每片 15–30 min |
| 两条 on-policy GRPO | 2 x 1 GPU | 各 20–60 min |
| dev3 六个核心 identities | 最多 4 GPU 并行 | 1–3 h |

单作业默认 1 GPU、4 CPU、24 GiB、24 h 上限；同时最多 4 GPU。正常长作业按预计完成时间检查，
不持续轮询。确定性实现失败只修直接根因；算法/数据门失败不得通过增加 successor 刷结果。

## 14. 停止条件

任一条件成立即停止对应方向：

- 教师资格未过门；
- 教师 correction replay pass < 95%；
- same-state/same-item纠错不足；
- 发现 target/hidden/internal-state 泄漏；
- `pi_1` 只在 correction_train 提升、monitor 不提升；
- `pi_1 <= pi_rehearsal`；
- corrective SFT 引起 retention 下降 > 0.5 pp；
- `pi_2 <= pi_1` 或 RL-only flips 不净正；
- post-update KL > 0.01、pre-update replay clip > 0.01 或出现非有限 loss/gradient；
- dev3 被用于调参后，不得继续作为未见晋级集。

本计划不允许在失败后自动增加教师规模、任务数、K、seed、epoch 或训练步数。新的变量必须形成新的
明确假设和独立计划。

## 15. 产物与实现里程碑

建议输出根：

```text
outputs/m6_monotonic_posttraining_v1/phase10_student_state_teacher_correction_v1/
  phase10_exposure_union.json
  split_lock.json
  readiness/
  smoke_8task/
  single_update_probe/
  teacher_qualification/
  correction_train_student_rollouts/
  correction_manifest.json
  correction_corpus/
  rehearsal_sft/
  state_self_sft/              # optional, only if exact-state control feasible
  corrective_sft/
  correction_train_eval/
  correction_monitor_a/
  correction_monitor_b/
  rehearsal_rl/
  corrective_rl/
  dev3_eval/
  final_stats.json
```

实现顺序：

1. 完整 exposure-union registry、split/roster builder 与 overlap tests；
2. student failure-state extractor、state-matched self probe 与 exact prefix replay tests；
3. teacher suffix collector、query provenance、public-only/leakage tests；
4. 8-task端到端 smoke：prefix replay -> teacher interaction -> fresh strict replay -> token mask；
5. correction corpus builder、multi-source task sampler、task-balanced suffix mask 与恢复测试；
6. matched single-update teacher/rehearsal probe；
7. rehearsal/corrective SFT wrappers 与 train/monitor paired stats；
8. 复用现有 Phase4 online GRPO，分别从 `pi_rehearsal`/`pi_1` 启动并各自使用同起点 reference；
9. 500-task dev3 paired evaluator、failure transition 与 interaction 归因统计；
10. 结果追加到迭代技术报告，逐 Job 工程失败追加到失败账本。

每个产物必须绑定 producer Git、输入自哈希、model/adapter/prompt/tokenizer/environment identity、
task order、seed 和角色。必要身份检查保留，但不扩张为新的通用审计系统。

## 16. 预期可证伪结论

本计划至少可以区分四种结果：

1. **教师无法纠正学生状态**：外部教师在当前环境/动作合同下不够强；停止。
2. **教师纠正能训练，但不迁移**：数据只记忆局部状态；停止或重做任务覆盖，不进入 RL。
3. **教师纠错蒸馏提升、GRPO无增量**：保留 `pi_1`，明确否定当前 RL 额外价值。
4. **教师纠错蒸馏与学生GRPO均提升**：首次建立可归因的 Raw < SFT < teacher-corrected SFT < RL
   链路，再请求批准进入更大规模和未打开切分。

该设计的关键不是让教师代替学生完成任务，而是让教师只在学生真实访问、真实失败的状态上提供
严格可验证的纠正信息；随后所有策略梯度仍由学生自己的在线轨迹产生。

## 17. 多智能体审查记录

2026-08-15 由三个独立只读 reviewer 完成审查：

| reviewer | 初审结论 | 主要阻断问题 | 修订处置 |
|---|---|---|---|
| data/evaluation | 方向正确，未达到 execution-ready | 历史 exposure 未完整枚举；self control 非同状态；64/128 task 无法判断 +1 pp；资格与 corpus 门不一致 | 新增 fail-closed exposure union；state-matched self probe；monitor A/B；500-task dev3；资格门改为20/32 |
| optimization | 算法边界正确，四项阻断 | control 无法匹配；缺少 control->RL 分支；现有 trainer 不支持多源 task loss；统计功效不足 | 新增 rehearsal/state-self feasibility；2x2 RL；明确新 multi-source trainer 与 pi0 reference；paired CI 门 |
| reward/credit | conditional approval | public-only selector 混入 oracle 风险；teacher query 泄漏与 replay 分母；homogeneous batch 合同；小样本安全门 | 纠错点与终局分类分离；query provenance；fresh replay/y-minus replay；10 fixed batches/最多10 updates；failure transition 与 CI |

审查共识确认：

- teacher suffix 只进入 corrective SFT，student on-policy GRPO 边界正确；
- 首轮不增加 DPO，不改变 strict binary reward，不重新引入 tail/preterminal credit；
- 在 exposure registry、8-task smoke、single-update probe、control feasibility、统计门全部完成前，
  本计划不授权提交正式训练；
- 资格门过严导致 fail-fast、teacher 数据随 `pi_1` 产生 state shift、单一 SFT seed 的机制噪声属于
  可接受 pilot 风险，但不得扩写为通用或显著性结论。

第二次复核又补充并已关闭以下合同：无 eligible failure state 的 qualification task 计 fail；GRPO
loss 恒除以4个 attempted group；纠错 SFT 只用冻结 CE retention、不保留 Raw/reference 二选一；
RL 原子 batch 恢复；所有 purchase failure 共用 public pre-Buy selector；monitor 同时要求
`pi_1 > pi_0`；关键失败类使用数值化 paired non-inferiority 门。

最终复核状态：data/evaluation 为 `conditional execution-ready`，optimization 为 `PASS`，
reward/credit 为 `PASS`。这里的 conditional 表示可以进入实现与 readiness，不表示可以跳过前置门
直接提交正式纠错训练或 RL。

经修订，三个 reviewer 的阻断意见均已转化为显式 readiness gate。实现阶段若无法满足其中任一项，
必须停止或降级结论，不得静默放宽。

## 18. Phase10-B：多 Specialist On-policy Distillation 后续计划

### 18.1 与 Phase10-A 教师纠错的边界

本节是原计划 readiness 失败后的后续方法修订，不把既有 corrective SFT 更名为 OPD。

两者的数据和优化边界不同：

```text
Phase10-A corrective SFT（已在 readiness 停止）
  teacher 自己生成 suffix action
  -> 录取 suffix 成为离线 CE label
  -> student 对 teacher sequence 做 suffix-only SFT

Phase10-B multi-Specialist OPD（本节计划）
  unified Student 自己采样 action/token，保持 behavior on-policy
  -> 在 Student 实际访问的完全相同 prefix 上查询一个冻结 Specialist
  -> Specialist 只返回该 Student token space 上的 logits/probability target
  -> Student 在自己的 on-policy token 上最小化 distillation loss
  -> 所有 Specialist 能力重新汇总到单一 Student
```

Phase10-A 已采样的 teacher suffix 可以作为未来 Specialist 构建或离线诊断的候选输入，但不得直接
计入 Phase10-B 的 on-policy distillation batch。Phase10-B 的 behavior token 必须由当前统一 Student
生成；若 Specialist token 被直接作为环境动作执行，该 batch 必须标记为 teacher-forced/off-policy，
不得进入 OPD 主臂。

### 18.2 冻结统一 Student

最终部署始终只有一个 Student：

| identity | 模型 | 权威远端路径 | 用途 |
|---|---|---|---|
| `pi_raw` | Qwen3.5-4B，无 adapter | `/data/share/model/Qwen3.5-4B` | Raw 基线 |
| `pi_0` | Qwen3.5-4B + M6 SFT LoRA | `/home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter` | OPD Student 起点与 behavior policy |
| `pi_opd` | Qwen3.5-4B + OPD adapter | 由 `$PHASE10B_ROOT/opd_student/run_report.json::final_adapter` 解析 | 汇总多个 Specialist 后的统一 Student |
| `pi_opd_rl` | `pi_opd` + 可选 Student-only GRPO adapter | 由 `$PHASE10B_ROOT/opd_rl/run_report.json::final_adapter` 解析 | 最终候选 |

OPD 不把多个 Specialist adapter 合并到部署模型，也不在推理阶段调用路由器或 Specialist。所有外部
能力必须在训练期蒸馏回 Qwen3.5-4B。

### 18.3 首轮多 Specialist 的具体模型与职责

远端 2026-08-15 只读盘点确认以下模型存在。首轮冻结为三个候选 Specialist：

| identity | 专项职责 | 模型 | 权威远端路径 | 首轮状态 |
|---|---|---|---|---|
| `S_nav` | 搜索词、结果页选择、翻页/返回与页面导航 | Qwen3.5-9B | `/data/share/model/Qwen3.5-9B` | 候选；须通过导航片资格门 |
| `S_match` | 商品约束理解、属性核对、同商品 option 选择 | Qwen3.5-35B-A3B | `/data/share/model/Qwen3.5-35B-A3B` | 候选；须通过匹配片资格门 |
| `S_finish` | 动作失败恢复、预算控制、购买时机与终止判断 | Qwen3.6-35B-A3B-FP8 | `/data/share/model/Qwen3.6-35B-A3B-FP8` | 候选；须通过收尾片资格门 |

`S_finish` 可以在路由层分为 `recovery` 与 `purchase` 两个逻辑角色，但首轮共享同一冻结模型权重，
不能把两个 prompt 角色计成两个独立 foundation model。首轮不同时增加新的 Specialist LoRA；先验证
现有强模型在预注册任务片上是否已经形成相对 `pi_0` 的专项优势。只有资格通过后，文档才称其为
有效 Specialist；参数更大本身不是资格证据。

选择理由：

- `S_nav` 使用 9B 控制高频搜索/导航状态的推理成本；
- `S_match` 使用同系列 35B-A3B 处理最依赖语义和多约束组合的 item/option 判断；
- `S_finish` 使用当前已部署的 Qwen3.6-35B 候选处理低频但高价值的恢复、预算与购买决策；
- 最终 Student 仍为 4B，避免多模型在线部署。

### 18.4 tokenizer、词表和 chat-template 兼容门

四个模型的 `tokenizer.json` 在远端实测 SHA256 均为：

```text
5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
```

四者 `vocab_size=248320`，满足 token-level OPD 的首要词表条件。Qwen3.5-4B、Qwen3.5-9B、
Qwen3.5-35B-A3B 的 `tokenizer_config.json` SHA256 为：

```text
316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8
```

Qwen3.6-35B-A3B-FP8 的 `tokenizer_config.json` SHA256 不同：

```text
5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b
```

因此正式实现必须统一使用 Student 的 tokenizer、special-token ID 与 chat template 生成 token IDs，
再把完全相同的 token prefix 输入 Specialist。资格采样前必须通过以下 fail-closed 探针：

1. 四个模型对 Student chat template 生成的 BOS/EOS/system/user/assistant token ID 解释一致；
2. canonical WebShop action 在四模型上的 token ID 序列逐位一致；
3. 同一 Student prefix 的 attention mask、position 与 assistant action 边界一致；
4. Specialist 返回 logits 的最后一维严格为 `248320`，token index 与 Student 一一对应；
5. 在固定 prefix 上，full-vocab 或 top-k+rest-mass 重建后的概率有限且总和为 1；
6. 任一模型需要自己的 chat template 重编码时，不允许进入 token-level OPD，只能降级为 sequence
   teacher，并从 OPD 主实验剔除。

`S_finish` 的 Hugging Face FP8 forward 还依赖首次从
`kernels-community/finegrained-fp8@version=4` 下载 Triton kernel，而当前登录/计算节点均无外网。
这属于运行后端限制，不作为 token 不兼容结论。其对齐门改用 vLLM 原生 FP8 loader：仍输入完全相同的
Student token IDs，运行时 config vocab 必须为 `248320`，返回 top-64 raw log-prob 必须全有限，并显式
记录 `top64_probability_mass + rest_mass = 1`。该 fallback 不允许给其他模型放宽 tokenizer/action/prefix
逐位一致门；若 vLLM 也不能接受 Student token prefix，则剔除 `S_finish`。正式 OPD 若采用压缩 target，
三个 Specialist 必须统一使用同一 top-k+rest 表示，且在 single-update 前用可 full-vocab 前向的模型
冻结最小 `k` 和近似误差门，不能让不同 Specialist 使用不同损失。

### 18.5 Specialist 资格与冻结路由

三个候选必须在任务采样和训练前冻结互斥的专项资格片。每个资格片只判断对应能力，不在同一批任务
上选择模型或调 prompt。建议每片至少 24 个 fresh development-only task、K4、18/15，与 `pi_0`
使用完全相同 task/seed/budget：

| Specialist | 资格片 | 最小通过门 |
|---|---|---|
| `S_nav` | 需要搜索、翻页或多页面导航的任务 | strict point gain 相对 `pi_0 >= +5 pp`，paired net flips > 0，schema/action 不恶化 |
| `S_match` | 多属性、价格及 option 约束任务 | strict point gain `>= +5 pp`，same-item/option 正向 flips > 0，zero-score purchase 不增加 |
| `S_finish` | partial、horizon、动作失败恢复及 Buy Now 边界任务 | strict point gain `>= +5 pp`，recovery/purchase 正向 flips > 0，premature-buy 不增加 |

资格结果必须同时报告 pass@1/K、任务级 paired flips、失败转移和 token/step 成本。未通过的候选不会被
其他模型自动替换；若只通过一个 Specialist，本轮降级为 single-specialist OPD feasibility，不宣称
multi-Specialist consolidation。至少两个候选通过才允许进入多 Specialist OPD 主臂。

实现时每个 24-task 专项片拆成一对独立 collection：`student_<specialist>` 与 `<specialist>`；不能用
一个 72-task Student collection 代替三个配对 Student arm，因为 vLLM sampling seed 包含 group ID，
后两个专项片会因此失去逐 rollout 配对。六个 arm 均冻结 `seed=20260851`、K4、18/15、相同 task order；
Student 使用 `pi_0`，Specialist 不加载 Student adapter。资格输出固定为：

```text
$PHASE10B_ROOT/specialist_qualification/
  rosters/{student_S_nav,S_nav,student_S_match,S_match,student_S_finish,S_finish}.json
  nav/{student,S_nav}/
  match/{student,S_match}/
  finish/{student,S_finish}/
  {nav,match,finish}/pair_report.json
  report.json
```

`S_match` 的 BF16 checkpoint 约 70GB，资格推理冻结为 2-GPU tensor parallel；`S_nav`、`S_finish` 和
三个 Student arm 各使用 1 GPU。该差异只解决模型装载，不改变 task、seed、K、horizon、prompt 或性能
门；资格报告必须记录 `tensor_parallel_size`。三个 Specialist 的完整文件 manifest 固定写入
`$PHASE10B_ROOT/base_model_manifests/{S_nav,S_match,S_finish}.json`，collection 启动前逐文件复核。

路由器首轮不训练，只依据 action turn 开始前的公开状态确定一个 Specialist：

```text
search/results/list page                          -> S_nav
item page，仍需核对商品属性或选择 option          -> S_match
action failure / repeated state / budget warning -> S_finish(recovery)
item 已选择 option，下一关键动作接近 Buy Now      -> S_finish(purchase)
其他状态                                          -> no-specialist，使用 pi_0 retention/self target
```

路由输入只允许 page type、公开 available actions、公开 selected marker、已执行动作是否 success、
剩余 model/env budget 和公开状态 hash。禁止使用 terminal score、target ASIN、隐藏 goal fields、未来
结果或“哪个 Specialist 最后答对了”的事后选择。每个 action turn 只查询一个 Specialist；不能查询
全部模型后按输出置信度择优，否则会引入额外模型选择变量和三倍推理成本。

### 18.6 正式 OPD 数据流与损失

Student 是唯一 behavior policy：

```text
s_t, a_<t ~ pi_student
j_t = frozen_public_router(s_t)
a_t ~ pi_student(. | s_t, a_<t)
q_t = pi_specialist_j(. | exact_student_prefix)
```

Specialist 只对 Student 已访问的 exact prefix 返回 logits，不替换 Student 已采样 token。只在 assistant
canonical action token 上计算蒸馏；system、instruction、环境 observation、历史 Student action 与
padding 全部 mask。主损失冻结为 teacher-to-student token KL/交叉熵：

```text
L_opd = mean_task mean_turn mean_action_token
        KL(stopgrad(q_specialist^T(. | x_student)) || pi_student(. | x_student))
```

其中温度 `T`、full-vocab 或 top-k 近似、top-k 大小、rest-mass 重建、retention 权重和 optimizer
updates 必须在 single-update probe 前冻结，不能在 monitor 后修改。首轮建议先采用 full-vocab online
logits；若显存/带宽不允许，才使用经过概率质量误差探针的 top-k+rest-mass 表示。

为防遗忘，未路由状态和固定 old-SFT 状态只允许使用同一冻结 `pi_0` retention target。不得把
Specialist 离线 suffix、official dense reward 或未来结果作为辅助梯度混入 OPD 主臂。每个 batch 报告
Student behavior tokens、各 Specialist routed turns/tokens、KL、entropy、top-k retained mass、参数变化和
old-SFT fixed-state drift。

### 18.7 最小可证伪执行路径

```text
Gate 0  phase10b exposure union 与 fresh split
Gate 1  四模型 tokenizer/logit alignment CPU/GPU probe
Gate 2  三个候选 Specialist 独立资格测试
Gate 3  8-task x K4 端到端 OPD smoke，optimizer_steps=0
Gate 4  相同 on-policy batch 的 single-update OPD vs no-op/continued-SFT control
Gate 5  40-task、10个冻结 K4x4 batch 的最小 OPD pilot
Gate 6  独立 monitor A/B paired 评测
Gate 7  仅 OPD 确认提升后，运行 Student-only GRPO
Gate 8  一次性 fresh dev paired 最终评测
```

8-task smoke 必须证明：Student 自己产生全部 behavior token；路由只读公开状态；每个 action turn 只命中
零或一个 Specialist；四模型 exact token prefix hash 一致；Specialist logits 有限；蒸馏 mask 只覆盖
Student assistant action token；`optimizer_steps=0`。

single-update probe 使用同一冻结 Student on-policy batch 比较：

- OPD arm：接受路由 Specialist 的 token distribution；
- no-op/control arm：相同更新/采样/任务/token 预算，只使用冻结 `pi_0` self/retention target。

必须观察到 OPD routed-token NLL/KL 下降、prefix/observation 梯度为 0、参数真实变化、old-SFT
fixed-state KL 不超过 `0.01`，且 control 与 OPD 的实际 labeled token/update 预算匹配。任一失败不进入
40-task pilot。

最小 pilot 沿用低成本结构：40 个 fresh OPD train tasks，K4/task，4 task groups/batch，10 个冻结
batches，18/15，dropout=0；具体 LR 在 single-update 前冻结。全 batch 无 Specialist 路由 token 时执行
零更新且不替换任务。不得为了追到 10 个有效 update 动态补采。

### 18.8 OPD 后的 Student-only GRPO

只有 `pi_opd` 在两个独立 monitor 上均优于 `pi_0`，并通过 retention/failure 门，才允许继续 RL：

```text
pi_opd
  -> pi_opd 自己生成 K4 x 4-task on-policy trajectories
  -> strict-binary single-epoch trajectory group-normalized policy gradient
  -> reference = frozen pi_opd
  -> pi_opd_rl
```

RL 阶段不查询任何 Specialist，Specialist logits/corpus 不产生辅助梯度。这样分别测量：

```text
OPD gain = pi_opd - pi_0
RL-after-OPD gain = pi_opd_rl - pi_opd
```

若要声称“OPD 使 RL 更有效”，仍须保留与 `pi_0 -> pi_0_rl_control` 的同配置 2x2 对照；否则只能
声称“OPD 后的 RL 是否有额外增量”。

### 18.9 冻结性能门与停止条件

首轮至少满足：

- multi-Specialist 资格：至少 2/3 候选各自在预注册专项片通过；
- OPD monitor A、B 上 `pi_opd - pi_0` 均为正；合并 point gain `>= +1 pp`、paired net flips > 0、
  bootstrap `P(delta>0) >= 0.8`；
- partial、zero-score purchase、horizon、premature-buy 各自 point delta `<= +0.5 pp`，schema/action
  不增加；
- old-SFT retention point delta `> -0.5 pp`，fixed-state KL `<0.01`；
- 路由覆盖至少两个 Specialist，任一 Specialist 不得贡献超过 routed action tokens 的 70%；
- tokenizer/logit alignment rejection=0，非有限 logit/loss/gradient=0；
- 最终 fresh paired dev 上要求 OPD point gain `>= +1 pp` 且 95% CI 下界 `>0`；样本不足时结论为
  inconclusive，不扩写为成功链路。

任一以下条件成立即停止：只有一个或零个 Specialist 资格通过；Qwen3.6 无法使用 Student token IDs
对齐 logits；router 需要终局/隐藏信息；Student behavior 被 Specialist token 替换；OPD 只降低训练
KL但 monitor strict 不增；收益完全来自单一 Specialist 或单一 task bucket；retention/failure 安全门
失败。失败后不得通过增加 Specialist 数量、查询全部模型择优、增加 K/seed/步数或打开 holdout 刷门。

### 18.10 输出根与实现里程碑

新增输出根与旧 corrective-SFT 根分离：

```text
PHASE10B_ROOT = /home/wushaohua/data/MiniWebWork-RL/outputs/m6_monotonic_posttraining_v1/phase10b_multi_specialist_opd_v1

$PHASE10B_ROOT/
  exposure_union.json
  split_lock.json
  model_tokenizer_manifest.json
  specialist_qualification/{nav,match,finish}/
  router_manifest.json
  smoke_8task/
  single_update_probe/
  opd_train_roster.json
  opd_student/
  monitor_a/
  monitor_b/
  opd_rl/
  final_dev/
  final_stats.json
```

实现顺序冻结为：

1. 生成包含 Phase10-A readiness 任务的新 exposure union 与互斥角色；
2. 实现 Student-token-prefix 多模型 logits alignment probe；
3. 实现三个候选 Specialist 的独立资格采样和 paired evaluator；
4. 实现只读公开状态的 turn-level frozen router；
5. 实现 Student behavior rollout、Specialist logits 查询、assistant-action-only OPD mask；
6. 完成 8-task zero-update smoke 与 matched single-update probe；
7. 仅在全部前门通过后提交 40-task 最小 OPD pilot；
8. monitor A/B 确认后才执行可选 Student-only GRPO；
9. fresh paired dev 一次性评测 `Raw / pi_0 / pi_opd / pi_opd_rl`；
10. 把每轮假设、唯一变化、结果和停止决策写入技术报告，逐 Job 工程错误写入失败账本。

本节只批准进入实现、资格和 readiness，不授权绕过门禁直接提交正式 OPD 或 RL。它的核心目标是
验证多个更强模型能否在 Student 自己访问的前缀上提供互补 token distribution，并把这些能力压回
单一 Qwen3.5-4B，而不是把教师 sequence 冒充成 on-policy 数据。

### 18.11 执行结果：Gate2停止（2026-08-15）

Gate0 exposure/split与Gate1四模型Student-token logit alignment均通过；Gate2六个配对推理作业也全部
工程成功。但冻结资格门结果为`0/3`：`S_nav`增益`+4.167 pp`未达到`+5 pp`；`S_match`相对Student
`-11.458 pp`且无same-item option正向flip；`S_finish`仅`+2.083 pp`、任务级净flip为0并增加nonstrict
purchase。聚合报告SHA为`835da0da...c0f2e6`，decision=`stop_opd`。

按本节原始停止条件，Gate3零更新smoke虽已实现和测试，但不得提交；single-update、40-task pilot、
monitor、OPD后GRPO及正式OPD训练全部不执行。该结果确认的是“当前三份现成Specialist不具备训练前
资格”，不是OPD算法在合格Specialist下必然无效。若未来更换或专项训练Specialist，必须新建fresh
qualification片并重新从Gate1/Gate2开始，不能复用本轮已查看的72个资格task。

## 19. Phase10-C：同系列Qwen3.5专项SFT教师再准入

### 19.1 冻结模型边界

Phase10-B证明未经WebShop专项适配的更大基座不自动构成有效Specialist。下一轮不再使用
Qwen3.6-35B-A3B-FP8；Phase10-B中的Qwen3.6记录只作为历史结果保留，不进入Phase10-C模型池。

统一Student与三个候选教师冻结为：

| identity | 基座 | 初始化/计划adapter | 角色 |
|---|---|---|---|
| `pi_0` | Qwen3.5-4B | 现有M6 SFT adapter：`outputs/m6_monotonic_posttraining_v1/mini/pilot_sft/final_adapter` | 唯一behavior Student与OPD起点 |
| `S_nav_sft` | Qwen3.5-9B | `$PHASE10C_ROOT/teacher_sft/S_nav/final_adapter` | 搜索、结果页选择与导航 |
| `S_match_sft` | Qwen3.5-35B-A3B | `$PHASE10C_ROOT/teacher_sft/S_match/final_adapter` | 属性、价格、商品与option匹配 |
| `S_finish_sft` | Qwen3.5-35B-A3B | `$PHASE10C_ROOT/teacher_sft/S_finish/final_adapter` | 恢复、预算与购买边界 |

`S_match_sft`与`S_finish_sft`共享同一个Qwen3.5-35B-A3B foundation checkpoint，但使用互不混合的
专项数据、独立LoRA adapter和独立资格片；它们是两个专项策略，不宣称为两个独立foundation model。
最终部署仍只有Qwen3.5-4B Student。

### 19.2 教师指导SFT Student，而不是Raw Student

Phase10-C主链冻结为：

```text
Qwen3.5-4B Raw -> 当前SFT -> pi_0 Student

Qwen3.5-9B/35B-A3B -> 专项高质量SFT -> S_*_sft

pi_0自己生成on-policy trajectory/action token
  -> frozen public router选择一个已合格SFT Specialist
  -> Specialist在exact pi_0 token prefix上返回distribution target
  -> assistant-action-only OPD更新pi_0
  -> pi_opd
  -> 可选Student-only on-policy GRPO
```

Raw 4B只保留为`Raw < SFT < OPD`链路的基线评测身份，不作为主OPD behavior policy，也不接收
Specialist target。原因是Raw→OPD的大部分增益会与动作格式、页面交互和基础模仿能力混杂，无法回答
“OPD能否在SFT之后继续提升”；同时Raw访问状态质量更低，会把昂贵Specialist查询浪费在SFT已经解决的
基础错误上。若未来资源允许，Raw→OPD只能作为独立诊断control，不能替代`pi_0 -> pi_opd`主臂。

### 19.3 教师SFT数据要求

教师SFT数据必须给教师带来学生SFT语料中没有的专项信息，不能简单复制现有156-task学生SFT corpus：

- 训练状态优先来自Student公开失败/不确定边界，但teacher prompt不得读取target ASIN、隐藏答案或
  promotion/holdout；
- label必须由更强推理、公开约束规则或人工/程序化纠错产生，并经WebShop fresh-session重放验证；
- `S_nav_sft`监督搜索/导航action，`S_match_sft`监督商品与option决策，`S_finish_sft`监督恢复和购买边界；
- 教师SFT train、教师qualification、OPD train、monitor A/B和final dev必须task、goal-index与normalized
  instruction互斥；Phase10-B已查看的72个qualification task全部进入exposure union；
- 教师训练结束后必须先在全新专项片与同条件`pi_0`配对资格；至少2/3教师满足相对`pi_0 >= +5 pp`、
  task-level net flips为正和专项安全门，才重新启用零更新OPD smoke。

在教师资格通过前，不实现或提交正式OPD Student更新。教师SFT成功只表示产生候选教师；是否能指导
Student仍由新资格片、零更新target smoke和matched single-update共同决定。

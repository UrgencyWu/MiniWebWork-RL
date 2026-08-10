# M5 WebShop 长程信用分配研究：冻结方案

> 状态：`preflight_only`
> 决策日期：2026-08-10
> 机器合同：`data/m5_webshop_study_v1.json`
> 正式训练开关：关闭；只有新的 clean-SHA readiness 与独立授权可以打开

## 1. 最终决定

项目不再扩充算法清单，也不再把自造 MiniWebWork 任务当作正式效果基准。M5
只回答一个问题：

> 在相同的 verified SFT 起点、WebShop 训练任务、K=4 采样、模型、优化器和
> 500,000 generated-action-token 预算下，基于公开状态锚点的 step credit，是否
> 比把终局 GRPO advantage 广播到整条轨迹更高效？

正式训练只有两个在线方法：

1. `multi_turn_grpo`：每个 K4 组按终局二元奖励标准化，同一轨迹的全部动作 turn
   使用相同 macro advantage；
2. `anchor_gigpo`：保留相同 macro advantage，并在多个轨迹首次到达同一公开
   state anchor 时加入 discounted-return micro advantage；没有共享信息时
   精确退化为 macro advantage。

每个方法只做 3 个预注册 seed：`20260801/02/03`。不加入 PPO、RLOO、GSPO、
RSFT 或 critic，以免面试项目变成“大而全的算法清单”。项目深度集中在多轮轨迹、
稀疏终奖、信用分配、严格 on-policy logprob、同卡 rollout/learner 切换、Slurm
恢复、数据隔离和统计检验。

## 2. 为什么选择 WebShop

| 来源 | 冻结版本 | 正式角色 | 许可处理 |
|---|---|---|---|
| Princeton WebShop | `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` | 原始 benchmark 与任务语义 | MIT，保留署名 |
| Agent-R1 | `b124aa46534cbf2fb8bc8af11405774984c42ac7` | 2026 年仍在使用的 FastAPI/SQLite/Lucene WebShop 环境 | MIT，固定代码提交 |
| Agent-R1-data | `1e624211d47dc1d66758a056fa8ee1017d72de6f` | 1,181,430 商品、12,087 目标和检索索引 | 依照原 benchmark 条款，仅研究使用，不重新分发原始大文件 |

WebShop 同时满足三点：它是公开、可执行、rule-verifiable 的多轮 Agent 环境；任务
量足够形成稳定训练信号；它不需要 WebArena 的多套 Docker 网站和频繁 reset。
集群没有 Docker/Podman/Singularity/Apptainer，因此 WebArena 会把项目重心转成
基础设施运维，并阻碍六个单卡训练并行。

Agent-R1 源码优先用 Git fetch 获取该 40 位提交；最多两次、每次含连接阶段硬
超时 60 秒。针对校园出口对 Git smart HTTP 偶发超时，唯一备用路径是同一提交
的 GitHub codeload 归档；其当前精确大小
`1,628,704` bytes、SHA-256 `07e6a35a...7d57b`、226 个 members 以及 WebShop
recipe 内容树 SHA-256 `bf79abaf...51c1f` 全部冻结；完整 178-file 源码树另锁为
`f45b0e09...ecd9a`。内容树算法明确冻结为 repository-relative path、byte size 与
逐文件 SHA-256 的有序记录。归档逐 member 拒绝绝对路径、`..`、symlink/hardlink，安全
展开后仍重算两层内容树；服务禁写 bytecode，因此备用路径不放宽源码身份。

其他候选的处理如下：

- Mind2Web 是离线轨迹，不能提供本研究需要的可执行在线 reward；
- AgentInstruct 的 351 条 WebShop 轨迹在固定 data card 中没有明确数据许可证，
  且本地审计发现 4 条命中冻结 test 目标，正式语料完全排除；
- VisualAgentBench/WebArena 与 CUA-Gym 会同时改变视觉模态、动作空间和部署栈；
- 原 MiniWebWork v2 保留为“数据制度失败诊断”，不能进入 M5 正式表格。

## 3. 数据切分与泄漏修复

上游约定是 `goals[0:500]` 为测试，其余 11,587 条为训练。M5 在上游训练部分
再切出开发集：

| 角色 | 原始索引 | 原始数量 | 精确文本去重后可用 | 可更新模型？ |
|---|---:|---:|---:|---:|
| frozen test | `[0, 500)` | 500 | 500 | 否 |
| dev | `[500, 1000)` | 500 | 499 | 否 |
| train | `[1000, 12087)` | 11,087 | 10,885 | 是 |

原始 `goals.json` 含 11,884 个唯一规范化 instruction；203 行是后出现的重复项。
直接按索引使用会产生 1 个 test–dev、15 个 test–train、12 个 dev–train 的精确
文本交集。冻结修复规则是：

> 测试 500 条全部保留；dev/train 中，如果规范化 instruction 已在更小
> `goal_index` 出现，则排除当前行。

因此 test 未被删改，eligible 三个角色的精确 instruction 交集为零。完整排除索引、
可用 roster hash 和源 `goals.json` SHA-256 固定在
`data/m5_webshop_split_exclusions_v1.json`。测试内容在最终评测前只能进入机械式
字节/重复隔离审计，不能进入 SFT、rollout、调参、checkpoint 选择或 preflight
指标。

WebShop 的商品在任务间会复用，这是 benchmark 的既有任务生成机制；本研究不把
商品重叠伪装成任务独立。统计分析以 task 为 cluster，避免把同一任务的 K4 rollout
当成 4 个独立样本。

## 4. 上游环境的两项加固

代码审查发现 Agent-R1 WebShop 的默认 HTTP 接口有两条不能直接用于正式研究的
行为：

1. `/reset` 的 `info` 返回目标 ASIN；
2. engine 会接受数据库中存在、但当前页面 `available_actions` 未列出的 ASIN 点击。

M5 适配层因此采用 fail-closed 白名单：

- prompt 只读取 instruction、公开 observation、page type、最近公开动作结果和
  最多 256 个公开动作；绝不序列化 reset 的 ASIN 或 verifier target；
- `search[...]` 只在公开 search 模板存在时允许；每个 `click[...]` 必须逐字出现于
  当前经过上限约束的公开 action list；未列出的点击在客户端被记为 policy failure，
  不发送给上游 engine；
- credit anchor 只由动作前的 task-scoped 公开 MDP state 构成：task/instruction、
  page type、prompt 可见 observation、截断标记、最多 256 个公开动作和 terminal；
- exact prompt-token SHA-256 单独绑定 behavior logprob 血缘，不能进入 credit group；
  因此前序动作不同的轨迹如果重新到达同一公开状态，仍能形成 step 对照；episode
  ID、step index、history、目标 ASIN、verifier 和运行时 URL 均不能影响 anchor；
- 可选 `thefuzz`/spaCy 会改变上游 reward 计算，隔离 server 环境明确要求二者不存在，
  并固定 Python `3.12.13`、Java `21.0.10`，把完整 `pip freeze` 写入审计工件。

训练环境另有两个明确处置。`requests==2.32.5` 与误装的 `chardet 6` 会产生运行时
警告，M5 将 chardet 收紧到 `5.2.0` 并要求 warning 为零。`vllm==0.17.0` 的包元数据
仍声明 `transformers<5`，而冻结 learner 使用 `transformers==5.14.1`；这不是静默忽略：
CPU gate 要求 `pip check` 只剩这一条精确登记的 exception，并确认 vLLM 原生注册
`Qwen3_5ForConditionalGeneration`；随后真实 GPU gate 还必须通过确定性生成、behavior
logprob、sleep/wake、adapter reload 和 learner update，任一失败即停止正式训练。

这部分是项目最重要的工程论点之一：benchmark 可用不等于 benchmark 默认实现可以
不经审查地用于因果对比。

## 5. Shared verified SFT

SFT 不引入外部轨迹。专家只读取被冻结 goal 的 oracle metadata 生成标签，但每一步
都必须由当前公开 action list 支持，并由官方环境最终验证 reward=1：

```text
search[goal.query]
→ 在最多 5 页公开 top-50 搜索结果中找到并点击 goal.asin
→ 逐一点击公开可见的 goal_options
→ click[Buy Now]
→ retain only reward=1 and task_score=1
```

正式语料按 seed `20260810` 对 eligible roster 做 SHA-256 排序，顺序扫描并保留前
4,000 个 verified train 任务与前 400 个 verified dev 任务。每个 action turn 是一个
completion-only 样本；completion 只有 `{"command":"..."}`，无 CoT、理由或隐藏字段。

训练合同：Qwen3.5-4B、LoRA `r=16/alpha=32`、LR `2e-4`、effective batch 16、
max sequence 8192、2–3 epochs。微批只在 `[1,2,4,8]` 中选择满足至少 15% VRAM
余量的最大值。累计有效 completion-label token exposure 必须达到 250,000，
zero-label 比例必须为 0；三轮仍达不到则 fail，不复制样本伪造数据量。checkpoint
选择只使用 dev NLL、teacher-forced schema/exact action 和 64-task closed-loop dev。

## 6. 在线训练的唯一受控变量

两个方法共享以下设置：

- base/shared SFT adapter、task order 和 sampling seed；
- K=4，temperature=1、top-p=1、top-k=0；
- max 15 environment steps、18 model turns、每 turn 最多 128 新 token；
- 每次迭代最多 32 个 task group；每个 seed 总计 500,000 generated action tokens；
- 无效输出、policy failure 和零优势 token 全部计入成本；infra failure 为 `reward=null`
  并整组重采；
- behavior-policy staleness=0；2 policy epochs、trajectory minibatch 4、LR `5e-6`、
  clip epsilon 0.2、gradient clip 1.0；
- token mean → turn mean → trajectory mean → K4 group mean 的层级归一化。

GRPO 使用 K4 终奖的 population-standardized advantage。`anchor_gigpo` 使用相同
macro 项，另加 `gamma=0.95`、权重 `omega=1.0` 的 first-visit public-state micro
项；anchor 少于两个不同轨迹或 return 无方差时 micro 精确为 0。这使“信用分配”
成为唯一方法差异，而不是同时改变 reward shaping、采样器或训练预算。

具体地，轨迹 `i` 在动作前状态 `s_t` 的稀疏回报为
`G(i,t)=0.95^(T_i-t-1) * R_i`。对同一 task 的 K4 中首次到达相同 `s_t` 的不同轨迹
做 population standardization 得到 `A_micro`，最终
`A_turn=A_macro+1.0*A_micro`。循环回到同一状态只取每条轨迹第一次，避免一条卡住
的轨迹通过重复点击给自己增加样本权重。

这与固定 Agent-R1 GiGPO 的核心一致：都按动作前 observation 聚合 discounted
return，而不是按完整 prompt 聚合。M5 有两项明确的工程化收紧：将公开 action list
纳入状态，并做 per-trajectory first-visit 去重；因此名称是 `anchor_gigpo`，不声称与
上游实现逐行等价。若把 prompt token 放进 anchor，轨迹分叉后几乎无法再次成组，
会把所谓“长程信用分配”退化成主要只作用于第一步，故已在冻结前排除。

## 7. 评测与预期结论

正式测试在所有 8 个推理身份冻结后只打开一次：raw base、shared SFT、两个方法各
3 seed。500 个 test task 每个 K=4，共 16,000 条轨迹。

主指标是 binary success；WebShop dense task score 只作次指标。报告还包括类别/约束
分层、invalid action、环境步数、generated tokens、wall time、失败分类、训练曲线、
public-anchor 覆盖和有效 optimizer token 比例。置信区间使用 task-cluster bootstrap；
方法比较使用相同任务与 seed 的 paired permutation，不把 rollout 当独立样本。

预期不是承诺“GiGPO 一定更高”。项目需要得到以下三类可审计结论之一：

1. 在相同成本下，anchor credit 显著提高成功率或更早达到同一成功率；
2. 成功率无显著差异，但有效 optimizer token、样本效率或方差改善；
3. 没有改善，并能用 anchor 覆盖、信号稀疏度和失败轨迹解释原因。

第三种仍是合格结果；选择性汇报正向 seed 不是合格结果。

## 8. Slurm 与资源计划

所有作业上限 24 小时，逻辑 run 通过同一 root 续跑，不声称固定 sbatch 次数。

| 角色 | GPU | CPU | 内存 |
|---|---:|---:|---:|
| clean-SHA CPU 回归 | 0 | 2 | 8 GiB |
| 数据下载/字节审计 | 0 | 1 | 8 GiB |
| training runtime 修复/审计 | 0 | 2 | 8 GiB |
| server 环境安装 | 0 | 2 | 20 GiB |
| shared WebShop service（初始 4 workers） | 0 | 8 | 48 GiB |
| verified SFT corpus（4 workers） | 0 | 4 | 8 GiB |
| SFT | 1 | 4 | 32 GiB |
| 每个 online run | 1 | 4 | 24 GiB |
| 每个 frozen eval | 1 | 3 | 20 GiB |

六个 online run 可以并行：合计 6 GPU/24 CPU/144 GiB；加共享服务后是
6 GPU/32 CPU/192 GiB。相比为每个 GPU job 启动独立 8-CPU server，这一设计同时
保留 GPU 并行和 CPU 调度余量。Lucene/JVM 与 SQLite cache 是 per-process 成本，
因此不预设 16 workers；先用 4，随后在相同 8 CPU/48 GiB 内比较 2/4/8 workers 与
32/64 lanes，正式值取稳定且更快者；
generation 阶段 GPU 利用率中位数要求至少 60%，learner 阶段至少 80%。

service 同样受 24 小时上限约束。集群的 `scontrol` 对普通用户禁权，因此到期前
5 分钟不做伪 requeue，而是由当前作业用 `sbatch` 提交带 `afterany:<parent_job_id>`
依赖的后继 allocation；日志记录 parent/successor 血缘。每次 allocation 都重新校验
8.37 GB runtime 与隔离环境，并要求审计 content hash 与初始冻结工件完全一致。客户端
将重启窗口记为 infrastructure failure 并重采整个原子 K4 group，不能把服务中断
写成 reward=0。所有 CPU-only 作业显式隐藏 GPU。runtime、data、environment、
health、逐任务 SFT record、corpus 与 token-audit 工件均同时写入 clean 40 位 Git SHA
和协议 SHA-256，不能只靠 Slurm 日志反推代码血缘。

逻辑 GPU 工作量是 7 个训练 run（1 SFT + 6 online）和 8 个 eval run，共 15 个；
24 小时恢复可能增加 Slurm allocation 数。稳定情况下预计 5–8 个自然日；给环境
兼容、恢复和一次协议修订留余量，保守计划为 7–10 天。

## 9. 正式准入门槛

正式 SFT 或在线训练提交前必须全部满足：

1. 25 个上游 runtime 文件逐字节匹配 8,367,150,707 bytes 的 lock；SQLite 商品
   1,181,430、goal 12,087 且索引连续；
2. split exclusion lock 可复现，eligible 跨角色精确 instruction 交集为零；
3. 4,000/400 SFT task 100% public-action-valid、reward=1、zero label=0、无截断，
   三 epoch 能达到 250k label exposure；
4. 至少 32 个 SFT-policy K4 train group：infra valid ≥99%，mixed-reward group ≥20%，
   success 在 3%–70%；
5. 每组初始状态形成 shared anchor，至少 2% turn 获得非零 informative micro
   credit，且至少 5% 有效 K4 group 存在一个由不同轨迹共享的非初始状态；
6. 两种 learner 均至少完成 2 次非零更新，loss/gradient 有限，有效 optimizer
   action-token 比例 ≥15%；
7. 32/64 lanes benchmark、generation/learner GPU 利用率、VRAM 与 OOM 门槛通过；
8. 真实 Slurm 中断后 same-root 恢复，token ledger、adapter、optimizer、sampler 和
   K4 原子组血缘不漂移；
9. 最终 CPU 全回归在 clean Git SHA 上通过，生成 self-hashed readiness；
10. 单独版本化 authorization 才可把正式训练开关打开。

任一门槛失败时只产生诊断工件，不允许“先跑正式任务再补文档”。

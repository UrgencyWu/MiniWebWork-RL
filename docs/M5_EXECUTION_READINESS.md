# M5 执行与准入状态

> 最后更新：2026-08-10
> 当前结论：推荐方案已冻结；正式训练仍为 `NOT_READY`。当前只允许 CPU 数据、
> server 和小规模 preflight 作业。

## 作业依赖

```text
clean M5 preflight SHA
├─ scheduled full CPU regression (2 CPU)
├─ CPU data download + exact audit (1 CPU)
├─ CPU training-runtime repair + audit (2 CPU)
└─ CPU server-env setup + upstream pin (2 CPU)
       ↓ all three pass
shared WebShop service (24 CPU, 8/16 serialized workers, renewable 24h)
       ↓
verified SFT corpus + tokenizer audit (8 CPU / 8 workers)
       ↓
SFT microbatch/short-train preflight (1 GPU)
       ↓
base-signal K4 + GRPO/GiGPO learner + 32/64-lane + real-resume preflights
       ↓
clean-SHA full CPU regression + self-hashed readiness
       ↓ separate authorization
formal SFT → six parallel online runs → eight frozen evaluations → analysis
```

## 当前 checklist

| 门禁 | 状态 | 证据/说明 |
|---|---|---|
| 唯一 benchmark 与方法矩阵 | PASS（本地） | `data/m5_webshop_study_v1.json` |
| 训练 runtime | IMPLEMENTED / PENDING（Slurm） | 修复 chardet 警告；只容许已冻结的 vLLM/Transformers metadata exception，真实 generation 仍须 GPU gate |
| 上游版本/25 文件 hash lock | PASS（合同）/ PENDING（远端全文件） | `data/m5_webshop_upstream_lock_v1.json` |
| 原始重复文本审计与排除 lock | PASS（本地真实 goals） | 500 test、499 dev、10,885 train；eligible overlap=0 |
| Prompt target 泄漏防护 | PASS（CPU 单测） | reset ASIN 不进入 prompt |
| 未公开 ASIN shortcut 防护 | PASS（CPU 单测） | 未列出的 click 不发送 HTTP |
| Verified oracle 实现 | PASS（CPU mock）/ PENDING（full env） | top-50 public navigation + reward=1 only |
| 隔离 Python 3.12.13/Java 21.0.10/Pyserini server | IMPLEMENTED / PENDING（Slurm） | 独立环境，不污染训练 env |
| 4,000/400 SFT corpus | PENDING | 依赖 data + service |
| 8192 token/250k exposure audit | PENDING | 依赖 corpus + Qwen tokenizer |
| SFT GPU preflight | PENDING | 正式 SFT 禁止 |
| K4 signal/credit/optimizer gate | PENDING | 含非初始状态汇合率；正式 online 禁止 |
| 8/16 workers、32/64 lanes 与 GPU telemetry | PENDING | 充分供给 CPU；HTTP 5xx 必须为零 |
| 真实 24h 中断恢复 | PENDING | 必须至少一次 scheduler 级恢复 |
| clean-SHA readiness | PENDING | 必须 self-hashed 且无 unmet gate |
| 正式 authorization | CLOSED | preflight 通过后另行生成 |

## 当前允许执行的入口

```text
scripts/run_m5_webshop_cpu_regression_job.sh
scripts/run_m5_webshop_data_preflight_job.sh
scripts/run_m5_webshop_server_setup_job.sh
scripts/run_m5_training_runtime_setup_job.sh
scripts/run_m5_webshop_service_job.sh
scripts/run_m5_webshop_service_health_job.sh
scripts/run_m5_webshop_concurrency_preflight_job.sh
scripts/run_m5_webshop_sft_corpus_job.sh
```

这些入口不提交正式训练。正式 SFT、online 与 frozen test 脚本在 readiness 完成前
不应存在可绕过的开放路径。

## 已发现并关闭的集群差异

- 集群的 `scontrol` 仅允许 `slurmadmin` 执行，service 不再调用
  `scontrol requeue`；到期前 5 分钟由当前 allocation 使用普通用户可执行的 `sbatch`
  提交 `afterany:<parent_job_id>` 后继作业，日志同时记录 parent/successor job id；
- 所有 CPU-only 入口显式设置空 `CUDA_VISIBLE_DEVICES`，即使隔离 server 的
  Pyserini 依赖树带有 Torch/CUDA wheel，也不能触碰未申请的 GPU；
- 每个新 service allocation 不仅重算 runtime/environment 审计，还必须与初始冻结
  audit 的 `content_sha256` 完全相同；隔离 `JAVA_HOME`、`JVM_PATH`、`PATH` 与
  禁写 bytecode 必须在 allocation 审计之前导出，不能依赖登录 shell；
- 上游 Git fetch 使用固定提交、HTTP/1.1、两次有限重试，并对每次尝试施加 60 秒
  硬超时（含连接阶段）；若校园出口持续
  阻断 Git smart HTTP，只允许退到同一提交的 GitHub codeload 归档，且归档
  SHA-256、字节数、member 数和 `recipes/webshop` 内容树 hash 均已写入机器合同；
  完整 178-file 源码树也独立锁定；部分 Git checkout 移入可恢复 quarantine，绝不
  退回浮动分支或未审计代码。内容树使用已版本化的 repository-relative
  path/size/file-SHA256 记录算法，服务禁写 bytecode，防止源码树在 allocation 间漂移；
- 已存在的 Agent-R1 Git 源只有在 HEAD 精确等于冻结提交，且 tracked、untracked
  与 ignored 状态均为空时才无网络复用；随后仍执行完整源码树与 WebShop 子树 hash
  审计。重复 preflight 因而不依赖 GitHub 可用性，也不会接受本地残留；
- 真实并发审计发现冻结上游在每个 worker 内共享一个 SQLite connection、Lucene
  searcher 与 mutable cache，而 FastAPI 会把同步 endpoint 放入线程池；同 worker
  并发可令 SQLite 查询返回损坏值并产生 HTTP 500。上游源码 hash 保持不变，仓库
  ASGI 包装只在 worker 内串行 HTTP 请求，8/16 个 worker 之间仍并行；健康探针用
  connection-closing 并发波覆盖每个 PID，且分别保留 `health_workers_8.json` 与
  `health_workers_16.json`；压力门槛要求 HTTP 5xx 比例严格为零；
- SFT corpus collector 申请 8 CPU 且开 8 个 worker；共享服务申请 24 CPU/96 GiB，
  可在 8/16 process worker 间做吞吐选择，不让 CPU 服务拖慢 6 个 GPU run；
- runtime、data、server environment、health、service-stress、逐任务 SFT record、
  corpus 和 token audit 都同时嵌入当前 clean 40 位 Git SHA 与协议 SHA-256；不能
  只靠 Slurm 日志反推代码血缘。

## 停止条件

出现下列任一情况立即停在 preflight，并新建协议版本，而不是原地修改已绑定工件：

- 上游 hash、商品/goal 数或 Lucene index 不一致；
- 可选依赖导致 reward 语义漂移；
- 公开动作 oracle 无法稳定获得 4,000/400 个 verified task；
- SFT 后 K4 几乎全成功或全失败，mixed group 低于 20%；
- micro credit 覆盖低于 2%，或含 shared non-initial state 的有效 K4 group 低于 5%；
- GPU 利用率未达门槛且增加 lanes 仍由 CPU server 限制；
- 中断恢复改变 token 成本、组原子性或 adapter/optimizer lineage。

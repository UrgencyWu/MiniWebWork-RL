# MiniWebWork-RL

轻量级浏览器智能体强化训练项目，面向采购调研任务。

## 当前阶段：M1.1

**确定性采购环境 (Deterministic Procurement Environment)**

已完成：
- M1.0: 环境建立与浏览器烟雾测试 → PASS
- M1.1: 采购数据库、任务集、网站、Verifier、Slurm E2E → PASS

## 已冻结技术范围

| 项目 | 选型 |
|---|---|
| 本地 Web 服务 | FastAPI + Uvicorn + Jinja2 |
| 数据库 | SQLite (6 供应商, 24 商品) |
| 浏览器自动化 | Playwright + Chromium (headless) |
| LLM 推理 | Qwen3.5-4B via Transformers |
| 训练微调 | LoRA PEFT |
| 强化训练 | TRL (GRPO) |
| 容器化 | 不使用 Docker |
| 集群调度 | Slurm |

## 环境要求

- Python 3.11
- PyTorch 2.10.0+cu128
- CUDA 12.8+
- Conda 26+

## Conda 环境

```bash
conda activate miniwebwork
```

## 初始化数据库

```bash
python -m miniwebwork.cli init-db
# 或
bash scripts/init_db.sh
```

## 重置数据库

```bash
python -m miniwebwork.cli reset-db
# 或
bash scripts/reset_db.sh
```

## 启动网站

```bash
python -m miniwebwork.webapp
# 或
bash scripts/run_procurement_site.sh
```

默认监听 `http://127.0.0.1:18080`。

## 运行测试

```bash
# 全部单元/集成测试
python -m pytest -q

# 任务验证
python -m miniwebwork.cli validate-tasks

# 种子数据验证
python -m miniwebwork.cli validate-seed

# M1.0 浏览器烟雾测试
python -m miniwebwork.browser_smoke
```

## 运行 Slurm E2E

```bash
# 采购端到端测试
sbatch scripts/slurm/m1_1_procurement_e2e.sbatch

# M1.0 浏览器烟雾测试
sbatch scripts/slurm/m1_browser_smoke.sbatch

# M1.0 模型烟雾测试
sbatch scripts/slurm/m1_model_smoke.sbatch
```

## 任务文件位置

| 文件 | 说明 |
|---|---|
| `data/tasks/tasks_public.jsonl` | 用户可见任务 (不含答案) |
| `data/tasks/tasks_oracle.jsonl` | 私有 Oracle (含约束、正确答案) |
| `data/seed/suppliers.json` | 供应商种子数据 |
| `data/seed/products.json` | 商品种子数据 |
| `data/seed/manifest.json` | 种子数据清单和哈希 |

## Verifier 使用方式

```bash
python -m miniwebwork.cli verify --task-id TASK-001 --episode-id EP-XXXX
```

## 当前未实现

- Agent 推理循环 (ReAct)
- Function Calling
- DOM 压缩 / Accessibility Tree
- SFT 数据构造
- LoRA 训练
- GRPO 强化训练
- 多模态视觉
- 多浏览器并发
- vLLM 推理服务

## 已知限制

- 仅通过 Slurm 使用 GPU，不可在登录节点直接运行 GPU 任务
- 单任务最多使用 2 张 GPU
- 任务最长运行 24 小时
- 无 Docker 环境
- 采购数据为虚构，不连接真实采购网站

## 下一阶段 (M1.2)

将采购网站封装为标准 Agent Environment：
- 文本 DOM observation
- 固定 JSON action space
- step/reset 接口
- 基于规则的无模型 Agent 基线

# MiniWebWork-RL

轻量级浏览器智能体强化训练项目，面向采购调研任务。

## 当前阶段：M1.0

**环境建立与最小浏览器烟雾测试**

本阶段目标：快速、可复现地跑通浏览器 Agent 强化训练之前的基础环境闭环。

## 已冻结技术范围

| 项目 | 选型 |
|---|---|
| 本地 Web 服务 | FastAPI + Uvicorn |
| 数据库 | SQLite |
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

环境基于 `qwen9B` 克隆，包含所有核心 ML 依赖。

## 本地网站启动

```bash
python -m miniwebwork.webapp
# 或
bash scripts/run_local_site.sh
```

默认监听 `http://127.0.0.1:18080`。

## 浏览器烟雾测试

```bash
python -m miniwebwork.browser_smoke
```

在 Slurm 作业中运行：

```bash
sbatch scripts/slurm/m1_browser_smoke.sbatch
```

## 模型烟雾测试

```bash
sbatch scripts/slurm/m1_model_smoke.sbatch
```

## 尚未实现

- 采购业务网站
- 商品/供应商数据库
- Agent 推理循环 (ReAct)
- SFT 数据构造
- LoRA 训练
- GRPO 强化训练
- 多模态视觉
- 多浏览器并发

## 已知限制

- 仅通过 Slurm 使用 GPU，不可在登录节点直接运行 GPU 任务
- 单任务最多使用 2 张 GPU
- 任务最长运行 24 小时
- 无 Docker 环境

## 下一阶段 (M1.1)

建立最小采购网站数据模型、商品页面、供应商页面和第一批确定性采购任务。

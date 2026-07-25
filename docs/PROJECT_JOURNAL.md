# MiniWebWork-RL 项目全记录

> 从零到 M1.1 确定性采购环境的完整过程，包含每一步的关键决策、命令、结果和教训。

---

## 0. 项目定位

**目标**：构建轻量级浏览器智能体强化训练系统，面向采购调研任务。

**终端路线**：
本地采购网站 → 文本 DOM / Accessibility Tree → Playwright 浏览器动作 → 多轮 Agent 轨迹 → 自动终态验证 → LoRA SFT → 简化 Agentic RL / GRPO

**约束**：
- 不使用 Docker（服务器无 sudo 权限）
- 所有 GPU 任务必须通过 Slurm
- 模型优先从 ModelScope 获取
- 单任务最多 2 张 GPU，最长 24 小时

---

## 1. 第零步：服务器环境审计（2026-07-24）

### 1.1 硬件

| 项目 | 实际值 |
|---|---|
| 主机 | user-NF5468M6 |
| OS | Ubuntu 22.04.5 LTS, Kernel 6.8.0 |
| CPU | 2× Intel Xeon Gold 6330 (28C/56T each), 112 vCPU |
| RAM | 377 GB (315 GB available) |
| GPU | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition, 95 GB each |
| CUDA Driver | 595.71.05, CUDA 13.2 |
| 磁盘 | 系统 1.8T (946G free), 数据盘 29T (24T free) |

### 1.2 软件环境

| 组件 | 状态 |
|---|---|
| Python | 3.11.14 (Conda) |
| PyTorch | 2.10.0+cu128, CUDA available=True, 8 GPU |
| Transformers | 5.14.1 |
| ModelScope | 1.34.0 |
| vLLM | 0.17.0 |
| Conda | 26.1.1, 8 个环境 |
| Docker | 未安装（也不使用） |
| Slurm | 25.05, partition=compute, 单节点 |
| Playwright | 未安装 |
| Chromium | 未安装 |

### 1.3 关键发现

- GPU 0/1/6/7 被 vLLM 占用，GPU 2/3/4/5 可用
- nvcc 未安装（不影响 PyTorch 使用）
- Git LFS 未安装
- `/dev/shm` 189 GB（足够 Chromium）
- 无代理限制，GitHub/ModelScope/PyPI 均可访问
- 现有模型缓存：`/data/share/model/Qwen3.5-4B` (8.8 GB)

### 1.4 可行性结论

**等级 B**：可开始，但需采用简化部署（无 Docker、优先用现有环境）

**文档**：`docs/M1_0_ENVIRONMENT_REPORT.md`

---

## 2. M1.0：环境建立与最小浏览器烟雾测试（2026-07-24 ~ 07-25）

### 2.1 Conda 环境

从 `qwen9B`（PyTorch 2.10.0+cu128, Transformers 5.14.1, ModelScope 1.34.0, 全功能环境）克隆：

```bash
conda create -n miniwebwork --clone qwen9B -y
```

**决策**：克隆而非新建。理由：qwen9B 已包含所有 ML 核心依赖，克隆比从零 conda install + pip install 更快且不会引入版本冲突。

### 2.2 浏览器依赖

```bash
pip install playwright           # 1.61.0
python -m playwright install chromium   # Chrome 149.0.7827.55
```

Chromium 缓存在 `~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`。

### 2.3 TRL

```bash
pip install --dry-run trl        # 先检查兼容性，确认无 PyTorch/CUDA 降级
pip install trl                  # 1.9.0，安全安装
```

### 2.4 项目结构

```
MiniWebWork-RL/
├── src/miniwebwork/
│   ├── webapp.py          # FastAPI 烟雾测试页面（/ + /health）
│   ├── browser_smoke.py   # Playwright 无头 Chromium 测试
│   └── model_smoke.py     # Qwen3.5-4B 最小生成测试
├── tests/test_webapp.py
├── scripts/slurm/m1_browser_smoke.sbatch
├── scripts/slurm/m1_model_smoke.sbatch
└── docs/
```

### 2.5 测试结果

| 测试 | 结果 | 细节 |
|---|---|---|
| 单元测试 | 9/9 PASS | FastAPI TestClient |
| 本地浏览器烟雾 | PASS | Chromium 启动 → 填写输入 → 点击按钮 → 验证 DOM → 截图 |
| Slurm 浏览器烟雾 | PASS (Job 926, 4s) | 同上，在 Slurm 作业内完成 |
| Slurm 模型烟雾 | PASS (Job 925, 14s) | Qwen3.5-4B from /data/share/model, GPU RTX PRO 6000, 32 tokens generated |

### 2.6 教训

- `$0` 在 sbatch 脚本中指向 Slurm spool 路径，不是脚本实际位置。改用 `$SLURM_SUBMIT_DIR`。
- Uvicorn 0.41 不支持 `--no-reload`，去掉即可（默认不重载）。
- M1.0 浏览器烟雾 URL 从 `/` 变为 `/smoke`（在 M1.1 中创建采购首页后），需同步更新测试。

### 2.7 基线提交

```bash
git init
git commit -m "feat: establish MiniWebWork-RL M1.0 runtime baseline"
# commit 2366389
```

**文档**：`docs/M1_0_COMMAND_LOG.md`, `docs/M1_0_ENVIRONMENT_REPORT.md`

---

## 3. M1.1：确定性采购环境（2026-07-25）

### 3.1 架构设计

```
data/seed/{suppliers,products}.json   →  SQLite 初始化
data/tasks/{public,oracle}.jsonl       →  任务定义
FastAPI + Jinja2                       →  服务端渲染
Playwright                             →  浏览器 E2E
Verifier (纯 Python)                   →  终态验证
Slurm                                  →  作业调度
```

### 3.2 数据库

6 个供应商（华北/华南/华东/西北，含认证/非认证，评分 3.6~4.9）。24 个商品（GPU/服务器/存储/网络，价格 12,000~520,000 元，显存 12~80 GB）。

所有数据虚构。种子文件含 SHA-256 哈希，可验证完整性。`PRAGMA foreign_keys = ON`。

### 3.3 任务设计（15 个）

| 类型 | 数量 | 示例 |
|---|---|---|
| exact_product | 3 | 查找型号 CC-A100X-80G |
| cheapest_feasible | 8 | 价格 ≤25000 + 有库存 GPU，选最低价 |
| highest_rating_supplier | 2 | GPU≥32GB≤80000，选最高评分供应商 |
| no_feasible_product | 2 | GPU≥64GB≤50000 有库存 → 无解 |

**关键发现**：手写 Oracle 时漏算了 PRD-002（48GB, 45000元），导致 TASK-005 和 TASK-014 的 expected_product_id 写错。通过 `validate-tasks` 重新计算发现并纠正。**任务答案必须通过代码验证，不能仅靠人工判断。**

### 3.4 网站实现（11 条路由）

采用 FastAPI + Jinja2 服务端渲染，所有关键元素使用 `data-testid` 稳定选择器。后端筛选通过 SQL WHERE 子句实现（非前端隐藏）。

### 3.5 Verifier

纯 Python 实现，无 LLM 依赖：
1. 从 Oracle 读取约束 → 重新计算可行商品集合
2. 从数据库读取最终提交
3. 逐条检查约束违规
4. 重新计算最优目标
5. 输出结构化结果（16 种 failure reason codes）

### 3.6 遇到的 Bug

1. **certified_only 默认值错误**：`parse_constraints` 将 `certified_only` 默认设为 `False`，但 `False is not None`，导致所有任务查询都加了 `s.certified = 0`。修复：默认值改为 `None`，仅在显式设置时过滤。

2. **筛选表单丢失 episode_id**：Playwright 提交筛选表单时，URL 中的 `episode_id` 和 `task_id` 被丢失。修复：在模板中添加 `<input type="hidden">` 字段，并改为直接构造 URL 导航。

3. **Oracle 计算错误**：见 3.3。

### 3.7 测试结果

| 层级 | 数量 | 结果 |
|---|---|---|
| 单元测试（数据/任务/提交/Verifier） | 53 | PASS |
| M1.0 回归（Web 路由） | 9 | PASS |
| Slurm E2E（Playwright 4 cases） | 4 | PASS (Job 930, 18s) |

### 3.8 提交

```bash
git commit -m "feat: build deterministic procurement environment for M1.1"
# commit 095f0b7, 40 files, +3481/-118 lines
```

**文档**：`docs/M1_1_*.md`（5 份规格和报告文件）

---

## 4. 当前状态总览

### 4.1 已完成

| 阶段 | 内容 | 状态 |
|---|---|---|
| 审计 | 服务器环境完整审计 | ✅ |
| M1.0 | Conda 环境、Playwright/Chromium、Qwen3.5-4B 加载、Slurm 验证 | ✅ |
| M1.1 | 采购数据库、15 个任务、Web 网站、Verifier、Slurm E2E | ✅ |

### 4.2 技术栈

| 层 | 技术 |
|---|---|
| 环境 | Conda (miniwebwork), Python 3.11.14 |
| AI/ML | PyTorch 2.10.0+cu128, Transformers 5.14.1, Qwen3.5-4B |
| Web | FastAPI 0.135.1, Uvicorn 0.41.0, Jinja2 3.1.6 |
| 数据库 | SQLite 3.51.1 |
| 浏览器 | Playwright 1.61.0, Chromium 149.0.7827.55 |
| 训练 | TRL 1.9.0, PEFT 0.19.1, Accelerate 1.13.0 |
| 调度 | Slurm 25.05 |
| 测试 | pytest 9.0.2 |

### 4.3 数据规模

| 实体 | 数量 |
|---|---|
| 供应商 | 6 |
| 商品 | 24 |
| 任务 | 15 |
| 测试 | 62 |
| 路由 | 11 |
| 代码文件 | 17 |

### 4.4 待实施（不可提前执行）

| 阶段 | 内容 |
|---|---|
| M1.2 | Agent Environment 封装（step/reset/observation/action space） |
| M1.3 | 文本 DOM / Accessibility Tree |
| M2.0 | Agent 推理循环（ReAct）、轨迹采集、SFT 数据构造 |
| M2.1 | LoRA SFT 训练 |
| M3.0 | GRPO 强化训练 |

### 4.5 已知未解决

- vLLM 0.17.0 与 transformers 5.14.1 版本不匹配（预存在，vLLM 仍正常运行）
- 无 Git LFS（模型已通过 ModelScope 本地缓存，不影响使用）
- 无 nvcc（不影响 PyTorch 预编译包）

---

## 5. 快速上手

```bash
# 环境
conda activate miniwebwork

# 初始化
python -m miniwebwork.cli init-db
python -m miniwebwork.cli validate-tasks

# 启动网站
python -m miniwebwork.webapp

# 测试
python -m pytest -q

# Slurm E2E
sbatch scripts/slurm/m1_1_procurement_e2e.sbatch
```

---

*最后更新：2026-07-25*
*当前阶段：M1.1 PASS*

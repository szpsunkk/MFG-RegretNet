# 📚 实验文档与脚本索引

完整的 RQ1-RQ6 实验运行指南、脚本和文档索引。

---

## 🚀 快速开始

**推荐阅读顺序**：

1. **[RQ_COMMANDS.md](RQ_COMMANDS.md)** ⭐ - 所有命令汇总（10 分钟）
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考卡（3 分钟）
3. **[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)** - 完整运行指南（30 分钟）

**一键运行**：
```bash
bash run_all_rq.sh              # 完整运行（8 小时）
QUICK=1 bash run_all_rq.sh      # 快速测试（40 分钟）
```

---

## 📖 文档清单

### 主要文档（新增）

| 文档 | 大小 | 说明 | 适合 |
|------|------|------|------|
| **[RQ_COMMANDS.md](RQ_COMMANDS.md)** | 12KB | 所有命令汇总 + 快速 FAQ | ⭐ 推荐首读 |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | 6KB | 快速参考卡（表格形式） | 查命令时 |
| **[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)** | 17KB | 完整运行指南（含环境、FAQ） | 深入学习 |
| **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** | 12KB | 改进总结（技术细节） | 开发者 |

### 原有文档（重要）

| 文档 | 说明 |
|------|------|
| **[实验思路.md](实验思路.md)** | 实验设计与理论依据（3.1-3.7 对应 RQ1-RQ6） |
| **[README.md](README.md)** | 项目总览（已更新快速开始） |
| **[docs/EXPERIMENT_DESIGN_STEPBYSTEP.md](docs/EXPERIMENT_DESIGN_STEPBYSTEP.md)** | 逐步实验设计 |
| **[docs/EXPERIMENT_PLAN_PRIVACY_PAPER.md](docs/EXPERIMENT_PLAN_PRIVACY_PAPER.md)** | Phase 1-5 详细计划 |
| **[docs/FL_MARKET_STYLE_RQ.md](docs/FL_MARKET_STYLE_RQ.md)** | 原始论文对照 |

### 实验过程文档

| 文档 | 说明 |
|------|------|
| **[exp_rq/RQ1_EXPERIMENT_PROCESS.md](exp_rq/RQ1_EXPERIMENT_PROCESS.md)** | RQ1 实验细节 |
| **[exp_rq/RQ2_PROCESS.md](exp_rq/RQ2_PROCESS.md)** | RQ2 实验细节 |
| **[exp_rq/RQ3_PROCESS.md](exp_rq/RQ3_PROCESS.md)** | RQ3 社会福利定义 |
| **[exp_rq/RQ4_PROCESS.md](exp_rq/RQ4_PROCESS.md)** | RQ4 FL 收敛细节 |
| **[exp_rq/RQ5_PROCESS.md](exp_rq/RQ5_PROCESS.md)** | RQ5 隐私-效用分析 |

---

## 🛠️ 脚本清单

### 主入口脚本（新增）

```
run_all_rq.sh                           ⭐ 主入口：一键运行所有 RQ1-RQ6 + 消融
├── 支持参数：
│   ├── QUICK=1                        # 快速模式
│   ├── --only rq1,rq2                # 只运行指定实验
│   ├── SKIP_RQ4=1                    # 跳过 RQ4
│   ├── MFG_CKPT=path                 # 指定 checkpoint
│   └── OUT=path                      # 指定输出目录
└── 输出：run/privacy_paper/ + logs/
```

### 各 RQ 脚本

```
scripts/
├── run_rq1_complete.sh                ✅ RQ1 完整流程（基线 + 神经 + 全部表与图）
├── run_rq1_paper.sh                   ✅ RQ1 论文主表/图 A/B（7 方法）
├── run_rq2_paper.sh                   ✅ RQ2 可扩展性（log-log + 内存 + 延迟）
├── run_rq3_complete.sh                ✅ RQ3 拍卖效率（收益 + 福利，图 1/2/3）
├── run_rq4_paper.sh                   ✅ RQ4 FL 收敛（MNIST + CIFAR，图 A/B/C/D）
├── run_rq5_paper.sh                   ✅ RQ5 隐私-效用（Pareto，图 A/B/C/D/E）
├── run_rq6_paper.sh                   🆕 RQ6 鲁棒性（虚假报价 + 遗憾增长）
└── run_ablation_study.sh              🆕 消融研究（MFG 组件对比）
```

### 快捷方式（项目根目录）

```
run_rq1.sh  →  scripts/run_rq1_complete.sh
run_rq3.sh  →  scripts/run_rq3_complete.sh
run_all_rq.sh  →  主入口（新增）
```

### Python 实验脚本

```
exp_rq/
├── rq1_incentive_compatibility.py     # RQ1 主脚本
├── rq1_paper_table_figures.py        # RQ1 论文表/图生成
├── rq1_convergence_curve.py          # RQ1 收敛曲线
├── rq1_figure_c_training_rounds.py   # RQ1 图 C（训练过程）
├── rq1_figure_d_regret_distribution.py # RQ1 图 D（遗憾分布）
├── rq2_paper_benchmark.py            # RQ2 基准测试
├── rq2_plot_paper_figures.py         # RQ2 绘图
├── rq3_paper_complete.py             # RQ3 完整流程
├── rq4_fl_benchmark.py               # RQ4 FL 训练
├── rq4_plot_paper_figures.py         # RQ4 绘图
├── rq5_fl_benchmark.py               # RQ5 FL 训练
├── rq5_plot_paper_figures.py         # RQ5 绘图
├── rq6_robustness.py                 # RQ6 鲁棒性（已改进）
└── ablation_study.py                 # 消融研究
```

---

## 📊 命令对照表

| 实验 | 主命令 | 快速模式 | 输出目录 | 用时（完整/快速） |
|------|--------|----------|----------|-------------------|
| **所有** | `bash run_all_rq.sh` | `QUICK=1 bash run_all_rq.sh` | `run/privacy_paper/` | 8h / 40min |
| **RQ1** | `bash run_rq1.sh` | `QUICK=1 bash run_rq1.sh` | `run/privacy_paper/rq1/` | 45min / 5min |
| **RQ2** | `bash scripts/run_rq2_paper.sh` | `QUICK=1 bash scripts/run_rq2_paper.sh` | `run/privacy_paper/rq2/` | 20min / 3min |
| **RQ3** | `bash run_rq3.sh` | `QUICK=1 bash run_rq3.sh` | `run/privacy_paper/rq3/` | 45min / 5min |
| **RQ4** | `bash scripts/run_rq4_paper.sh` | `QUICK=1 bash scripts/run_rq4_paper.sh` | `run/privacy_paper/rq4/` | 3h / 10min |
| **RQ5** | `bash scripts/run_rq5_paper.sh` | `QUICK=1 bash scripts/run_rq5_paper.sh` | `run/privacy_paper/rq5/` | 3h / 10min |
| **RQ6** | `bash scripts/run_rq6_paper.sh` | `QUICK=1 bash scripts/run_rq6_paper.sh` | `run/privacy_paper/rq6/` | 20min / 2min |
| **消融** | `bash scripts/run_ablation_study.sh` | `QUICK=1 bash scripts/run_ablation_study.sh` | `run/privacy_paper/ablation/` | 20min / 2min |

---

## 🎯 使用场景指南

### 场景 1：首次运行（完整复现）

**目标**：复现论文所有实验

**步骤**：
1. 阅读 [RQ_COMMANDS.md](RQ_COMMANDS.md)（10 分钟）
2. 按照"训练模型"章节训练 checkpoint（2-4 小时）
3. 运行 `QUICK=1 bash run_all_rq.sh` 验证流程（40 分钟）
4. 运行 `bash run_all_rq.sh` 完整实验（8 小时）
5. 查看 `run/privacy_paper/` 下的所有结果

**参考文档**：
- [RQ_COMMANDS.md](RQ_COMMANDS.md) - 训练模型 + 运行命令
- [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) - 环境配置 + FAQ

---

### 场景 2：快速验证流程

**目标**：快速验证脚本是否正常工作

**步骤**：
1. `QUICK=1 bash run_all_rq.sh`（40 分钟）
2. 检查 `run/privacy_paper/` 下的输出文件
3. 查看 `run/privacy_paper/logs/*.log` 确认无错误

**参考文档**：
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速命令查找

---

### 场景 3：调试单个实验

**目标**：单独运行某个 RQ 并查看详细输出

**步骤**（以 RQ1 为例）：
1. `QUICK=1 bash scripts/run_rq1_paper.sh`（5 分钟）
2. 检查 `run/privacy_paper/rq1/` 输出
3. 查看日志：`cat run/privacy_paper/logs/rq1.log`
4. 若正常，运行完整版：`bash scripts/run_rq1_paper.sh`

**参考文档**：
- [exp_rq/RQ1_EXPERIMENT_PROCESS.md](exp_rq/RQ1_EXPERIMENT_PROCESS.md) - RQ1 详细说明
- [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) - FAQ 章节

---

### 场景 4：生成论文图表

**目标**：只生成论文需要的主要图表，跳过耗时实验

**步骤**：
1. 运行 `SKIP_RQ4=1 SKIP_RQ5=1 bash run_all_rq.sh`（2 小时）
2. 或分别运行：
   ```bash
   bash scripts/run_rq1_paper.sh   # 主表 + 图 A/B
   bash scripts/run_rq2_paper.sh   # 图 1/2/3
   bash scripts/run_rq3_complete.sh # 图 1/2/3
   ```
3. 查看 `run/privacy_paper/*/figures/*.png`

**参考文档**：
- [RQ_COMMANDS.md](RQ_COMMANDS.md) - 输出目录结构

---

### 场景 5：重新生成图表

**目标**：在已有实验数据基础上，重新绘制图表（不重跑实验）

**步骤**：
1. RQ2：`python exp_rq/rq2_plot_paper_figures.py --input run/privacy_paper/rq2/rq2_paper_data.json --out-dir run/privacy_paper/rq2/figures`
2. RQ4：`python exp_rq/rq4_plot_paper_figures.py --rq4-dir run/privacy_paper/rq4`
3. RQ5：`python exp_rq/rq5_plot_paper_figures.py --rq5-dir run/privacy_paper/rq5 --dataset-filter MNIST`

**参考文档**：
- [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) - "重新生成图表"章节

---

### 场景 6：理解实验设计

**目标**：理解每个 RQ 的实验设计、指标定义、理论依据

**阅读顺序**：
1. [实验思路.md](实验思路.md) - 总体设计（3.1-3.7 章节）
2. [exp_rq/RQ*_PROCESS.md](exp_rq/) - 各 RQ 的详细说明
3. [docs/EXPERIMENT_DESIGN_STEPBYSTEP.md](docs/EXPERIMENT_DESIGN_STEPBYSTEP.md) - 逐步设计

**关键章节**：
- 实验思路 3.1 → RQ1（激励相容性）
- 实验思路 3.2 → RQ2（可扩展性）
- 实验思路 3.3 → RQ3（拍卖效率）
- 实验思路 3.4 → RQ4（FL 收敛性）
- 实验思路 3.5 → RQ5（隐私-效用）
- 实验思路 3.6 → RQ6（鲁棒性）
- 实验思路 3.7 → 消融研究

---

## 🔧 技术细节

### 主要改进

1. ✅ **统一入口**：`run_all_rq.sh` 一键运行所有实验
2. ✅ **完善脚本**：新增 RQ6、消融研究脚本
3. ✅ **完整文档**：3 个层次的文档（快速/完整/理论）
4. ✅ **灵活配置**：支持快速模式、部分运行、跳过指定实验
5. ✅ **错误处理**：Checkpoint 检查、依赖验证、详细日志

详见 [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)

### RQ6 改进

- 新增 `--seed` 参数（支持多种子）
- 新增遗憾与 IR 违反率计算
- 结果追加模式（支持多次运行）
- 详细输出（包含所有关键指标）

详见 [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) 的"RQ6 改进"章节

---

## 📁 输出结构

```
run/privacy_paper/
├── rq1/                    # RQ1: 激励相容性
│   ├── table_rq1_paper.md
│   └── figure_rq1_paper_*.png
├── rq2/                    # RQ2: 可扩展性
│   ├── rq2_paper_data.json
│   └── figures/
├── rq3/                    # RQ3: 拍卖效率
│   └── figure_rq3_*.png
├── rq4/                    # RQ4: FL 收敛性
│   ├── raw/
│   └── figures/
├── rq5/                    # RQ5: 隐私-效用
│   ├── raw/
│   └── figures/
├── rq6/                    # RQ6: 鲁棒性
│   └── rq6_results.json
├── ablation/               # 消融研究
│   ├── ablation_table.csv
│   └── ablation_table.md
└── logs/                   # 运行日志
    ├── rq1.log
    ├── rq2.log
    ├── rq3.log
    ├── rq4.log
    ├── rq5.log
    ├── rq6.log
    └── ablation.log
```

---

## ❓ 常见问题（FAQ）

### Q1: 从哪个文档开始看？

**A**: 推荐顺序：
1. [RQ_COMMANDS.md](RQ_COMMANDS.md) - 10 分钟，了解所有命令
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 3 分钟，快速查命令
3. [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) - 30 分钟，深入学习

### Q2: 如何快速验证流程？

**A**: `QUICK=1 bash run_all_rq.sh`（40 分钟）

### Q3: 缺少 checkpoint 怎么办？

**A**: 见 [RQ_COMMANDS.md](RQ_COMMANDS.md) 的"训练模型"章节

### Q4: 如何只运行部分实验？

**A**: `bash run_all_rq.sh --only rq1,rq2` 或 `SKIP_RQ4=1 bash run_all_rq.sh`

### Q5: 如何查看运行日志？

**A**: `tail -f run/privacy_paper/logs/rq1.log`

**更多 FAQ**：见 [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) 的"常见问题"章节

---

## 🔗 相关链接

- **GitHub**: https://github.com/your-repo/FL-Market
- **论文（IEEE BigData 2022）**: https://ieeexplore.ieee.org/document/10020232
- **论文（ArXiv）**: https://arxiv.org/abs/2106.04384

---

**最后更新**: 2026-03-21  
**版本**: v1.0  
**维护者**: FL-Market Team

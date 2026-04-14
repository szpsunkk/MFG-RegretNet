# main_m.pdf 论文实验一键运行指南

本文档提供 **main_m.pdf（MFG-RegretNet 论文）** 所有实验（RQ1-RQ6 + 消融研究）的完整运行命令和说明。

## 📋 目录

- [环境准备](#环境准备)
- [训练模型](#训练模型)
- [一键运行所有实验](#一键运行所有实验)
- [分别运行各实验](#分别运行各实验)
  - [RQ1: 激励相容性](#rq1-激励相容性)
  - [RQ2: 可扩展性](#rq2-可扩展性)
  - [RQ3: 拍卖效率](#rq3-拍卖效率)
  - [RQ4: FL 收敛性](#rq4-fl-收敛性)
  - [RQ5: 隐私-效用权衡](#rq5-隐私-效用权衡)
  - [RQ6: 鲁棒性](#rq6-鲁棒性)
  - [消融研究](#消融研究)
- [快速测试模式](#快速测试模式)
- [输出结构](#输出结构)
- [常见问题](#常见问题)

---

## 🔧 环境准备

### 1. 安装依赖

```bash
# 基础依赖（PyTorch、NumPy 等）
pip install torch torchvision numpy

# Phase 2/4/5 必需
pip install cvxpy

# 可选：绘图
pip install matplotlib seaborn
```

### 2. 验证环境（可选）

```bash
# Phase 1：数据与基础管线
python run_phase1_full_check.py

# Phase 2：基线 PAC/VCG 验证
python run_phase2_verify.py

# Phase 3：MFG-RegretNet 前向验证
python run_phase3_verify.py
```

---

## 🏋️ 训练模型

在运行实验前，需要先训练神经网络模型：

### 训练 MFG-RegretNet（必需）

```bash
# 快速试跑（2 个 epoch，验证流程）
python train_mfg_regretnet.py --num-epochs 2 --num-examples 1024

# 论文规模（200 轮，约 10 万样本）
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10 --n-items 1

# 若需多个 N 的模型（RQ2 可扩展性需要）
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 50 --n-items 1
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 100 --n-items 1

# 断点续训
python train_mfg_regretnet.py --resume path/to/checkpoint.pt --num-epochs 200
```

**输出**：checkpoint 保存在 `result/mfg_regretnet_privacy_*_checkpoint.pt`

### 训练 RegretNet（RQ1/消融需要）

```bash
# 训练普通 RegretNet（无 b_MFG）
python train_regretnet_privacy.py --num-epochs 200 --n-agents 10 --n-items 1
```

### 训练 DM-RegretNet（RQ1 需要）

```bash
# 训练 DM-RegretNet（用于 FL 模型交易场景）
python train_dm_regretnet_privacy.py --num-epochs 200 --n-agents 10 --n-items 1
```

---

## 🚀 一键运行所有实验

### 完整运行（所有 RQ1-RQ6 + 消融）

```bash
# 使用默认配置（需要预先训练好 MFG-RegretNet）
bash run_all_rq.sh

# 指定 checkpoint 路径
MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt \
  bash run_all_rq.sh

# 包含 RegretNet 和 DM-RegretNet（RQ1 需要）
MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt \
  REGRETNET_CKPT=result/regretnet_n10.pt \
  DM_REGRETNET_CKPT=result/dm_regretnet_n10.pt \
  bash run_all_rq.sh
```

### 只运行部分实验

```bash
# 只运行 RQ1 和 RQ2
bash run_all_rq.sh --only rq1,rq2

# 跳过 RQ4（耗时较长）
SKIP_RQ4=1 bash run_all_rq.sh

# 跳过多个实验
SKIP_RQ4=1 SKIP_RQ5=1 SKIP_RQ6=1 bash run_all_rq.sh
```

### 快速测试模式（小数据量）

```bash
# 所有实验使用小数据量（快速验证流程）
QUICK=1 bash run_all_rq.sh

# 指定输出目录
OUT=run/test_run QUICK=1 bash run_all_rq.sh
```

---

## 📊 分别运行各实验

### RQ1: 激励相容性

**目标**：评估平均事后遗憾（Regret）和 IR 违反率

**基线方法**：PAC、VCG、CSRA、MFG-Pricing、RegretNet、DM-RegretNet、MFG-RegretNet（Ours）

```bash
# 完整运行（包含所有基线 + 神经方法 + 全部表与图）
bash run_rq1.sh
# 或
bash scripts/run_rq1_complete.sh

# 缺 RegretNet/DM 权重时自动训练
REGRETNET_AUTO_TRAIN=1 DM_REGRETNET_AUTO_TRAIN=1 bash run_rq1.sh

# 仅论文主表 + 主图 A/B（7 方法）
bash scripts/run_rq1_paper.sh

# 关闭可选图 C/D
RQ1_FIG_CD=0 bash run_rq1.sh

# 仅解析基线（无神经网络）
python exp_rq/rq1_incentive_compatibility.py \
  --num-profiles 1000 \
  --seeds 42,43,44 \
  --n-agents 10 \
  --budget 50.0
```

**输出**：
- `run/privacy_paper/rq1/table_rq1_paper.md` - 论文主表（mean±std，t-test）
- `run/privacy_paper/rq1/figure_rq1_paper_regret.png` - 遗憾对比图
- `run/privacy_paper/rq1/figure_rq1_paper_ir.png` - IR 违反率对比图
- `run/privacy_paper/rq1/figure_rq1_regret_vs_pga_rounds.png` - 遗憾收敛曲线
- 可选：`figure_rq1_paper_regret_vs_epoch.png`（图 C）、`figure_rq1_paper_regret_distribution.png`（图 D）

---

### RQ2: 可扩展性

**目标**：评估计算时间随客户端数 N 的增长

**图表**：图 1 log-log 时间、图 2 内存/通信、图 3 堆叠延迟

```bash
# 完整运行
bash scripts/run_rq2_paper.sh

# 快速模式（仅 N=10,50,100）
QUICK=1 bash scripts/run_rq2_paper.sh

# 自定义 N 列表
N_LIST="10,50,100,200" bash scripts/run_rq2_paper.sh

# 跳过大规模 N（N>100）
bash scripts/run_rq2_paper.sh --skip-large-n
```

**输出**：
- `run/privacy_paper/rq2/figures/figure_rq2_1_time_vs_N_loglog.png`
- `run/privacy_paper/rq2/figures/figure_rq2_2_memory_comm.png`
- `run/privacy_paper/rq2/figures/figure_rq2_3_stacked_latency.png`
- `run/privacy_paper/rq2/rq2_paper_data.json` - 原始数据

---

### RQ3: 拍卖效率

**目标**：评估收益效率（η_rev）和社会福利（W̄）

**图表**：图 1 柱状图、图 2 随训练轮次、图 3 预算扫描

```bash
# 完整运行（包含所有图表）
bash run_rq3.sh
# 或
bash scripts/run_rq3_complete.sh

# 指定 checkpoint
MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt \
  REGRETNET_CKPT=result/regretnet_n10.pt \
  bash run_rq3.sh

# 仅运行评估脚本
python exp_rq/rq3_paper_complete.py \
  --n-agents 10 \
  --budget 50.0 \
  --num-profiles 1000 \
  --seeds 42,43,44
```

**输出**：
- `run/privacy_paper/rq3/figure_rq3_1_bar.png` - 收益与福利柱状图
- `run/privacy_paper/rq3/figure_rq3_2_training.png` - 随训练轮次变化
- `run/privacy_paper/rq3/figure_rq3_3_budget_sweep.png` - 预算扫描

---

### RQ4: FL 收敛性

**目标**：评估联邦学习精度收敛（MNIST + CIFAR-10）

**数据集**：MNIST、CIFAR-10；Dirichlet α ∈ {0.1, 0.5}

```bash
# 完整运行（MNIST + CIFAR-10，3 seeds，80 轮）
bash scripts/run_rq4_paper.sh

# 快速模式（20 轮，1 seed）
QUICK=1 bash scripts/run_rq4_paper.sh

# 自定义参数
SEEDS=5 ROUNDS=100 bash scripts/run_rq4_paper.sh

# 额外参数（必须放在 -- 后）
bash scripts/run_rq4_paper.sh -- --budget-rate 0.8 --pac

# 单独运行某个配置
python exp_rq/rq4_fl_benchmark.py \
  --dataset MNIST \
  --alpha 0.5 \
  --seed 0 \
  --rounds 80 \
  --rnd-step 5 \
  --out-dir run/privacy_paper/rq4
```

**输出**：
- `run/privacy_paper/rq4/figures/figure_rq4_A_mnist_cifar_alpha05.png` - 主图 A
- `run/privacy_paper/rq4/figures/figure_rq4_B_mnist_alpha_comparison.png` - 图 B（α 对比）
- `run/privacy_paper/rq4/figures/figure_rq4_C_cifar_alpha_comparison.png` - 图 C（α 对比）
- `run/privacy_paper/rq4/figures/figure_rq4_D_convergence_rate.png` - 图 D（收敛速度）

---

### RQ5: 隐私-效用权衡

**目标**：评估不同预算下的隐私-效用 Pareto 前沿

**预算档位**：B ∈ {0.35, 0.7, 1.05, 1.4} × 基准预算

```bash
# 完整运行（MNIST，多预算，3 seeds）
bash scripts/run_rq5_paper.sh

# 快速模式
QUICK=1 bash scripts/run_rq5_paper.sh

# 自定义参数
SEEDS=5 ROUNDS=80 BUDGET_RATES="0.3,0.6,1.0,1.4" bash scripts/run_rq5_paper.sh

# 同时运行 MNIST 和 CIFAR-10
DATASETS="MNIST CIFAR10" bash scripts/run_rq5_paper.sh

# 额外参数
bash scripts/run_rq5_paper.sh -- --pac --rounds 40

# 单独运行
python exp_rq/rq5_fl_benchmark.py \
  --dataset MNIST \
  --alpha 0.5 \
  --seed 0 \
  --rounds 60 \
  --budget-rates "0.35,0.7,1.05,1.4" \
  --out-dir run/privacy_paper/rq5
```

**输出**：
- `run/privacy_paper/rq5/figures/figure_rq5_A_pareto_*.png` - Pareto 前沿（图 A）
- `run/privacy_paper/rq5/figures/figure_rq5_B_privacy_cost_*.png` - 隐私成本分布（图 B）
- `run/privacy_paper/rq5/figures/figure_rq5_C_utility_boxplot_*.png` - 效用分布（图 C）
- `run/privacy_paper/rq5/figures/figure_rq5_D_budget_vs_accuracy_*.png` - 预算-精度曲线（图 D）
- `run/privacy_paper/rq5/figures/figure_rq5_E_burden_distribution_*.png` - 负担分布（图 E）

---

### RQ6: 鲁棒性

**目标**：评估虚假报价和勾结攻击下的鲁棒性

**攻击参数**：虚假报价比例 δ ∈ {0.1, 0.3, 0.5}

```bash
# 完整运行
bash scripts/run_rq6_paper.sh

# 快速模式
QUICK=1 bash scripts/run_rq6_paper.sh

# 自定义参数
FALSE_RATIOS="0.1,0.2,0.3,0.4,0.5" SEEDS=5 bash scripts/run_rq6_paper.sh

# 单独运行
python exp_rq/rq6_robustness.py \
  --out-dir run/privacy_paper/rq6 \
  --n-agents 10 \
  --budget 50.0 \
  --num-profiles 1000 \
  --false-ratio 0.1 \
  --seed 42
```

**输出**：
- `run/privacy_paper/rq6/rq6_results.json` - 所有配置的结果汇总

---

### 消融研究

**目标**：评估 MFG 组件和增强拉格朗日的影响

**对比**：RegretNet（无 b_MFG）vs MFG-RegretNet（有 b_MFG）

```bash
# 完整运行
bash scripts/run_ablation_study.sh

# 快速模式
QUICK=1 bash scripts/run_ablation_study.sh

# 指定 checkpoint
REGRETNET_CKPT=result/regretnet_n10.pt \
  MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt \
  bash scripts/run_ablation_study.sh

# 单独运行
python exp_rq/ablation_study.py \
  --out-dir run/privacy_paper/ablation \
  --n-agents 10 \
  --budget 50.0 \
  --num-profiles 1000 \
  --regretnet-ckpt result/regretnet_n10.pt \
  --mfg-regretnet-ckpt result/mfg_regretnet_privacy_200_checkpoint.pt
```

**输出**：
- `run/privacy_paper/ablation/ablation_table.csv` - 消融对比表
- `run/privacy_paper/ablation/ablation_table.md` - Markdown 格式表格

---

## ⚡ 快速测试模式

所有脚本都支持 `QUICK=1` 环境变量来快速验证流程：

```bash
# 使用小数据量和少量轮次
QUICK=1 bash run_all_rq.sh
QUICK=1 bash run_rq1.sh
QUICK=1 bash scripts/run_rq2_paper.sh
QUICK=1 bash scripts/run_rq3_complete.sh
QUICK=1 bash scripts/run_rq4_paper.sh
QUICK=1 bash scripts/run_rq5_paper.sh
QUICK=1 bash scripts/run_rq6_paper.sh
QUICK=1 bash scripts/run_ablation_study.sh
```

快速模式会自动调整以下参数：
- 减少样本数（1000 → 300）
- 减少训练轮次（80 → 20）
- 减少种子数（3 → 1）
- 限制 N 列表（10,50,100,200,400 → 10,50,100）

---

## 📁 输出结构

完整运行后，输出目录结构如下：

```
run/privacy_paper/
├── rq1/                                      # RQ1: 激励相容性
│   ├── table_rq1_paper.md                   # 论文主表
│   ├── figure_rq1_paper_regret.png          # 遗憾对比图
│   ├── figure_rq1_paper_ir.png              # IR 违反率图
│   ├── figure_rq1_regret_vs_pga_rounds.png  # 遗憾收敛曲线
│   ├── figure_rq1_paper_regret_vs_epoch.png # 图 C（可选）
│   └── figure_rq1_paper_regret_distribution.png # 图 D（可选）
├── rq2/                                      # RQ2: 可扩展性
│   ├── rq2_paper_data.json                  # 原始数据
│   └── figures/
│       ├── figure_rq2_1_time_vs_N_loglog.png
│       ├── figure_rq2_2_memory_comm.png
│       └── figure_rq2_3_stacked_latency.png
├── rq3/                                      # RQ3: 拍卖效率
│   ├── figure_rq3_1_bar.png
│   ├── figure_rq3_2_training.png
│   └── figure_rq3_3_budget_sweep.png
├── rq4/                                      # RQ4: FL 收敛性
│   ├── raw/                                  # 原始运行数据
│   └── figures/
│       ├── figure_rq4_A_mnist_cifar_alpha05.png
│       ├── figure_rq4_B_mnist_alpha_comparison.png
│       ├── figure_rq4_C_cifar_alpha_comparison.png
│       └── figure_rq4_D_convergence_rate.png
├── rq5/                                      # RQ5: 隐私-效用权衡
│   ├── raw/
│   └── figures/
│       ├── figure_rq5_A_pareto_*.png
│       ├── figure_rq5_B_privacy_cost_*.png
│       ├── figure_rq5_C_utility_boxplot_*.png
│       ├── figure_rq5_D_budget_vs_accuracy_*.png
│       └── figure_rq5_E_burden_distribution_*.png
├── rq6/                                      # RQ6: 鲁棒性
│   └── rq6_results.json
├── ablation/                                 # 消融研究
│   ├── ablation_table.csv
│   └── ablation_table.md
└── logs/                                     # 运行日志
    ├── rq1.log
    ├── rq2.log
    ├── rq3.log
    ├── rq4.log
    ├── rq5.log
    ├── rq6.log
    └── ablation.log
```

---

## ❓ 常见问题

### 1. 缺少 checkpoint 怎么办？

**问题**：运行实验时提示 `MFG checkpoint not found`

**解决**：
```bash
# 训练 MFG-RegretNet
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400

# 或者在运行时指定路径
MFG_CKPT=path/to/checkpoint.pt bash run_all_rq.sh
```

### 2. RQ1 需要多个 checkpoint 吗？

**需要**：RQ1 评估 7 个方法，建议准备：
- MFG-RegretNet（必需）：`result/mfg_regretnet_privacy_*_checkpoint.pt`
- RegretNet（可选）：`result/regretnet_n10.pt`
- DM-RegretNet（可选）：`result/dm_regretnet_n10.pt`

缺少时 `run_rq1_complete.sh` 可自动训练（设置 `REGRETNET_AUTO_TRAIN=1`）

### 3. RQ4/RQ5 运行很慢？

**原因**：需要训练 FL 模型（80 轮 × 多种子 × 多数据集）

**解决**：
```bash
# 使用快速模式
QUICK=1 bash scripts/run_rq4_paper.sh  # 20 轮，1 种子

# 或减少轮次
ROUNDS=40 bash scripts/run_rq4_paper.sh
```

### 4. 如何只生成论文主图？

```bash
# RQ1 主表 + 图 A/B（不含图 C/D）
RQ1_FIG_CD=0 bash scripts/run_rq1_paper.sh

# RQ4 主图 A（α=0.5）
FIG_A_ALPHA=0.5 bash scripts/run_rq4_paper.sh
```

### 5. 内存不足（OOM）？

**解决**：
```bash
# 减少 batch size（修改脚本中的 --batch-size 参数）
# 或使用 CPU（影响速度）
export CUDA_VISIBLE_DEVICES=""
```

### 6. 如何验证某个实验的正确性？

```bash
# Phase 验证脚本
python run_phase1_full_check.py  # 数据接口
python run_phase2_verify.py      # PAC/VCG
python run_phase3_verify.py      # MFG-RegretNet

# 快速模式试跑
QUICK=1 bash scripts/run_rq1_paper.sh
```

### 7. 如何重新生成图表（不重跑实验）？

```bash
# RQ2 仅绘图
python exp_rq/rq2_plot_paper_figures.py \
  --input run/privacy_paper/rq2/rq2_paper_data.json \
  --out-dir run/privacy_paper/rq2/figures

# RQ4 仅绘图
python exp_rq/rq4_plot_paper_figures.py \
  --rq4-dir run/privacy_paper/rq4 \
  --fig-a-alpha 0.5

# RQ5 仅绘图
python exp_rq/rq5_plot_paper_figures.py \
  --rq5-dir run/privacy_paper/rq5 \
  --dataset-filter MNIST
```

---

## 📚 相关文档

- **实验设计**：`实验思路.md` - 详细实验设计与理论依据
- **逐步指南**：`docs/EXPERIMENT_DESIGN_STEPBYSTEP.md` - 分步实验流程
- **过程文档**：
  - `exp_rq/RQ1_EXPERIMENT_PROCESS.md` - RQ1 实验细节
  - `exp_rq/RQ2_PROCESS.md` - RQ2 实验细节
  - `exp_rq/RQ3_PROCESS.md` - RQ3 社会福利定义
  - `exp_rq/RQ4_PROCESS.md` - RQ4 FL 收敛细节
  - `exp_rq/RQ5_PROCESS.md` - RQ5 隐私-效用分析
- **FL-Market 对标**：`docs/FL_MARKET_STYLE_RQ.md` - 原始论文实验对照

---

## 🎯 推荐工作流

### 首次运行（完整复现）

```bash
# 1. 训练模型（约 2-4 小时）
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10
python train_regretnet_privacy.py --num-epochs 200 --n-agents 10
python train_dm_regretnet_privacy.py --num-epochs 200 --n-agents 10

# 2. 快速验证（约 10 分钟）
QUICK=1 bash run_all_rq.sh

# 3. 完整运行（约 8-12 小时，取决于硬件）
bash run_all_rq.sh
```

### 单个实验调试

```bash
# 1. 快速模式验证流程
QUICK=1 bash scripts/run_rq1_paper.sh

# 2. 检查输出
ls run/privacy_paper/rq1/

# 3. 完整运行
bash scripts/run_rq1_paper.sh
```

### 论文图表生成

```bash
# 生成论文所有主图
bash run_all_rq.sh

# 或分别生成
bash scripts/run_rq1_paper.sh   # RQ1 主表 + 图 A/B
bash scripts/run_rq2_paper.sh   # RQ2 图 1/2/3
bash scripts/run_rq3_complete.sh # RQ3 图 1/2/3
bash scripts/run_rq4_paper.sh   # RQ4 图 A/B/C/D
bash scripts/run_rq5_paper.sh   # RQ5 图 A/B/C/D/E
```

---

**完整实验预计运行时间**（参考配置：NVIDIA RTX 3090，32GB RAM）：
- RQ1：30-60 分钟
- RQ2：15-30 分钟
- RQ3：30-60 分钟
- RQ4：2-4 小时（取决于 FL 轮次）
- RQ5：2-4 小时（多预算档位）
- RQ6：15-30 分钟
- 消融：15-30 分钟
- **总计：约 6-10 小时**

使用 `QUICK=1` 模式可缩短到 **30-60 分钟**。

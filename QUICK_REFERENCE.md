# 🚀 实验快速参考卡

## 一键运行所有实验

```bash
# 完整运行（需要先训练模型）
bash run_all_rq.sh

# 快速测试（30-60 分钟）
QUICK=1 bash run_all_rq.sh

# 只运行部分实验
bash run_all_rq.sh --only rq1,rq2,rq3
```

---

## 各实验一键命令

| 实验 | 命令 | 输出 | 用时 |
|------|------|------|------|
| **RQ1** 激励相容性 | `bash run_rq1.sh` | `run/privacy_paper/rq1/` | 30-60min |
| **RQ2** 可扩展性 | `bash scripts/run_rq2_paper.sh` | `run/privacy_paper/rq2/figures/` | 15-30min |
| **RQ3** 拍卖效率 | `bash run_rq3.sh` | `run/privacy_paper/rq3/` | 30-60min |
| **RQ4** FL 收敛性 | `bash scripts/run_rq4_paper.sh` | `run/privacy_paper/rq4/figures/` | 2-4h |
| **RQ5** 隐私-效用 | `bash scripts/run_rq5_paper.sh` | `run/privacy_paper/rq5/figures/` | 2-4h |
| **RQ6** 鲁棒性 | `bash scripts/run_rq6_paper.sh` | `run/privacy_paper/rq6/` | 15-30min |
| **消融研究** | `bash scripts/run_ablation_study.sh` | `run/privacy_paper/ablation/` | 15-30min |

---

## 训练模型（必需）

```bash
# MFG-RegretNet（必需，所有实验）
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10

# RegretNet（RQ1/消融）
python train_regretnet_privacy.py --num-epochs 200 --n-agents 10

# DM-RegretNet（RQ1）
python train_dm_regretnet_privacy.py --num-epochs 200 --n-agents 10
```

---

## 快速模式（所有实验通用）

```bash
QUICK=1 bash run_rq1.sh
QUICK=1 bash scripts/run_rq2_paper.sh
QUICK=1 bash scripts/run_rq3_complete.sh
QUICK=1 bash scripts/run_rq4_paper.sh
QUICK=1 bash scripts/run_rq5_paper.sh
QUICK=1 bash scripts/run_rq6_paper.sh
QUICK=1 bash scripts/run_ablation_study.sh
```

---

## 环境变量配置

```bash
# 指定 checkpoint
MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt \
  REGRETNET_CKPT=result/regretnet_n10.pt \
  DM_REGRETNET_CKPT=result/dm_regretnet_n10.pt \
  bash run_all_rq.sh

# 指定输出目录
OUT=run/test_run bash run_all_rq.sh

# 跳过某些实验
SKIP_RQ4=1 SKIP_RQ5=1 bash run_all_rq.sh
```

---

## RQ1-RQ6 核心命令总结

### RQ1: 激励相容性（7 方法对比）

```bash
# 完整（包含图 C/D）
bash run_rq1.sh

# 仅主表 + 图 A/B
bash scripts/run_rq1_paper.sh

# 关闭图 C/D
RQ1_FIG_CD=0 bash run_rq1.sh
```

### RQ2: 可扩展性（N vs 时间）

```bash
# 默认 N=10,50,100,200,400
bash scripts/run_rq2_paper.sh

# 快速（N=10,50,100）
QUICK=1 bash scripts/run_rq2_paper.sh

# 自定义 N
N_LIST="10,50,100" bash scripts/run_rq2_paper.sh
```

### RQ3: 拍卖效率（收益 + 福利）

```bash
# 完整（图 1/2/3）
bash run_rq3.sh

# 等价命令
bash scripts/run_rq3_complete.sh
```

### RQ4: FL 收敛性（MNIST + CIFAR-10）

```bash
# 默认：80 轮，3 seeds，α∈{0.1,0.5}
bash scripts/run_rq4_paper.sh

# 快速：20 轮，1 seed
QUICK=1 bash scripts/run_rq4_paper.sh

# 自定义
SEEDS=5 ROUNDS=100 bash scripts/run_rq4_paper.sh
```

### RQ5: 隐私-效用权衡

```bash
# 默认：MNIST，预算档位 0.35,0.7,1.05,1.4
bash scripts/run_rq5_paper.sh

# 快速
QUICK=1 bash scripts/run_rq5_paper.sh

# 多数据集
DATASETS="MNIST CIFAR10" bash scripts/run_rq5_paper.sh
```

### RQ6: 鲁棒性（虚假报价）

```bash
# 默认：δ∈{0.1,0.3,0.5}
bash scripts/run_rq6_paper.sh

# 自定义
FALSE_RATIOS="0.1,0.2,0.3" bash scripts/run_rq6_paper.sh
```

### 消融研究（MFG 组件）

```bash
# 自动查找 checkpoint
bash scripts/run_ablation_study.sh

# 指定 checkpoint
REGRETNET_CKPT=result/regretnet_n10.pt \
  MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt \
  bash scripts/run_ablation_study.sh
```

---

## 常用检查命令

```bash
# 查看已训练的模型
ls -lh result/*.pt

# 查看实验结果
tree run/privacy_paper/ -L 2

# 查看日志
tail -f run/privacy_paper/logs/rq1.log

# 清理旧结果（谨慎！）
rm -rf run/privacy_paper/
```

---

## 重新生成图表（不重跑）

```bash
# RQ2
python exp_rq/rq2_plot_paper_figures.py \
  --input run/privacy_paper/rq2/rq2_paper_data.json \
  --out-dir run/privacy_paper/rq2/figures

# RQ4
python exp_rq/rq4_plot_paper_figures.py \
  --rq4-dir run/privacy_paper/rq4

# RQ5
python exp_rq/rq5_plot_paper_figures.py \
  --rq5-dir run/privacy_paper/rq5 \
  --dataset-filter MNIST
```

---

## 分步验证（调试）

```bash
# Phase 1-3 验证
python run_phase1_full_check.py   # 数据接口
python run_phase2_verify.py       # PAC/VCG
python run_phase3_verify.py       # MFG-RegretNet

# Phase 4-5 评估
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000
python run_phase5_tables_figures.py --input run/privacy_paper/phase4_summary.json
```

---

## 论文图表清单

### RQ1
- ✅ `table_rq1_paper.md` - 主表（7 方法，mean±std，t-test）
- ✅ `figure_rq1_paper_regret.png` - 遗憾对比（图 A）
- ✅ `figure_rq1_paper_ir.png` - IR 违反率（图 B）
- ✅ `figure_rq1_paper_regret_vs_epoch.png` - 训练过程（图 C，可选）
- ✅ `figure_rq1_paper_regret_distribution.png` - 遗憾分布（图 D，可选）

### RQ2
- ✅ `figure_rq2_1_time_vs_N_loglog.png` - log-log 时间（图 1）
- ✅ `figure_rq2_2_memory_comm.png` - 内存/通信（图 2）
- ✅ `figure_rq2_3_stacked_latency.png` - 堆叠延迟（图 3）

### RQ3
- ✅ `figure_rq3_1_bar.png` - 收益与福利柱图（图 1）
- ✅ `figure_rq3_2_training.png` - 随训练轮次（图 2）
- ✅ `figure_rq3_3_budget_sweep.png` - 预算扫描（图 3）

### RQ4
- ✅ `figure_rq4_A_*.png` - MNIST+CIFAR α=0.5（主图 A）
- ✅ `figure_rq4_B_*.png` - MNIST α 对比（图 B）
- ✅ `figure_rq4_C_*.png` - CIFAR α 对比（图 C）
- ✅ `figure_rq4_D_*.png` - 收敛速度（图 D）

### RQ5
- ✅ `figure_rq5_A_pareto_*.png` - Pareto 前沿（图 A）
- ✅ `figure_rq5_B_privacy_cost_*.png` - 隐私成本（图 B）
- ✅ `figure_rq5_C_utility_boxplot_*.png` - 效用分布（图 C）
- ✅ `figure_rq5_D_budget_vs_accuracy_*.png` - 预算-精度（图 D）
- ✅ `figure_rq5_E_burden_distribution_*.png` - 负担分布（图 E）

### RQ6
- ✅ `rq6_results.json` - 鲁棒性评估数据

### 消融
- ✅ `ablation_table.csv` - 消融对比表
- ✅ `ablation_table.md` - Markdown 格式

---

**完整文档**: `RUN_EXPERIMENTS.md`  
**实验设计**: `实验思路.md`  
**代码仓库**: https://github.com/your-repo/FL-Market

# RQ1–RQ5 实验运行说明

在**仓库根目录**执行；需已安装 `torch`、项目依赖；神经机制需 `result/` 下有与 **N=10, n_items=1** 匹配的 checkpoint（RQ1/RQ3/RQ4/RQ5 会自动探测，也可手动指定）。

**RegretNet / DM-RegretNet 是否要提前训练？**

- **只跑 `run_rq1_complete.sh`**：不必提前跑；缺 ckpt 时会自动执行 `train_regretnet_privacy.py` 与 `train_dm_regretnet_privacy.py`（可设 `REGRETNET_AUTO_TRAIN=0`、`DM_REGRETNET_AUTO_TRAIN=0` 关闭）。
- **跑 `run_rq1_paper.sh` 或 RQ2/RQ3/RQ4/RQ5**：不会自动训练；若希望表/图里出现 RegretNet、DM-RegretNet，需**提前**在根目录执行：
  ```bash
  python train_regretnet_privacy.py --n-agents 10 --n-items 1 --num-epochs 10
  python train_dm_regretnet_privacy.py --n-agents 10 --n-items 1 --num-epochs 10
  ```
  产出：`result/regretnet_privacy_*_checkpoint.pt`、`result/dm_regretnet_privacy_*_checkpoint.pt`。

---

## RQ1 激励与主表/图

| 用途 | 命令 |
|------|------|
| **论文主表 + 图 A/B**（遗憾、IR%） | `bash scripts/run_rq1_paper.sh` |
| **完整流程**（IC 表、柱图、t-test、主表、可选图 C/D） | `bash scripts/run_rq1_complete.sh` |
| **轻量**（仅 incentive 相关） | `bash scripts/run_rq1_full.sh` |

常用环境变量：

```bash
export OUT=run/privacy_paper/rq1
export SEEDS=42,43,44,45,46
export MFG_CKPT=result/mfg_regretnet_privacy_200_checkpoint.pt
export REGRETNET_CKPT=...   # 可选
export DM_REGRETNET_CKPT=... # 可选
IR_LOG=1 bash scripts/run_rq1_paper.sh   # IR 图纵轴对数
```

单跑 Python（示例）：

```bash
python exp_rq/rq1_paper_table_figures.py --out-dir run/privacy_paper/rq1 --seeds 42,43,44
python exp_rq/rq1_incentive_compatibility.py --out-dir run/privacy_paper/rq1 --seeds 42,43
```

---

## RQ2 可扩展性（时间 / 显存 / 通信）

```bash
bash scripts/run_rq2_paper.sh
QUICK=1 bash scripts/run_rq2_paper.sh
N_LIST="10,50,100,200" bash scripts/run_rq2_paper.sh
```

输出：`run/privacy_paper/rq2/rq2_paper_data.json`，图在 `run/privacy_paper/rq2/figures/`。

---

## RQ3 收益 η_rev 与社会福利 W̄

```bash
bash scripts/run_rq3_complete.sh
```

常用环境变量：`OUT`、`N_AGENTS`、`BUDGET`、`NUM_PROFILES`、`SEEDS`、`MFG_CKPT`、`REGRETNET_CKPT`、`DM_REGRETNET_CKPT`、`RQ3_NO_FIG2=1`、`RQ3_NO_FIG3=1`。

```bash
python exp_rq/rq3_paper_complete.py --out-dir run/privacy_paper/rq3 --n-agents 10 --budget 50 --seeds 42,43,44
```

---

## RQ4 FL 精度 vs 通信轮次

```bash
bash scripts/run_rq4_paper.sh
QUICK=1 bash scripts/run_rq4_paper.sh
# 额外参数须写在 -- 之后：
bash scripts/run_rq4_paper.sh -- --budget-rate 0.8 --pac
# 论文 Algorithm 2（高斯权重 + ε-FedAvg）：
bash scripts/run_rq4_paper.sh -- --pag-fl-alg2
```

输出：`run/privacy_paper/rq4/raw/*.json`，图：`run/privacy_paper/rq4/figures/`。  
仅画图：`python exp_rq/rq4_plot_paper_figures.py --rq4-dir run/privacy_paper/rq4`

---

## RQ5 隐私–效用（Pareto、柱图、箱线等）

```bash
bash scripts/run_rq5_paper.sh
QUICK=1 bash scripts/run_rq5_paper.sh
SEEDS=3 DATASETS="MNIST CIFAR10" bash scripts/run_rq5_paper.sh
bash scripts/run_rq5_paper.sh -- --pac --rounds 40
```

`DATASETS` 用**空格**分隔（勿用逗号）。可选：`MFG_CKPT`、`FL_LR`、`--pag-fl-alg2` 等（经 `--` 传入）。

手动作图：

```bash
python exp_rq/rq5_plot_paper_figures.py --rq5-dir run/privacy_paper/rq5 --dataset-filter MNIST
```

---

## 对照表

| RQ | 一键脚本 | 主要 Python |
|----|-----------|-------------|
| RQ1 主表/图 | `scripts/run_rq1_paper.sh` | `exp_rq/rq1_paper_table_figures.py` |
| RQ1 完整 | `scripts/run_rq1_complete.sh` | `rq1_incentive_compatibility.py` 等 |
| RQ2 | `scripts/run_rq2_paper.sh` | `exp_rq/rq2_paper_benchmark.py` → `rq2_plot_paper_figures.py` |
| RQ3 | `scripts/run_rq3_complete.sh` | `exp_rq/rq3_paper_complete.py` |
| RQ4 | `scripts/run_rq4_paper.sh` | `exp_rq/rq4_fl_benchmark.py` → `rq4_plot_paper_figures.py` |
| RQ5 | `scripts/run_rq5_paper.sh` | `exp_rq/rq5_fl_benchmark.py` → `rq5_plot_paper_figures.py` |

更细的指标与注意事项见：`exp_rq/RQ1_EXPERIMENT_PROCESS.md`、`RQ2_PROCESS.md`、`RQ3_PROCESS.md`、`RQ4_PROCESS.md`、`RQ5_PROCESS.md`，以及 `docs/RQ1_RQ5_EXECUTION_AUDIT.md`、`docs/MAIN_PDF_ALGORITHM_MAP.md`（Alg.2 FL）。

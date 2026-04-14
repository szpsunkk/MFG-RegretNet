# RQ5：隐私–效用（FL）

## 一键

```bash
bash scripts/run_rq5_paper.sh
QUICK=1 bash scripts/run_rq5_paper.sh   # 冒烟
```

## 指标（与论文章节对齐）

| 符号 | 实现 |
|------|------|
| **$\bar{\epsilon}$**（图 A/D 横轴/左轴） | 每轮对「当轮 $\epsilon_i>0$ 且有数据客户端」取均值，再对 **FL 轮次时间平均** |
| **$\mathcal{A}_{\mathrm{test}}$** | 最后一轮全局模型在测试集准确率 |
| **客户端 $\epsilon_i^{\mathrm{out}}$**（图 C） | **最后一轮** 拍卖给出的各客户端 $\epsilon$ |
| **Gini** | 对 $\epsilon_i^{\mathrm{out}}>0$ 的客户端计算基尼（箱线图脚注） |
| **图 E** | $\|\mathbf{w}^{(t)}-\mathbf{w}^{(t-1)}\|_2$（全局参数向量差） |

## 预算 $B$

每轮：`budget = max_cost(bids) × B`。**同一随机 bid 序列** 下扫描多个 $B$，得到 Pareto 前沿（图 A）；不同 $B$ 对应图 D。

## 输出

| 路径 | 内容 |
|------|------|
| `rq5/raw/MNIST_a0.5_s0.json` | 该种子下各 `(method, B)` 一条记录 |
| `figures/figure_rq5_A_pareto_eps_acc.png` | 图 A（仅 MNIST 或仅 CIFAR 时） |
| `figures/figure_rq5_*_MNIST.png` / `*_CIFAR10.png` | **同一 `raw/` 含多数据集时** 按数据集分文件（脚本已自动分两次作图） |
| `rq5_summary.json` 或 `rq5_summary_MNIST.json` … | 汇总 JSON |

`run_rq5_paper.sh` 结束作图时与训练阶段相同，对 `for ds in $DATASETS` 的**同一批数据集名**各跑一次 `--dataset-filter`（勿用 `read <<<`，否则多行/多数据集易与训练不一致）。单数据集也传 filter，避免 `raw/` 残留其它数据集的 JSON。

## 与 RQ4 / RQ1

- **RQ4**：时间维收敛；**RQ5**：固定总轮数下 **隐私强度 vs 精度** 及 **负担分布**。
- **RQ1**：若 Ours 为保 IR 收紧部分 $\epsilon$，RQ5 应强调在 **可接受 IR** 下仍处合理 Pareto；不可只报精度最高。

## 单跑与作图

```bash
python exp_rq/rq5_fl_benchmark.py --dataset MNIST --seed 0 --budget-rates 0.5,1.0,1.5
python exp_rq/rq5_plot_paper_figures.py --rq5-dir run/privacy_paper/rq5 --b-ref 1.0
```

可选：`--pac`、`--skip-regretnet`、`--mfg-ckpt`、`--regretnet-ckpt`。

旧脚本 `exp_rq/rq5_privacy_utility.py` 仅机制层 PAC 成本估计，**不含 FL**；论文主图请以本管线为准。

**Ours 在五张图上不占优时怎么办**：见 **`exp_rq/RQ5_IMPROVING_OURS.md`**；快速重训 MFG：`bash scripts/train_mfg_for_rq5.sh`。Shell 支持 `MFG_CKPT`、`FL_LR` 环境变量。

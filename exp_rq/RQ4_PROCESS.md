# RQ4：FL 测试精度 vs 轮次（MNIST / CIFAR-10）

## 一键运行

```bash
# 完整（默认 3 seeds × 80 rounds × 2 α × 2 数据集，耗时较长）
bash scripts/run_rq4_paper.sh

# 快速冒烟（1 seed、20 轮）
QUICK=1 bash scripts/run_rq4_paper.sh
```

需已训练 **MFG-RegretNet**（Ours）与 **RegretNet**；脚本会从 `result/` 自动解析 checkpoint（与 RQ1 相同逻辑）。若缺失则跳过对应曲线。

## 输出

| 文件 | 说明 |
|------|------|
| `run/privacy_paper/rq4/raw/{MNIST\|CIFAR10}_a{α}_s{seed}.json` | 每种子原始曲线 |
| `run/privacy_paper/rq4/rq4_aggregated.json` | 跨种子 mean±std |
| `run/privacy_paper/rq4/figures/figure_rq4_A_acc_vs_round.png` | 图 A：精度 vs 轮次（双面板，α 由 `FIG_A_ALPHA`） |
| `.../figure_rq4_B_final_acc_by_alpha.png` | 图 B：最终精度，α=0.1/0.5 分组柱 |
| `.../figure_rq4_C_delta_accuracy.png` | 图 C：ΔA = A_NoDP − A_method |
| `.../figure_rq4_D_train_loss.png` | 图 D：训练损失（附录） |

## 方法说明

- **Ours**：MFG-RegretNet 分配 ε；FL 默认 **Laplace 梯度 + ConvlAggr**；与正文 Algorithm 2 一致时用 `--pag-fl-alg2`（**高斯加在本地权重** + **ε 加权 FedAvg**，见 `docs/MAIN_PDF_ALGORITHM_MAP.md`）
- **稳定曲线（默认已调）**：每轮 **多轮次本地训练**（`--local-epochs` 默认 2、小批量 `--local-batch-size` 64）；分类头输出 **logits** + `CrossEntropyLoss`（与单步 `log_softmax` 混用相比更易收敛）。若仍抖，可加 **`--budget-rate 0.8`** 固定每轮预算倍率，或略降 `--fl-lr`。
- **CSRA / MFG-Pricing / RegretNet**：与 RQ1/RQ3 一致  
- **Uniform-DP**：各客户端相同 ε（默认 2.555），无拍卖  
- **No-DP (upper)**：极大 ε，作上界参照  

每轮 **预算** 为 `max_cost(bids) × U[min_budget_rate, max_budget_rate]`，**同一随机预算序列** 对所有方法共用。固定预算：`bash scripts/run_rq4_paper.sh -- --budget-rate 0.8`。

## 与 RQ1/RQ3 衔接（写 caption）

RQ4 只报告 **精度**；可注明：在 RQ1 **IR**、RQ3 **平均社会福利** 前提下，Ours 在下游 FL 精度上仍具竞争力。若 RegretNet 精度更高但 IR 更差，讨论 **精度–激励权衡**。

## 单数据集 / 仅画图

```bash
python exp_rq/rq4_fl_benchmark.py --dataset MNIST --alpha 0.5 --seed 0 --rounds 50
python exp_rq/rq4_plot_paper_figures.py --rq4-dir run/privacy_paper/rq4 --fig-a-alpha 0.5
# 若 raw 里同时有普通 JSON 与 *_pagalg2.json：默认只聚合普通曲线；画 Alg.2 加 --prefer-pagalg2
```

可选：`--pac`、`--skip-regretnet`、`--mfg-ckpt`、`--regretnet-ckpt`、**`--pag-fl-alg2`**（及 `--delta-dp`、`--eps-min-alg2`）。

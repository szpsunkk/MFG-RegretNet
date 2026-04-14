# RQ1 实验过程说明（可复现）

## 研究问题

机制是否更接近 **DSIC（占优策略激励相容）** 与 **IR（个体理性）**：平均事后遗憾越小越接近真实报价最优；IR 违反率越低越好。

## 1. 数据准备

| 项目 | 设定 |
|------|------|
| 估值 | \(v_i \sim \mathrm{Uniform}[0,1]\) |
| 隐私预算参数 | \(\varepsilon_i \sim \mathrm{Uniform}[0.1,5]\) |
| 成本 | \(c_i = v_i \cdot \varepsilon_i^{\mathrm{alloc}}\)（分配到的隐私损失 × 真估值） |
| 客户端数 \(N\) | 默认 10 |
| 总预算 \(B\) | 默认 50 |
| 仿真 profile 数 | 默认 1000 / seed |
| 随机种子 | 默认 5 个：42,43,44,45,46 |

> 当前管线为**合成 bid**（与隐私论文一致）。若要做 MNIST/CIFAR/Shakespeare 子表，需替换 `build_privacy_paper_batch` 的数据源后复跑同一脚本。

## 2. 基线与 Ours

| 方法 | 类型 | 说明 |
|------|------|------|
| PAC / VCG / CSRA | 解析机制 | 网格搜索单代理最优偏离，估计 ex-post regret |
| RegretNet | 神经拍卖 | 需与 \(N,n_{\mathrm{items}}\) 一致的 checkpoint |
| DM-RegretNet | 神经 | `train_dm_regretnet_privacy.py` → `dm_regretnet_privacy_*_checkpoint.pt`；与 RegretNet 同架构、独立种子 |
| MFG-Pricing | 解析基线 | 均值场式定价：`pay_i=B·ε_i/Σε_j`，`pl_i=ε_i`；(v,ε) 网格上单代理偏离 |
| **Ours** | MFG-RegretNet | `train_mfg_regretnet.py` 训练得到 `.pt` |

**训练 Ours（若尚无权重）：**

```bash
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10 --n-items 1
# 产出示例：result/mfg_regretnet_privacy_200_checkpoint.pt
```

**训练 RegretNet 基线（与 MFG 同分布；`run_rq1_complete.sh` 在缺权重时可自动跑）：**

```bash
python train_regretnet_privacy.py --num-epochs 10 --num-examples 32768
# 产出：result/regretnet_privacy_<epoch>_checkpoint.pt
python train_dm_regretnet_privacy.py --num-epochs 10 --num-examples 32768
# 产出：result/dm_regretnet_privacy_<epoch>_checkpoint.pt
```

## 3. 指标（与脚本对应）

| 指标 | 含义 |
|------|------|
| \(\bar{\mathrm{rgt}}\) | 平均归一化事后遗憾（神经：PGA 找偏离；解析：网格） |
| \(\mathcal{V}_{\mathrm{IR}}\) (%) | \(u_i=p_i-v_i\varepsilon_i^{\mathrm{alloc}}<0\) 的 agent-slot 占比 |
| Truthful (%) | 归一化遗憾低于阈值 ≈ 诚实最优 |
| Bid CV | 神经：偏离后报价 \(v\) 的跨代理变异系数；解析：支付 CV |
| 显著性 | 论文表：Ours vs 各法 paired t-test（per-seed 平均遗憾） |

## 4. 一键脚本执行顺序

1. **环境**：Python3 + PyTorch + **cvxpy** + **scipy** + **matplotlib**
2. **阶段 A** — `rq1_incentive_compatibility.py`：机制对比表、遗憾柱图、Welch t-test（RegretNet vs MFG）、**PGA 收敛曲线**（需至少一个神经 ckpt）
3. **阶段 B** — `rq1_paper_table_figures.py`：论文主表（7 方法 × 4 列）、图 A/B、paired t-test、脚注
4. **可选 图 C / 图 D**（`RQ1_FIG_CD=1` 默认）— `rq1_figure_c_training_rounds.py`：横轴为 **checkpoint 训练 epoch**（可对应「每轮联邦前更新机制」的时间轴 proxy），纵轴为 \(\bar{\mathrm{rgt}}\) 与 IR%；解析基线为水平参考线。**图 D** — `rq1_figure_d_regret_distribution.py`：每方法「客户端×profile」归一化遗憾的**箱线图**（可选 `--violin`），看长尾是否更短。

## 5. 产出文件（默认目录 `run/privacy_paper/rq1/`）

| 文件 | 内容 |
|------|------|
| `table_rq1.csv` / `.md` | 阶段 A 表 |
| `rq1_statistics.json` | 阶段 A 统计 + t-test |
| `figure_rq1_regret_bar.png` | 阶段 A 遗憾柱图 |
| `figure_rq1_regret_vs_pga_rounds.png` | 遗憾随 PGA 步数 |
| `rq1_convergence_curve.json` | 收敛曲线数据 |
| `table_rq1_paper.md` / `.csv` | 论文主表 |
| `rq1_paper.json` | 论文表 JSON + paired t-test |
| `figure_rq1_paper_regret.png` / `_ir.png` | 论文图 A/B |
| `figure_rq1_paper_regret_vs_epoch.png` | 图 C：遗憾+IR% vs 训练 epoch |
| `rq1_figure_c.json` | 图 C 数据 |
| `figure_rq1_paper_regret_distribution.png` | 图 D：遗憾分布（箱线） |
| `rq1_figure_d.json` | 图 D 分位数摘要 |
| `RQ1_last_run.md` | **本次运行的参数与时间戳记录** |

## 6. 一键命令

```bash
./scripts/run_rq1_complete.sh
```

可选环境变量见 `scripts/run_rq1_complete.sh` 头部注释。

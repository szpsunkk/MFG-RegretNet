# FL-Market

The code, data, and models for our paper accepted by IEEE BigData 2022: "FL-Market: Trading Private Models in Federated Learning".

- IEEE Explore: https://ieeexplore.ieee.org/document/10020232  
- Arxiv: https://arxiv.org/abs/2106.04384  

---

## Privacy Paper (MFG-RegretNet) 实验流程

本仓库包含与 **Privacy as Commodity** 论文（main_m.pdf）对应的完整实验管线（RQ1–RQ6 + 消融研究），用于复现论文所有评估与表格/图。

### 📚 快速开始

**一键运行所有实验（RQ1-RQ6）**：
```bash
# 完整运行（需要预先训练模型）
bash run_all_rq.sh

# 快速测试模式（30-60 分钟）
QUICK=1 bash run_all_rq.sh
```

**详细文档**：
- 🚀 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考卡（所有命令一览）
- 📖 **[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)** - 完整实验运行指南（含环境配置、训练、FAQ）
- 📝 **[实验思路.md](实验思路.md)** - 实验设计与理论依据
- 🔬 **[docs/EXPERIMENT_DESIGN_STEPBYSTEP.md](docs/EXPERIMENT_DESIGN_STEPBYSTEP.md)** - 逐步实验设计

**按《实验思路.md》跑全量实验**：使用 **`run_all_experiments.py`** 一键跑 Phase 4 → Phase 5，并可选跑 RQ4（FL 收敛）、RQ5（隐私-效用）、RQ6（鲁棒性）、消融。配置见 **`config.yaml`**；RQ4–RQ6 与消融脚本在 **`exp_rq/`** 下（见实验思路 3.4–3.7）。

### 环境与依赖

- **Python 3.6+**，建议使用虚拟环境。
- 安装项目依赖（含 PyTorch、numpy 等）；**Phase 2 / Phase 4 / Phase 5** 需要 **cvxpy**（与 `experiments` / `aggregation` 一致）。
- 可选：**torchvision**（Phase 1 中 MNIST/CIFAR 相关检查）、**matplotlib**（Phase 5 画图）。

```bash
pip install torch numpy  # 其他依赖按项目原有方式安装
pip install cvxpy       # Phase 2/4/5 必需
pip install matplotlib  # Phase 5 画图
```

---

### 1. Phase 1：数据与基础管线（可选验证）

验证隐私论文的 bid 生成、成本定义与 FL 数据接口是否正常：

```bash
# 快速检查（无 torchvision 时部分项会跳过）
python run_phase1_full_check.py
```

---

### 2. Phase 2：基线 PAC / VCG（可选验证）

验证 PAC、VCG 的 plosses/payments 形状与预算可行性，并可选运行 auction 集成测试（需 cvxpy）：

```bash
python run_phase2_verify.py
```

---

### 3. Phase 3：MFG-RegretNet 训练

训练 MFG-RegretNet（论文 Algorithm 2，增广拉格朗日）。得到 checkpoint 后可用于 Phase 4 的 RQ1/RQ2/RQ3。

```bash
# 快速试跑（2 个 epoch，小数据）
python train_mfg_regretnet.py --num-epochs 2 --num-examples 1024

# 论文规模（200 轮，约 10 万样本；可按需调整 --n-agents）
python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10 --n-items 1
```

- 训练日志与 checkpoint 会写入 `run/` 下（具体路径见脚本或 tensorboard 输出）。
- 若需从断点续训：`python train_mfg_regretnet.py --resume path/to/checkpoint.pt ...`

**可选**：先验证 MFG-RegretNet 前向与预算可行性（需 cvxpy）：

```bash
python run_phase3_verify.py
```

---

### 4. Phase 4：系统评估与 RQ

在隐私论文设定下（v~U[0,1]、ε~U[0.1,5]、c=v·ε、固定预算 B）评估 RQ1（regret/IR）、RQ2（N vs 时间）、RQ3（收益与 BF）。默认会跑 RQ1、RQ2、RQ3；可用 `--skip-rq1` / `--skip-rq2` / `--skip-rq3` 跳过对应项。

**仅 PAC / VCG（无需 checkpoint）：**

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 --seeds 42,43,44 --n-list 10,50,100
```

**含 RegretNet、MFG-RegretNet（需先完成 Phase 3 得到 checkpoint）：**

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 \
  --regretnet-ckpt path/to/regretnet.pt \
  --mfg-regretnet-ckpt path/to/mfg_regretnet.pt
```

**让 RQ2 图中 MFG-RegretNet 有多个 N 的点**：神经模型架构与 N 绑定，若只传一个 checkpoint 则 RQ2 只在对应 N 有一个点。若对多个 N 分别训练了 MFG-RegretNet，可用 `--mfg-regretnet-ckpt-by-n` 传入多个 checkpoint，例如：

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 --n-list 10,50,100 \
  --mfg-regretnet-ckpt-by-n "10:result/mfg_10.pt,50:result/mfg_50.pt,100:result/mfg_100.pt"
```

- 结果写入 `--out-dir`（默认 `run/privacy_paper/`）下的 **phase4_summary.json**，供 Phase 5 使用。

---

### 5. Phase 5：表格与图表

根据 Phase 4 的 **phase4_summary.json** 生成 RQ1/RQ2/RQ3 的表格（CSV + Markdown）与 RQ2 的 N–时间图。

```bash
# 默认从 run/privacy_paper/phase4_summary.json 读入，输出到 run/privacy_paper/tables/ 与 figures/
python run_phase5_tables_figures.py
```

**常用选项：**

```bash
# 指定输入与输出目录
python run_phase5_tables_figures.py --input run/privacy_paper/phase4_summary.json --out-dir run/privacy_paper

# 只生成表格，不生成图
python run_phase5_tables_figures.py --no-figures

# 若另有 RQ4 准确率数据（rounds + 各方法每轮准确率），可生成准确率曲线图
python run_phase5_tables_figures.py --accuracy-json path/to/accuracy.json
```

**输出说明：**

| 输出 | 说明 |
|------|------|
| `run/privacy_paper/tables/table_rq1.csv`, `table_rq1.md` | RQ1：方法 × (mean_regret, mean_ir_violation) |
| `run/privacy_paper/tables/table_rq2.*` | RQ2：N vs 每轮时间（表格） |
| `run/privacy_paper/tables/table_rq3.*` | RQ3：方法 × (mean_revenue, bf_rate, revenue_efficiency, mean_social_welfare) |
| `run/privacy_paper/figures/figure_rq2_time_vs_n.png` | RQ2：N vs 时间曲线图 |
| **RQ2 论文三张图（log-log / 内存通信 / 堆叠延迟）** | **`bash scripts/run_rq2_paper.sh`**（`exp_rq/RQ2_PROCESS.md`） |

---

### 推荐完整流程（简要）

1. **安装依赖**（含 `cvxpy`、可选 `matplotlib`）：
   ```bash
   pip install torch torchvision numpy cvxpy matplotlib
   ```

2. **训练模型**（约 2-4 小时）：
   ```bash
   # MFG-RegretNet（必需）
   python train_mfg_regretnet.py --num-epochs 200 --num-examples 102400 --n-agents 10
   
   # RegretNet 和 DM-RegretNet（RQ1 需要，可选自动训练）
   python train_regretnet_privacy.py --num-epochs 200 --n-agents 10
   python train_dm_regretnet_privacy.py --num-epochs 200 --n-agents 10
   ```

3. **运行所有实验**（约 6-10 小时）：
   ```bash
   bash run_all_rq.sh
   ```

4. **查看结果**：
   ```bash
   ls run/privacy_paper/*/figures/*.png    # 所有图表
   ls run/privacy_paper/*/table_*.md       # 所有表格
   ```

**快速测试**（30-60 分钟）：
```bash
QUICK=1 bash run_all_rq.sh
```

更细的实验设定、指标定义与复检记录见 **[RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md)** 和 **docs/EXPERIMENT_PLAN_PRIVACY_PAPER.md**。

---

### 按《实验思路.md》一键跑全量实验

实验思路见项目根目录 **`实验思路.md`**。已实现对应脚本与配置：

| 内容 | 脚本/配置 |
|------|-----------|
| **RQ1（实验思路 3.1，推荐）** | `exp_rq/rq1_incentive_compatibility.py`：多 seed、std、t 检验、柱状图 |
| RQ1–RQ3 + 表格/图 | `run_phase4_eval.py --skip-rq2 --skip-rq3` → `run_phase5_tables_figures.py` |
| **RQ4（MNIST/CIFAR 图 A–D）** | **`bash scripts/run_rq4_paper.sh`**（见 `exp_rq/RQ4_PROCESS.md`）；旧 NSL-KDD 曲线仍可用 `exp_rq/rq4_fl_convergence.py --sample` |
| **FL-market.pdf 对标图**（预算–误差界、无效梯度率、RQ4 精度曲线、参数 M） | **`bash scripts/run_fl_market_style.sh`**；说明见 **`docs/FL_MARKET_STYLE_RQ.md`** |
| **RQ5（隐私–精度 Pareto + 负担分布）** | **`bash scripts/run_rq5_paper.sh`**（`exp_rq/RQ5_PROCESS.md`）；旧机制-only 脚本仍见 `rq5_privacy_utility.py` |
| **RQ3（收益+社会福利，图1/2/3）** | `./run_rq3.sh`；**W̄ 定义**见 `exp_rq/RQ3_PROCESS.md`（T 轮时间平均后再跨种子 mean±std） |
| RQ6 鲁棒性 | `exp_rq/rq6_robustness.py` |
| 消融（MFG vs RegretNet） | `exp_rq/ablation_study.py` |
| 配置 | `config.yaml` |
| 一键入口 | `run_all_experiments.py` |

**一键运行（需先准备好 checkpoint）：**

```bash
python run_all_experiments.py --mfg-regretnet-ckpt result/mfg_regretnet_n10_200_checkpoint.pt
```

可选：`--run-rq4`（生成 RQ4 示例曲线）、`--run-rq5`、`--run-rq6`、`--run-ablation`，以及 `--log-scale`（RQ2 对数坐标图）。

**RQ1 一键（基线 + 神经 + 全部表与图，推荐在项目根目录执行）：**

```bash
cd /path/to/FL-Market
./run_rq1.sh
# 或：bash run_rq1.sh   /   bash scripts/run_rq1_complete.sh
```

一次完成：**基线** PAC / VCG / CSRA / **MFG-Pricing**；**神经** RegretNet、DM-RegretNet、MFG-RegretNet（Ours）；**表** `table_rq1*`、`table_rq1_paper.md`；**图** `figure_rq1_regret_bar.png`、`figure_rq1_paper_regret.png`、`figure_rq1_paper_ir.png`、`figure_rq1_regret_vs_pga_rounds.png`；**可选图 C/D**（默认开启，`RQ1_FIG_CD=0` 可关）：`figure_rq1_paper_regret_vs_epoch.png`（遗憾+IR% 随 checkpoint 训练 epoch）、`figure_rq1_paper_regret_distribution.png`（各方法遗憾分布箱线/长尾）。输出目录默认 **`run/privacy_paper/rq1/`**。

- 缺 RegretNet / DM 权重时默认**自动训练**（关掉：`REGRETNET_AUTO_TRAIN=0`、`DM_REGRETNET_AUTO_TRAIN=0`）；MFG 需已有 `result/mfg_regretnet_privacy_*_checkpoint.pt` 或设 `MFG_CKPT`。
- 过程说明：`exp_rq/RQ1_EXPERIMENT_PROCESS.md`；本次运行记录：`run/privacy_paper/rq1/RQ1_last_run.md`

**仅完成 RQ1（分步）：**

```bash
# 仅解析基线 PAC / VCG / CSRA（无 checkpoint）
python exp_rq/rq1_incentive_compatibility.py --num-profiles 1000 --seeds 42,43,44

# 含 RegretNet + MFG-RegretNet（需与 N=10 一致的 checkpoint）
python exp_rq/rq1_incentive_compatibility.py --num-profiles 1000 --seeds 42,43,44 \
  --regretnet-ckpt result/regretnet_n10.pt \
  --mfg-regretnet-ckpt result/mfg_regretnet_n10_200_checkpoint.pt

# 一键：MFG 默认识别 result/mfg_regretnet_privacy_*_checkpoint.pt（与 train_mfg_regretnet.py 一致）
# RegretNet 需: REGRETNET_CKPT=path/to.pt ./scripts/run_rq1_full.sh
./scripts/run_rq1_full.sh
```

输出目录：`run/privacy_paper/rq1/`（`table_rq1.csv`、`table_rq1.md`、`rq1_statistics.json`、`figure_rq1_regret_bar.png`；加 `--convergence-curve` 或跑脚本另有 `figure_rq1_regret_vs_pga_rounds.png`）。

**RQ1 论文主表 + 主图 A/B（7 方法：PAC/VCG/CSRA、RegretNet、DM-RegretNet、MFG-Pricing、Ours）：**

```bash
# MFG：不设则自动用 result/mfg_regretnet_privacy_* 里 epoch 最大的；勿写死 _200_ 若你只训了 10 轮
# RegretNet / DM：`run_rq1_complete.sh` 可自动训练（见 REGRETNET_AUTO_TRAIN、DM_REGRETNET_AUTO_TRAIN）
export REGRETNET_CKPT=path/to/regretnet.pt   # 可选
export DM_REGRETNET_CKPT=path/to/dm.pt       # 可选；或 python3 train_dm_regretnet_privacy.py
./scripts/run_rq1_paper.sh
# IR 纵轴对数：IR_LOG=1 ./scripts/run_rq1_paper.sh
```

产出：`table_rq1_paper.md`（$\bar{\mathrm{rgt}}$、IR%、Truthful%、Bid CV，mean±std，5 seeds + paired t-test）、`figure_rq1_paper_regret.png`、`figure_rq1_paper_ir.png`。子表按 MNIST/CIFAR 需另接数据源复制该脚本参数。

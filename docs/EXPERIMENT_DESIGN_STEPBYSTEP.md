# 实验设计：提出算法与所有基线对比（逐步执行）

本文档给出**一步一步**可执行的实验流程，确保 **MFG-RegretNet（提出算法）** 与 **所有基线** 在 RQ1–RQ4 上均有对比结果，并产出论文可用的表格与图。

---

## 一、方法列表（提出算法 + 所有基线）

| 类型 | 方法名 | 说明 | 是否需要训练/checkpoint |
|------|--------|------|--------------------------|
| **提出算法** | **MFG-RegretNet** | 输入含 b_MFG，预算投影，增广拉格朗日训练 | ✅ 需训练，得到 checkpoint |
| 基线 1 | **PAC** | 论文 Algorithm 1，解析 Nash 阈值机制 | ❌ 无需 checkpoint |
| 基线 2 | **VCG** | 采购 VCG，真实报价、BF | ❌ 无需 checkpoint |
| 基线 3 | **RegretNet** | 无 b_MFG 的深度拍卖网络 | ✅ 需训练（同设定下），得到 checkpoint |

**说明**：MFG-Pricing [26] 若难以复现，可仅在文中说明，实验表与图中以 PAC、VCG、RegretNet 为基线即可。

---

## 二、评估维度与产出（与论文 RQ 对应）

| RQ | 内容 | 指标 | 产出 |
|----|------|------|------|
| **RQ1** | 激励相容 | 平均 regret、IR 违反率 | 表：方法 × (regret, IR) |
| **RQ2** | 可扩展性 | N vs 每轮墙钟时间 | 表 + 图：N vs time |
| **RQ3** | 拍卖收益 | 平均收益、BF 率 | 表：方法 × (revenue, bf_rate) |
| **RQ4** | FL 精度（可选） | 最终测试准确率 | 表/图：方法 × accuracy |

每一张表/图中都应包含：**PAC、VCG、RegretNet、MFG-RegretNet** 四列（或四条曲线），即提出算法与所有基线的对比。

---

## 三、实验设置（统一）

- **隐私估值**：\(v_i \sim U[0,1]\)
- **隐私预算**：\(\epsilon_i \sim U[0.1, 5]\)
- **成本**：\(c(v_i,\epsilon_i) = v_i \cdot \epsilon_i\)
- **预算 B**：如 50（可再跑 B=10, 100 做敏感性）
- **N（智能体数）**：主实验 N=10；RQ2 需 N ∈ {10, 50, 100}（或 10, 50, 100, 500）
- **Seeds**：如 42, 43, 44（RQ1/RQ3 取平均）
- **Profile 数**：如 1000（可 2000 更稳）

---

## 四、逐步执行流程

### Step 0：环境与依赖

```bash
# 建议 conda 环境，Python 3.8+
pip install torch numpy cvxpy cvxpylayers  # 及其他项目依赖
# 若做 RQ2 多 N 的 MFG 曲线，需能跑 N=50, 100 的 aggregation（已用 psd_wrap 处理大 N）
```

确认可运行：

```bash
python run_phase4_eval.py --n-agents 10 --num-profiles 100 --skip-rq2 --skip-rq3
# 应得到 phase4_summary.json，且 rq1 中有 PAC、VCG 两行
```

---

### Step 1：仅基线 PAC、VCG（无 checkpoint）

目的：确认 RQ1/RQ2/RQ3 流程畅通，且 PAC/VCG 结果合理。

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 \
  --seeds 42,43,44 --n-list 10,50,100
```

- 输出：`run/privacy_paper/phase4_summary.json`
- 应包含：rq1（PAC, VCG）、rq2（PAC, VCG 在 N=10,50,100）、rq3（PAC, VCG）
- RegretNet、MFG-RegretNet 此时未加入，表中无这两行属正常

---

### Step 2：训练 MFG-RegretNet（提出算法）

主实验 N=10，建议 200 epoch；快速试跑可用 5–20 epoch。

```bash
# 主实验：N=10，200 epoch
python train_mfg_regretnet.py --n-agents 10 --n-items 1 --num-epochs 200 \
  --name mfg_regretnet_n10

# 快速试跑（仅验证流程）
python train_mfg_regretnet.py --n-agents 10 --n-items 1 --num-epochs 5 \
  --name mfg_regretnet_n10
```

- Checkpoint 路径：`result/mfg_regretnet_n10_200_checkpoint.pt`（或最后一轮，如 `..._5_checkpoint.pt`）

若要做 **RQ2 多 N 曲线**，需对 N=50、N=100 各训练一个：

```bash
python train_mfg_regretnet.py --n-agents 50 --n-items 1 --num-epochs 200 --name mfg_regretnet_n50
python train_mfg_regretnet.py --n-agents 100 --n-items 1 --num-epochs 200 --name mfg_regretnet_n100
```

得到：`result/mfg_regretnet_n50_200_checkpoint.pt`、`result/mfg_regretnet_n100_200_checkpoint.pt`。

---

### Step 3：训练 RegretNet（基线 3，与提出算法同设定）

RegretNet 需在**同一设定**（v, ε, B, N, n_items=1）下训练，否则对比不公平。当前仓库中 `train.py` 使用 NSL-KDD 等数据；若暂无“隐私论文数据 + RegretNet”训练脚本，可二选一：

- **方案 A**：在现有 `train_mfg_regretnet.py` 上增加 `--model RegretNet` 分支，用同一 `generate_privacy_paper_bids` 数据训练 RegretNet，保存 checkpoint（如 `result/regretnet_n10_200_checkpoint.pt`）。
- **方案 B**：暂时不跑 RegretNet，表中只对比 **PAC、VCG、MFG-RegretNet**；在论文中注明“RegretNet 需同数据训练，留作补充实验”。

若采用方案 A，训练命令形如（需脚本支持）：

```bash
# 示例（需项目内提供 train_regretnet_privacy.py 或 train_mfg_regretnet.py --model RegretNet）
python train_mfg_regretnet.py --model RegretNet --n-agents 10 --n-items 1 --num-epochs 200 --name regretnet_n10
```

得到：`result/regretnet_n10_200_checkpoint.pt`。

---

### Step 4：运行完整 Phase 4（提出算法 + 所有基线）

在 Step 1–3 完成后，一次性跑 RQ1/RQ2/RQ3，**包含 PAC、VCG、RegretNet、MFG-RegretNet**。

**4.1 仅 N=10（单点 MFG-RegretNet）**

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 \
  --seeds 42,43,44 --n-list 10,50,100 \
  --mfg-regretnet-ckpt result/mfg_regretnet_n10_200_checkpoint.pt \
  --regretnet-ckpt result/regretnet_n10_200_checkpoint.pt
```

若暂无 RegretNet checkpoint，去掉 `--regretnet-ckpt` 即可；表中会缺 RegretNet 一行。

**4.2 RQ2 需要 MFG 在 N=10,50,100 各有一点（三条曲线）**

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 \
  --seeds 42,43,44 --n-list 10,50,100 \
  --mfg-regretnet-ckpt-by-n "10:result/mfg_regretnet_n10_200_checkpoint.pt,50:result/mfg_regretnet_n50_200_checkpoint.pt,100:result/mfg_regretnet_n100_200_checkpoint.pt" \
  --regretnet-ckpt result/regretnet_n10_200_checkpoint.pt
```

- 输出仍为 `run/privacy_paper/phase4_summary.json`，其中：
  - **rq1**：PAC, VCG, RegretNet(若有), MFG-RegretNet
  - **rq2**：PAC, VCG 在 10,50,100；RegretNet 仅在 N=10（单点）；MFG-RegretNet 在 10,50,100（三点）
  - **rq3**：PAC, VCG, RegretNet(若有), MFG-RegretNet

---

### Step 5：生成表格与图（Phase 5）

用 Phase 4 的同一份 `phase4_summary.json` 生成论文用表与图。

```bash
python run_phase5_tables_figures.py --input run/privacy_paper/phase4_summary.json --out-dir run/privacy_paper
```

产出：

| 文件 | 内容 | 方法覆盖 |
|------|------|----------|
| `tables/table_rq1.csv`, `table_rq1.md` | RQ1：regret、IR | PAC, VCG, RegretNet(若有), MFG-RegretNet |
| `tables/table_rq2.csv`, `table_rq2.md` | RQ2：N vs 时间 | 同上 |
| `tables/table_rq3.csv`, `table_rq3.md` | RQ3：收益、BF 率 | 同上 |
| `figures/figure_rq2_time_vs_n.png` | RQ2 曲线图 | 同上，MFG 若提供多 N checkpoint 则为曲线 |

此时应得到**提出算法与所有已跑基线的对比**；若缺 RegretNet，表中仅无 RegretNet 一行，其余三方法齐全。

---

### Step 6（可选）：RQ4 FL 精度

若论文含 RQ4（隐私机制对下游 FL 精度的影响），需：

1. 在同一数据集（如 MNIST/CIFAR-10 非 IID）上，对每种机制（PAC、VCG、RegretNet、MFG-RegretNet）跑 FL，记录每轮测试准确率。
2. 将结果整理为 JSON：`{"rounds": [1,10,...], "methods": {"PAC": [...], "VCG": [...], "RegretNet": [...], "MFG-RegretNet": [...]}}`。
3. 运行：

```bash
python run_phase5_tables_figures.py --accuracy-json path/to/accuracy.json
```

得到 `figures/figure_rq4_accuracy_vs_round.png` 及可选表格，完成 RQ4 的**提出算法与所有基线**对比。

---

## 五、检查清单：确保“提出算法 vs 所有基线”

- [ ] **RQ1 表**：至少含 PAC、VCG、MFG-RegretNet；若有 RegretNet checkpoint，则四行齐全。
- [ ] **RQ2 表/图**：PAC、VCG、MFG-RegretNet（单点或多 N）；若有 RegretNet，则出现其单点。
- [ ] **RQ3 表**：同上，四方法（或三方法）均有 mean_revenue、bf_rate。
- [ ] **RQ4（若做）**：每条曲线/每行对应一方法，含 MFG-RegretNet 与全部基线。
- [ ] 论文中所有“与基线对比”的表格/图，均来自上述 `phase4_summary.json` 与 Phase 5 输出，无遗漏基线。

---

## 六、命令速查（复制即用）

假设已训练好：

- `result/mfg_regretnet_n10_200_checkpoint.pt`
- `result/mfg_regretnet_n50_200_checkpoint.pt`
- `result/mfg_regretnet_n100_200_checkpoint.pt`
- `result/regretnet_n10_200_checkpoint.pt`（可选）

**一次性跑齐 RQ1+RQ2+RQ3（提出算法 + 所有基线）：**

```bash
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 \
  --seeds 42,43,44 --n-list 10,50,100 \
  --mfg-regretnet-ckpt-by-n "10:result/mfg_regretnet_n10_200_checkpoint.pt,50:result/mfg_regretnet_n50_200_checkpoint.pt,100:result/mfg_regretnet_n100_200_checkpoint.pt" \
  --regretnet-ckpt result/regretnet_n10_200_checkpoint.pt
```

**生成表与图：**

```bash
python run_phase5_tables_figures.py
```

完成后，`run/privacy_paper/tables/` 与 `run/privacy_paper/figures/` 中即为**提出算法与所有基线的对比结果**，可直接用于论文实验部分。

# RQ5 里如何让「Ours」更好看、故事更站得住

五张图若都不理想，通常**不是画图 bug**，而是：**FL 下游只认「有效噪声预算」**，而 MFG/拍卖可能在 IR/预算约束下把部分客户端压到 **ε≈0**（Laplace 噪声 \(\propto 1/\varepsilon\) 爆炸），或 **ε 分布极不均匀**，导致全局模型比 CSRA/RegretNet/均匀 DP 更差。

下面分三层：**机制训练** → **FL 实验设置** → **论文叙事**。

---

## 1. 机制侧：重训 MFG（优先）

RQ5 用的是 **MFG-RegretNet 的分配**，与 RQ1/RQ3 同一套权重。若 RQ3 为抬收益/福利调过 λ，未必对 **FL 精度**友好。

```bash
# 预设：略增 participant welfare，减轻「有人 ε 过小」
bash scripts/train_mfg_for_rq5.sh

# 网格（示例）
for W in 0.004 0.008 0.012; do
  LAMBDA_W=$W NAME=mfg_rq5_w${W} bash scripts/train_mfg_for_rq5.sh
done
```

对每个 checkpoint 跑一小段 RQ5（`QUICK=1` 或单 B），选 **Pareto 上最靠右上或同样 ε̄ 下精度最高** 的权重，再跑正式 `run_rq5_paper.sh`：

```bash
export MFG_CKPT=result/mfg_regretnet_privacy_rq5tune_200_checkpoint.pt   # 按实际改
bash scripts/run_rq5_paper.sh
```

**直觉**：`lambda_participant_welfare` 略大 → 少出现「为保 IR/预算把某些人压到几乎不参与有效训练」的极端分配（需在 **RQ1 上仍验 IR**）。

---

## 2. FL 与预算扫描（实验设计）

| 旋钮 | 建议 |
|------|------|
| **轮数** | `ROUNDS=80~120`；60 轮 MNIST/CIFAR 可能未收敛，大家挤在一起 |
| **Dirichlet α** | 先试 `ALPHA=0.5`（比 0.1 容易拉开方法差异）；非 IID 故事可另开附录 |
| **B 档位** | 覆盖「低预算难训」与「高预算饱和」：`0.3,0.6,1.0,1.4` 或更密 |
| **学习率** | `FL_LR=0.015`（MNIST）或略调；`bash scripts/run_rq5_paper.sh` 前 `export FL_LR=...` |
| **均匀 DP 基线** | `--uniform-eps` 与 Ours 的 **时间平均 ε̄** 对齐时再跑一次，避免基线天然占「高噪声预算」便宜 |

```bash
ROUNDS=100 FL_LR=0.012 BUDGET_RATES="0.35,0.55,0.75,1.0,1.25" bash scripts/run_rq5_paper.sh
```

---

## 3. 五张图分别能讲什么（占不占优都要会写）

| 图 | 若 Ours 不占优时的用法 |
|----|------------------------|
| **A Pareto** | 强调 **同 B 下轨迹**；若 Ours **更左下** = 更省平均 ε、精度换隐私，配合 RQ1「为 IR 收紧 ε」；若略逊，报 **ε 效率** = acc/ε̄（可正文表） |
| **B 双柱** | 明确写：**不能只看精度柱，要看 ε 柱**——谁更「敢花隐私」；Ours 若 ε 更低、精度略低 = **隐私更省** |
| **C 箱线** | **公平性**：Ours **Gini 更低** → 少数据客户端不被榨干 ε；这是独立贡献 |
| **D B 扫描** | 看 **单调性**：随 B 增大，Ours 的 acc / ε̄ 是否 **可控、可预期**（工程可部署） |
| **E 更新范数** | 辅助：**更平滑** = 训练更稳；不必与 A 同时赢 |

---

## 4. 预期管理（审稿视角）

- **RegretNet 不保 IR**：可在讨论写 **精度–激励权衡**（与 RQ1 一致）。  
- **Pareto 全面碾压**很难：FL + DP 噪声下，**公平 IR + 预算** 会自然收缩部分 ε。  
- 最小可接受叙事：**在可比 ε̄ 或可比公平性（Gini）下，Ours 更合理**；或 **多给一点 B 后 Ours 上升更快**（图 D）。

---

## 5. 检查清单

1. [ ] RQ1 上对 **新 MFG 权重** 再跑一遍 IR/regret。  
2. [ ] RQ5 用 **显式 `MFG_CKPT`**，避免解析到旧 checkpoint。  
3. [ ] 至少 **2 个 seed** + **足够轮数** 再判断 Pareto。  
4. [ ] 正文 **图 A caption** 写清 ε̄ 定义（参与者时间平均，见 `RQ5_PROCESS.md`）。

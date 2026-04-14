# RQ3 实验过程与代码说明

## 研究问题

在固定预算 **B** 下，各机制的总支付（收益）**R**、预算可行率（BF）、以及**参与者社会福利**的对比；并与 **Ours（MFG-RegretNet）** 对照。

## 社会福利定义（与论文一致）

在每一轮 FL 拍卖轮次 **t**（代码中用独立采样的一个 **profile** 表示该轮市场状态）中，客户端 **i** 获得支付 **p_i^(t)**，隐私产出 **ε_i^{(t),out}**，真实估值 **v_i**，成本 **c(v_i,ε)=v_i·ε**。

- **单轮社会福利**  
  \[
  \mathcal{W}^{(t)} = \sum_{i=1}^{N}\left( p_i^{(t)} - v_i\,\varepsilon_i^{(t),\mathrm{out}} \right).
  \]

- **每个随机种子 s**（不同数据划分 / 随机性）：在 **T** 轮上取**时间平均**  
  \[
  \bar{\mathcal{W}}_s = \frac{1}{T}\sum_{t=1}^{T} \mathcal{W}^{(t)}.
  \]  
  实现中 **T = num_profiles**（每种子下生成 T 个 profile，依次视作 t=1,…,T）。

- **报告指标**  
  \[
  \mathbb{E}_s[\bar{\mathcal{W}}_s] \pm \mathrm{std}_s(\bar{\mathcal{W}}_s)
  \]  
  即对多种子下的 **W̄_s** 再取 **mean ± std**（样本标准差，ddof=1）。

同理，**总收益**对每轮定义 **R^(t)=∑_i p_i^(t)**，再 **R̄_s=(1/T)∑_t R^(t)**，报告 **mean_s(R̄_s) ± std_s**；**η_rev,s = R̄_s / B**，再对种子汇总 **mean ± std**。

**BF 率**：每个种子上，满足 **R^(t)≤B** 的轮次 t 所占比例，再对种子取平均。

## 代码入口

| 脚本 | 作用 |
|------|------|
| `run_phase4_eval.py` | Phase4 中 RQ3，写 `phase4_summary.json` 的 `rq3` 列表（dict，含 std） |
| `exp_rq/rq3_paper_complete.py` | 论文图 1/2/3 + `table_rq3_paper.md` |
| `run_rq3.sh` / `scripts/run_rq3_complete.sh` | 一键跑 `rq3_paper_complete` |

核心实现：**`run_phase4_eval.rq3_revenue_privacy_paper`**（按上式逐种子算 **W̄_s、R̄_s**，再汇总）。

## 图说明

1. **图1**：η_rev 与 **W̄** 的柱图，**误差棒 = 跨种子 std**。  
2. **图2**：神经机制在不同 **checkpoint epoch** 下，**单个种子**的 **R̄_s、W̄_s**（仍为 T 轮时间平均）；点线表示解析基线在同一度量下的均值。  
3. **图3**：不同预算 **B** 下，**每种方法**在多种子上的 **W̄_s** 的 **mean ± std**（errorbar）。

## 参数建议

- **seeds**：≥3，否则 std=0。  
- **num_profiles（T）**：越大，**W̄_s** 的蒙特卡洛方差越小，但算力越大。

---

## 为何图上 Ours 可能弱于 VCG / CSRA？

1. **训练目标不同**：MFG-RegretNet 主要压 **遗憾 + IR**（激励相容），并含与隐私损失相关的项；**并未**像拍卖理论中的最优机制那样最大化 **买方收益** 或 **总支付**。VCG 等在相同预算下往往 **更“敢付钱”**，故 **η_rev** 更高很常见。  
2. **社会福利 W**：\(W=\sum_i(p_i-v_i\varepsilon_i^{\mathrm{out}})\)。支付高、隐私成本低则 W 高；若 Ours 为保 IR 而 **保守支付**，W 可能低于高支付基线——**不一定代表机制差**，而是 **收益—激励—隐私** 的折中。  
3. **图2**：若 checkpoint 来自 **少 epoch** 训练，曲线未收敛也会落后；应用 **足够大 epoch** 的权重。  

**论文中可写**：Ours 优先 **策略证明性与 IR**，RQ3 上 **收益略低** 可与 **RQ1 遗憾/IR** 对照，说明 **不同目标下的权衡**。

---

## 希望提升 RQ3 曲线时的训练建议（合规、可复现）

在 **不删 regret/IR 主损失** 的前提下，对 **MFG-RegretNet** 增加两项 **可选辅助项**（`regretnet.py` + `train_mfg_regretnet.py`）：

| 参数 | 含义 |
|------|------|
| `--lambda-revenue-util` | 鼓励 **总支付接近预算**（提高 η_rev），建议从 **0.05** 起试 |
| `--lambda-participant-welfare` | 鼓励 ** truthful 效用和**（与 W 同向），建议 **1e-3** 量级起试 |

示例（在原有 epoch/数据量基础上微调后重训，再跑 `./run_rq3.sh`）：

```bash
python3 train_mfg_regretnet.py \
  --num-epochs 200 --num-examples 102400 --n-agents 10 --n-items 1 \
  --lambda-revenue-util 0.06 \
  --lambda-participant-welfare 0.002 \
  --name mfg_regretnet_privacy_rq3tune
```

若 IR 变差，略 **增大 `--rho-ir`** 或 **减小** 上述 λ。最终以 **RQ1 遗憾/IR** 与 **RQ3** 联合选点。

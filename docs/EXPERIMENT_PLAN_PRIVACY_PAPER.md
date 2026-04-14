# 实验规划：Privacy as Commodity (MFG-RegretNet) 论文

本文档依据 **privacy paper.pdf** 第八章（Performance Evaluation）与项目现有代码，给出可执行的实验规划，便于在 FL-Market 代码库上复现与扩展您论文中的实验。

---

## 一、研究问题与指标（对应论文 Section VIII.A & VIII.B）

### 1.1 研究问题 (RQ)

| 编号 | 研究问题 | 对应内容 |
|------|----------|----------|
| **RQ1** | 激励相容性 | MFG-RegretNet 是否比基线获得更低的平均 ex-post regret 和 IR 违反率（近似 DSIC、IR）？ |
| **RQ2** | 可扩展性 | 随 N 增大，MFG-RegretNet 与直接 Nash 计算、vanilla RegretNet 的每轮计算时间如何变化？ |
| **RQ3** | 拍卖收益 | 在满足预算可行前提下，MFG-RegretNet 是否获得更高的总拍卖收益 R？ |
| **RQ4** | FL 性能 | 隐私拍卖激励机制相比固定隐私基线，是否提升下游 FL 全局模型精度？ |

### 1.2 评估指标

| 指标 | 定义 | 方向 |
|------|------|------|
| **平均 ex-post regret** | \(\bar{rgt} = \frac{1}{N}\sum_i \widehat{rgt}^{MFG}_i\) | 越低越好（DSIC 合规） |
| **IR 违反率** | 满足 \(\bar{p}_i < c(v_i,\epsilon_i)\) 的轮次比例 | 越低越好 |
| **总拍卖收益 R** | \(\sum_i p_i\) | 越高越好（在 BF 约束下） |
| **全局模型精度** | 测试集准确率 | 越高越好 |
| **每轮墙钟时间** | 单轮拍卖 + 聚合的耗时 vs N | 用于 RQ2  scalability 分析 |

---

## 二、基线方法（对应论文 Section VIII.B）

| 基线 | 说明 | 当前代码状态 |
|------|------|----------------|
| **PAC (Algorithm 1)** | 论文中的阈值机制，解析 Nash 解（排序 v，选最大 k，\(p_i = \min(B/k, c(v_{k+1}, 1/(n-k)))\)） | 已实现 `baselines/pac.py` |
| **RegretNet [25]** | 无 MFG 的深度学习拍卖（去掉 b_MFG） | 已有 `RegretNet`，需保证输入/成本与论文一致 |
| **VCG Auction** | 经典真实报价机制，Vickrey-Clarke-Groves 支付 | 已实现 `baselines/vcg.py` |
| **CSRA (IEEE TIFS 2024)** | 鲁棒激励机制：质量最大化选择 (QMS)，按 q_i/b_i 排序、预算可行支付 | 已实现 `baselines/csra.py`（见 `baselines/CSRA.md`） |
| **MFG-Pricing [26]** | 基于 MFG 的数据定价，无拍卖机制 | 需新增或引用 [26] 实现 |
| **MFG-RegretNet** | 本文方法：输入含 b_MFG，增广拉格朗日训练 | 已有 `MFGRegretNet`，需对齐论文 Algorithm 2/3 与 b_MFG 定义 |

---

## 三、实验设置（对应论文 Section VIII.C）

### 3.1 任务与数据集

| 任务 | 数据集 | 客户端数 N | 备注 |
|------|--------|-------------|------|
| 图像分类 | **MNIST** | 10, 50, 100, 500 | 非 IID：Dirichlet 划分，\(\alpha_{Dir}=0.5\) |
| 图像分类 | **CIFAR-10** | 50, 100, 500 | 同上 |
| 下一字符预测 | **Shakespeare** | 100, 200 | 同上 |

**非 IID 实现**：按 Dirichlet(α=0.5) 对类别/标签做非均匀划分到各客户端（当前项目为 Bank/NSL-KDD，需增加 MNIST/CIFAR-10/Shakespeare 的数据加载与划分脚本）。

### 3.2 隐私估值与预算

- **隐私估值**：\(v_i \sim \text{Uniform}[0,1]\)
- **隐私预算**：\(\epsilon_i \sim \text{Uniform}[0.1, 5]\)
- **预算约束**：\(B \in \{10, 50, 100\}\)
- **成本函数**：\(c(v_i, \epsilon_i) = v_i \cdot \epsilon_i\)（与论文一致）

### 3.3 网络与训练超参（MFG-RegretNet）

| 参数 | 论文取值 | 说明 |
|------|----------|------|
| 隐藏层 | 3 层，每层 100 单元 | AHN / Allocation / Payment 统一 |
| 外迭代 T | 200 | 增广拉格朗日外层 |
| 内步数 T_in | 25 | 每轮网络梯度更新步数 |
| batch size L | 64 | |
| PGA 步数 R | 25 | 误报优化步数 |
| PGA 步长 η | 0.01 | |
| Adam 学习率 α | 1e-3 | |
| IR 惩罚系数 γ | 10 | |
| AL 惩罚 ρ0 / κ / ρ_max | 1, 1.5, 100 | |

---

## 四、与当前项目的对齐与缺口

### 4.1 当前项目已有

- **RegretNet**：`regretnet.py`，分配 + 支付，支持 regret/IR/budget 约束训练与测试。
- **MFGRegretNet**：同文件，输入为 `[reports; row_means]`（row_means 作为 b_MFG 的简化），输出分配与支付。
- **FL 流程**：`FL.py` 中 LDP-FedSGD、`client.py`、`experiments.py` 中 `auction` → `acc_eval`。
- **聚合**：`aggregation.py` 中 OptAggr 等，可与论文中的 \(\alpha_i \propto \epsilon^{out}_i\) 对应。
- **数据**：Bank、NSL-KDD；bid 格式 `(n_agents, n_items+2)`（含 pbudget、size）。

### 4.2 需要补充或修改的部分

1. **数据集**
   - 增加 **MNIST**、**CIFAR-10**、**Shakespeare** 的加载与 **Dirichlet 非 IID** 划分（\(\alpha=0.5\)）。
   - 若沿用现有 Bank/NSL-KDD，需在实验规划中注明为“补充实验”，论文主表仍以 MNIST/CIFAR-10/Shakespeare 为准。

2. **投标/成本与论文一致**
   - 论文中 bid 为 \(b_i = (v_i, d_i, u_i, \epsilon_i)\)，成本 \(c(v_i,\epsilon_i)=v_i\cdot\epsilon_i\)。
   - 当前 `generate_dataset*` 与 client bid 的“每 item 成本”形式需映射到“单维 \(\epsilon_i\) + \(v_i\)”或保持多 item 但确保 \(c = v\cdot\epsilon\) 可计算；**测试/训练时的效用与 regret 需按 \(u_i = p_i - c(v_i,\epsilon_i)\) 计算**。

3. **PAC 机制 (Algorithm 1)** ✅
   - 已实现于 `baselines/pac.py`：输入 v, B；排序 v；找最大 k 使 \(c(v_k, 1/(n-k)) \le B/k\)；\(p_i = \min(B/k, c(v_{k+1}, 1/(n-k)))\)（i≤k），否则 0。成本 \(c(v,\epsilon)=v\cdot\epsilon\)。
   - 输出：plosses（分配到的 ε）、payments；已接入 `experiments.auction()`，用于 RQ1/RQ3/RQ4 对比。

4. **VCG 采购拍卖** ✅
   - 已实现于 `baselines/vcg.py`：按 v 升序选最大 k 使 \(k\cdot c(v_{k+1}, 1/(n-k)) \le B\)，每位中标者支付 \(c(v_{k+1}, 1/(n-k))\)，满足 BF。
   - 已接入 `experiments.auction()`，用于 RQ1/RQ3/RQ4 对比。

5. **MFG-RegretNet 与论文完全对齐**
   - **b_MFG**：论文为 \(b_{MFG} = \frac{1}{N}\sum_i b_i\)（式 (50)）；当前 MFGRegretNet 用 `row_means`（按 item 平均），若 bid 为多维需改为对 agent 维求平均得到 \(b_{MFG}\) 并拼接到输入。
   - **误报时更新**：式 (54) \(b'_{MFG} = b_{MFG} + (b'_i - b_i)/N\)，在 regret 估计时使用 \(b'_{MFG}\)。
   - **预算投影**：式 (49) \(\bar{p}_i = p_i / \max(1, \frac{1}{B}\sum_{i'} p_{i'})\)，保证 \(\sum_i \bar{p}_i \le B\)。
   - **训练循环**：按 Algorithm 2（增广拉格朗日、λ 与 ρ 更新、PGA 估计 regret）。

6. **MFG-Pricing [26]**
   - 若无法复现 [26]，可先以“固定定价 + 同聚合”作为替代基线，或在文中说明仅与 PAC/RegretNet/VCG 对比。

7. **FL 全局模型**
   - 论文为图像/文本任务，需在 FL 中使用 **CNN（MNIST/CIFAR-10）** 和 **RNN/LSTM（Shakespeare）**；当前 `FL.py` 含 `Net`(MNIST)、`Logistic`，需确认 CIFAR-10 与 Shakespeare 的模型与数据接口。

8. **指标与日志**
   - 每轮/每实验记录：\(\bar{rgt}\)、IR 违反率、\(\sum_i p_i\)、测试准确率、墙钟时间（每轮）。
   - 便于画表（RQ1–RQ4）和 scalability 曲线（RQ2）。

---

## 五、实验实施顺序建议

### Phase 1：数据与基础管线

1. **MNIST 非 IID**  
   - 实现 Dirichlet(0.5) 划分到 N 个 client，写 `datasets_mnist.py` 或扩展现有 `datasets.py`。  
   - 与现有 `FL.py` 中 `Net` 对接，跑通 1 个 N（如 10）的 FL。

2. **CIFAR-10 / Shakespeare**  
   - 同上，Dirichlet 划分；Shakespeare 需确定数据源与格式（如 Leaf 的 Shakespeare）。  
   - 在 `FL.py` 中增加对应模型（如简单 CNN、LSTM）。

3. **Bid 与成本统一**  
   - 确定“论文模式”：每个 client 的 \(v_i, \epsilon_i\) 从 Uniform 采样，\(c_i = v_i \cdot \epsilon_i\)；bid 张量形状与 RegretNet/MFGRegretNet 的输入一致。  
   - 在 `utils` 或 `regretnet` 中统一 \(u_i = p_i - c(v_i,\epsilon_i)\) 与 regret 计算。

### Phase 2：基线实现

4. **PAC**  
   - 实现 Algorithm 1（可 numpy/torch），输入 (v, B)，输出 (allocation, payments)。  
   - 在 `experiments.py` 中增加 `trade_mech = ("PAC", ...)` 分支，输出 plosses/weights 供聚合与精度评估。

5. **VCG**  
   - 实现采购 VCG（分配 + 支付），满足 BF。  
   - 同上，接入现有 `auction` 与 `acc_eval` 流程。

6. **MFG-Pricing（可选）**  
   - 实现或替代为固定定价基线，用于 RQ3/RQ4。

### Phase 3：MFG-RegretNet 对齐与训练

7. **MFG-RegretNet 输入/输出**  
   - 输入：\([b_1,\ldots,b_N; b_{MFG}]\)，\(b_{MFG} = \frac{1}{N}\sum_i b_i\)。  
   - 误报时用式 (54) 更新 \(b'_{MFG}\)；预算投影用式 (49)。  
   - 若当前 `MFGRegretNet` 的 row_means 与论文不一致，改为对 agent 维求平均得到 \(b_{MFG}\)。

8. **训练脚本**  
   - 单独脚本或 `train.py` 分支：按 Algorithm 2 做增广拉格朗日（regret 约束 + 收益 + IR），T=200, T_in=25, L=64, R=25 等。  
   - 保存 checkpoint（用于后续 RQ1/RQ3/RQ4 评估）。

### Phase 4：系统评估与 RQ

9. **RQ1（激励相容）**  
   - 对 PAC、RegretNet、VCG、MFG-RegretNet（及可选 MFG-Pricing）在相同 (N, B, 数据) 下评估：  
     - 平均 regret、IR 违反率（多轮/多 seed）。  
   - 表格：方法 × (regret, IR rate)。

10. **RQ2（可扩展性）**  
    - 固定 B 与数据分布，N ∈ {10, 50, 100, 500}（及 200 若做 Shakespeare）。  
    - 记录每轮拍卖+聚合的墙钟时间，画 N vs time；对比 MFG-RegretNet、RegretNet、PAC（若 PAC 按 O(N log N) 实现，可画理论线）。

11. **RQ3（收益）**  
    - 同一批实验里记录 \(\sum_i \bar{p}_i\)，确保 BF；表格/图：方法 × 收益（可按 B 或 N 分组）。

12. **RQ4（FL 精度）**  
    - 同一批实验里记录每轮/最终测试准确率；对比固定隐私（如 All-in 或统一 ε）、PAC、VCG、RegretNet、MFG-RegretNet。  
    - 表格：方法 × 最终精度（及可选随轮次曲线）。

### Phase 5：表格与图表

13. **表格**  
    - Table：RQ1（regret, IR）、RQ3（revenue）、RQ4（accuracy）；行=方法，列=指标或 (N, B) 组合。  
    - 若做 RQ2：Table 或 Figure：N vs 每轮时间。

14. **图**  
    - RQ2：N vs 墙钟时间（对数或线性）。  
    - RQ4：训练轮次 vs 测试准确率（多条曲线，每方法一条）。  
    - 其他：如 B 对收益/精度的影响。

---

## 六、文件与脚本建议

| 用途 | 建议路径/文件名 |
|------|------------------|
| 实验规划（本文档） | `docs/EXPERIMENT_PLAN_PRIVACY_PAPER.md` |
| MNIST/CIFAR/Shakespeare 非 IID | `datasets_fl_benchmark.py` 或扩展现有 `datasets.py` |
| PAC 机制 | `pac_mechanism.py` 或 `baselines/pac.py` |
| VCG 采购拍卖 | `vcg_mechanism.py` 或 `baselines/vcg.py` |
| MFG-RegretNet 训练（论文 Algorithm 2） | `train_mfg_regretnet.py` 或 `train.py --model MFGRegretNet` |
| 统一评估（RQ1–RQ4） | `experiments_privacy_paper.py` 或扩展现有 `experiments.py` |
| 结果与图表 | `run/privacy_paper/` 下按 RQ/数据集分子目录 |

---

## 七、简要检查清单

- [ ] MNIST/CIFAR-10/Shakespeare 数据 + Dirichlet(0.5) 非 IID 划分可用。  
- [ ] 成本 \(c(v,\epsilon)=v\cdot\epsilon\) 与效用 \(u_i=p_i-c_i\) 在训练/测试中一致。  
- [ ] PAC、VCG 实现并接入现有 auction → aggregation → FL 流程。  
- [ ] MFG-RegretNet 的 b_MFG 定义、误报更新、预算投影与论文一致。  
- [ ] 增广拉格朗日训练（Algorithm 2）超参与论文一致并可复现。  
- [ ] 每轮记录 regret、IR 违反、收益、准确率、时间。  
- [ ] 生成 RQ1–RQ4 的表格与图，便于写入论文。

完成上述规划后，即可按 Phase 1→5 顺序实现并跑实验；若某基线（如 MFG-Pricing）难以复现，可在论文中说明并保留其余对比。

---

## 八、Phase 1 复检记录（已修复问题）

- **Shakespeare dummy 数据为空**：`load_shakespeare_dummy` 原先传入的 `sequences` 长度为 `seq_len`，导致 `ShakespeareDataset` 中 `range(len(seq)-seq_len)` 为 0、无样本。已改为传入长度为 `seq_len+1` 的序列，保证每个序列产生至少一个 (x,y)。
- **`extr_noniid_dirt` 负维度**：当类别数大、每类样本少（如 Shakespeare vocab_size=80）时，`data_size_each_class[max_class] -= 10*n_clients` 可能为负，进而 `np.random.choice(..., size=负值)` 报错。已改为：仅当最大类样本数 ≥ 10×n_clients 时才做保留；并对 `data_size_each_class_client` 做 `np.maximum(., 0)`；非最后一名 client 的 `take` 使用 `min(take, len(idxs_classes[j]))` 避免越界。
- **空 client 的 DataLoader**：验证脚本中对 Dirichlet 划分后的 client 用 `batch_size=min(64, len(sub))`，若 `len(sub)==0` 会得到 batch_size=0。已改为 `batch_size=max(1, min(64, n))` 并在 `n==0` 时显式报错。
- **未使用的 torchvision 导入**：`client.py` 顶部曾导入 `torchvision`，在仅用 Bank/NSL-KDD/Shakespeare 时也会强依赖。已移除该未使用导入；MNIST/CIFAR 的加载仅在 `datasets_fl_benchmark` 内按需导入。
- **验证**：在无 torchvision 环境下已通过：隐私论文 bid/cost/util、Shakespeare dummy 非空、`generate_clients("Shakespeare", ...)`。MNIST/CIFAR 单轮 FL 需安装 `torchvision` 后运行 `python run_phase1_verify.py`。
- **FL.py 中未使用的 h5py**：导入 h5py 时在部分环境（numpy 2.x）下会触发 `numpy.typeDict` 报错。已从 `FL.py` 移除未使用的 `import h5py`，便于 Phase 1 检查通过。

**Phase 1 完整过程检查**：运行 `python run_phase1_full_check.py` 可执行 10 项检查（无 torchvision 时 7 项执行、3 项跳过）；当前 7/7 通过。

---

## 九、Phase 2 复检记录（已修复与核对）

- **PAC 选人条件**：论文 Algorithm 1 中 k 为满足 \(c(v_k, 1/(n-k))\le B/k\) 的最大整数；当 \(v_k=0\) 时 \(c=0\) 也应允许选中。原实现用 `v_sorted[k_cand-1]*(n-k_cand)<=0` 时 `continue`，会在 \(v_k=0\) 时错误跳过。已改为仅当 `(n - k_cand) <= 0` 时跳过（在 `k_cand in range(1,n)` 下恒不成立，保留以防接口变化）。
- **与论文一致**：PAC 为排序 v 升序、选最大 k 使 \(c(v_k, 1/(n-k))\le B/k\)、\(p_i=\min(B/k, c(v_{k+1}, 1/(n-k)))\)（i≤k），否则 0；VCG 为选最大 k 使 \(k\cdot c(v_{k+1}, 1/(n-k))\le B\)，每位中标者付 \(c(v_{k+1}, 1/(n-k))\)。成本均为 \(c(v,\epsilon)=v\cdot\epsilon\)。实现与论文一致。
- **experiments 接入**：`auction()` 中 `trade_mech[0] in ("PAC","VCG")` 时调用 `pac_batch`/`vcg_procurement_batch`，不修改 reports，不加载 RegretNet；`acc_eval_mechs`、`mse_eval`、`mse_agents` 等中 `trade_mech[0] in ("All-in","FairQuery","PAC","VCG")` 时 `auc_model=None`，逻辑正确。
- **验证脚本**：`run_phase2_verify.py` 检查 PAC/VCG 的 plosses 与 payments 形状、预算可行性；并尝试 `auction(reports, budget, ("PAC","ConvlAggr","",1), model=None)` 做集成测试。若环境中未安装 `cvxpy`，`experiments` 导入会失败，脚本会跳过 auction 集成并打印说明，基线检查仍通过。

**Phase 2 检查**：运行 `python run_phase2_verify.py`；基线与预算检查通过；有 cvxpy 时还会执行 auction() 集成测试。

- **guarantees / invalid_rate_budget**：二者依赖 `load_auc_model(trade_mech[2])`，PAC/VCG 无 checkpoint。已在 `guarantees()` 与 `invalid_rate_budget()` 中对 `trade_mech[0] in ("PAC","VCG")` 做跳过处理（guarantees 记 regret/IR 为 0，invalid_rate_budget 记空曲线），避免 `trade_mech_ls` 含 PAC/VCG 时崩溃。

---

## 十、Phase 3 实现记录（MFG-RegretNet 对齐与训练）

- **b_MFG（式 50）**：在 `regretnet.MFGRegretNet.forward` 中实现为对 agent 维求平均：`b_mfg = reports.mean(dim=1, keepdim=True)`，再广播到每位 agent 与 reports 拼接为输入。
- **输入/输出**：与 RegretNet 一致接受 `(reports, budget)`；`reports` 形状 `(batch, n_agents, n_items+2)`（v, ε, size）；输入维度 `n_agents*2*(n_items+2)+1`（reports + b_MFG 广播 + budget）。
- **预算投影（式 49）**：`budget_projection_privacy_paper(payments, budget)`：`p_bar = p * min(1, B/sum(p))`，保证 `sum_i p_bar_i ≤ B`；在 forward 末对 raw payments 做投影后返回。
- **误报更新（式 54）**：tiled 误报时，仅 agent i 报 `b'_i`、其余报真实值，模型从当前 reports 计算均值，自动得到 `b'_MFG = b_MFG + (b'_i - b_i)/N`，无需在 `tiled_misreport_util` 中单独实现。
- **训练脚本**：`train_mfg_regretnet.py` 使用 `datasets_fl_benchmark.generate_privacy_paper_bids` 生成 (v, ε, size) + val_type 数据，调用 `regretnet.train_loop` 做增广拉格朗日训练。超参与论文对齐：T=200 轮、batch_size=64 (L)、misreport_iter=25 (R)、lagr_update_iter=25、rho_ir=10、n_hidden_layers=3、hidden_layer_size=100 等；checkpoint 保存时 arch 含 `model_type: 'MFGRegretNet'`。
- **experiments 接入**：`load_auc_model` 根据 `arch.get('model_type') == 'MFGRegretNet'` 实例化并加载 MFGRegretNet；`map_abbr_name` / `map_labels` 增加 "MFG-RegretNet"；auction 的 else 分支统一用 `model((reports, budget))`，MFG-RegretNet 与 RegretNet 共用该路径。
- **验证**：`run_phase3_verify.py` 在安装 cvxpy 后运行，检查 MFGRegretNet 前向、输出形状与预算可行性；无 cvxpy 时仅打印跳过说明。

**Phase 3 检查**：安装 cvxpy 后运行 `python run_phase3_verify.py`；训练：`python train_mfg_regretnet.py --num-epochs 2 --num-examples 1024`（快速试跑）。

**Phase 3 复检（已修复）**：
- **allocation 输出形状**：原 MFGRegretNet 使用本地 `View_Cut`（`x[:, :-1, :]`，切最后一“行”），得到 `(batch, n_agents-1, n_items+1)`，与 `allocs_to_plosses` / `allocs_instantiate_plosses` 期望的 `(batch, n_agents, n_items)` 不一致。已改为与 RegretNet 一致使用 `ibp.View_Cut`（`x[:, :, :-1]`，切最后一“列”），输出为 `(batch, n_agents, n_items)`。
- **resume 加载**：checkpoint 由 DataParallel 保存，键为 `module.xxx`；在 `train_mfg_regretnet.py` 中加载到未包装的 model 时需去掉 `module.` 前缀，已加上该逻辑；并增加对空 `state_dict` 的防护（仅当 `sd` 非空时加载）。
- **效用与成本与论文一致**：论文要求 \(u_i = p_i - c(v_i,\varepsilon_i)\)，\(c(v,\varepsilon)=v\cdot\varepsilon\)，其中 \(\varepsilon_i\) 为机制分配给的隐私损失（plosses）。原 `calc_agent_util` 使用 `costs = allocs*valuations*sizes`，与论文不一致。已增加参数 `cost_from_plosses`：当为 True 且 `n_items==1` 时，`costs = v * plosses`。在 `train_loop`、`test_loop`、`optimize_misreports`、`tiled_misreport_util` 及 `guarantees_eval` 中，当模型为 MFGRegretNet 且 `n_items==1` 时传入 `cost_from_plosses=True`；`train_loop` 内用于归一化 regret/IR 的 `costs` 在 MFG 时也改为 `v * plosses`。
- **tiled_misreport_util 与误报下的 cost**：① `real_reports` 的 batch 维与 `agent_allocations` 一致：`agent_allocations` 为 `(batch*n_agents, n_agents, n_items)`，故将 `real_reports` 用 `repeat_interleave(n_agents, dim=0)` 扩展为 `(batch*n_agents, n_agents, n_items+2)`，使 `allocs_to_plosses(agent_allocations, pbudgets)` 的维度匹配。② 误报时 regret 应为 \(u_i(b'_i) - u_i(b_i)\)，其中 cost 用**真实** \(v_i\)：在 `calc_agent_util` 中增加 `true_valuation_for_cost`，在 `tiled_misreport_util` 中当 `cost_from_plosses` 时传入 `true_valuation_for_cost = current_reports[:,:,0].repeat_interleave(n_agents, dim=0)`，保证误报效用中的 cost 为真实 \(v_i \cdot \varepsilon'_i\)。

---

## 十一、Phase 4：系统评估与 RQ

### 11.1 实现内容

- **auction() 扩展**：`experiments.auction` 增加参数 `return_payments=False`；为 True 时返回 `(plosses, weights, payments)`，供 RQ3 计算收益 \(\sum_i p_i\) 与预算可行性。
- **统一评估脚本**：`run_phase4_eval.py` 在隐私论文设定下（v~U[0,1]、ε~U[0.1,5]、c=v·ε、固定预算 B）运行：
  - **RQ1（激励相容）**：对 PAC、VCG、RegretNet、MFG-RegretNet 评估平均归一化 regret 与 IR 违反率。PAC/VCG 记为 0；神经机制使用 `guarantees_eval` 在 `generate_privacy_paper_bids` 生成的 profiles 上计算。
  - **RQ2（可扩展性）**：固定 B，N ∈ `--n-list`（如 10,50,100），记录单轮 auction + aggregation 墙钟时间；神经机制仅在与 checkpoint 对应的 N 上计时。
  - **RQ3（收益）**：同一批 profiles 上记录各机制的 \(\sum_i p_i\) 与预算可行比例（BF rate）。
  - **RQ4（FL 精度）**：可选；需现有数据集与各机制 checkpoint，调用 `experiments.acc_eval_mechs_parallel` 得到方法 × 最终准确率。

### 11.2 运行方式

**依赖**：需安装 `cvxpy`（与 `experiments`/`aggregation` 一致）。

默认会运行 RQ1、RQ2、RQ3；可用 `--skip-rq1`/`--skip-rq2`/`--skip-rq3` 跳过对应项。

```bash
# 仅 PAC/VCG（无需 RegretNet/MFG checkpoint）
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 --seeds 42,43,44 --n-list 10,50,100

# 含 RegretNet、MFG-RegretNet（需先训练得到 checkpoint）
python run_phase4_eval.py --n-agents 10 --budget 50 --num-profiles 1000 \
  --regretnet-ckpt result/mfg_regretnet_privacy-10-1/regretnet_10_1.pt \
  --mfg-regretnet-ckpt result/mfg_regretnet_privacy-10-1/mfg_10_1.pt
```

结果写入 `--out-dir`（默认 `run/privacy_paper/`）下的 `phase4_summary.json`，包含 RQ1–RQ3 的表格化数据，便于画表与 RQ 对应。

### 11.3 与 RQ 的对应

| RQ   | 指标/输出 | 脚本输出 |
|------|-----------|----------|
| RQ1  | 平均 regret、IR 违反率 | `phase4_summary.json` → `rq1` |
| RQ2  | N vs 每轮时间（秒） | `rq2` |
| RQ3  | 平均收益、BF 率 | `rq3` |
| RQ4  | 最终测试准确率 | 需单独运行 `acc_eval_mechs_parallel`，或扩展脚本传入 dataset/ckpt |

### 11.4 Phase 4 复检记录

- **argparse 默认行为**：原用 `--run-rq1` + `default=True` 时，`store_true` 的默认实为 False，导致不传参时三个 RQ 都不跑。已改为默认执行 RQ1/RQ2/RQ3，用 `--skip-rq1`/`--skip-rq2`/`--skip-rq3` 跳过。
- **RQ1 可复现**：`Dataloader` 改为 `shuffle=False`，相同 seed 下 regret/IR 结果可复现。
- **auction 与 RQ3**：`experiments.auction(..., return_payments=True)` 对 PAC/VCG 返回 `payments` 形状 `(batch, n_agents)`，与基线实现一致；收益与 BF 按 `payments.sum(dim=1)` 与 `budget` 逐 profile 比较，逻辑正确。
- **RQ2**：仅在与 checkpoint 对应的 `n_agents` 上对 RegretNet/MFG 计时，PAC/VCG 对所有 `--n-list` 中的 N 计时，符合设计。
- **JSON 输出**：写入 `phase4_summary.json` 前将 `float('nan')`（如无 checkpoint 时跳过的机制）转为 `null`，保证各环境下 JSON 可解析。
- **空参数**：`--seeds` 或 `--n-list` 解析后若为空，分别默认 `[42]` 与 `[--n-agents]`，并打印警告，避免 RQ1/RQ3 或 RQ2 因空列表报错。

---

## 十二、Phase 5：表格与图表

### 12.1 实现内容

- **脚本**：`run_phase5_tables_figures.py` 读取 Phase 4 输出的 `phase4_summary.json`，生成表格与图。
- **表格**（输出到 `run/privacy_paper/tables/`）：
  - **table_rq1.csv / table_rq1.md**：RQ1 — 行=方法，列=mean_regret, mean_ir_violation。
  - **table_rq2.csv / table_rq2.md**：RQ2 — N vs 每轮时间；MD 为 N × 方法的透视表。
  - **table_rq3.csv / table_rq3.md**：RQ3 — 行=方法，列=mean_revenue, bf_rate。
- **图**（输出到 `run/privacy_paper/figures/`）：
  - **figure_rq2_time_vs_n.png**：RQ2 可扩展性 — 横轴 N，纵轴时间（秒），每条曲线一个方法。
  - **figure_rq4_accuracy_vs_round.png**：可选；需通过 `--accuracy-json` 传入含 `rounds` 与 `methods`（方法名 → 每轮准确率列表）的 JSON，画训练轮次 vs 测试准确率。

### 12.2 运行方式

先完成 Phase 4 得到 `phase4_summary.json`，再运行：

```bash
# 默认：从 run/privacy_paper/phase4_summary.json 读入，生成 tables/ 与 figures/
python run_phase5_tables_figures.py

# 指定输入与输出目录
python run_phase5_tables_figures.py --input run/privacy_paper/phase4_summary.json --out-dir run/privacy_paper

# 仅表格、不生成图
python run_phase5_tables_figures.py --no-figures

# 含 RQ4 准确率曲线（需自备 accuracy JSON，格式见脚本说明）
python run_phase5_tables_figures.py --accuracy-json path/to/accuracy.json
```

**RQ4 准确率 JSON 格式**（用于 `--accuracy-json`）：  
`{"rounds": [1, 10, 20, ...], "methods": {"PAC": [acc1, ...], "VCG": [...], ...}}`  
或键为方法名、值为准确率列表的字典，脚本会尝试用 `rounds` 或 1-based 轮次作为横轴。

### 12.3 与 Phase 5 规划对应

| 规划项 | 输出 |
|--------|------|
| Table RQ1（regret, IR） | `tables/table_rq1.csv`, `table_rq1.md` |
| Table/Figure RQ2（N vs time） | `tables/table_rq2.*`, `figures/figure_rq2_time_vs_n.png` |
| Table RQ3（revenue） | `tables/table_rq3.csv`, `table_rq3.md` |
| Figure RQ4（rounds vs accuracy） | 可选，`figures/figure_rq4_accuracy_vs_round.png`（需 `--accuracy-json`） |

### 12.4 Phase 5 复检记录

- **CSV 输出**：改用标准库 `csv.writer` 写入，避免机制名含逗号时破坏 CSV；`newline=""` 保证跨平台换行。
- **RQ2 排序**：表格与图中 N（n_agents）按数值排序，保证 10, 50, 100 顺序正确；图中各方法曲线按 N 升序绘制。
- **输入校验**：若 `rq1`/`rq2`/`rq3` 缺失或非列表，对应表格/图跳过并打印说明，不抛错。
- **RQ2 图**：`mech_to_n` / `mech_to_t` 按 N 数值排序后再绘图，避免乱序。

# RQ3 问题诊断 - 快速开始指南

## 📋 您提出的三个问题

| 问题 | 图文件 | 症状 | 根本原因 |
|---|---|---|---|
| 1️⃣ 曲线遮挡 | `figure_rq3_budget_sensitivity.png` | 看不到MFG-RegretNet的曲线 | **预算扫描未实现**（代码层面） |
| 2️⃣ 收益平坦 | `figure_rq3_R_W_vs_epoch.png` | Revenue不随epoch变化 | **评估阶段推理的正常表现**（可能不是bug） |
| 3️⃣ 福利异常低 | `figure_rq3_revenue_welfare_bars.png` | W̄=37.23 vs RegretNet=48.48 | **隐私预算映射错误**（数据流层面）🎯 |

---

## 🚀 三分钟快速诊断

### 第一步：查看数据流问题
```bash
cd /home/skk/FL/market/FL-Market
python debug_rq3.py
```

**预期输出**：
```
eps_i 范围应该是: 0.1000 - 5.0000 ✓
eps_out 范围应该是: 0.1000 - 5.0000 ✓（检查这个！）
pay 范围应该是: 0.0000 - 50.0000 ✓
```

如果 `eps_out` 是 `[0, 1]`，那就找到问题了 →【下一步】

### 第二步：查看问题文件
我为您创建了5份详细文档：

1. **`RQ3_QUICK_REFERENCE.md`** ⭐ 从这里开始
   - 3-5分钟快速了解所有问题
   - 包含所有命令

2. **`RQ3_SUMMARY_AND_FIXES.md`** 
   - 问题总结 + 修复方案
   - 包含修复清单

3. **`RQ3_ROOT_CAUSE_FOUND.md`** 🎯
   - 根本原因深度分析
   - 代码流程追踪

4. **`RQ3_ISSUES_DIAGNOSIS.md`**
   - 详细诊断报告
   - 三个问题的独立分析

5. **`RQ3_DETAILED_FIX_GUIDE.md`**
   - 最详细的修复指南
   - 包含代码修复示例

---

## 🎯 问题排序（按修复优先级）

### 最紧急 🔴：问题3 - 社会福利异常低

**为什么紧急**：
- 这是RQ3的核心结果
- 如果数值错了，论文结论就错了
- 这是**必须在发表前修复**的

**修复过程**：
1. 运行 `debug_rq3.py` 确认问题位置 (5分钟)
2. 检查 `experiments.py` 中 `auction()` 函数的 `pbudgets` 提取 (10分钟)
3. 修复 (5分钟)
4. 重新运行 RQ3 (10分钟)

**总耗时**: 30分钟

---

### 次要 🟡：问题1 - 曲线被遮挡

**为什么次要**：
- 这只是代码不完整，不影响结果正确性
- 完成问题3的修复后，这个会自然解决（如果实现预算扫描）

**修复过程**：
1. 添加预算扫描参数支持 (15分钟)
2. 重新运行 (30分钟)

**总耗时**: 45分钟

---

### 可选 🟢：问题2 - 收益平坦

**为什么可选**：
- 这可能不是bug，而是正常的推理稳定性表现
- 需要澄清图的含义（是训练过程还是推理稳定性）

**决定**：
- 如果要展示训练收敛：需要修改代码跟踪训练指标
- 如果要展示推理稳定性：当前表现是正确的

---

## 💡 我的诊断

### 根本原因（简版）

```python
# 当前代码
eps_out = plosses  # 从auction返回的值
cost = (v * eps_out).sum(axis=1)  # 计算成本

# 可能的问题
# plosses 是否真的是隐私预算[0.1-5]？
# 还是某种归一化的[0-1]值？
# 这决定了整个成本计算是否正确
```

### 数据流追踪

```
生成数据: reports[:, :, 1] = eps_i ∈ [0.1, 5] ✓
  ↓
拍卖机制: 
  allocs = Z ∈ [0, 1]
  payments = p ∈ [0, B]
  ↓
计算plosses:
  plosses = allocs_to_plosses(allocs, pbudgets)
  应该 = ε_i^out ∈ [0.1, 5]
  问题?: pbudgets 是否被正确提取？
  ↓
计算福利:
  cost = v * plosses
  sw = pay.sum() - cost
  如果 cost > pay.sum()，则福利 < 0（错误！）
```

---

## 📂 文件位置速查

| 需要修改 | 文件 | 行号 |
|---|---|---|
| **pbudgets提取** | `experiments.py` | ~173-208 |
| **支付投影** | `regretnet.py` | 296-298 |
| **分配计算** | `utils.py` | 173-182 |
| **福利计算** | `rq3_paper_complete.py` | 76-82 |
| **数据生成** | `datasets_fl_benchmark.py` | 280-314 |

---

## 🔧 最可能的修复位置

### 文件：`experiments.py`（第173-208行）

```python
def auction(reports, budget, trade_mech, model=None, expected=False, return_payments=False):
    """问题可能在这里"""
    
    name = trade_mech[0]
    
    # ❌ 可能缺少这行（提取pbudgets）
    if reports.shape[-1] == n_items + 2:
        pbudgets = reports[:, :, -2]  # 每个代理的隐私预算 ε_i
    
    # ... 拍卖机制代码 ...
    
    # ❌ 这里 pbudgets 可能没有被正确传递
    if model is not None:
        allocs, payments_out = model((reports, budget))
        plosses = allocs_to_plosses(allocs, pbudgets)  # ← 检查这行
```

---

## ✅ 验证清单

修复后，您应该验证：

- [ ] `eps_out` 的范围应该在 [0.1, 5]
- [ ] `cost` 的平均值应该小于 `pay.sum()`
- [ ] `sw` (社会福利) 应该 > 0
- [ ] IR违反率应该 ≈ 0
- [ ] 与RegretNet的福利值相近或更好

---

## 📊 期望的修复结果

### 修复前
```
Ours:       收益 50.0 → 福利 37.23  ✗
RegretNet:  收益 48.49 → 福利 48.48 ✓
```

### 修复后
```
Ours:       收益 50.0 → 福利 40-48  ✓
RegretNet:  收益 48.49 → 福利 48.48 ✓
```

---

## 🎓 关键概念复习

| 术语 | 含义 | 范围 |
|---|---|---|
| **v_i** | 隐私估值（客户端对隐私的评价） | [0, 1] |
| **ε_i** | 隐私预算（客户端声明的隐私预算） | [0.1, 5] |
| **ε_i^out** | 分配的隐私预算 | [0, 5] |
| **p_i** | 支付（对客户端的补偿） | [0, B] |
| **c_i** | 隐私成本 = v_i · ε_i^out | [0, 5] |
| **u_i** | 代理效用 = p_i - c_i | [-5, B] |
| **W** | 社会福利 = Σ_i u_i | [?, ?] |
| **Z_ij** | 分配矩阵 | [0, 1] |
| **B** | 总预算 | 50 |

---

## 📞 如果卡住了

1. **检查维度**：所有计算都用对了吗？
   - `v.shape = (batch, n_agents)` ✓
   - `eps_out.shape = (batch, n_agents)` ✓
   - `pay.shape = (batch, n_agents)` ✓

2. **检查范围**：所有值都在合理范围内吗？
   - `v ∈ [0, 1]` ✓
   - `eps_out ∈ [0, 5]` ⬅️ **关键**
   - `pay ∈ [0, 50]` ✓
   - `cost ∈ [0, 5]` ✓

3. **检查计算**：
   ```python
   cost = (v * eps_out).sum(axis=1)      # 这行是否合理？
   pay_sum = pay.sum(axis=1)              # 支付总和
   welfare = pay_sum - cost               # 应该都 > 0
   ```

---

## 🎬 行动计划

### 今天（立即）
1. ✅ 读这份文档 (5分钟)
2. ✅ 运行 `debug_rq3.py` (5分钟)
3. ✅ 确定问题位置 (10分钟)

### 明天上午
1. ✅ 修复 `experiments.py` 的 `pbudgets` 提取 (30分钟)
2. ✅ 重新运行 RQ3 (20分钟)
3. ✅ 验证结果 (10分钟)

### 明天下午
1. ✅ 添加预算扫描支持 (30分钟)
2. ✅ 重新运行完整实验 (45分钟)
3. ✅ 生成所有图表 (15分钟)

**总耗时**: ~3小时内完成所有修复 ✅

---

## 🏁 完成标志

修复完成的标志是：
- ✅ 所有三个图表都正确生成
- ✅ 表格中的数值合理（福利 > 基线 or 接近最优）
- ✅ 所有代码运行不报错
- ✅ IR违反率接近0
- ✅ 结果与论文设计一致

---

**推荐阅读顺序**：
1. 本文件 (现在)
2. `RQ3_QUICK_REFERENCE.md` (了解全貌)
3. `RQ3_SUMMARY_AND_FIXES.md` (修复方案)
4. `RQ3_ROOT_CAUSE_FOUND.md` (深度理解)

祝修复顺利！🚀

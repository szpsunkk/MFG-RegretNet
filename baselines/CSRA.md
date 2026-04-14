# CSRA: 差分隐私联邦学习的鲁棒激励机制

## 算法概述

CSRA (Client Selection with Reverse Auction) [1] 是一种为差分隐私联邦学习 (DPFL) 设计的鲁棒激励机制，能够在存在不诚实客户端的情况下进行客户端选择和奖励分配。

## 核心思想

CSRA 包含两个主要组件：

1. **质量最大化选择 (CSRA-QMS)**：基于客户端的历史诚实记录、隐私预算和数据集大小，选择参与客户端并分配奖励
2. **不诚实客户端检测 (CSRA-DCD)**：通过分析暴露的模型更新的方差来检测和排除不诚实客户端

## 算法流程

### 第一步：质量最大化选择 (CSRA-QMS)

**目标函数：**

$$\max_{\{x_i^t\}, \{r_i^t\}} \sum_{i=1}^{N} q(H_i^t, \epsilon_i^t, |D_i|)$$

**约束条件：**
- $x_i^t \in \{0, 1\}$（客户端选择指示变量）
- $\sum_{i=1}^{N} x_i^t r_i^t \leq B$（奖励预算约束）
- $r_i^t \geq x_i^t b_i^t$（奖励可行性约束）

**客户端质量评估：**

$$q(H_i^t, \epsilon_i^t, |D_i|) = p(H_i^t) \cdot \frac{\epsilon_i^t}{\sum_{m=1}^{M} \epsilon_m} \cdot \frac{|D_i|}{\sum_{i=1}^{N} |D_i|}$$

其中诚实概率为：

$$p(H_i^t) = \begin{cases} 1, & t = 1 \\ \frac{\sum_{\tau=1}^{t-1} h_i^{\tau}}{t-1}, & t > 1 \end{cases}$$

**客户端排序与选择：**

按质量与出价比例排序：

$$\frac{q(H_1^t, \epsilon_1^t, |D_1|)}{b_1^t} > \cdots > \frac{q(H_k^t, \epsilon_k^t, |D_k|)}{b_k^t}$$

**奖励分配：**

$$r_i^t = \frac{b_{k+1}^t}{q(H_{k+1}^t, \epsilon_{k+1}^t, |D_{k+1}|)} q(H_i^t, \epsilon_i^t, |D_i|), \quad 1 \leq i \leq k$$

### 第二步：不诚实客户端检测 (CSRA-DCD)

#### 粗粒度检测

计算模型更新方差与噪声方差的比值：

$$\Phi_i^t = \frac{\hat{\sigma}_i^2}{2(\Delta f_i / \epsilon_i^t)^2} = \frac{\sigma_i^2 + 2(\Delta f_i / \bar{\epsilon}_i^t)^2}{2(\Delta f_i / \epsilon_i^t)^2}$$

若 $\Phi_i^t > \delta$（阈值），则将客户端 $i$ 标记为可疑客户端，加入集合 $I$。

#### 细粒度检测

对具有相同隐私预算的客户端进行聚类：

1. 计算相同隐私预算客户端间的余弦相似度
2. 使用 k-means 将其分为两个簇 $C_1$ 和 $C_2$
3. 根据以下规则确定不诚实客户端集合 $C$：

当 $|C_1| = |C_2|$ 时：
$$C = \arg\max_{\bar{C} \in \{C_1, C_2\}} \sum_{i \in \bar{C} \cap I} \Phi_i^t$$

当 $|C_1| \neq |C_2|$ 时：
$$C = \arg\max_{\bar{C} \in \{C_1, C_2\}} |\bar{C} \cap I|$$

### 第三步：模型聚合

排除不诚实客户端后的全局模型更新：

$$\omega^{t+1} = \sum_{x_i^t=1 \cap h_i^t=1} \frac{\epsilon_i^t |D_i|}{|D|} \omega_i^t$$

其中 $|D| = \sum_{x_i^t=1 \cap h_i^t=1} \epsilon_i^t |D_i|$

## 算法性质

CSRA 满足以下重要性质 [1]：

- **真实性 (Truthfulness)**：客户端只有在诚实报价时才能最大化效用
- **个体理性 (Individual Rationality)**：参与客户端的效用非负
- **预算可行性 (Budget Feasibility)**：总奖励不超过预算
- **计算效率 (Computational Efficiency)**：所有计算在多项式时间内完成

## 参考文献

[1] Y. Yang, M. Hu, Y. Zhou, X. Liu, and D. Wu, "CSRA: Robust incentive mechanism design for differentially private federated learning," IEEE Transactions on Information Forensics and Security, vol. 19, 2024.
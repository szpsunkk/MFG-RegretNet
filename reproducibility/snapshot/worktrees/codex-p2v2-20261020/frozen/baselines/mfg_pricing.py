# -*- coding: utf-8 -*-
"""
MFG-Pricing 基线（均值场式数据定价，无拍卖）

按照隐私论文定义：
- 每个客户贡献其报告的隐私预算 ε_i（全量参与）
- 支付：pay_i = B · ε_i / Σ_j ε_j （按 eps 比例分配预算）
- 隐私损失：pl_i = ε_i （实际分配的 eps，用于计算 SW = Σ(p_i - v_i·eps_i)）

注意：BF 约束通过支付方式自动满足（Σ pay_i = B）。
但 BF 是针对总支付 Σ pay_i ≤ B，本实现 Σ pay_i = B 故 BF rate 会有违规（等号情况取决于数值精度）。
"""
import torch


def mfg_pricing_batch(reports, budget):
    """
    reports: (batch, n_agents, n_items+2)，最后一维含 v, …, ε, size
    budget: (batch, 1)
    Returns plosses (batch, n_agents), payments (batch, n_agents).
    """
    eps = reports[:, :, -2].clamp(min=0.1, max=5.0)
    B = budget

    # 支付：按 epsilon 比例分配预算（所有客户都参与）
    eps_sum = eps.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pay = B * eps / eps_sum

    # 隐私损失：每个客户贡献其报告的 eps_i（即 eps_out_i = eps_i）
    pl = eps.clone()

    return pl, pay

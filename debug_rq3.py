#!/usr/bin/env python3
"""
调试脚本：验证RQ3中的数据流和计算正确性
"""

import torch
import numpy as np
from datasets_fl_benchmark import generate_privacy_paper_bids
from experiments import auction
from baselines.pac import pac_batch

def debug_rq3():
    print("="*60)
    print("RQ3 数据流调试")
    print("="*60)
    
    # 参数
    n_agents = 10
    n_items = 1
    batch_size = 5
    budget_val = 50.0
    seed = 42
    
    # 生成数据
    print("\n1. 数据生成")
    print("-"*60)
    reports = generate_privacy_paper_bids(
        n_agents=n_agents,
        n_items=n_items,
        num_profiles=batch_size,
        v_min=0.0, v_max=1.0,
        eps_min=0.1, eps_max=5.0,
        seed=seed
    )
    print(f"reports.shape: {reports.shape}")  # (batch, n_agents, 3)
    print(f"reports[0, 0, :]: {reports[0, 0, :]}")  # [v, ε, size]
    
    v = reports[:, :, 0].numpy()
    eps_i = reports[:, :, 1].numpy()
    
    print(f"\nv (隐私估值):")
    print(f"  范围: {v.min():.4f} - {v.max():.4f}")
    print(f"  样本: {v[0, :3]}")
    
    print(f"\neps_i (每个代理的隐私预算):")
    print(f"  范围: {eps_i.min():.4f} - {eps_i.max():.4f}")
    print(f"  样本: {eps_i[0, :3]}")
    
    budget = budget_val * torch.ones(batch_size, 1)
    print(f"\nbudget (总预算):")
    print(f"  值: {budget[0].item()}")
    
    # 测试PAC机制（基线）
    print("\n2. PAC机制")
    print("-"*60)
    plosses_pac, payments_pac = pac_batch(reports, budget)
    print(f"plosses_pac.shape: {plosses_pac.shape}")  # (batch, n_agents)
    print(f"payments_pac.shape: {payments_pac.shape}")
    
    print(f"\nPAC 隐私预算分配:")
    eps_out_pac = plosses_pac.numpy()
    print(f"  范围: {eps_out_pac.min():.4f} - {eps_out_pac.max():.4f}")
    print(f"  样本: {eps_out_pac[0, :3]}")
    
    # 计算PAC的社会福利
    pay_pac = payments_pac.numpy()
    print(f"\nPAC 支付:")
    print(f"  范围: {pay_pac.min():.4f} - {pay_pac.max():.4f}")
    print(f"  样本: {pay_pac[0, :3]}")
    print(f"  总和: {pay_pac.sum(axis=1)}")
    
    cost_pac = (v * eps_out_pac).sum(axis=1)
    sw_pac = pay_pac.sum(axis=1) - cost_pac
    
    print(f"\nPAC 社会福利计算:")
    print(f"  成本: {cost_pac}")
    print(f"  支付总和: {pay_pac.sum(axis=1)}")
    print(f"  福利: {sw_pac}")
    print(f"  平均福利: {sw_pac.mean():.4f}")
    
    # 检查IR
    cost_per_agent_pac = v * eps_out_pac
    ir_violation_pac = np.maximum(0, cost_per_agent_pac - pay_pac)
    ir_rate_pac = (ir_violation_pac > 1e-6).mean()
    print(f"  IR违反率: {ir_rate_pac:.4f}")
    
    # 现在测试auction函数
    print("\n3. auction() 函数返回值")
    print("-"*60)
    trade_mech_pac = ["PAC", "ConvlAggr", "", n_items]
    out = auction(reports, budget, trade_mech_pac, model=None, return_payments=True)
    
    if len(out) == 3:
        plosses, weights, payments = out
        print(f"返回三元组: (plosses, weights, payments)")
    else:
        print(f"警告：返回值长度 = {len(out)}")
    
    print(f"plosses.shape: {plosses.shape}")
    print(f"weights.shape: {weights.shape}")
    print(f"payments.shape: {payments.shape}")
    
    # 检查plosses是否与pac_batch一致
    print(f"\nplosses vs plosses_pac:")
    print(f"  plosses[0]: {plosses[0, :3]}")
    print(f"  plosses_pac[0]: {plosses_pac[0, :3]}")
    print(f"  差异: {(plosses - plosses_pac).abs().max().item():.2e}")
    
    # 检查payments是否与pac_batch一致
    print(f"\npayments vs payments_pac:")
    print(f"  payments[0]: {payments[0, :3]}")
    print(f"  payments_pac[0]: {payments_pac[0, :3]}")
    print(f"  差异: {(payments - payments_pac).abs().max().item():.2e}")
    
    print("\n" + "="*60)
    print("调试完成")
    print("="*60)

if __name__ == "__main__":
    debug_rq3()

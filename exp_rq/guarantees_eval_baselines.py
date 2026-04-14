# -*- coding: utf-8 -*-
"""
PAC / VCG / CSRA 的 RQ1 指标：与 guarantees_eval 一致的定义。
在离散网格上近似「单代理最优偏离」后的 ex-post regret 与 IR（机制不可微，不能用 PGA）。

u_i^truth = p_i - v_i * eps_alloc_i（真实成本 = 真 v × 分配到的隐私损失）
偏离：代理 i 单独改报价 (v', eps')，其余保持真实；CSRA / MFG-Pricing 对 (v, eps) 二维网格；PAC/VCG 仅 v。
"""
from __future__ import division, print_function

import torch

from baselines.pac import pac_batch
from baselines.vcg import vcg_procurement_batch
from baselines.csra import csra_qms_batch
from baselines.mfg_pricing import mfg_pricing_batch


def _dispatch(mech_name):
    if mech_name == "PAC":
        return pac_batch
    if mech_name == "VCG":
        return vcg_procurement_batch
    if mech_name == "CSRA":
        return csra_qms_batch
    if mech_name == "MFG-Pricing":
        return mfg_pricing_batch
    raise ValueError("unknown mech %r" % (mech_name,))


def guarantees_eval_procurement_baseline(
    reports, budget, mech_name, v_grid_n=25, eps_grid_n=12, eps_max=5.0, eps_min=0.1
):
    """
    reports: (B, N, n_items+2)，隐私设定 n_items=1 时为 (B,N,3) v,eps,size
    budget: (B, 1)
    返回 regret_norm, ir_norm，形状 (B, N)。分母：已分配用 v·ε_alloc；未分配用 v·ε_bid（避免 pl=0 时爆炸）。
    """
    fn = _dispatch(mech_name)
    device = reports.device
    dtype = reports.dtype
    B, N, D = reports.shape
    v_true = reports[:, :, 0]

    pl0, pay0 = fn(reports, budget)
    util0 = pay0 - v_true * pl0
    # 真实隐私成本 c = v * ε_alloc；未中标者 pl0≈0 时不能用 1e-10 做分母（会把遗憾放大到 1e9 量级）
    eps_bid = reports[:, :, -2].clamp(min=0.1, max=5.0)
    actual_cost = v_true * pl0
    denom = torch.where(
        actual_cost > 1e-5,
        actual_cost,
        (v_true * eps_bid).clamp(min=0.02),
    )
    denom = denom.clamp(min=1e-4)

    max_u = util0.clone()
    v_grid = torch.linspace(0.0, 1.0, v_grid_n, device=device, dtype=dtype)

    if mech_name in ("PAC", "VCG"):
        # 需在 batch 维复制网格：每行 profile 独立，形状 (B, v_grid_n)
        v_tile = v_grid.unsqueeze(0).expand(B, v_grid_n)
        for i in range(N):
            Rexp = reports.unsqueeze(1).expand(B, v_grid_n, N, D).contiguous().clone()
            Rexp[:, :, i, 0] = v_tile
            Rflat = Rexp.view(B * v_grid_n, N, D)
            Bflat = budget.unsqueeze(1).expand(B, v_grid_n, 1).contiguous().view(B * v_grid_n, 1)
            pl, pay = fn(Rflat, Bflat)
            pl = pl.view(B, v_grid_n, N)
            pay = pay.view(B, v_grid_n, N)
            u_i = pay[:, :, i] - v_true[:, i].unsqueeze(1) * pl[:, :, i]
            max_u[:, i] = torch.maximum(max_u[:, i], u_i.max(dim=1)[0])
    elif mech_name in ("CSRA", "MFG-Pricing"):
        # (v, eps) 二维网格
        eps_grid = torch.linspace(eps_min, eps_max, eps_grid_n, device=device, dtype=dtype)
        G = v_grid_n * eps_grid_n
        g_idx = torch.arange(G, device=device)
        gv = g_idx // eps_grid_n
        ge = g_idx % eps_grid_n
        v_rep = v_grid[gv]
        eps_rep = eps_grid[ge]
        v_tile = v_rep.unsqueeze(0).expand(B, G)
        e_tile = eps_rep.unsqueeze(0).expand(B, G)
        for i in range(N):
            Rexp = reports.unsqueeze(1).expand(B, G, N, D).contiguous().clone()
            Rexp[:, :, i, 0] = v_tile
            Rexp[:, :, i, -2] = e_tile
            Rflat = Rexp.view(B * G, N, D)
            Bflat = budget.unsqueeze(1).expand(B, G, 1).contiguous().view(B * G, 1)
            pl, pay = fn(Rflat, Bflat)
            pl = pl.view(B, G, N)
            pay = pay.view(B, G, N)
            u_i = pay[:, :, i] - v_true[:, i].unsqueeze(1) * pl[:, :, i]
            max_u[:, i] = torch.maximum(max_u[:, i], u_i.max(dim=1)[0])
    else:
        raise ValueError("unknown mech %r" % (mech_name,))

    regrets = torch.clamp(max_u - util0, min=0.0)
    ir_violation = -torch.clamp(util0, max=0.0)
    return regrets / denom, ir_violation / denom

#!/usr/bin/env python3
"""
RQ1：激励相容性 - 完全重写版本

修复问题：
1. 基线方法的遗憾计算（网格搜索实现）
2. 神经网络方法的遗憾计算（PGA 实现）
3. IR 违反率计算（统一效用函数）
4. 数据收集和聚合逻辑

指标：
  - 平均归一化事后遗憾 (normalized ex-post regret)
  - IR 违反率 (Individual Rationality violation rate)
  - 诚实报价率 (Truthful bidding rate)
  - 支付稳定性 (Payment CV)
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def compute_utility(payments, costs):
    """计算效用：u_i = p_i - c_i"""
    return payments - costs


def compute_regret_grid_search(mechanism_fn, reports, budget, v_grid_n=15, eps_grid_n=7):
    """
    对基线方法（PAC, VCG, CSRA, MFG-Pricing）使用网格搜索计算遗憾
    
    Args:
        mechanism_fn: 机制函数，输入 (reports, budget)，输出 (plosses, payments)
        reports: [batch, n_agents, 3] (v, eps, size)
        budget: [batch, 1]
        v_grid_n: v 的网格点数
        eps_grid_n: epsilon 的网格点数
    
    Returns:
        regrets: [batch, n_agents] - 归一化遗憾
        ir_violations: [batch, n_agents] - IR 违反（0/1）
    """
    batch_size, n_agents, _ = reports.shape
    device = reports.device
    
    # 真实报价下的结果
    plosses_true, payments_true = mechanism_fn(reports, budget)
    costs_true = reports[:, :, 0] * plosses_true
    utility_true = compute_utility(payments_true, costs_true)
    
    # 对每个 agent 进行网格搜索
    regrets = torch.zeros(batch_size, n_agents, device=device)
    
    # 创建网格
    v_grid = torch.linspace(0.0, 1.0, v_grid_n, device=device)
    eps_grid = torch.linspace(0.1, 5.0, eps_grid_n, device=device)
    
    for i in range(n_agents):
        max_utility_misreport = utility_true[:, i].clone()
        
        # 遍历所有可能的虚假报价
        for v_fake in v_grid:
            for eps_fake in eps_grid:
                # 创建虚假报价
                reports_fake = reports.clone()
                reports_fake[:, i, 0] = v_fake
                reports_fake[:, i, 1] = eps_fake
                
                # 运行机制
                plosses_fake, payments_fake = mechanism_fn(reports_fake, budget)
                
                # 计算虚假报价下的效用（使用真实成本）
                costs_fake = reports[:, i, 0] * plosses_fake[:, i]
                utility_fake = compute_utility(payments_fake[:, i], costs_fake)
                
                # 更新最大效用
                max_utility_misreport = torch.maximum(max_utility_misreport, utility_fake)
        
        # 计算遗憾
        regret_i = torch.clamp(max_utility_misreport - utility_true[:, i], min=0)
        
        # 归一化（除以成本）
        costs_norm = torch.clamp(costs_true[:, i], min=1e-6)
        regrets[:, i] = regret_i / costs_norm
    
    # IR 违反
    ir_violations = (utility_true < 0).float()
    
    return regrets, ir_violations


def compute_regret_pga(model, reports, budget, val_type, misreport_iter=25, lr=0.1):
    """
    对神经网络方法使用投影梯度上升（PGA）计算遗憾
    
    Args:
        model: 神经网络模型
        reports: [batch, n_agents, 3] (v, eps, size)
        budget: [batch, 1]
        val_type: [batch, n_agents, 2] (v, eps) - 真实类型
        misreport_iter: PGA 迭代次数
        lr: 学习率
    
    Returns:
        regrets: [batch, n_agents] - 归一化遗憾
        ir_violations: [batch, n_agents] - IR 违反（0/1）
    """
    from regretnet import MFGRegretNet
    from utils import allocs_to_plosses
    
    device = reports.device
    batch_size, n_agents = reports.shape[0], reports.shape[1]
    
    # 判断是否是 MFG-RegretNet（cost 从 plosses 计算）
    cost_from_plosses = isinstance(model, MFGRegretNet)
    
    # 真实报价下的结果
    with torch.no_grad():
        allocs_true, payments_true = model((reports, budget))
        
        if cost_from_plosses:
            pbudgets = reports[:, :, -2]
            plosses_true = allocs_to_plosses(allocs_true, pbudgets)
            costs_true = val_type[:, :, 0] * plosses_true
        else:
            costs_true = torch.sum(allocs_true * val_type[:, :, :1], dim=2) * reports[:, :, -1]
        
        utility_true = compute_utility(payments_true, costs_true)
    
    # 初始化虚假报价（从真实报价开始）
    misreports = reports.clone().detach()
    misreports.requires_grad = True
    
    # PGA 优化虚假报价
    for _ in range(misreport_iter):
        if misreports.grad is not None:
            misreports.grad.zero_()
        
        # 虚假报价下的结果
        allocs_fake, payments_fake = model((misreports, budget))
        
        # 计算虚假报价下的效用（使用真实成本）
        if cost_from_plosses:
            pbudgets_fake = misreports[:, :, -2]
            plosses_fake = allocs_to_plosses(allocs_fake, pbudgets_fake)
            costs_fake = val_type[:, :, 0] * plosses_fake
        else:
            costs_fake = torch.sum(allocs_fake * val_type[:, :, :1], dim=2) * misreports[:, :, -1]
        
        utility_fake = compute_utility(payments_fake, costs_fake)
        
        # 最大化效用
        loss = -utility_fake.sum()
        loss.backward()
        
        # 梯度上升
        with torch.no_grad():
            misreports[:, :, 0] -= lr * misreports.grad[:, :, 0]
            misreports[:, :, 1] -= lr * misreports.grad[:, :, 1]
            
            # 投影到有效范围
            misreports[:, :, 0].clamp_(0.0, 1.0)
            misreports[:, :, 1].clamp_(0.1, 5.0)
        
        misreports = misreports.detach()
        misreports.requires_grad = True
    
    # 最终虚假报价下的效用
    with torch.no_grad():
        allocs_final, payments_final = model((misreports, budget))
        
        if cost_from_plosses:
            pbudgets_final = misreports[:, :, -2]
            plosses_final = allocs_to_plosses(allocs_final, pbudgets_final)
            costs_final = val_type[:, :, 0] * plosses_final
        else:
            costs_final = torch.sum(allocs_final * val_type[:, :, :1], dim=2) * misreports[:, :, -1]
        
        utility_misreport = compute_utility(payments_final, costs_final)
    
    # 计算遗憾
    regret = torch.clamp(utility_misreport - utility_true, min=0)
    
    # 归一化
    costs_norm = torch.clamp(costs_true, min=1e-6)
    regrets = regret / costs_norm
    
    # IR 违反
    ir_violations = (utility_true < 0).float()
    
    return regrets, ir_violations


def evaluate_baseline_mechanism(mech_name, n_agents, n_items, budget, num_profiles, seeds, batch_size=256):
    """评估基线方法（PAC, VCG, CSRA, MFG-Pricing）"""
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import DEVICE
    from datasets import Dataloader
    from baselines import pac_batch, vcg_procurement_batch, csra_qms_batch
    from baselines.mfg_pricing import mfg_pricing_batch
    
    mechanism_map = {
        "PAC": pac_batch,
        "VCG": vcg_procurement_batch,
        "CSRA": csra_qms_batch,
        "MFG-Pricing": mfg_pricing_batch,
    }
    
    if mech_name not in mechanism_map:
        return None
    
    mechanism_fn = mechanism_map[mech_name]
    results_per_seed = []
    
    print(f"  Evaluating {mech_name:15s} ...", end="", flush=True)
    
    for seed_idx, seed in enumerate(seeds):
        # 生成数据
        reports, bud, val_type = build_privacy_paper_batch(
            num_profiles, n_agents, n_items, budget, seed, DEVICE
        )
        
        loader = Dataloader(
            torch.cat([reports, val_type], dim=2),
            batch_size=batch_size,
            shuffle=False
        )
        
        all_regrets = []
        all_ir_violations = []
        
        for batch in loader:
            rep = batch[:, :, :-2].to(DEVICE)
            b = budget * torch.ones(rep.shape[0], 1, device=DEVICE)
            
            # 计算遗憾和 IR 违反
            regrets, ir_violations = compute_regret_grid_search(
                mechanism_fn, rep, b,
                v_grid_n=11,  # 减少网格点以加快速度
                eps_grid_n=6
            )
            
            all_regrets.append(regrets.detach().cpu().numpy())
            all_ir_violations.append(ir_violations.detach().cpu().numpy())
        
        # 聚合当前种子的结果
        regrets_flat = np.concatenate(all_regrets).ravel()
        ir_flat = np.concatenate(all_ir_violations).ravel()
        
        results_per_seed.append({
            "seed": seed,
            "mean_regret": float(regrets_flat.mean()),
            "mean_ir_violation": float(ir_flat.mean() * 100),  # 转换为百分比
        })
        
        print(f" seed{seed_idx+1}/{ len(seeds)}", end="", flush=True)
    
    print(f" DONE")
    return results_per_seed


def evaluate_neural_mechanism(mech_name, ckpt_path, n_agents, n_items, budget, num_profiles, seeds, batch_size=256):
    """评估神经网络方法（RegretNet, DM-RegretNet, MFG-RegretNet）"""
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import DEVICE, load_auc_model
    from datasets import Dataloader
    
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    
    try:
        model = load_auc_model(ckpt_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"  [ERROR] {mech_name} load failed: {e}")
        return None
    
    results_per_seed = []
    
    print(f"  Evaluating {mech_name:15s} ...", end="", flush=True)
    
    for seed_idx, seed in enumerate(seeds):
        # 生成数据
        reports, bud, val_type = build_privacy_paper_batch(
            num_profiles, n_agents, n_items, budget, seed, DEVICE
        )
        
        loader = Dataloader(
            torch.cat([reports, val_type], dim=2),
            batch_size=batch_size,
            shuffle=False
        )
        
        all_regrets = []
        all_ir_violations = []
        
        for batch in loader:
            rep = batch[:, :, :-2].to(DEVICE)
            vt = batch[:, :, -2:].to(DEVICE)
            b = budget * torch.ones(rep.shape[0], 1, device=DEVICE)
            
            # 计算遗憾和 IR 违反
            regrets, ir_violations = compute_regret_pga(
                model, rep, b, vt,
                misreport_iter=25,
                lr=0.1
            )
            
            all_regrets.append(regrets.detach().cpu().numpy())
            all_ir_violations.append(ir_violations.detach().cpu().numpy())
        
        # 聚合当前种子的结果
        regrets_flat = np.concatenate(all_regrets).ravel()
        ir_flat = np.concatenate(all_ir_violations).ravel()
        
        results_per_seed.append({
            "seed": seed,
            "mean_regret": float(regrets_flat.mean()),
            "mean_ir_violation": float(ir_flat.mean() * 100),  # 转换为百分比
        })
        
        print(f" seed{seed_idx+1}/{len(seeds)}", end="", flush=True)
    
    print(f" DONE")
    return results_per_seed


def aggregate_results(results_per_seed):
    """聚合多个种子的结果"""
    if not results_per_seed:
        return None
    
    regrets = [r["mean_regret"] for r in results_per_seed]
    ir_violations = [r["mean_ir_violation"] for r in results_per_seed]
    
    return {
        "mean_regret_mean": float(np.mean(regrets)),
        "mean_regret_std": float(np.std(regrets, ddof=1)) if len(regrets) > 1 else 0.0,
        "ir_violation_mean": float(np.mean(ir_violations)),
        "ir_violation_std": float(np.std(ir_violations, ddof=1)) if len(ir_violations) > 1 else 0.0,
        "per_seed": results_per_seed,
    }


def main():
    parser = argparse.ArgumentParser(description="RQ1 Incentive Compatibility - Rewritten")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-items", type=int, default=1)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--num-profiles", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--regretnet-ckpt", type=str, default="")
    parser.add_argument("--dm-regretnet-ckpt", type=str, default="")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper/rq1_rewritten")
    args = parser.parse_args()
    
    # 解析种子
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip().isdigit()]
    if not seeds:
        seeds = [42, 43, 44]
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 60)
    print("RQ1: Incentive Compatibility - Rewritten Version")
    print("=" * 60)
    print(f"N agents: {args.n_agents}")
    print(f"Budget: {args.budget}")
    print(f"Num profiles: {args.num_profiles}")
    print(f"Seeds: {seeds}")
    print("=" * 60)
    print()
    
    # 评估所有方法
    all_results = {}
    
    # 基线方法
    for mech in ["PAC", "VCG", "CSRA", "MFG-Pricing"]:
        results = evaluate_baseline_mechanism(
            mech, args.n_agents, args.n_items, args.budget,
            args.num_profiles, seeds, args.batch_size
        )
        if results:
            all_results[mech] = aggregate_results(results)
    
    # 神经网络方法
    neural_methods = [
        ("RegretNet", args.regretnet_ckpt),
        ("DM-RegretNet", args.dm_regretnet_ckpt),
        ("MFG-RegretNet", args.mfg_regretnet_ckpt),
    ]
    
    for mech_name, ckpt_path in neural_methods:
        if ckpt_path and os.path.isfile(ckpt_path):
            results = evaluate_neural_mechanism(
                mech_name, ckpt_path, args.n_agents, args.n_items,
                args.budget, args.num_profiles, seeds, args.batch_size
            )
            if results:
                all_results[mech_name] = aggregate_results(results)
        else:
            print(f"  Skipping {mech_name:15s} (no checkpoint)")
    
    # 保存结果
    output = {
        "config": {
            "n_agents": args.n_agents,
            "n_items": args.n_items,
            "budget": args.budget,
            "num_profiles": args.num_profiles,
            "seeds": seeds,
        },
        "results": all_results,
    }
    
    out_path = os.path.join(args.out_dir, "rq1_rewritten_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # 打印结果表格
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Method':<15} {'Regret (mean±std)':<25} {'IR% (mean±std)':<25}")
    print("-" * 60)
    
    for mech_name in ["PAC", "VCG", "CSRA", "RegretNet", "DM-RegretNet", "MFG-Pricing", "MFG-RegretNet"]:
        if mech_name in all_results:
            r = all_results[mech_name]
            regret_str = f"{r['mean_regret_mean']:.4f} ± {r['mean_regret_std']:.4f}"
            ir_str = f"{r['ir_violation_mean']:.2f} ± {r['ir_violation_std']:.2f}"
            print(f"{mech_name:<15} {regret_str:<25} {ir_str:<25}")
    
    print("=" * 60)
    print(f"✓ Results saved to: {out_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

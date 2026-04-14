#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3: Invalid Gradient Rate 实验 (类似 FL-market.pdf 图8)

Invalid Gradient Rate = 拍卖机制分配所有客户端 ε=0 的频率
- 随机机制 (RegretNet, M-RegretNet) 可能产生全零分配
- 确定性机制 (DM-RegretNet, MFG-RegretNet) 应该避免这种情况

实验设置:
- 横轴: Financial Budget Factor (0.2 - 2.0)
- 纵轴: Invalid Gradient Rate (0 - 1, 越低越好)
- 数据集: MNIST, CIFAR-10 (IID 和 Non-IID)

一键运行:
  python run_invalid_gradient_experiment.py --datasets MNIST CIFAR10 --n-trials 200
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from datasets_fl_benchmark import generate_privacy_paper_bids
from utils import generate_max_cost


def _load_auc(path):
    if not path or not os.path.isfile(path):
        return None
    from experiments import load_auc_model
    return load_auc_model(path)


def compute_invalid_gradient_rate(reports, budget, trade_mech, auc_model, device):
    """
    计算 invalid gradient rate
    Invalid = 所有客户端的 epsilon 都为 0（无法进行有效的FL训练）
    """
    from experiments import auction
    
    n_agents = reports.shape[1]
    n_items = reports.shape[2] - 2
    
    reports = reports.to(device).float()
    budget = budget.to(device).float()
    
    # 运行拍卖
    if trade_mech[0] in ("PAC", "VCG", "CSRA", "MFG-Pricing", "All-in"):
        plosses, _ = auction(reports, budget, trade_mech, model=None)
    else:
        # 确保模型的 n_items 匹配
        if auc_model is not None:
            expected_items = auc_model.n_items if hasattr(auc_model, 'n_items') else 1
            if expected_items != n_items:
                print(f"Warning: Model expects {expected_items} items but got {n_items}, adjusting...")
                # 调整 reports 维度
                if n_items < expected_items:
                    # 扩展到expected维度
                    padding = torch.zeros((reports.shape[0], n_agents, expected_items - n_items), device=device)
                    reports_padded = torch.cat([reports[:, :, :-2], padding, reports[:, :, -2:]], dim=2)
                    reports = reports_padded
                else:
                    # 截断到expected维度
                    reports = torch.cat([reports[:, :, :expected_items], reports[:, :, -2:]], dim=2)
        
        auc = auc_model.to(device) if auc_model is not None else None
        plosses, _ = auction(reports, budget, trade_mech, model=auc)
    
    # 检查是否所有 epsilon 都为 0
    invalid = (plosses.sum(dim=1) == 0).float().mean().item()
    return invalid


def run_experiment_single_config(
    dataset,
    alpha,
    n_trials,
    n_agents,
    budget_factors,
    mfg_ckpt,
    regretnet_ckpt,
    dm_ckpt,
    mreg_ckpt,
):
    """运行单个配置的实验"""
    device = torch.device("cpu")
    
    print(f"\n{'='*80}")
    print(f"Running: {dataset}, α={alpha}, n_trials={n_trials}")
    print(f"{'='*80}")
    
    # 定义要测试的机制
    mfg = _load_auc(mfg_ckpt)
    reg = _load_auc(regretnet_ckpt)
    dm = _load_auc(dm_ckpt)
    mreg = _load_auc(mreg_ckpt)
    
    mechanisms = []
    if mfg is not None:
        mechanisms.append(("MFG-RegretNet (ours)", ["MFG-RegretNet", "ConvlAggr", mfg_ckpt, 1], mfg))
    if reg is not None:
        mechanisms.append(("RegretNet", ["RegretNet", "ConvlAggr", regretnet_ckpt, 1], reg))
    if mreg is not None:
        mechanisms.append(("M-RegretNet", ["M-RegretNet", "ConvlAggr", mreg_ckpt, 8], mreg))
    if dm is not None:
        mechanisms.append(("DM-RegretNet", ["DM-RegretNet", "ConvlAggr", dm_ckpt, 8], dm))
    
    # 添加 baseline（不需要模型）
    mechanisms.append(("PAC", ["PAC", "ConvlAggr", "", 1], None))
    mechanisms.append(("VCG", ["VCG", "ConvlAggr", "", 1], None))
    mechanisms.append(("CSRA", ["CSRA", "ConvlAggr", "", 1], None))
    mechanisms.append(("MFG-Pricing", ["MFG-Pricing", "ConvlAggr", "", 1], None))
    
    # 结果存储
    results = {mech[0]: [] for mech in mechanisms}
    
    # 对每个 budget factor 进行实验
    for budget_factor in budget_factors:
        print(f"\nBudget Factor = {budget_factor:.2f}")
        
        # 每个方法的 invalid rate
        invalid_rates = {mech[0]: [] for mech in mechanisms}
        
        # 多次试验
        for trial in range(n_trials):
            if trial % 50 == 0:
                print(f"  Trial {trial}/{n_trials}...")
            
            # 生成随机拍卖输入
            seed = int(time.time() * 1000000) % 1000000 + trial
            reports = generate_privacy_paper_bids(n_agents, 1, 1, seed=seed)
            if reports.dim() == 2:
                reports = reports.unsqueeze(0)
            reports = reports.to(device)
            
            # 计算 budget
            max_cost = generate_max_cost(reports)
            budget = max_cost * budget_factor
            
            # 对每个机制计算 invalid rate
            for mech_name, mech_spec, mech_model in mechanisms:
                invalid = compute_invalid_gradient_rate(
                    reports, budget, mech_spec, mech_model, device
                )
                invalid_rates[mech_name].append(invalid)
        
        # 计算平均 invalid rate
        for mech_name in invalid_rates:
            avg_invalid = np.mean(invalid_rates[mech_name])
            results[mech_name].append(avg_invalid)
            print(f"    {mech_name:30s}: {avg_invalid:.4f}")
    
    return results


def plot_invalid_gradient_4panel(all_data, output_path, title="Invalid Gradient Rate vs Budget Factor"):
    """绘制 4 子图"""
    method_styles = {
        'MFG-RegretNet (ours)': {'color': '#d32f2f', 'linestyle': '-', 'linewidth': 3.0, 'marker': 'o', 'markersize': 7, 'zorder': 10},
        'RegretNet': {'color': '#ff9800', 'linestyle': '--', 'linewidth': 2.5, 'marker': 'v', 'markersize': 6, 'zorder': 5},
        'M-RegretNet': {'color': '#ffc107', 'linestyle': '-.', 'linewidth': 2.5, 'marker': '^', 'markersize': 6, 'zorder': 5},
        'DM-RegretNet': {'color': '#4caf50', 'linestyle': '-', 'linewidth': 2.5, 'marker': 's', 'markersize': 6, 'zorder': 5},
        'CSRA': {'color': '#1976d2', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'p', 'markersize': 5, 'zorder': 4},
        'MFG-Pricing': {'color': '#7b1fa2', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'D', 'markersize': 4, 'zorder': 4},
        'PAC': {'color': '#00838f', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'h', 'markersize': 5, 'zorder': 4},
        'VCG': {'color': '#388e3c', 'linestyle': '-', 'linewidth': 2.0, 'marker': '*', 'markersize': 7, 'zorder': 4},
    }
    
    method_order = ['RegretNet', 'M-RegretNet', 'MFG-RegretNet (ours)', 'DM-RegretNet', 'PAC', 'VCG', 'CSRA', 'MFG-Pricing']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    configs = [
        (("MNIST", 0.5), axes[0, 0], "MNIST (α=0.5, IID)"),
        (("MNIST", 0.1), axes[0, 1], "MNIST (α=0.1, Non-IID)"),
        (("CIFAR10", 0.5), axes[1, 0], "CIFAR-10 (α=0.5, IID)"),
        (("CIFAR10", 0.1), axes[1, 1], "CIFAR-10 (α=0.1, Non-IID)"),
    ]
    
    for (dataset_key, ax, subtitle) in configs:
        if dataset_key not in all_data:
            print(f"Warning: {dataset_key} not found")
            continue
        
        block = all_data[dataset_key]
        budget_factors = np.array(block['budget_factors'])
        methods = block['methods']
        
        for method_name in method_order:
            if method_name not in methods:
                continue
            
            invalid_rates = np.array(methods[method_name])
            if len(invalid_rates) == 0:
                continue
            
            style = method_styles.get(method_name, {})
            
            ax.plot(budget_factors, invalid_rates, 
                    label=method_name,
                    color=style.get('color', '#333333'),
                    linestyle=style.get('linestyle', '-'),
                    linewidth=style.get('linewidth', 2.0),
                    marker=style.get('marker'),
                    markersize=style.get('markersize', 5),
                    markevery=max(1, len(budget_factors) // 8),
                    alpha=style.get('alpha', 1.0),
                    zorder=style.get('zorder', 5))
        
        ax.set_xlabel('Financial Budget Factor', fontsize=14)
        ax.set_ylabel('Invalid Gradient Rate', fontsize=14)
        ax.set_title(subtitle, fontsize=15, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=0.2, right=2.0)
        ax.set_ylim(bottom=0, top=1.0)
        
        ax.legend(loc='upper right', framealpha=0.9, 
                 edgecolor='gray', fancybox=True, 
                 ncol=1, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 4-panel figure: {output_path}")
    plt.close()


def plot_single_dataset(data, dataset, alpha, output_path):
    """单个数据集的独立图"""
    method_styles = {
        'MFG-RegretNet (ours)': {'color': '#d32f2f', 'linestyle': '-', 'linewidth': 3.5, 'marker': 'o', 'markersize': 8},
        'RegretNet': {'color': '#ff9800', 'linestyle': '--', 'linewidth': 2.5, 'marker': 'v', 'markersize': 6},
        'M-RegretNet': {'color': '#ffc107', 'linestyle': '-.', 'linewidth': 2.5, 'marker': '^', 'markersize': 6},
        'DM-RegretNet': {'color': '#4caf50', 'linestyle': '-', 'linewidth': 2.5, 'marker': 's', 'markersize': 6},
        'CSRA': {'color': '#1976d2', 'linestyle': '-', 'linewidth': 2.2, 'marker': 'p', 'markersize': 6},
        'MFG-Pricing': {'color': '#7b1fa2', 'linestyle': '-', 'linewidth': 2.2, 'marker': 'D', 'markersize': 5},
        'PAC': {'color': '#00838f', 'linestyle': '-', 'linewidth': 2.2, 'marker': 'h', 'markersize': 6},
        'VCG': {'color': '#388e3c', 'linestyle': '-', 'linewidth': 2.2, 'marker': '*', 'markersize': 7},
    }
    
    method_order = ['RegretNet', 'M-RegretNet', 'MFG-RegretNet (ours)', 'DM-RegretNet', 'PAC', 'VCG', 'CSRA', 'MFG-Pricing']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    budget_factors = np.array(data['budget_factors'])
    methods = data['methods']
    
    for method_name in method_order:
        if method_name not in methods:
            continue
        invalid_rates = np.array(methods[method_name])
        if len(invalid_rates) == 0:
            continue
        
        style = method_styles.get(method_name, {})
        ax.plot(budget_factors, invalid_rates, 
                label=method_name,
                color=style.get('color'),
                linestyle=style.get('linestyle'),
                linewidth=style.get('linewidth'),
                marker=style.get('marker'),
                markersize=style.get('markersize', 6),
                markevery=max(1, len(budget_factors) // 10),
                alpha=style.get('alpha', 1.0),
                zorder=10 if method_name == 'MFG-RegretNet (ours)' else 5)
    
    iid_str = "IID" if alpha >= 0.3 else "Non-IID"
    ax.set_xlabel('Financial Budget Factor', fontsize=14)
    ax.set_ylabel('Invalid Gradient Rate', fontsize=14)
    ax.set_title(f'{dataset} Invalid Gradient Rate (α={alpha}, {iid_str})', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fancybox=True, fontsize=11)
    ax.set_xlim(left=0.2, right=2.0)
    ax.set_ylim(bottom=0, top=1.0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary_table(all_data):
    """打印汇总表"""
    print("\n" + "="*100)
    print("Invalid Gradient Rate Summary (at Budget Factor = 1.0)".center(100))
    print("="*100)
    
    for key in sorted(all_data.keys()):
        dataset, alpha = key
        block = all_data[key]
        methods = block['methods']
        budget_factors = np.array(block['budget_factors'])
        
        idx = np.argmin(np.abs(budget_factors - 1.0))
        
        iid_str = "IID" if alpha >= 0.3 else "Non-IID"
        print(f"\n【{dataset}, α={alpha} ({iid_str})】Budget Factor = {budget_factors[idx]:.2f}")
        sorted_methods = sorted(methods.items(), key=lambda x: x[1][idx])
        for method_name, rates in sorted_methods:
            rate_at_1 = rates[idx]
            marker = " ✓" if method_name == 'MFG-RegretNet (ours)' else ""
            print(f"  {method_name:30s}: {rate_at_1:.4f} ({rate_at_1*100:.2f}%){marker}")


def main():
    parser = argparse.ArgumentParser(description="RQ3: Invalid Gradient Rate Experiment")
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "CIFAR10"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1])
    parser.add_argument("--n-trials", type=int, default=200, help="Number of trials per budget factor")
    parser.add_argument("--n-agents", type=int, default=10, help="Number of agents per auction")
    parser.add_argument("--n-budget-points", type=int, default=15, help="Number of budget factor points")
    parser.add_argument("--mfg-ckpt", type=str, default="result/mfg_regretnet_privacy_200_checkpoint.pt")
    parser.add_argument("--regretnet-ckpt", type=str, default="result/regretnet_privacy_200_checkpoint.pt")
    parser.add_argument("--dm-ckpt", type=str, default="result/dm_regretnet_privacy_200_checkpoint.pt")
    parser.add_argument("--mreg-ckpt", type=str, default="model/8-reg_nslkdd_iid.pt")
    parser.add_argument("--out-dir", type=str, default="run/paper_invalid_gradient")
    args = parser.parse_args()
    
    print("="*100)
    print("RQ3: Invalid Gradient Rate 实验 (类似 FL-market.pdf 图8)".center(100))
    print("="*100)
    print(f"Datasets: {args.datasets}")
    print(f"Alphas: {args.alphas}")
    print(f"Trials per budget: {args.n_trials}")
    print(f"Budget points: {args.n_budget_points}")
    print("="*100)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Budget factors
    budget_factors = np.linspace(0.2, 2.0, args.n_budget_points)
    
    # 运行所有配置
    all_data = {}
    for dataset in args.datasets:
        for alpha in args.alphas:
            key = (dataset, alpha)
            
            results = run_experiment_single_config(
                dataset=dataset,
                alpha=alpha,
                n_trials=args.n_trials,
                n_agents=args.n_agents,
                budget_factors=budget_factors,
                mfg_ckpt=args.mfg_ckpt,
                regretnet_ckpt=args.regretnet_ckpt,
                dm_ckpt=args.dm_ckpt,
                mreg_ckpt=args.mreg_ckpt,
            )
            
            data = {
                'budget_factors': budget_factors.tolist(),
                'methods': results,
                'n_trials': args.n_trials,
            }
            all_data[key] = data
            
            # 保存单个配置
            out_file = os.path.join(args.out_dir, f"{dataset}_alpha{alpha}_invalid_rate.json")
            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n✓ Saved: {out_file}")
            
            # 生成单独的图
            iid_str = "iid" if alpha >= 0.3 else "niid"
            plot_single_dataset(data, dataset, alpha, 
                              os.path.join(args.out_dir, f"invalid_rate_{dataset.lower()}_{iid_str}.png"))
    
    # 绘制 4 子图
    print("\n" + "="*100)
    print("Generating combined 4-panel figure...")
    plot_invalid_gradient_4panel(all_data, os.path.join(args.out_dir, "invalid_rate_4panels.png"))
    
    # 打印汇总表
    print_summary_table(all_data)
    
    print("\n" + "="*100)
    print("✅ Invalid Gradient Rate 实验完成！".center(100))
    print("="*100)
    print(f"\n输出目录: {args.out_dir}/")
    print("\n一键运行命令:")
    print(f"  python {os.path.basename(__file__)} --n-trials 200")


if __name__ == "__main__":
    main()

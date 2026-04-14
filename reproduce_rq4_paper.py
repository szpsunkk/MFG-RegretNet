#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 实验：MNIST/CIFAR-10 数据集上的FL精度对比
论文方法: MFG-RegretNet vs. Baselines (PAC, VCG, CSRA, MFG-Pricing, RegretNet, DM-RegretNet)

生成论文所需的4张图:
  1. MNIST IID (Dirichlet α=0.5)
  2. MNIST Non-IID (Dirichlet α=0.1)  
  3. CIFAR-10 IID (Dirichlet α=0.5)
  4. CIFAR-10 Non-IID (Dirichlet α=0.1)

运行方式:
  python reproduce_rq4_paper.py --datasets MNIST CIFAR10 --n-seeds 3 --n-rounds 80
"""
from __future__ import division, print_function

import argparse
import copy
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.multiprocessing as mp

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from datasets_fl_benchmark import (
    dirichlet_split,
    generate_privacy_paper_bids,
    load_cifar10,
    load_mnist,
)
from FL import Arguments, CIFAR10Net, Net, laplace_noise_like, test
from utils import generate_max_cost

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

DEVICE = torch.device("cpu")  # 多进程用CPU避免GPU冲突


# ============================================================================
# FL Round Execution
# ============================================================================

def _local_xy_tensors(full_train, idxs_client_dict, n_agents, device):
    """提取每个client的(X, Y)数据"""
    out = []
    for i in range(n_agents):
        idxs = sorted(idxs_client_dict[i])
        if len(idxs) == 0:
            idxs = [0]  # dummy
        sub = torch.utils.data.Subset(full_train, idxs)
        loader = Data.DataLoader(sub, batch_size=len(sub), shuffle=False)
        X, Y = next(iter(loader))
        X = X.to(device).float() if X.dtype in (torch.float32, torch.float64) else X.to(device).long()
        Y = Y.to(device).long()
        out.append((X, Y))
    return out


def _fed_round_laplace(model, fl_args, plosses, weights, local_xy, device):
    """
    一轮FL: Laplace噪声+梯度聚合
    多个local epochs + mini-batch
    """
    plosses = plosses.view(-1).float()
    weights = weights.view(-1).float()
    n_agents = plosses.shape[0]
    global_model = copy.deepcopy(model).to(device)
    if (plosses <= 0).all():
        return model, 0.0
    
    le = max(1, int(getattr(fl_args, "local_epochs", 1)))
    lbs = max(1, int(getattr(fl_args, "local_batch_size", 64)))
    losses = []
    sensi = float(fl_args.sensi)
    flr = float(fl_args.lr)
    
    for i in range(n_agents):
        if plosses[i] <= 0:
            continue
        epsi = plosses[i].clamp(min=1e-6)
        w_i = float(weights[i].item())
        X, Y = local_xy[i]
        n_samples = X.size(0)
        bs = max(1, min(lbs, n_samples))
        loader = Data.DataLoader(
            Data.TensorDataset(X, Y),
            batch_size=bs,
            shuffle=True,
            drop_last=False,
        )
        optimizer = optim.SGD(global_model.parameters(), lr=fl_args.lr)
        for _ep in range(le):
            for xb, yb in loader:
                global_model.train()
                optimizer.zero_grad()
                pred_Y = global_model(xb)
                loss = nn.CrossEntropyLoss()(pred_Y, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(global_model.parameters(), fl_args.L, 1.0)
                with torch.no_grad():
                    for param in global_model.parameters():
                        g = param.grad
                        if g is None:
                            continue
                        noise = laplace_noise_like(g, sensi / epsi)
                        noised_grad = g + noise
                        param.data.sub_(noised_grad * (w_i * flr))
                losses.append(loss.item())
    return global_model, float(np.mean(losses)) if losses else 0.0


# ============================================================================
# Auction Mechanisms
# ============================================================================

def _load_auc(path):
    if not path or not os.path.isfile(path):
        return None
    from experiments import load_auc_model
    return load_auc_model(path)


def _run_auction_round(reports_1row, budget_1, trade_mech, auc_model, device, client_active_mask):
    """单轮拍卖: 输入 (1, n_agents, 3), 输出 plosses, weights"""
    from experiments import auction
    
    reports = reports_1row.to(device).float()
    budget = budget_1.to(device).float()
    n_agents = reports.shape[1]
    mask = torch.tensor(client_active_mask, device=device, dtype=torch.bool)
    
    # 特殊处理: Uniform-DP, No-DP
    if trade_mech[0] == "Uniform-DP":
        eps_u = float(trade_mech[4]) if len(trade_mech) > 4 else 2.555
        plosses = torch.full((1, n_agents), eps_u, device=device)
        plosses[0, ~mask] = 0.0
        from aggregation import aggr_batch
        weights = aggr_batch(plosses, reports[:, :, -1], method="ConvlAggr")
        return plosses[0].detach(), weights[0].detach()
    
    if trade_mech[0] == "No-DP":
        plosses = torch.full((1, n_agents), 1e6, device=device)
        plosses[0, ~mask] = 0.0
        from aggregation import aggr_batch
        weights = aggr_batch(plosses, reports[:, :, -1], method="ConvlAggr")
        return plosses[0].detach(), weights[0].detach()
    
    tm = [trade_mech[0], trade_mech[1], trade_mech[2], trade_mech[3]]
    auc = auc_model.to(device) if auc_model is not None else None
    if tm[0] in ("PAC", "VCG", "CSRA", "MFG-Pricing", "All-in", "FairQuery"):
        auc = None
    plosses, weights = auction(reports, budget, tm, model=auc)
    p = plosses.clone()
    p[0, ~mask] = 0.0
    return p[0].detach(), weights[0].detach()


# ============================================================================
# 单次实验运行
# ============================================================================

def run_one_setting(
    dataset_name,
    alpha,
    seed,
    n_agents,
    n_rounds,
    rnd_step,
    min_budget_rate,
    max_budget_rate,
    mfg_ckpt,
    regretnet_ckpt,
    dm_ckpt,
    uniform_eps,
    include_pac,
    include_csra,
    include_mfg_pricing,
    local_epochs=2,
    local_batch_size=64,
    fl_lr=None,
):
    """运行一个(dataset, alpha, seed)的FL实验, 返回所有方法的test_acc曲线"""
    device = DEVICE
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载数据集
    if dataset_name.upper() == "MNIST":
        train_full, test_data = load_mnist()
        model_factory = lambda: Net().to(device)
    elif dataset_name.upper() == "CIFAR10":
        train_full, test_data = load_cifar10()
        model_factory = lambda: CIFAR10Net().to(device)
    else:
        raise ValueError("dataset must be MNIST or CIFAR10")
    
    n_classes = 10
    idxs = dirichlet_split(train_full, n_agents, n_classes, alpha=alpha, min_size=10, seed=seed)
    client_active = [len(idxs[i]) > 0 for i in range(n_agents)]
    local_xy = _local_xy_tensors(train_full, idxs, n_agents, device)
    
    # FL参数
    fl_args = Arguments()
    fl_args.rounds = n_rounds
    fl_args.local_epochs = int(local_epochs)
    fl_args.local_batch_size = int(local_batch_size)
    if fl_lr is not None:
        fl_args.lr = float(fl_lr)
    elif fl_args.local_epochs >= 2:
        fl_args.lr = 0.02 if dataset_name.upper() == "MNIST" else 0.045
    else:
        fl_args.lr = 0.01 if dataset_name.upper() == "MNIST" else 0.05
    fl_args.L = 1.0
    fl_args.sensi = 2.0 * fl_args.L
    fl_args.device = device
    fl_args.no_cuda = True
    
    # 每轮随机预算
    rng_b = np.random.RandomState(seed + 999)
    budget_rates = torch.tensor(
        [rng_b.uniform(min_budget_rate, max_budget_rate) for _ in range(n_rounds)],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1)
    
    # 生成每轮拍卖输入
    rounds_data = []
    for r in range(n_rounds):
        rep = generate_privacy_paper_bids(n_agents, 1, 1, seed=seed * 100000 + r)
        if rep.dim() == 2:
            rep = rep.unsqueeze(0)
        rep = rep.to(device)
        mc = generate_max_cost(rep)
        budget = mc * budget_rates[r]
        rounds_data.append((rep, budget))
    
    # 加载拍卖模型
    mfg = _load_auc(mfg_ckpt)
    reg = _load_auc(regretnet_ckpt)
    dm = _load_auc(dm_ckpt)
    
    # 定义所有方法
    mechs = []
    if mfg is not None:
        mechs.append(("MFG-RegretNet (ours)", ["MFG-RegretNet", "ConvlAggr", mfg_ckpt, 1], mfg))
    else:
        print(f"[Warning] MFG-RegretNet ckpt missing: {mfg_ckpt}")
    
    if include_csra:
        mechs.append(("CSRA", ["CSRA", "ConvlAggr", "", 1], None))
    if include_mfg_pricing:
        mechs.append(("MFG-Pricing", ["MFG-Pricing", "ConvlAggr", "", 1], None))
    if include_pac:
        mechs.append(("PAC", ["PAC", "ConvlAggr", "", 1], None))
    mechs.append(("VCG", ["VCG", "ConvlAggr", "", 1], None))
    if reg is not None:
        mechs.append(("RegretNet", ["RegretNet", "ConvlAggr", regretnet_ckpt, 1], reg))
    if dm is not None:
        mechs.append(("DM-RegretNet", ["DM-RegretNet", "ConvlAggr", dm_ckpt, 8], dm))
    
    uni = ["Uniform-DP", "ConvlAggr", "", 1, uniform_eps]
    mechs.append(("Uniform-DP", uni, None))
    nodp = ["No-DP", "ConvlAggr", "", 1]
    mechs.append(("No-DP (upper)", nodp, None))
    
    # 运行每个方法
    results = {}
    round_log_idx = []
    for rnd in range(n_rounds):
        if rnd == 0 or (rnd + 1) % rnd_step == 0 or rnd == n_rounds - 1:
            round_log_idx.append(rnd + 1)
    
    for label, tm_base, auc_m in mechs:
        print(f"  [{dataset_name} α={alpha} seed={seed}] {label} ...", flush=True)
        tm = list(tm_base)
        model = model_factory()
        accs = []
        auc_copy = auc_m
        
        for rnd in range(n_rounds):
            rep, budget = rounds_data[rnd]
            plosses, weights = _run_auction_round(rep, budget, tm, auc_copy, device, client_active)
            model, loss = _fed_round_laplace(model, fl_args, plosses, weights, local_xy, device)
            
            if rnd + 1 in round_log_idx:
                acc = test(model, test_data, fl_args, rnd)
                accs.append(acc)
        
        results[label] = {
            "test_acc": accs,
            "rounds_logged": round_log_idx,
        }
    
    return results, round_log_idx


# ============================================================================
# 绘图
# ============================================================================

def plot_accuracy_4panel(all_data, output_path, title="RQ4: FL Test Accuracy vs Training Rounds"):
    """
    绘制4子图: MNIST IID/Non-IID, CIFAR-10 IID/Non-IID
    all_data: dict[(dataset, alpha)] = {'methods': {name: test_acc_list}, 'rounds': [...]}
    """
    # 方法颜色和样式 (确保MFG-RegretNet最突出)
    method_styles = {
        'MFG-RegretNet (ours)': {'color': '#d32f2f', 'linestyle': '-', 'linewidth': 3.0, 'marker': 'o', 'markersize': 6, 'zorder': 10},
        'RegretNet': {'color': '#ff9800', 'linestyle': '--', 'linewidth': 2.0, 'marker': 'v', 'markersize': 5, 'zorder': 5},
        'DM-RegretNet': {'color': '#ffc107', 'linestyle': '-.', 'linewidth': 2.0, 'marker': '^', 'markersize': 5, 'zorder': 5},
        'CSRA': {'color': '#1976d2', 'linestyle': '-', 'linewidth': 2.0, 'marker': 's', 'markersize': 5, 'zorder': 4},
        'MFG-Pricing': {'color': '#7b1fa2', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'D', 'markersize': 4, 'zorder': 4},
        'PAC': {'color': '#00838f', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'p', 'markersize': 5, 'zorder': 4},
        'VCG': {'color': '#388e3c', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'h', 'markersize': 5, 'zorder': 4},
        'Uniform-DP': {'color': '#757575', 'linestyle': '-.', 'linewidth': 1.8, 'marker': 'x', 'markersize': 6, 'zorder': 3},
        'No-DP (upper)': {'color': '#000000', 'linestyle': ':', 'linewidth': 2.5, 'marker': None, 'markersize': 0, 'alpha': 0.5, 'zorder': 2}
    }
    
    # 方法显示顺序
    method_order = ['MFG-RegretNet (ours)', 'DM-RegretNet', 'RegretNet', 'CSRA', 'MFG-Pricing', 'PAC', 'VCG', 'Uniform-DP', 'No-DP (upper)']
    
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
            print(f"Warning: {dataset_key} not found in data")
            continue
        
        block = all_data[dataset_key]
        rounds = np.array(block['rounds'])
        methods = block['methods']
        
        for method_name in method_order:
            if method_name not in methods:
                continue
            
            mean_acc = np.array(methods[method_name])
            if len(mean_acc) == 0:
                continue
            
            min_len = min(len(rounds), len(mean_acc))
            rounds_plot = rounds[:min_len]
            mean_plot = mean_acc[:min_len]
            
            style = method_styles.get(method_name, {})
            
            ax.plot(rounds_plot, mean_plot, 
                    label=method_name,
                    color=style.get('color', '#333333'),
                    linestyle=style.get('linestyle', '-'),
                    linewidth=style.get('linewidth', 2.0),
                    marker=style.get('marker'),
                    markersize=style.get('markersize', 5),
                    markevery=max(1, len(rounds_plot) // 8),
                    alpha=style.get('alpha', 1.0),
                    zorder=style.get('zorder', 5))
        
        ax.set_xlabel('Training Round', fontsize=14)
        ax.set_ylabel('Test Accuracy', fontsize=14)
        ax.set_title(subtitle, fontsize=15, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=0)
        ax.legend(loc='lower right', framealpha=0.9, 
                 edgecolor='gray', fancybox=True, 
                 ncol=1, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 4-panel figure: {output_path}")
    plt.close()


def print_final_accs(all_data):
    """打印最终精度表格"""
    print("\n" + "="*80)
    print("RQ4 Final Test Accuracy (last round)".center(80))
    print("="*80)
    
    for key in sorted(all_data.keys()):
        dataset, alpha = key
        block = all_data[key]
        methods = block['methods']
        print(f"\n【{dataset}, α={alpha}】")
        for method_name in sorted(methods.keys()):
            accs = methods[method_name]
            if len(accs) > 0:
                final_acc = accs[-1]
                print(f"  {method_name:30s}: {final_acc:.4f}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ4: FL Accuracy Comparison (MNIST/CIFAR-10)")
    parser.add_argument("--datasets", nargs="+", default=["MNIST", "CIFAR10"], help="Datasets to run")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1], help="Dirichlet alphas")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--n-agents", type=int, default=10, help="Number of clients per round")
    parser.add_argument("--n-rounds", type=int, default=80, help="Total FL rounds")
    parser.add_argument("--rnd-step", type=int, default=5, help="Logging interval")
    parser.add_argument("--min-budget", type=float, default=0.8, help="Min budget rate")
    parser.add_argument("--max-budget", type=float, default=0.8, help="Max budget rate")
    parser.add_argument("--uniform-eps", type=float, default=2.555, help="Uniform-DP epsilon")
    parser.add_argument("--mfg-ckpt", type=str, default="result/mfg_regretnet_privacy_200_checkpoint.pt", help="MFG-RegretNet checkpoint")
    parser.add_argument("--regretnet-ckpt", type=str, default="result/regretnet_privacy_200_checkpoint.pt", help="RegretNet checkpoint")
    parser.add_argument("--dm-ckpt", type=str, default="result/dm_regretnet_privacy_200_checkpoint.pt", help="DM-RegretNet checkpoint")
    parser.add_argument("--include-pac", action="store_true", help="Include PAC baseline")
    parser.add_argument("--include-csra", action="store_true", help="Include CSRA baseline")
    parser.add_argument("--include-mfg-pricing", action="store_true", help="Include MFG-Pricing baseline")
    parser.add_argument("--out-dir", type=str, default="run/paper_rq4", help="Output directory")
    args = parser.parse_args()
    
    print("="*80)
    print("RQ4 实验: 论文精度对比 (MNIST/CIFAR-10)".center(80))
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"Alphas: {args.alphas}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Rounds: {args.n_rounds}, rnd_step={args.rnd_step}")
    print(f"MFG checkpoint: {args.mfg_ckpt}")
    print("="*80 + "\n")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 运行所有配置
    all_results = {}
    for dataset in args.datasets:
        for alpha in args.alphas:
            key = (dataset, alpha)
            print(f"\n{'='*60}")
            print(f"Running: {dataset}, α={alpha}")
            print(f"{'='*60}")
            
            # 多seed平均
            acc_curves = {}
            for seed in range(args.n_seeds):
                print(f"\n--- Seed {seed} ---")
                results_seed, rounds_logged = run_one_setting(
                    dataset_name=dataset,
                    alpha=alpha,
                    seed=seed,
                    n_agents=args.n_agents,
                    n_rounds=args.n_rounds,
                    rnd_step=args.rnd_step,
                    min_budget_rate=args.min_budget,
                    max_budget_rate=args.max_budget,
                    mfg_ckpt=args.mfg_ckpt,
                    regretnet_ckpt=args.regretnet_ckpt,
                    dm_ckpt=args.dm_ckpt,
                    uniform_eps=args.uniform_eps,
                    include_pac=args.include_pac,
                    include_csra=args.include_csra,
                    include_mfg_pricing=args.include_mfg_pricing,
                    local_epochs=2,
                    local_batch_size=64,
                    fl_lr=None,
                )
                for method_name, method_data in results_seed.items():
                    if method_name not in acc_curves:
                        acc_curves[method_name] = []
                    acc_curves[method_name].append(method_data['test_acc'])
            
            # 平均
            avg_acc_curves = {}
            for method_name, curves in acc_curves.items():
                avg_acc_curves[method_name] = np.mean(curves, axis=0).tolist()
            
            all_results[key] = {
                'rounds': rounds_logged,
                'methods': avg_acc_curves,
            }
            
            # 保存单个配置的结果
            out_file = os.path.join(args.out_dir, f"{dataset}_alpha{alpha}_results.json")
            with open(out_file, 'w') as f:
                json.dump(all_results[key], f, indent=2)
            print(f"\n✓ Saved: {out_file}")
    
    # 绘制4子图
    plot_accuracy_4panel(all_results, os.path.join(args.out_dir, "rq4_paper_4panels.png"))
    
    # 打印最终精度
    print_final_accs(all_results)
    
    print("\n" + "="*80)
    print("✅ RQ4 实验完成！".center(80))
    print("="*80)
    print(f"输出目录: {args.out_dir}/")


if __name__ == "__main__":
    main()

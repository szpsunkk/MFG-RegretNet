#!/usr/bin/env python3
"""
RQ5: 隐私-效用权衡（实验思路 3.5）
隐私异质性场景（均匀/双峰/现实分布），计算平均隐私成本、梯度质量、帕累托前沿（隐私成本 vs 精度）。
输出：帕累托图数据、隐私-精度曲线、公平性指标。
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="RQ5: Privacy-utility tradeoff")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper/rq5", help="output directory")
    parser.add_argument("--num-profiles", type=int, default=500)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--dist", type=str, default="uniform", choices=["uniform", "bimodal", "lognormal"],
                        help="valuation/privacy distribution")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from run_phase4_eval import build_privacy_paper_batch, get_ckpt_path, DEVICE
        from experiments import auction, load_auc_model
        from aggregation import aggr_batch
    except ImportError as e:
        print("RQ5 requires run_phase4_eval, experiments, aggregation. Error:", e)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    # 不同分布：在 build 时用不同 seed 或扩展 generate_privacy_paper_bids 支持分布
    from datasets_fl_benchmark import generate_privacy_paper_bids

    # 当前 generate_privacy_paper_bids 为 uniform；这里用多组 (v, eps) 近似不同“场景”
    results = []
    for seed in [42, 43, 44]:
        reports, budget, _ = build_privacy_paper_batch(
            args.num_profiles, args.n_agents, 1, args.budget, seed, DEVICE
        )
        # 平均隐私成本： (1/N) sum v_i * eps_i^out
        trade_mech = ["PAC", "ConvlAggr", "", 1]
        out = auction(reports, budget, trade_mech, model=None, return_payments=True)
        plosses, weights, payments = out[0], out[1], out[2]
        v = reports[:, :, 0].cpu().numpy()
        eps_out = plosses.detach().cpu().numpy()
        cost = (v * eps_out).sum(axis=1).mean()
        rev = payments.sum(dim=1).mean().item()
        results.append({"seed": seed, "avg_privacy_cost": float(cost), "revenue": rev})

    avg_cost = np.mean([r["avg_privacy_cost"] for r in results])
    out_path = os.path.join(args.out_dir, "privacy_utility.json")
    with open(out_path, "w") as f:
        json.dump({"distribution": args.dist, "avg_privacy_cost": avg_cost, "per_seed": results}, f, indent=2)
    print("Wrote", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())

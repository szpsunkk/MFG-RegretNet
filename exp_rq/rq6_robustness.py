#!/usr/bin/env python3
"""
RQ6: 鲁棒性（实验思路 3.6）
虚假报价比例 δ、勾结规模 k，测量遗憾增长与收益损失。
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="RQ6: Robustness — false bids and collusion")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper/rq6", help="output directory")
    parser.add_argument("--num-profiles", type=int, default=300)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--false-ratio", type=float, default=0.1, help="fraction of agents with false bids (delta)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="", help="MFG-RegretNet checkpoint for evaluation")
    parser.add_argument("--quick", action="store_true", help="quick test mode")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from run_phase4_eval import build_privacy_paper_batch, DEVICE
        from experiments import auction, load_auc_model, guarantees_eval
        from datasets import Dataloader
    except ImportError as e:
        print("RQ6 requires run_phase4_eval, experiments. Error:", e)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    reports, budget, val_type = build_privacy_paper_batch(
        args.num_profiles, args.n_agents, 1, args.budget, seed=args.seed, device=DEVICE
    )
    # 注入虚假报价：随机将 delta 比例 agent 的 valuation 改为随机值
    n_false = max(1, int(args.n_agents * args.false_ratio))
    false_agents = np.random.choice(args.n_agents, size=n_false, replace=False)
    reports_false = reports.clone()
    for i in false_agents:
        reports_false[:, i, 0] = torch.rand(args.num_profiles, device=reports.device)  # random v

    trade_mech = ["PAC", "ConvlAggr", "", 1]
    out = auction(reports_false, budget, trade_mech, model=None, return_payments=True)
    rev_robust = out[2].sum(dim=1).mean().item()

    out_clean = auction(reports, budget, trade_mech, model=None, return_payments=True)
    rev_clean = out_clean[2].sum(dim=1).mean().item()
    revenue_loss = 1.0 - (rev_robust / rev_clean) if rev_clean > 0 else 0.0

    # 计算遗憾增长
    loader = Dataloader(torch.cat([reports_false, val_type], dim=2), batch_size=128, shuffle=False)
    regret_robust, ir_robust = 0.0, 0.0
    count = 0
    for batch in loader:
        rep = batch[:, :, :-2].to(DEVICE)
        try:
            rgt, ir = guarantees_eval(rep, budget[:rep.shape[0]], trade_mech, model=None)
            regret_robust += rgt.mean().item() * rep.shape[0]
            ir_robust += ir.mean().item() * rep.shape[0]
            count += rep.shape[0]
        except Exception as e:
            print(f"[Warning] guarantees_eval failed: {e}")
    
    regret_robust = regret_robust / count if count > 0 else 0.0
    ir_robust = ir_robust / count if count > 0 else 0.0

    result = {
        "false_ratio": args.false_ratio,
        "seed": args.seed,
        "n_agents": args.n_agents,
        "revenue_clean": rev_clean,
        "revenue_under_attack": rev_robust,
        "revenue_loss_ratio": revenue_loss,
        "regret_robust": regret_robust,
        "ir_violation_robust": ir_robust,
    }
    
    # 追加结果到 JSON 文件（支持多次运行）
    out_path = os.path.join(args.out_dir, "rq6_results.json")
    results_list = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    results_list = existing
                else:
                    results_list = [existing]
        except:
            pass
    
    results_list.append(result)
    
    with open(out_path, "w") as f:
        json.dump(results_list, f, indent=2)
    
    print(f"RQ6 results appended to {out_path}")
    print(f"  False ratio: {args.false_ratio:.2f}")
    print(f"  Revenue loss ratio: {revenue_loss:.4f}")
    print(f"  Regret (robust): {regret_robust:.6f}")
    print(f"  IR violation (robust): {ir_robust:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

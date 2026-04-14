#!/usr/bin/env python3
"""
消融研究（实验思路 3.7）
1) MFG 组件：移除 b_MFG（RegretNet）vs 完整 MFG-RegretNet
2) 增强拉格朗日：不同 ρ 的影响
输出：消融表格（regret, revenue, time 等）
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Ablation study: MFG component, AL rho")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper/ablation", help="output directory")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--num-profiles", type=int, default=500)
    parser.add_argument("--regretnet-ckpt", type=str, default="", help="RegretNet (no b_MFG)")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="", help="MFG-RegretNet (with b_MFG)")
    args = parser.parse_args()

    try:
        from run_phase4_eval import (
            build_privacy_paper_batch, rq1_guarantees_privacy_paper, rq3_revenue_privacy_paper,
            get_ckpt_path, DEVICE,
        )
    except ImportError as e:
        print("Ablation requires run_phase4_eval. Error:", e)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [42, 43]

    trade_mech_ls = [
        ["PAC", "ConvlAggr", "", 1],
        ["VCG", "ConvlAggr", "", 1],
    ]
    name_to_display = {}
    if args.regretnet_ckpt and os.path.isfile(args.regretnet_ckpt):
        trade_mech_ls.append(["RegretNet", "ConvlAggr", args.regretnet_ckpt, 1])
        name_to_display["RegretNet"] = "RegretNet (no b_MFG)"
    if args.mfg_regretnet_ckpt and os.path.isfile(args.mfg_regretnet_ckpt):
        trade_mech_ls.append(["MFG-RegretNet", "ConvlAggr", args.mfg_regretnet_ckpt, 1])
        name_to_display["MFG-RegretNet"] = "MFG-RegretNet (with b_MFG)"

    rq1 = rq1_guarantees_privacy_paper(
        trade_mech_ls, args.n_agents, 1, args.budget, args.num_profiles, seeds, batch_size=128
    )
    rq3 = rq3_revenue_privacy_paper(
        trade_mech_ls, args.n_agents, 1, args.budget, args.num_profiles, seeds
    )

    rows = []
    for (name, mean_r, mean_ir) in rq1:
        display_name = name_to_display.get(name, name)
        row = {"mechanism": display_name, "mean_regret": mean_r, "mean_ir_violation": mean_ir}
        for r3 in rq3:
            if r3.get("mechanism") == name:
                row["mean_revenue"] = r3.get("mean_revenue")
                row["std_revenue"] = r3.get("std_revenue")
                row["bf_rate"] = r3.get("bf_rate")
                row["revenue_efficiency"] = r3.get("revenue_efficiency")
                row["std_revenue_efficiency"] = r3.get("std_revenue_efficiency")
                row["mean_social_welfare"] = r3.get("mean_social_welfare")
                row["std_social_welfare"] = r3.get("std_social_welfare")
                break
        rows.append(row)

    out_path = os.path.join(args.out_dir, "ablation_table.json")
    with open(out_path, "w") as f:
        json.dump({"ablation": rows}, f, indent=2)
    print("Wrote", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())

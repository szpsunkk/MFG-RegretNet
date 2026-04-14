#!/usr/bin/env python3
"""
RQ1 补充：遗憾随「误报优化步数」(PGA rounds) 变化曲线。
实验思路中的「遗憾随轮数收敛」在机制评估中等价为：misreport_iter 增大时，估计的 ex-post regret 趋于稳定。
输出：figure_rq1_regret_vs_pga_rounds.png
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-items", type=int, default=1)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--num-profiles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--regretnet-ckpt", type=str, default="")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    parser.add_argument("--misreport-iters", type=str, default="1,3,5,10,15,20,25,35,50")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper/rq1")
    args = parser.parse_args()

    iters = [int(x.strip()) for x in args.misreport_iters.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    # 未传或路径无效时自动选 result/mfg_regretnet_privacy_*_checkpoint.pt（epoch 最大）
    if not args.mfg_regretnet_ckpt or not os.path.isfile(args.mfg_regretnet_ckpt):
        import glob
        import re
        cand = glob.glob("result/mfg_regretnet_privacy_*_checkpoint.pt")

        def _epoch(p):
            m = re.search(r"privacy_(\d+)_checkpoint", p)
            return int(m.group(1)) if m else 0

        cand = sorted(cand, key=_epoch)
        if cand:
            args.mfg_regretnet_ckpt = cand[-1]
            print("[INFO] MFG ckpt (auto):", args.mfg_regretnet_ckpt)

    if not args.regretnet_ckpt or not os.path.isfile(args.regretnet_ckpt):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt
            _r = resolve_regretnet_ckpt(args.n_agents, args.n_items)
            if _r:
                args.regretnet_ckpt = _r
                print("[INFO] RegretNet ckpt (auto):", _r)
        except Exception:
            pass

    from run_phase4_eval import build_privacy_paper_batch
    from experiments import guarantees_eval, DEVICE
    from datasets import Dataloader
    import torch

    reports, bud, val_type = build_privacy_paper_batch(
        args.num_profiles, args.n_agents, args.n_items, args.budget, args.seed, DEVICE
    )
    loader = Dataloader(
        torch.cat([reports, val_type], dim=2),
        batch_size=args.batch_size,
        shuffle=False,
    )

    curves = {}

    def eval_mech(name, path):
        if not path or not os.path.isfile(path):
            return None
        mech = (name, "ConvlAggr", path, args.n_items)
        regrets = []
        for R in iters:
            rs = []
            for batch in loader:
                rep = batch[:, :, :-2].to(DEVICE)
                vt = batch[:, :, -2:].to(DEVICE)
                b = args.budget * torch.ones(rep.shape[0], 1, device=DEVICE)
                r, _ = guarantees_eval(rep, b, vt, mech, misreport_iter=R, lr=1e-1)
                rs.append(r.detach().cpu().numpy().ravel())
            regrets.append(float(np.concatenate(rs).mean()))
        return regrets

    if args.regretnet_ckpt and os.path.isfile(args.regretnet_ckpt):
        c = eval_mech("RegretNet", args.regretnet_ckpt)
        if c:
            curves["RegretNet"] = c
    if args.mfg_regretnet_ckpt and os.path.isfile(args.mfg_regretnet_ckpt):
        c = eval_mech("MFG-RegretNet", args.mfg_regretnet_ckpt)
        if c:
            curves["MFG-RegretNet"] = c

    if not curves:
        print("No valid neural checkpoints; skip convergence curve.")
        return 0

    with open(os.path.join(args.out_dir, "rq1_convergence_curve.json"), "w") as f:
        json.dump({"misreport_iters": iters, "mean_regret_by_iter": curves}, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#ff7f0e", "#d62728"]
        for i, (name, ys) in enumerate(curves.items()):
            ax.plot(iters, ys, marker="o", label=name, color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel("PGA steps (attack refinement)")
        ax.set_ylabel("Mean ex-post regret (normalized)")
        ax.set_title(
            "RQ1: Regret vs PGA steps (more steps → stronger attack; flatten = converged est.)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.out_dir, "figure_rq1_regret_vs_pga_rounds.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print("Wrote", p)
    except ImportError:
        print("matplotlib not installed")

    return 0


if __name__ == "__main__":
    sys.exit(main())

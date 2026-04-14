#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 可选图 D：各方法下「单客户端×单 profile」归一化遗憾的分布（箱线 / 小提琴）。

纵轴为每个 agent-slot 的事后归一化遗憾样本，用于对比长尾：Ours 是否不仅均值更低、高分位也更短。

输出：figure_rq1_paper_regret_distribution.png、rq1_figure_d.json
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

import numpy as np


def _collect_baseline(mech_name, loader, budget, v_grid, eps_grid, device):
    from exp_rq.guarantees_eval_baselines import guarantees_eval_procurement_baseline
    import torch

    regs = []
    for batch in loader:
        rep = batch[:, :, :-2].to(device)
        b = budget * torch.ones(rep.shape[0], 1, device=device)
        r, _ = guarantees_eval_procurement_baseline(
            rep, b, mech_name, v_grid_n=v_grid, eps_grid_n=eps_grid
        )
        regs.append(r.detach().cpu().numpy().ravel())
    return np.concatenate(regs)


def _collect_neural(path, loader, budget, device, n_items, misreport_iter):
    from experiments import load_auc_model, DEVICE
    from regretnet import MFGRegretNet
    from utils import optimize_misreports, tiled_misreport_util, calc_agent_util, allocs_to_plosses
    import torch

    model = load_auc_model(path).to(device)
    model.eval()
    cost_from_plosses = isinstance(model, MFGRegretNet) and n_items == 1
    regs = []
    for batch in loader:
        rep = batch[:, :, :-2].to(device)
        vt = batch[:, :, -2:].to(device)
        b = budget * torch.ones(rep.shape[0], 1, device=device)
        misreports = rep.clone().detach()
        optimize_misreports(
            model, rep, misreports, budget=b, val_type=vt,
            misreport_iter=misreport_iter, lr=1e-1, train=False,
            instantiation=True, cost_from_plosses=cost_from_plosses,
        )
        allocs, payments = model((rep, b))
        vals = rep[:, :, :-2]
        sizes = rep[:, :, -1]
        if cost_from_plosses:
            pbudgets = rep[:, :, -2].view(-1, rep.shape[1])
            plosses = allocs_to_plosses(allocs, pbudgets)
            costs = vals[:, :, 0] * plosses
        else:
            costs = torch.sum(allocs * vals, dim=2) * sizes
        truthful_util = calc_agent_util(
            rep, allocs, payments, instantiation=True, cost_from_plosses=cost_from_plosses
        )
        untruthful_util = tiled_misreport_util(
            misreports, rep, model, b, val_type=vt, instantiation=True, cost_from_plosses=cost_from_plosses
        )
        regrets = torch.clamp(untruthful_util - truthful_util, min=0)
        costs_safe = costs.clamp(min=1e-10)
        rnorm = regrets / costs_safe
        regs.append(rnorm.detach().cpu().numpy().ravel())
    return np.concatenate(regs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-items", type=int, default=1)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--num-profiles", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--misreport-iter", type=int, default=25)
    ap.add_argument("--baseline-v-grid", type=int, default=13)
    ap.add_argument("--baseline-eps-grid", type=int, default=6)
    ap.add_argument("--max-samples-plot", type=int, default=12000, help="绘图时每方法最多随机子样本数（加速）")
    ap.add_argument("--violin", action="store_true", help="小提琴图（需较多样本；否则箱线更稳）")
    ap.add_argument("--regretnet-ckpt", type=str, default="")
    ap.add_argument("--dm-regretnet-ckpt", type=str, default="")
    ap.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/rq1")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.regretnet_ckpt or not os.path.isfile(args.regretnet_ckpt):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt
            r = resolve_regretnet_ckpt(args.n_agents, args.n_items)
            if r:
                args.regretnet_ckpt = r
        except Exception:
            pass
    if not args.dm_regretnet_ckpt or not os.path.isfile(args.dm_regretnet_ckpt):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt
            r = resolve_dm_regretnet_ckpt(args.n_agents, args.n_items)
            if r:
                args.dm_regretnet_ckpt = r
        except Exception:
            pass
    if not args.mfg_regretnet_ckpt or not os.path.isfile(args.mfg_regretnet_ckpt):
        import glob
        import re
        cand = glob.glob("result/mfg_regretnet_privacy_*_checkpoint.pt")
        cand = sorted(cand, key=lambda p: int(re.search(r"_(\d+)_checkpoint", p).group(1)) if re.search(r"_(\d+)_checkpoint", p) else 0)
        if cand:
            args.mfg_regretnet_ckpt = cand[-1]

    from run_phase4_eval import build_privacy_paper_batch
    from experiments import DEVICE
    from datasets import Dataloader
    import torch

    def make_loader():
        rep, _, vt = build_privacy_paper_batch(
            args.num_profiles, args.n_agents, args.n_items, args.budget, args.seed, DEVICE
        )
        return Dataloader(torch.cat([rep, vt], dim=2), batch_size=args.batch_size, shuffle=False)

    data = []
    labels = []
    stats = {}

    order = [
        ("PAC", "baseline", "PAC"),
        ("VCG", "baseline", "VCG"),
        ("CSRA", "baseline", "CSRA"),
        ("MFG-Pricing", "baseline", "MFG-Pricing"),
    ]
    for disp, kind, mech in order:
        try:
            x = _collect_baseline(
                mech, make_loader(), args.budget, args.baseline_v_grid, args.baseline_eps_grid, DEVICE
            )
            x = np.clip(x, 0, np.nanpercentile(x, 99.9) + 1e-6)  # 极端值裁剪便于展示
            data.append(x)
            labels.append(disp)
            stats[disp] = {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "p50": float(np.percentile(x, 50)),
                "p90": float(np.percentile(x, 90)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
            }
        except Exception as e:
            print("[WARN] {}: {}".format(disp, e))

    neural_specs = [
        ("RegretNet", args.regretnet_ckpt),
        ("DM-RegretNet", args.dm_regretnet_ckpt),
        ("Ours", args.mfg_regretnet_ckpt),
    ]
    for disp, path in neural_specs:
        if not path or not os.path.isfile(path):
            continue
        try:
            x = _collect_neural(path, make_loader(), args.budget, DEVICE, args.n_items, args.misreport_iter)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            hi = np.nanpercentile(x, 99.9)
            x = np.clip(x, 0, hi + 1e-6)
            data.append(x)
            labels.append(disp)
            stats[disp] = {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "p50": float(np.percentile(x, 50)),
                "p90": float(np.percentile(x, 90)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
            }
        except Exception as e:
            print("[WARN] {}: {}".format(disp, e))

    if not data:
        print("rq1_figure_d: no methods, skip")
        return 0

    rng = np.random.RandomState(args.seed)
    plot_data = []
    for x in data:
        if len(x) > args.max_samples_plot:
            plot_data.append(rng.choice(x, size=args.max_samples_plot, replace=False))
        else:
            plot_data.append(x)

    out_json = os.path.join(args.out_dir, "rq1_figure_d.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"per_method_quantiles": stats, "seed": args.seed}, f, indent=2, ensure_ascii=False)

    # 与箱线图同源的表格（论文可直接贴）
    csv_path = os.path.join(args.out_dir, "table_rq1_regret_distribution.csv")
    md_path = os.path.join(args.out_dir, "table_rq1_regret_distribution.md")
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "n", "Mean", "p50", "p90", "p95", "p99"])
        for lab in labels:
            s = stats.get(lab, {})
            w.writerow(
                [
                    lab,
                    s.get("n", ""),
                    s.get("mean", ""),
                    s.get("p50", ""),
                    s.get("p90", ""),
                    s.get("p95", ""),
                    s.get("p99", ""),
                ]
            )
    hdr = "| Method | n | Mean | p50 | p90 | p95 | p99 |\n|--------|---:|-----:|----:|----:|----:|----:|"
    lines = [hdr]
    for lab in labels:
        s = stats.get(lab, {})
        lines.append(
            "| {} | {} | {:.4g} | {:.4g} | {:.4g} | {:.4g} | {:.4g} |".format(
                lab,
                s.get("n", ""),
                float(s.get("mean", 0)),
                float(s.get("p50", 0)),
                float(s.get("p90", 0)),
                float(s.get("p95", 0)),
                float(s.get("p99", 0)),
            )
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            "# RQ1 图 D：归一化事后遗憾分布（与 figure_rq1_paper_regret_distribution.png 同源）\n\n"
            "seed={}\n\n".format(args.seed) + "\n".join(lines) + "\n"
        )
    print("Wrote", csv_path, "and", md_path)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        pos = np.arange(1, len(labels) + 1)
        if args.violin:
            parts = ax.violinplot(plot_data, positions=pos, showmeans=True, showmedians=True, widths=0.65)
            for pc in parts["bodies"]:
                pc.set_alpha(0.55)
        else:
            bp = ax.boxplot(plot_data, positions=pos, widths=0.55, patch_artist=True, showfliers=True)
            for i, p in enumerate(bp["boxes"]):
                p.set_facecolor(plt.cm.Set3(i / max(len(labels), 1)))
                p.set_alpha(0.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Normalized ex-post regret (per client $\\times$ profile)")
        ax.set_title(
            "RQ1-D: Regret distribution across clients (tail / quantiles; not mean-only)"
        )
        pos_vals = [np.mean(d) for d in plot_data]
        if max(pos_vals) / (min([v for v in pos_vals if v > 1e-8] or [1e-8]) + 1e-12) > 25:
            ax.set_yscale("symlog", linthresh=max(1e-4, 0.02 * min([v for v in pos_vals if v > 0] or [0.01])))
        ax.grid(True, axis="y", alpha=0.35)
        fig.tight_layout()
        p = os.path.join(args.out_dir, "figure_rq1_paper_regret_distribution.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print("Wrote", p)
    except ImportError:
        print("matplotlib missing, skip figure D")

    return 0


if __name__ == "__main__":
    sys.exit(main())

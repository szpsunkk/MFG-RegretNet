#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 可选图 C：遗憾（及 IR%）随「训练轮次 / 拍卖相关轮次」变化。

将神经机制的 **checkpoint 保存 epoch** 视为横轴 t（与 FL 中「每轮联邦前更新拍卖器」的轮次对应）；
解析基线在图中为水平参考线（不随 t 变）。用于观察机制是否随训练稳定。

输出：figure_rq1_paper_regret_vs_epoch.png、rq1_figure_c.json
"""
from __future__ import division, print_function

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _subsample_epochs(items, max_points):
    """items: list of (epoch, path) sorted by epoch"""
    if len(items) <= max_points:
        return items
    idx = np.unique(
        np.linspace(0, len(items) - 1, num=max_points, dtype=int)
    )
    return [items[i] for i in idx]


def _filter_ckpts(pattern, n_agents, n_items, want_mfg):
    import torch

    out = []
    for p in glob.glob(pattern):
        m = re.search(r"_(\d+)_checkpoint\.pt$", p)
        if not m:
            continue
        ep = int(m.group(1))
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            if arch.get("n_agents") != n_agents or arch.get("n_items") != n_items:
                continue
            is_mfg = arch.get("model_type") == "MFGRegretNet"
            if want_mfg and not is_mfg:
                continue
            if not want_mfg and is_mfg:
                continue
        except Exception:
            continue
        out.append((ep, p))
    return sorted(out, key=lambda x: x[0])


def _neural_epoch_curve(ckpt_items, num_profiles, n_agents, n_items, budget, seed, batch_size, misreport_iter, regret_tol):
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import load_auc_model, DEVICE
    from regretnet import MFGRegretNet
    from utils import optimize_misreports, tiled_misreport_util, calc_agent_util, allocs_to_plosses
    from datasets import Dataloader
    import torch

    reports, _, val_type = build_privacy_paper_batch(
        num_profiles, n_agents, n_items, budget, seed, DEVICE
    )
    loader = Dataloader(torch.cat([reports, val_type], dim=2), batch_size=batch_size, shuffle=False)

    epochs, regrets, ir_pcts = [], [], []
    for ep, path in ckpt_items:
        try:
            model = load_auc_model(path).to(DEVICE)
        except Exception as e:
            print("[WARN] skip ckpt {}: {}".format(path, e))
            continue
        model.eval()
        cost_from_plosses = isinstance(model, MFGRegretNet) and n_items == 1
        regs, ir_l = [], []
        for batch in loader:
            rep = batch[:, :, :-2].to(DEVICE)
            vt = batch[:, :, -2:].to(DEVICE)
            b = budget * torch.ones(rep.shape[0], 1, device=DEVICE)
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
            regrets_t = torch.clamp(untruthful_util - truthful_util, min=0)
            costs_safe = costs.clamp(min=1e-10)
            rnorm = regrets_t / costs_safe
            ir_l.append((truthful_util < 0).float().mean().item() * 100.0)
            regs.append(rnorm.detach().flatten().cpu().numpy())
        epochs.append(ep)
        regrets.append(float(np.concatenate(regs).mean()))
        ir_pcts.append(float(np.mean(ir_l)))
    return epochs, regrets, ir_pcts


def _baseline_hline(mech_name, num_profiles, n_agents, n_items, budget, seed, batch_size, v_grid, eps_grid):
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import DEVICE
    from exp_rq.guarantees_eval_baselines import guarantees_eval_procurement_baseline
    from baselines.pac import pac_batch
    from baselines.vcg import vcg_procurement_batch
    from baselines.csra import csra_qms_batch
    from baselines.mfg_pricing import mfg_pricing_batch
    from datasets import Dataloader
    import torch

    fd = {
        "PAC": pac_batch,
        "VCG": vcg_procurement_batch,
        "CSRA": csra_qms_batch,
        "MFG-Pricing": mfg_pricing_batch,
    }[mech_name]

    reports, _, val_type = build_privacy_paper_batch(
        num_profiles, n_agents, n_items, budget, seed, DEVICE
    )
    loader = Dataloader(torch.cat([reports, val_type], dim=2), batch_size=batch_size, shuffle=False)
    regs, ir_l = [], []
    for batch in loader:
        rep = batch[:, :, :-2].to(DEVICE)
        b = budget * torch.ones(rep.shape[0], 1, device=DEVICE)
        v_true = rep[:, :, 0]
        pl0, pay0 = fd(rep, b)
        util0 = pay0 - v_true * pl0
        ir_l.append((util0 < 0).float().mean().item() * 100.0)
        r, _ = guarantees_eval_procurement_baseline(rep, b, mech_name, v_grid_n=v_grid, eps_grid_n=eps_grid)
        regs.append(r.detach().cpu().numpy().ravel())
    return float(np.concatenate(regs).mean()), float(np.mean(ir_l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-items", type=int, default=1)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--num-profiles", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--misreport-iter", type=int, default=25)
    ap.add_argument("--baseline-v-grid", type=int, default=15)
    ap.add_argument("--baseline-eps-grid", type=int, default=7)
    ap.add_argument("--max-ckpts-per-method", type=int, default=18, help="每类神经机制最多评估的 checkpoint 数（均匀抽样 epoch）")
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/rq1")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    series = {}  # name -> {epochs, regret, ir_pct}
    baselines = {}

    # Neural: MFG-RegretNet (Ours)
    mfg_list = _subsample_epochs(
        _filter_ckpts("result/mfg_regretnet_privacy_*_checkpoint.pt", args.n_agents, args.n_items, True),
        args.max_ckpts_per_method,
    )
    if mfg_list:
        e, r, ir = _neural_epoch_curve(
            mfg_list, args.num_profiles, args.n_agents, args.n_items, args.budget,
            args.seed, args.batch_size, args.misreport_iter, 0.02,
        )
        if e:
            series["Ours (MFG-RegretNet)"] = {"epochs": e, "mean_regret": r, "ir_pct": ir}

    for label, pattern, want_mfg in [
        ("RegretNet", "result/regretnet_privacy_*_checkpoint.pt", False),
        ("DM-RegretNet", "result/dm_regretnet_privacy_*_checkpoint.pt", False),
    ]:
        lst = _subsample_epochs(
            _filter_ckpts(pattern, args.n_agents, args.n_items, want_mfg),
            args.max_ckpts_per_method,
        )
        if lst:
            e, r, ir = _neural_epoch_curve(
                lst, args.num_profiles, args.n_agents, args.n_items, args.budget,
                args.seed, args.batch_size, args.misreport_iter, 0.02,
            )
            if e:
                series[label] = {"epochs": e, "mean_regret": r, "ir_pct": ir}

    for mech in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
        try:
            mr, mir = _baseline_hline(
                mech, args.num_profiles, args.n_agents, args.n_items, args.budget,
                args.seed, args.batch_size, args.baseline_v_grid, args.baseline_eps_grid,
            )
            baselines[mech] = {"mean_regret": mr, "ir_pct": mir}
        except Exception as ex:
            print("[WARN] baseline {}: {}".format(mech, ex))

    xmax = 1
    for s in series.values():
        if s["epochs"]:
            xmax = max(xmax, max(s["epochs"]))
    xmin = 0

    out_json = {
        "description": "X = checkpoint training epoch (proxy for FL/auction rounds); baselines = horizontal refs",
        "neural_series": series,
        "baselines": baselines,
    }
    with open(os.path.join(args.out_dir, "rq1_figure_c.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    if not series and not baselines:
        print("rq1_figure_c: no data, skip figure")
        return 0

    try:
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 0.9, max(3, len(series))))
        for i, (name, s) in enumerate(series.items()):
            ax0.plot(s["epochs"], s["mean_regret"], marker="o", ms=4, label=name, color=colors[i % 10])
            ax1.plot(s["epochs"], s["ir_pct"], marker="s", ms=4, label=name, color=colors[i % 10])

        bl_styles = [("--", "#555"), (":", "#777"), ("-.", "#999"), ("--", "#333")]
        for j, (mech, b) in enumerate(baselines.items()):
            sty, c = bl_styles[j % len(bl_styles)]
            ax0.hlines(b["mean_regret"], xmin, xmax, linestyles=sty, colors=c, label=mech + " (baseline)", linewidth=1.5)
            ax1.hlines(b["ir_pct"], xmin, xmax, linestyles=sty, colors=c, linewidth=1.5)

        ax0.set_ylabel(r"$\bar{\mathrm{rgt}}$ (mean norm. regret)")
        ax0.set_title("RQ1-C: Regret vs training epoch (neural ckpts; baselines = horizontal)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best", fontsize=8)
        ax1.set_ylabel(r"IR violation (\%)")
        ax1.set_xlabel("Training epoch $t$ (checkpoint save, proxy for FL/auction rounds)")
        ax1.set_title("IR violation rate vs epoch")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.out_dir, "figure_rq1_paper_regret_vs_epoch.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print("Wrote", p)
    except ImportError:
        print("matplotlib missing, skip figure C")

    return 0


if __name__ == "__main__":
    sys.exit(main())

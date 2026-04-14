#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FL-market.pdf–style figures mapped to the privacy-paper RQ setup (synthetic bids).

Produces (same axes / titles as original utils.plot_* helpers):
  1) Effect of financial budget on error bound  → plot_budget_mse
  2) Invalid gradient rate vs budget            → plot_budget_invalid_rate
  3) Model accuracy over FL training rounds      → plot_rnd_acc (from RQ4 raw JSON)
  4) Effect of parameter M                       → plot_m_guarantees (regret & IR vs M)

Mechanisms align with RQ4: MFG-RegretNet (Ours), CSRA, MFG-Pricing, RegretNet, optional PAC.
Neural mechanisms need checkpoints resolvable via rq1_ckpt_resolve.

Usage (repo root):
  # 1–2 + 4: compute sweeps + plots (can be slow)
  python exp_rq/fl_market_style_figures.py --out-dir run/privacy_paper/fl_market_style \\
      --n-agents 10 --n-profiles 4000 --nb-budget 20 --budget-step 0.1

  # 3 only: needs RQ4 JSON under --rq4-dir
  python exp_rq/fl_market_style_figures.py --out-dir run/privacy_paper/fl_market_style \\
      --only-plot --rq4-dir run/privacy_paper/rq4/raw

  MPLBACKEND=Agg recommended on servers.
"""
from __future__ import division, print_function

import argparse
import contextlib
import glob
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@contextlib.contextmanager
def _no_show():
    """utils.plot* calls plt.show(); suppress for batch runs."""
    import matplotlib.pyplot as plt

    saved = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = saved


def _patch_utils_fonts_if_no_times_new_roman():
    """Linux often lacks Times New Roman; avoid matplotlib findfont warnings from utils.plot."""
    try:
        from matplotlib import font_manager

        names = {f.name for f in font_manager.fontManager.ttflist}
        if "Times New Roman" in names:
            return
        import utils as u

        ff = "DejaVu Serif"
        u.LEGEND_FONT = {**u.LEGEND_FONT, "family": ff}
        u.LABEL_FONT = {**u.LABEL_FONT, "family": ff}
        u.TITLE_FONT = {**u.TITLE_FONT, "family": ff}
    except Exception:
        pass


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synthetic_profiles(n_agents, n_items, n_profiles, seed, device):
    from datasets_fl_benchmark import generate_privacy_paper_bids

    rep = generate_privacy_paper_bids(
        n_agents, n_items, n_profiles, seed=seed
    )
    return rep.to(device).float()


def _build_mechanisms(device, n_agents, n_items, include_pac, regret_ckpt, mfg_ckpt):
    """Return list of {label, trade_mech, model}. trade_mech: [auc, aggr, ckpt, n_items]."""
    from experiments import load_auc_model

    out = []
    if mfg_ckpt and os.path.isfile(mfg_ckpt):
        m = load_auc_model(mfg_ckpt).to(device)
        out.append(
            {
                "label": "Ours (MFG-RegretNet)",
                "trade_mech": ["MFG-RegretNet", "ConvlAggr", mfg_ckpt, n_items],
                "model": m,
            }
        )
    else:
        print("[fl_market_style] skip Ours: no MFG ckpt")

    out.append({"label": "CSRA", "trade_mech": ["CSRA", "ConvlAggr", "", n_items], "model": None})
    out.append({"label": "MFG-Pricing", "trade_mech": ["MFG-Pricing", "ConvlAggr", "", n_items], "model": None})
    if include_pac:
        out.append({"label": "PAC", "trade_mech": ["PAC", "ConvlAggr", "", n_items], "model": None})

    if regret_ckpt and os.path.isfile(regret_ckpt):
        m = load_auc_model(regret_ckpt).to(device)
        out.append(
            {
                "label": "RegretNet",
                "trade_mech": ["RegretNet", "ConvlAggr", regret_ckpt, n_items],
                "model": m,
            }
        )
    else:
        print("[fl_market_style] skip RegretNet: no ckpt")

    return out


def _mse_eval_batch(reports, budget, trade_mech, auc_model, expected=False):
    from experiments import mse_eval

    return mse_eval(reports, budget, trade_mech, L=1.0, expected=expected)


def _invalid_grad_rate_batch(reports, budget, tm, model):
    """
    Product of 'no privacy loss' probs across agents (same as experiments.invalid_rate_budget).
    Only defined for neural auctions with differentiable allocations.
    """
    import torch

    if tm[0] in ("PAC", "VCG", "CSRA", "MFG-Pricing", "All-in", "FairQuery"):
        return torch.zeros(reports.shape[0], device=reports.device)
    if model is None:
        return torch.zeros(reports.shape[0], device=reports.device)
    allocs, _payments = model((reports, budget))
    from utils import calc_full_allocs

    full_allocs = calc_full_allocs(allocs)
    return torch.prod(full_allocs[:, :, 0], dim=1)


def run_budget_error_and_invalid(
    mechanisms,
    n_agents,
    n_items,
    n_profiles,
    seed,
    nb_budget,
    budget_step,
    batch_size,
):
    """Returns dict with keys error_bound, invalid_rate (each: labels, budget_rates, curves)."""
    import torch
    from tqdm import tqdm

    from utils import generate_max_cost

    device = _device()
    torch.manual_seed(seed)
    reports = _synthetic_profiles(n_agents, n_items, n_profiles, seed, device)

    err_labels, err_x, err_y = [], [], []
    inv_labels, inv_x, inv_y = [], [], []

    for mech in mechanisms:
        label = mech["label"]
        tm = mech["trade_mech"]
        model = mech["model"]
        if tm[0] in ("PAC", "VCG", "CSRA"):
            auc = None
        else:
            auc = model

        b_rates = []
        eb_means = []
        ir_means = []

        for i in tqdm(range(nb_budget), desc="budget {}".format(label)):
            br = budget_step * (i + 1)
            ebs = []
            irs = []
            for start in range(0, n_profiles, batch_size):
                end = min(start + batch_size, n_profiles)
                rep = reports[start:end]
                mc = generate_max_cost(rep)
                budget = br * mc
                eb = _mse_eval_batch(rep, budget, tm, auc)
                eb = eb.detach().cpu().numpy()
                ebs.append(eb)

                ir = _invalid_grad_rate_batch(rep, budget, tm, auc)
                irs.append(ir.detach().cpu().numpy())

            eb_mean = float(np.mean(np.concatenate(ebs))) if len(ebs) else float("nan")
            ir_mean = float(np.mean(np.concatenate(irs))) if len(irs) else 0.0
            b_rates.append(br)
            eb_means.append(eb_mean)
            ir_means.append(ir_mean)

        err_labels.append(label)
        err_x.append(b_rates)
        err_y.append(eb_means)

        inv_labels.append(label)
        inv_x.append(b_rates)
        inv_y.append(ir_means)

    return {
        "error_bound": {"labels": err_labels, "x": err_x, "y": err_y},
        "invalid_rate": {"labels": inv_labels, "x": inv_x, "y": inv_y},
        "meta": {
            "n_agents": n_agents,
            "n_items": n_items,
            "n_profiles": n_profiles,
            "seed": seed,
            "nb_budget": nb_budget,
            "budget_step": budget_step,
        },
    }


def run_guarantees_vs_m(
    n_agents,
    m_items_list,
    n_profiles,
    seed,
    min_budget_rate,
    max_budget_rate,
    batch_size,
    misreport_iter,
):
    """
    One trade_mech per M with distinct checkpoint (n_items=M).
    PAC/VCG/CSRA get zero regret (appended once if requested).
    """
    import torch
    from tqdm import tqdm

    from datasets import Dataloader
    from experiments import guarantees_eval
    from utils import generate_max_cost

    from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt, resolve_regretnet_ckpt

    device = _device()
    torch.manual_seed(seed)

    trade_mech_ls = []
    labels_order = []

    for m in m_items_list:
        ck = resolve_mfg_regretnet_ckpt(n_agents, m)
        if not ck:
            print("[fl_market_style] M={}: no MFG-RegretNet ckpt, skip".format(m))
            continue
        trade_mech_ls.append(["MFG-RegretNet", "ConvlAggr", ck, m])
        labels_order.append("Ours (M={})".format(m))

    if not trade_mech_ls:
        print("[fl_market_style] No MFG ckpts for M-list; try training multi-item MFG or pass --m-items 1 only.")
        return {"m": [], "regret": [], "ir": [], "meta": {}}

    proflies_nb = n_profiles
    budget_rate = torch.rand((proflies_nb, 1), device=device) * (min_budget_rate - max_budget_rate) + max_budget_rate

    regret_ls = []
    ir_ls = []
    m_out = []

    for trade_mech in tqdm(trade_mech_ls, desc="guarantees vs M"):
        n_items = trade_mech[3]
        m_out.append(n_items)
        if trade_mech[0] in ("PAC", "VCG", "CSRA"):
            regret_ls.append(0.0)
            ir_ls.append(0.0)
            continue

        data = _synthetic_profiles(n_agents, n_items, proflies_nb, seed, device)
        val_type = torch.zeros(data.shape[0], n_agents, 2, device=device)
        full = torch.cat([data, val_type], dim=2)
        bs = min(batch_size, proflies_nb)
        loader = Dataloader(full, bs, shuffle=False)

        # guarantees_eval expects trade_mech[3] == n_items (full 4-tuple, see experiments.py)
        regrets = []
        irs = []
        for j, batch in enumerate(loader):
            reports = batch[:, :, :-2]
            vt = batch[:, :, -2:]
            max_cost = generate_max_cost(reports)
            b = max_cost * budget_rate[j * bs : (j + 1) * bs]
            regret, ir = guarantees_eval(
                reports,
                b,
                vt,
                trade_mech,
                misreport_iter=misreport_iter,
                lr=0.1,
            )
            regrets.append(regret.detach().cpu().numpy().ravel())
            irs.append(ir.detach().cpu().numpy().ravel())

        regret_ls.append(float(np.mean(np.concatenate(regrets))))
        ir_ls.append(float(np.mean(np.concatenate(irs))))

    return {
        "m": m_out,
        "regret": regret_ls,
        "ir": ir_ls,
        "labels": labels_order,
        "meta": {
            "n_agents": n_agents,
            "n_profiles": n_profiles,
            "misreport_iter": misreport_iter,
        },
    }


def _plot_rq4_accuracy_fl_style(rq4_raw_glob, out_png, title_suffix=""):
    """FL-market–style: x = training round, y = model accuracy (linear, 0–100%)."""
    import glob as glob_mod
    import matplotlib.pyplot as plt

    paths = sorted(glob_mod.glob(rq4_raw_glob))
    if not paths:
        print("[fl_market_style] No RQ4 JSON at", rq4_raw_glob)
        return

    # Aggregate seeds: mean acc per method (same logic as rq4_plot simplified)
    from exp_rq.rq4_plot_paper_figures import _filter_rq4_json_paths

    paths = _filter_rq4_json_paths(paths, prefer_pagalg2=False)
    by_method = {}
    rounds = None
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta") or {}
        rounds = meta.get("rounds_logged") or rounds
        for name, series in (d.get("methods") or {}).items():
            acc = np.array(series.get("test_acc") or [], dtype=np.float64)
            if acc.size == 0:
                continue
            by_method.setdefault(name, []).append(acc)

    if not by_method or rounds is None:
        print("[fl_market_style] RQ4 JSON missing methods or rounds_logged")
        return

    L = min(len(r) for rows in by_method.values() for r in rows)
    L = min(L, len(rounds))
    r = np.array(rounds[:L], dtype=np.float64)

    order = ["Ours", "CSRA", "MFG-Pricing", "RegretNet", "PAC", "Uniform-DP", "No-DP (upper)"]
    colors = {
        "Ours": "#1b5e20",
        "CSRA": "#1565c0",
        "MFG-Pricing": "#6a1b9a",
        "RegretNet": "#e65100",
        "Uniform-DP": "#757575",
        "PAC": "#00838f",
        "No-DP (upper)": "#000000",
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name in order:
        if name not in by_method:
            continue
        stack = np.stack([a[:L] for a in by_method[name]], axis=0)
        mu = stack.mean(0) * 100.0
        sd = stack.std(0) * 100.0
        c = colors.get(name, "#333333")
        ls = "--" if "No-DP" in name else "-"
        ax.plot(r, mu, label=name, color=c, ls=ls, lw=1.8)
        ax.fill_between(r, mu - sd, mu + sd, color=c, alpha=0.15)

    ax.set_xlabel("training round", fontsize=14)
    ax.set_ylabel("model accuracy", fontsize=14)
    ax.set_title("Model accuracy over FL training rounds (RQ4)" + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()
    print("Wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/fl_market_style")
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-items", type=int, default=1)
    ap.add_argument("--n-profiles", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nb-budget", type=int, default=20)
    ap.add_argument("--budget-step", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4000)
    ap.add_argument("--include-pac", action="store_true")
    ap.add_argument("--regret-ckpt", type=str, default="")
    ap.add_argument("--mfg-ckpt", type=str, default="")
    ap.add_argument("--rq4-dir", type=str, default="run/privacy_paper/rq4/raw")
    ap.add_argument("--m-items", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--min-budget-rate", type=float, default=0.1)
    ap.add_argument("--max-budget-rate", type=float, default=2.0)
    ap.add_argument("--misreport-iter", type=int, default=40)
    ap.add_argument("--only-plot", action="store_true", help="Only RQ4 accuracy + load saved JSON if present")
    ap.add_argument("--skip-budget-sweeps", action="store_true")
    ap.add_argument("--skip-m-guarantees", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "raw")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    _patch_utils_fonts_if_no_times_new_roman()

    from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt, resolve_regretnet_ckpt

    if not args.regret_ckpt:
        args.regret_ckpt = resolve_regretnet_ckpt(args.n_agents, args.n_items)
    if not args.mfg_ckpt:
        args.mfg_ckpt = resolve_mfg_regretnet_ckpt(args.n_agents, args.n_items)

    # --- Fig 3: RQ4 accuracy (always try)
    rq4_glob = os.path.join(args.rq4_dir, "*.json")
    _plot_rq4_accuracy_fl_style(
        rq4_glob,
        os.path.join(fig_dir, "figure_flm_RQ4_model_accuracy_vs_round.png"),
    )

    if args.only_plot:
        summ_path = os.path.join(raw_dir, "budget_sweeps.json")
        if os.path.isfile(summ_path):
            with open(summ_path, "r", encoding="utf-8") as f:
                sweeps = json.load(f)
            from utils import plot_budget_invalid_rate, plot_budget_mse

            with _no_show():
                plot_budget_mse(
                    sweeps["error_bound"]["x"],
                    sweeps["error_bound"]["y"],
                    sweeps["error_bound"]["labels"],
                    "Effect of financial budget on error bound (privacy-paper bids, RQ)",
                    os.path.join(fig_dir, "figure_flm_error_bound_vs_budget.png"),
                )
                plot_budget_invalid_rate(
                    sweeps["invalid_rate"]["x"],
                    sweeps["invalid_rate"]["y"],
                    sweeps["invalid_rate"]["labels"],
                    "Invalid gradient rate (privacy-paper bids, RQ)",
                    os.path.join(fig_dir, "figure_flm_invalid_grad_rate_vs_budget.png"),
                )
        gpath = os.path.join(raw_dir, "guarantees_vs_m.json")
        if os.path.isfile(gpath):
            with open(gpath, "r", encoding="utf-8") as f:
                g = json.load(f)
            if g.get("m"):
                from utils import plot_m_guarantees

                with _no_show():
                    plot_m_guarantees(
                        [g["m"], g["m"]],
                        [g["regret"], g["ir"]],
                        ["regret", "IR violation"],
                        "Effect of parameter M (privacy-paper bids, RQ)",
                        os.path.join(fig_dir, "figure_flm_effect_of_M.png"),
                    )
        return

    if not args.skip_budget_sweeps:
        mechs = _build_mechanisms(
            _device(),
            args.n_agents,
            args.n_items,
            args.include_pac,
            args.regret_ckpt,
            args.mfg_ckpt,
        )
        sweeps = run_budget_error_and_invalid(
            mechs,
            args.n_agents,
            args.n_items,
            args.n_profiles,
            args.seed,
            args.nb_budget,
            args.budget_step,
            args.batch_size,
        )
        with open(os.path.join(raw_dir, "budget_sweeps.json"), "w", encoding="utf-8") as f:
            json.dump(sweeps, f, indent=2)

        from utils import plot_budget_invalid_rate, plot_budget_mse

        with _no_show():
            plot_budget_mse(
                sweeps["error_bound"]["x"],
                sweeps["error_bound"]["y"],
                sweeps["error_bound"]["labels"],
                "Effect of financial budget on error bound (privacy-paper bids, RQ)",
                os.path.join(fig_dir, "figure_flm_error_bound_vs_budget.png"),
            )
            plot_budget_invalid_rate(
                sweeps["invalid_rate"]["x"],
                sweeps["invalid_rate"]["y"],
                sweeps["invalid_rate"]["labels"],
                "Invalid gradient rate (privacy-paper bids, RQ)",
                os.path.join(fig_dir, "figure_flm_invalid_grad_rate_vs_budget.png"),
            )

    if not args.skip_m_guarantees:
        g = run_guarantees_vs_m(
            args.n_agents,
            list(args.m_items),
            args.n_profiles,
            args.seed,
            args.min_budget_rate,
            args.max_budget_rate,
            args.batch_size,
            args.misreport_iter,
        )
        with open(os.path.join(raw_dir, "guarantees_vs_m.json"), "w", encoding="utf-8") as f:
            json.dump(g, f, indent=2)
        if g.get("m"):
            from utils import plot_m_guarantees

            with _no_show():
                plot_m_guarantees(
                    [g["m"], g["m"]],
                    [g["regret"], g["ir"]],
                    ["regret", "IR violation"],
                    "Effect of parameter M (privacy-paper bids, RQ)",
                    os.path.join(fig_dir, "figure_flm_effect_of_M.png"),
                )

    print("Done. Figures in", fig_dir)


if __name__ == "__main__":
    main()

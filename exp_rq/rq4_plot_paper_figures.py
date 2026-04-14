#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate RQ4 raw JSONs (multiple seeds) and write paper figures:
  Fig A: test accuracy vs round (MNIST | CIFAR), mean ± std, No-DP dashed
  Fig B: final test accuracy, grouped by Dirichlet alpha
  Fig C: ΔA = A_no_DP - A_method (privacy cost)
  Fig D: train loss vs round (appendix style)

Usage (from repo root):
  python exp_rq/rq4_plot_paper_figures.py --rq4-dir run/privacy_paper/rq4 --out-dir run/privacy_paper/rq4/figures
"""
from __future__ import division, print_function

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _filter_rq4_json_paths(paths, prefer_pagalg2=False):
    """Avoid mixing Laplace+ConvlAggr runs with *_pagalg2.json in one aggregate."""
    std = [p for p in paths if "_pagalg2" not in os.path.basename(p)]
    pag = [p for p in paths if "_pagalg2" in os.path.basename(p)]
    if std and pag:
        use = pag if prefer_pagalg2 else std
        print(
            "[rq4_plot] Mixed standard vs *_pagalg2.json → using %s. "
            "Opposite: pass %s."
            % (
                "Alg.2" if prefer_pagalg2 else "standard",
                "--prefer-pagalg2" if not prefer_pagalg2 else "(omit --prefer-pagalg2)",
            )
        )
        return use
    return paths


def _load_raw_files(raw_dir, prefer_pagalg2=False):
    paths = _filter_rq4_json_paths(
        sorted(glob.glob(os.path.join(raw_dir, "*.json"))), prefer_pagalg2=prefer_pagalg2
    )
    by_key = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d.get("methods"), dict) or not isinstance(d.get("meta"), dict):
            continue
        meta = d.get("meta") or {}
        ds = str(meta.get("dataset", "")).upper().replace("-", "")
        if not ds.strip():
            continue
        alpha = float(meta.get("alpha", 0.5))
        seed = int(meta.get("seed", -1))
        key = (ds, alpha)
        by_key.setdefault(key, []).append((seed, d))
    return by_key


def _aggregate(by_key):
    """Return dict (dataset, alpha) -> { rounds, methods: {m: {mean, std}} }."""
    out = {}
    for (ds, alpha), runs in by_key.items():
        if not runs:
            continue
        methods = {}
        rounds = None
        for seed, d in runs:
            mets = d.get("methods") or {}
            for name, series in mets.items():
                acc = np.array(series.get("test_acc") or [], dtype=np.float64)
                loss = np.array(series.get("train_loss") or [], dtype=np.float64)
                methods.setdefault(name, {"acc": [], "loss": []})
                methods[name]["acc"].append(acc)
                methods[name]["loss"].append(loss)
            r = d.get("meta", {}).get("rounds_logged")
            if r:
                rounds = r
        agg = {}
        for name, arrs in methods.items():
            acc_stack = np.stack(arrays_align(arrs["acc"]), axis=0) if arrs["acc"] else None
            loss_stack = np.stack(arrays_align(arrs["loss"]), axis=0) if arrs["loss"] else None
            if acc_stack is not None:
                Lm = acc_stack.shape[1]
                if loss_stack is not None:
                    Lm = min(Lm, loss_stack.shape[1])
                agg[name] = {
                    "test_acc_mean": acc_stack[:, :Lm].mean(0).tolist(),
                    "test_acc_std": acc_stack[:, :Lm].std(0).tolist(),
                    "train_loss_mean": loss_stack[:, :Lm].mean(0).tolist()
                    if loss_stack is not None
                    else [],
                    "train_loss_std": loss_stack[:, :Lm].std(0).tolist()
                    if loss_stack is not None
                    else [],
                }
        out[(ds, alpha)] = {"rounds": rounds or [], "methods": agg, "n_seeds": len(runs)}
    return out


def arrays_align(list_of_arr):
    """Truncate to min length across seeds."""
    if not list_of_arr:
        return []
    L = min(len(a) for a in list_of_arr)
    return [a[:L] for a in list_of_arr]


def _plot_fig_a(agg, datasets_order, out_path, fig_a_alpha=0.5, upper_name="No-DP (upper)"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for figures")
        return
    fig, axes = plt.subplots(1, len(datasets_order), figsize=(5.2 * len(datasets_order), 4.0), squeeze=False)
    colors = {
        "Ours": "#1b5e20",
        "CSRA": "#1565c0",
        "MFG-Pricing": "#6a1b9a",
        "RegretNet": "#e65100",
        "Uniform-DP": "#757575",
        "PAC": "#00838f",
    }
    order_plot = ["Ours", "CSRA", "MFG-Pricing", "RegretNet", "PAC", "Uniform-DP"]
    for ax_i, ds in enumerate(datasets_order):
        ax = axes[0, ax_i]
        plotted = False
        block = agg.get((ds, fig_a_alpha))
        if block:
            r = np.array(block["rounds"])
            mets = block["methods"]
            for name in order_plot:
                if name not in mets:
                    continue
                mu = np.array(mets[name]["test_acc_mean"])
                sd = np.array(mets[name]["test_acc_std"])
                if len(mu) == 0:
                    continue
                c = colors.get(name, "#333333")
                ax.plot(r[: len(mu)], mu * 100, label=name, color=c, lw=1.8)
                ax.fill_between(
                    r[: len(mu)],
                    (mu - sd) * 100,
                    (mu + sd) * 100,
                    color=c,
                    alpha=0.18,
                )
                plotted = True
            if upper_name in mets:
                mu = np.array(mets[upper_name]["test_acc_mean"])
                sd = np.array(mets[upper_name]["test_acc_std"])
                ax.plot(
                    r[: len(mu)],
                    mu * 100,
                    ls="--",
                    color="#000000",
                    lw=1.2,
                    label="No-DP (ref)",
                )
                ax.fill_between(r[: len(mu)], (mu - sd) * 100, (mu + sd) * 100, color="#999", alpha=0.12)
        ax.set_xlabel("Communication round $t$")
        ax.set_ylabel(r"Test accuracy $\mathcal{A}_{\mathrm{test}}$ (\%)")
        ax.set_title(r"({}) {}".format(chr(97 + ax_i), ds))
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=7, loc="lower right", ncol=2)
    # Avoid .format() here: LaTeX \mathcal{A}_{...} uses {} which breaks str.format
    fig.suptitle(
        r"RQ4 (Fig. A): $\mathcal{A}_{\mathrm{test}}$ vs round (Dirichlet $\alpha=%s$)" % fig_a_alpha,
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def _plot_fig_b(agg, datasets_order, alphas, out_path, upper_name="No-DP (upper)"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    methods_core = ["Ours", "CSRA", "MFG-Pricing", "RegretNet", "Uniform-DP", "PAC"]
    x = np.arange(len(methods_core))
    width = 0.35
    fig, axes = plt.subplots(1, len(datasets_order), figsize=(max(8, 2 + 2 * len(methods_core)), 4), squeeze=False)
    for di, ds in enumerate(datasets_order):
        ax = axes[0, di]
        for ai, alpha in enumerate(sorted(alphas)):
            offs = (ai - 0.5 * (len(alphas) - 1)) * width
            means, stds, labels = [], [], []
            block = agg.get((ds, alpha), {})
            mets = block.get("methods", {})
            for m in methods_core:
                if m not in mets:
                    means.append(0)
                    stds.append(0)
                    labels.append(m)
                    continue
                mu = np.array(mets[m]["test_acc_mean"])
                sd = np.array(mets[m]["test_acc_std"])
                if len(mu) == 0:
                    means.append(0)
                    stds.append(0)
                else:
                    means.append(mu[-1] * 100)
                    stds.append(sd[-1] * 100 if len(sd) else 0)
                labels.append(m)
            ax.bar(
                x + offs,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=r"$\alpha={}$".format(alpha),
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods_core, rotation=25, ha="right")
        ax.set_ylabel(r"Final $\mathcal{A}_{\mathrm{test}}$ (\%)")
        ax.set_title(ds)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("RQ4 (Fig. B): Final accuracy vs non-IID hardness", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def _plot_fig_c(agg, ds, alpha, out_path, upper_name="No-DP (upper)"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    block = agg.get((ds, alpha), {})
    mets = block.get("methods", {})
    if upper_name not in mets:
        print("Fig C: skip (no upper bound)")
        return
    ref = np.array(mets[upper_name]["test_acc_mean"])[-1]
    names = []
    deltas = []
    stds = []
    for name, v in mets.items():
        if name == upper_name:
            continue
        mu = np.array(v["test_acc_mean"])
        sd = np.array(v["test_acc_std"])
        if len(mu) == 0:
            continue
        names.append(name)
        deltas.append((ref - mu[-1]) * 100)
        stds.append(sd[-1] * 100 if len(sd) else 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, deltas, xerr=stds, capsize=3, color="steelblue", alpha=0.85)
    ax.set_xlabel(r"$\Delta\mathcal{A}$ = $\mathcal{A}_{\mathrm{no-DP}} - \mathcal{A}_{\mathrm{method}}$ (%)")
    ax.set_title("RQ4 (Fig. C): Privacy–utility gap @ final round — {}".format(ds))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def _plot_fig_d(agg, datasets_order, out_path, upper_name="No-DP (upper)"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    order_plot = ["Ours", "CSRA", "MFG-Pricing", "RegretNet", "Uniform-DP"]
    fig, axes = plt.subplots(1, len(datasets_order), figsize=(5 * len(datasets_order), 3.8), squeeze=False)
    for ax_i, ds in enumerate(datasets_order):
        ax = axes[0, ax_i]
        for (d, alpha), block in agg.items():
            if d != ds:
                continue
            r = np.array(block["rounds"])
            mets = block["methods"]
            for name in order_plot + [upper_name]:
                if name not in mets:
                    continue
                mu = np.array(mets[name].get("train_loss_mean") or [])
                if len(mu) == 0:
                    continue
                ls = "--" if name == upper_name else "-"
                ax.plot(r[: len(mu)], mu, label=name, ls=ls, lw=1.4)
        ax.set_xlabel("Round")
        ax.set_ylabel("Train loss (mean @ participants)")
        ax.set_title(ds + " (appendix)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("RQ4 (Fig. D): Training loss vs round", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq4-dir", type=str, default="run/privacy_paper/rq4")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--fig-a-alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for Fig A panels (both datasets)",
    )
    ap.add_argument(
        "--prefer-pagalg2",
        action="store_true",
        help="If raw mixes *._pagalg2.json with standard, plot Alg.2 runs (default: standard)",
    )
    args = ap.parse_args()
    raw_dir = os.path.join(args.rq4_dir, "raw")
    out_dir = args.out_dir or os.path.join(args.rq4_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    by_key = _load_raw_files(raw_dir, prefer_pagalg2=args.prefer_pagalg2)
    if not by_key:
        print("No raw JSON in", raw_dir)
        print("Run: bash scripts/run_rq4_paper.sh")
        return 1
    agg = _aggregate(by_key)
    agg_name = "rq4_aggregated_pagalg2.json" if args.prefer_pagalg2 else "rq4_aggregated.json"
    agg_path = os.path.join(args.rq4_dir, agg_name)
    serial = {str(k): v for k, v in agg.items()}
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)
    print("Wrote", agg_path)

    dss = sorted({d for (d, a) in agg})
    alphas = sorted({a for (d, a) in agg})
    if not dss:
        return 1
    _plot_fig_a(
        agg,
        dss,
        os.path.join(out_dir, "figure_rq4_A_acc_vs_round.png"),
        fig_a_alpha=args.fig_a_alpha,
    )
    if len(alphas) >= 1:
        _plot_fig_b(agg, dss, alphas, os.path.join(out_dir, "figure_rq4_B_final_acc_by_alpha.png"))
    if dss:
        _plot_fig_c(
            agg,
            dss[0],
            args.fig_a_alpha,
            os.path.join(out_dir, "figure_rq4_C_delta_accuracy.png"),
        )
    _plot_fig_d(agg, dss, os.path.join(out_dir, "figure_rq4_D_train_loss.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

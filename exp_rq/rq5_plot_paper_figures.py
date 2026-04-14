#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5 figures from rq5_fl_benchmark JSONs:
  Fig A: Pareto — mean ε̄ vs final test accuracy (lines per method across B)
  Fig B: (a) ε̄ bars (b) accuracy bars @ chosen B
  Fig C: boxplot of per-client ε^out by method @ B_ref
  Fig D: budget B vs ε̄ and vs accuracy (twin curves / two panels)
  Fig E: ||Δw||_2 vs round (appendix)

  python exp_rq/rq5_plot_paper_figures.py --rq5-dir run/privacy_paper/rq5
"""
from __future__ import division, print_function

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_COLORS = {
    "Ours": "#1b5e20",
    "CSRA": "#1565c0",
    "MFG-Pricing": "#6a1b9a",
    "RegretNet": "#e65100",
    "Uniform-DP": "#757575",
    "PAC": "#00838f",
    "No-DP (upper)": "#9e9e9e",
}


def _filter_rq5_json_paths(paths, prefer_pagalg2=False):
    std = [p for p in paths if "_pagalg2" not in os.path.basename(p)]
    pag = [p for p in paths if "_pagalg2" in os.path.basename(p)]
    if std and pag:
        use = pag if prefer_pagalg2 else std
        print(
            "[rq5_plot] Mixed JSON types → using %s (see --prefer-pagalg2)"
            % ("Alg.2" if prefer_pagalg2 else "standard")
        )
        return use
    return paths


def _norm_dataset_from_meta(meta):
    """Meta dataset string -> MNIST | CIFAR10 | other uppercase token."""
    ds = str((meta or {}).get("dataset", "")).upper().replace("-", "")
    if "CIFAR" in ds:
        return "CIFAR10"
    return ds or "UNKNOWN"


def _datasets_in_docs(docs):
    return {_norm_dataset_from_meta(d.get("meta")) for d in docs} - {"UNKNOWN"}


def _filter_docs_dataset(docs, dataset_filter):
    if not dataset_filter:
        return docs
    want = dataset_filter.strip().upper().replace("-", "")
    if "CIFAR" in want:
        want = "CIFAR10"
    out = [d for d in docs if _norm_dataset_from_meta(d.get("meta")) == want]
    return out


def _load_runs(raw_dir, prefer_pagalg2=False):
    """List of {meta, runs} from each file."""
    out = []
    paths = _filter_rq5_json_paths(
        sorted(glob.glob(os.path.join(raw_dir, "*.json"))), prefer_pagalg2=prefer_pagalg2
    )
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        runs = d.get("runs")
        if not isinstance(runs, list) or len(runs) == 0:
            continue
        if not isinstance(d.get("meta"), dict):
            continue
        out.append(d)
    return out


def _aggregate(docs, exclude_nopareto=("No-DP (upper)",)):
    """
    Returns:
      plot_a: method -> list of (eps_mean, acc_mean, eps_std, acc_std) per sorted B
      Bs: sorted budget rates
      by_method_B: (m,B) -> {eps, acc, eps_clients pooled}
    """
    by_mb = defaultdict(list)
    Bset = set()
    for d in docs:
        for r in d["runs"]:
            m, B = r["method"], round(float(r["budget_rate"]), 5)
            if m in exclude_nopareto:
                continue
            Bset.add(B)
            by_mb[(m, B)].append(r)
    Bs = sorted(Bset)
    methods = sorted({m for (m, _) in by_mb})
    plot_a = {}
    for m in methods:
        pts = []
        for B in Bs:
            rows = by_mb.get((m, B), [])
            if not rows:
                continue
            eps = [x["eps_bar_time_avg"] for x in rows]
            acc = [x["final_test_acc"] for x in rows]
            pts.append(
                (
                    float(np.mean(eps)),
                    float(np.mean(acc)),
                    float(np.std(eps)) if len(eps) > 1 else 0.0,
                    float(np.std(acc)) if len(acc) > 1 else 0.0,
                    B,
                )
            )
        if pts:
            plot_a[m] = pts
    return plot_a, Bs, by_mb


def plot_fig_a(plot_a, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Need matplotlib")
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for m, pts in plot_a.items():
        pts = sorted(pts, key=lambda x: x[4])
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ex = [p[2] for p in pts]
        ey = [p[3] * 100 for p in pts]
        c = _COLORS.get(m, "#333")
        ax.errorbar(xs, ys, xerr=ex, yerr=ey, fmt="o-", color=c, label=m, capsize=3, lw=1.5, ms=6)
    ax.set_xlabel(r"Time-averaged mean $\bar{\epsilon}$ (participants, larger = weaker privacy)")
    ax.set_ylabel(r"Final $\mathcal{A}_{\mathrm{test}}$ (\%)")
    ax.set_title("RQ5 (Fig. A): Privacy–accuracy tradeoff")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def _closest_B(by_mb, B_ref):
    Bs = sorted({float(b) for (_, b) in by_mb})
    if not Bs:
        return B_ref
    return min(Bs, key=lambda b: abs(b - float(B_ref)))


def plot_fig_b(by_mb, B_ref, methods_order, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    B0 = _closest_B(by_mb, B_ref)
    cand = sorted({m for (m, b) in by_mb if abs(float(b) - B0) < 1e-5 and m != "No-DP (upper)"})
    methods = [m for m in methods_order if m in cand] + [m for m in cand if m not in methods_order]
    if not methods:
        print("Fig B: no data near B=", B_ref)
        return
    eps_m, eps_s, acc_m, acc_s = [], [], [], []
    for m in methods:
        rows = by_mb.get((m, B0), [])
        if not rows:
            continue
        e = [r["eps_bar_time_avg"] for r in rows]
        a = [r["final_test_acc"] for r in rows]
        eps_m.append(float(np.mean(e)))
        eps_s.append(float(np.std(e)) if len(e) > 1 else 0.0)
        acc_m.append(float(np.mean(a)) * 100)
        acc_s.append(float(np.std(a)) * 100 if len(a) > 1 else 0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(methods))
    c = [_COLORS.get(m, "#555") for m in methods]
    ax1.bar(x, eps_m, yerr=eps_s, color=c, capsize=3, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right")
    ax1.set_ylabel(r"$\bar{\epsilon}$ (time avg.)")
    ax1.set_title("(a) Privacy consumption")
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar(x, acc_m, yerr=acc_s, color=c, capsize=3, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right")
    ax2.set_ylabel(r"$\mathcal{A}_{\mathrm{test}}$ (\%)")
    ax2.set_title("(b) Utility")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.suptitle("RQ5 (Fig. B): Privacy vs accuracy @ $B={}$".format(B0), y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def plot_fig_c(by_mb, B_ref, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    B0 = _closest_B(by_mb, B_ref)
    data, tick_lbl, names = [], [], []
    for m in sorted({k[0] for k in by_mb}):
        if m == "No-DP (upper)":
            continue
        rows = by_mb.get((m, B0), [])
        pooled = []
        for r in rows:
            pooled.extend(r.get("per_client_eps_out") or [])
        if pooled:
            data.append(pooled)
            gvals = [r.get("gini_eps_out", 0) for r in rows]
            gin = float(np.mean(gvals)) if gvals else 0.0
            names.append(m)
            tick_lbl.append("{}\nGini={:.2f}".format(m, gin))
    if not data:
        print("Fig C: skip")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        bp = ax.boxplot(data, tick_labels=tick_lbl, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=tick_lbl, patch_artist=True)
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(_COLORS.get(names[i], "#aaa"))
        p.set_alpha(0.7)
    ax.set_ylabel(r"Client $\epsilon_i^{\mathrm{out}}$ (final round)")
    ax.set_title("RQ5 (Fig. C): Distribution of privacy burden across clients @ $B={}$".format(B0))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def plot_fig_d(plot_a, Bs, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    focus = ["Ours", "CSRA", "MFG-Pricing", "Uniform-DP"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for m in focus:
        if m not in plot_a:
            continue
        pts = sorted(plot_a[m], key=lambda x: x[4])
        Bb = [p[4] for p in pts]
        eps = [p[0] for p in pts]
        acc = [p[1] * 100 for p in pts]
        c = _COLORS.get(m, "#333")
        ax1.plot(Bb, eps, "o-", color=c, label=m, lw=1.5)
        ax2.plot(Bb, acc, "s--", color=c, label=m, lw=1.5)
    ax1.set_xlabel("Budget rate $B$")
    ax1.set_ylabel(r"$\bar{\epsilon}$")
    ax1.set_title("(a) Privacy vs budget")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Budget rate $B$")
    ax2.set_ylabel(r"$\mathcal{A}_{\mathrm{test}}$ (\%)")
    ax2.set_title("(b) Accuracy vs budget")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("RQ5 (Fig. D): Budget scan", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def plot_fig_e(docs, by_mb, B_ref, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    B0 = _closest_B(by_mb, B_ref)
    focus = ["Ours", "CSRA", "Uniform-DP"]
    series = defaultdict(list)
    for d in docs:
        for r in d["runs"]:
            if abs(float(r["budget_rate"]) - B0) > 1e-4:
                continue
            if r["method"] not in focus:
                continue
            u = r.get("update_l2_norms") or []
            if u:
                series[r["method"]].append(np.array(u, dtype=float))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m in focus:
        arrs = series.get(m, [])
        if not arrs:
            continue
        L = min(len(a) for a in arrs)
        stack = np.stack([a[:L] for a in arrs], axis=0)
        t = np.arange(1, L + 1)
        mu, sd = stack.mean(0), stack.std(0)
        c = _COLORS.get(m, "#333")
        ax.plot(t, mu, label=m, color=c, lw=1.4)
        ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"$\|\mathbf{w}^{(t)}-\mathbf{w}^{(t-1)}\|_2$")
    ax.set_title("RQ5 (Fig. E, appendix): Global update norm @ $B={}$".format(B0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print("Wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq5-dir", type=str, default="run/privacy_paper/rq5")
    ap.add_argument("--b-ref", type=float, default=None, help="B for Fig B/C/E (default: median of observed B)")
    ap.add_argument(
        "--prefer-pagalg2",
        action="store_true",
        help="If raw mixes types, use *_pagalg2.json only",
    )
    ap.add_argument(
        "--dataset-filter",
        type=str,
        default="",
        metavar="DS",
        help="MNIST or CIFAR10. Required if raw mixes multiple datasets (else curves are wrong).",
    )
    args = ap.parse_args()
    raw_dir = os.path.join(args.rq5_dir, "raw")
    fig_dir = os.path.join(args.rq5_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    docs = _load_runs(raw_dir, prefer_pagalg2=args.prefer_pagalg2)
    if not docs:
        print("No JSON in", raw_dir)
        return 1
    uds = _datasets_in_docs(docs)
    multi_ds = len(uds) > 1
    if multi_ds and not (args.dataset_filter or "").strip():
        print(
            "[rq5_plot] ERROR: raw/ contains multiple datasets {}. "
            "Re-run with e.g. --dataset-filter MNIST and --dataset-filter CIFAR10 "
            "(separate calls), or use one dataset per output folder.".format(sorted(uds))
        )
        return 1
    filt = (args.dataset_filter or "").strip()
    docs = _filter_docs_dataset(docs, filt)
    if not docs:
        print("[rq5_plot] No JSON left after --dataset-filter", args.dataset_filter)
        return 1
    fig_tag = ("_" + _norm_dataset_from_meta({"dataset": filt})) if multi_ds else ""
    plot_a, Bs, by_mb = _aggregate(docs)
    if not Bs:
        print("[rq5_plot] No budget-rate points after aggregation.")
        return 1
    B_ref = args.b_ref if args.b_ref is not None else Bs[len(Bs) // 2]
    mo = ["Ours", "CSRA", "MFG-Pricing", "RegretNet", "PAC", "Uniform-DP"]
    plot_fig_a(plot_a, os.path.join(fig_dir, "figure_rq5_A_pareto_eps_acc{}.png".format(fig_tag)))
    plot_fig_b(by_mb, B_ref, mo, os.path.join(fig_dir, "figure_rq5_B_bars_privacy_utility{}.png".format(fig_tag)))
    plot_fig_c(by_mb, B_ref, os.path.join(fig_dir, "figure_rq5_C_client_eps_boxplot{}.png".format(fig_tag)))
    plot_fig_d(plot_a, Bs, os.path.join(fig_dir, "figure_rq5_D_budget_scan{}.png".format(fig_tag)))
    plot_fig_e(docs, by_mb, B_ref, os.path.join(fig_dir, "figure_rq5_E_update_l2{}.png".format(fig_tag)))
    if fig_tag and args.prefer_pagalg2:
        summ = "rq5_summary{}_pagalg2.json".format(fig_tag)
    elif fig_tag:
        summ = "rq5_summary{}.json".format(fig_tag)
    elif args.prefer_pagalg2:
        summ = "rq5_summary_pagalg2.json"
    else:
        summ = "rq5_summary.json"
    agg_path = os.path.join(args.rq5_dir, summ)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({"B_reference": B_ref, "budget_rates": Bs, "methods": list(plot_a.keys())}, f, indent=2)
    print("Wrote", agg_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

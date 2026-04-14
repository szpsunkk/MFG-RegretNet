#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3：拍卖收益 + 社会福利（定义见 exp_rq/RQ3_PROCESS.md）

  W^(t)=∑_i(p_i^(t)-v_i·ε_i^{(t),out})；W̄_s=(1/T)∑_t W^(t)（每种子）；
  报告 mean_s(W̄_s)±std_s(W̄_s)。R̄、η_rev 同理按轮次时间平均后再跨种子汇总。

  图1：η_rev 与 W̄（误差棒 = 跨种子 std）
  图2：R̄、W̄ 随训练 epoch（单种子展示轨迹；点线=基线）
  图3：预算扫描（W̄ 为每预算下多种子 mean±std）

一键：./run_rq3.sh 或 python3 exp_rq/rq3_paper_complete.py
"""
from __future__ import division, print_function

import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _display_name(key):
    return "Ours" if key == "MFG-RegretNet" else key


def _build_trade_mech_ls(n_items, rn, dm, mfg):
    ls = [
        ["PAC", "ConvlAggr", "", n_items],
        ["VCG", "ConvlAggr", "", n_items],
        ["CSRA", "ConvlAggr", "", n_items],
        ["MFG-Pricing", "ConvlAggr", "", n_items],
    ]
    if rn and os.path.isfile(rn):
        ls.append(["RegretNet", "ConvlAggr", rn, n_items])
    if dm and os.path.isfile(dm):
        ls.append(["DM-RegretNet", "ConvlAggr", dm, n_items])
    if mfg and os.path.isfile(mfg):
        ls.append(["MFG-RegretNet", "ConvlAggr", mfg, n_items])
    return ls


def rq3_eval_once(trade_mech, n_agents, n_items, budget, num_profiles, seed):
    """
    单个种子 s、T=num_profiles 轮：R̄_s、W̄_s 为对轮次的时间平均；η=R̄_s/B。
    """
    from run_phase4_eval import build_privacy_paper_batch, get_ckpt_path
    from experiments import auction, load_auc_model, DEVICE
    import torch

    name = trade_mech[0]
    if name in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
        model = None
    else:
        path = get_ckpt_path(trade_mech, n_agents)
        if not path or not os.path.isfile(path):
            return None
        model = load_auc_model(path).to(DEVICE)
    reports, bud, _ = build_privacy_paper_batch(
        num_profiles, n_agents, n_items, budget, seed, DEVICE
    )
    out = auction(reports, bud, trade_mech, model=model, return_payments=True, expected=True)
    plosses, _, payments = out[0], out[1], out[2]
    if payments.dim() == 2:
        rev = payments.sum(dim=1)
    else:
        rev = payments.sum(dim=-1)
    b = bud.squeeze()
    if b.dim() == 0:
        b = b.unsqueeze(0).expand(rev.shape[0])
    v = reports[:, :, 0].cpu().numpy()
    eps_out = plosses.detach().cpu().numpy()
    pay = payments.detach().cpu().numpy()
    cost = (v * eps_out).sum(axis=1)
    sw = pay.sum(axis=1) - cost
    bf = float((rev.cpu() <= b.cpu()).float().mean().item())
    mr = float(rev.mean().item())
    msw = float(sw.mean())
    eta = mr / float(budget) if budget > 0 else float("nan")
    return {"mean_revenue": mr, "bf_rate": bf, "eta_rev": eta, "mean_sw": msw}


def _list_neural_ckpts(pattern, n_agents, n_items, want_mfg, max_pts):
    import torch

    def _ep(p):
        m = re.search(r"_(\d+)_checkpoint\.pt$", p)
        return int(m.group(1)) if m else 0

    cand = sorted(glob.glob(pattern), key=_ep)
    out = []
    for p in cand:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            is_mfg = arch.get("model_type") == "MFGRegretNet"
            if want_mfg != is_mfg:
                continue
            na, ni = arch.get("n_agents"), arch.get("n_items")
            if na is not None and int(na) != int(n_agents):
                continue
            if ni is not None and int(ni) != int(n_items):
                continue
            out.append((_ep(p), p))
        except Exception:
            continue
    if len(out) > max_pts:
        idx = np.unique(np.linspace(0, len(out) - 1, max_pts, dtype=int))
        out = [out[i] for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-items", type=int, default=1)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--num-profiles", type=int, default=1000)
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--seed-fig2", type=int, default=42)
    ap.add_argument("--regretnet-ckpt", type=str, default="")
    ap.add_argument("--dm-regretnet-ckpt", type=str, default="")
    ap.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/rq3")
    ap.add_argument("--max-ckpts-fig2", type=int, default=15)
    ap.add_argument("--budget-multipliers", type=str, default="0.5,1.0,1.5,2.0")
    ap.add_argument("--num-profiles-fig3", type=int, default=600)
    ap.add_argument("--no-figure2", action="store_true")
    ap.add_argument("--no-figure3", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip().isdigit()]
    if not seeds:
        seeds = [42]

    rn = (args.regretnet_ckpt or "").strip()
    dm = (args.dm_regretnet_ckpt or "").strip()
    mfg = (args.mfg_regretnet_ckpt or "").strip()
    if not rn or not os.path.isfile(rn):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt
            r = resolve_regretnet_ckpt(args.n_agents, args.n_items)
            if r:
                rn = r
                print("[INFO] RegretNet:", rn)
        except Exception:
            pass
    if not dm or not os.path.isfile(dm):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt
            r = resolve_dm_regretnet_ckpt(args.n_agents, args.n_items)
            if r:
                dm = r
                print("[INFO] DM-RegretNet:", dm)
        except Exception:
            pass
    if not mfg or not os.path.isfile(mfg):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt
            r = resolve_mfg_regretnet_ckpt(args.n_agents, args.n_items)
            if r:
                mfg = r
                print("[INFO] MFG-RegretNet (Ours):", mfg)
        except Exception:
            pass

    trade_ls = _build_trade_mech_ls(args.n_items, rn, dm, mfg)
    from run_phase4_eval import rq3_revenue_privacy_paper

    rq3_rows = rq3_revenue_privacy_paper(
        trade_ls, args.n_agents, args.n_items, args.budget,
        num_profiles=args.num_profiles, seeds=seeds,
    )
    fig1_data = []
    for row in rq3_rows:
        name = row["mechanism"]
        fig1_data.append({
            "key": name,
            "display": _display_name(name),
            "mean_revenue": row["mean_revenue"],
            "std_revenue": row.get("std_revenue", 0.0),
            "bf_rate": row["bf_rate"],
            "revenue_efficiency": row["revenue_efficiency"],
            "std_revenue_efficiency": row.get("std_revenue_efficiency", 0.0),
            "mean_social_welfare": row["mean_social_welfare"],
            "std_social_welfare": row.get("std_social_welfare", 0.0),
            "num_rounds_T": row.get("num_rounds_T", args.num_profiles),
            "num_seeds": row.get("num_seeds", len(seeds)),
        })

    def _jdump(path, obj):
        def _fix(o):
            if isinstance(o, dict):
                return {k: _fix(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_fix(x) for x in o]
            if isinstance(o, float) and (o != o):
                return None
            return o

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_fix(obj), f, indent=2, ensure_ascii=False)

    _jdump(os.path.join(args.out_dir, "rq3_figure1_table.json"), fig1_data)

    # ---------- Figure 1 ----------
    try:
        import matplotlib.pyplot as plt

        labels = [x["display"] for x in fig1_data]
        etas = [0.0 if (isinstance(x["revenue_efficiency"], float) and x["revenue_efficiency"] != x["revenue_efficiency"]) else float(x["revenue_efficiency"]) for x in fig1_data]
        eta_err = [float(x.get("std_revenue_efficiency") or 0) for x in fig1_data]
        sws = [0.0 if (isinstance(x["mean_social_welfare"], float) and x["mean_social_welfare"] != x["mean_social_welfare"]) else float(x["mean_social_welfare"]) for x in fig1_data]
        sw_err = [float(x.get("std_social_welfare") or 0) for x in fig1_data]
        x = np.arange(len(labels))
        w = 0.38
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax0.bar(
            x - w / 2, etas, width=w, yerr=eta_err, capsize=3,
            label=r"$\eta_{\mathrm{rev}}=\bar R/B$", color="#4472c4", edgecolor="k",
        )
        ax0.set_ylabel(r"$\eta_{\mathrm{rev}}$ (mean $\pm$ std over seeds)")
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=22, ha="right")
        ax0.set_ylim(0, max(1.05, max(etas) * 1.15) if etas else (0, 1))
        ax0.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="B (if $\eta$=1)")
        ax0.set_title("(a) Revenue efficiency")
        ax0.grid(True, axis="y", alpha=0.3)
        ax0.legend(loc="upper right", fontsize=8)

        ax1.bar(x, sws, yerr=sw_err, capsize=3, color="#c55a11", edgecolor="k")
        ax1.axhline(0.0, color="k", linewidth=1.0)
        ax1.set_ylabel(
            r"$\bar{\mathcal{W}}$ = mean$_s(\frac{1}{T}\sum_t \mathcal{W}^{(t)})$ $\pm$ std$_s$"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=22, ha="right")
        ax1.set_title("(b) Time-averaged welfare / seed, then mean±std over seeds")
        ax1.grid(True, axis="y", alpha=0.3)
        fig.suptitle(
            "RQ3-Fig1: η_rev & W̄ (T auction rounds per seed; error bars = across-seed std)", y=1.03
        )
        fig.tight_layout()
        p1 = os.path.join(args.out_dir, "figure_rq3_revenue_welfare_bars.png")
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print("Wrote", p1)
    except ImportError:
        print("matplotlib missing, skip fig1")

    # baseline R, SW for fig2
    baseline_R = {}
    baseline_SW = {}
    for row in fig1_data:
        if row["key"] in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
            baseline_R[row["display"]] = row["mean_revenue"]
            baseline_SW[row["display"]] = row["mean_social_welfare"]

    # ---------- Figure 2 ----------
    fig2_json = {"series": {}, "baselines_R": baseline_R, "baselines_SW": baseline_SW}
    if not args.no_figure2:
        try:
            import matplotlib.pyplot as plt

            def sweep_neural(label, pattern, want_mfg, tm_prefix):
                items = _list_neural_ckpts(
                    pattern, args.n_agents, args.n_items, want_mfg, args.max_ckpts_fig2
                )
                if not items:
                    return None, None, None
                epochs, Rs, SWs = [], [], []
                for ep, path in items:
                    tm = [tm_prefix, "ConvlAggr", path, args.n_items]
                    m = rq3_eval_once(tm, args.n_agents, args.n_items, args.budget, min(args.num_profiles, 800), args.seed_fig2)
                    if m:
                        epochs.append(ep)
                        Rs.append(m["mean_revenue"])
                        SWs.append(m["mean_sw"])
                return epochs, Rs, SWs

            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
            cmap = ["#1f77b4", "#ff7f0e", "#d62728"]
            for i, (lab, pat, wm, pref) in enumerate([
                ("RegretNet", "result/regretnet_privacy_*_checkpoint.pt", False, "RegretNet"),
                ("DM-RegretNet", "result/dm_regretnet_privacy_*_checkpoint.pt", False, "DM-RegretNet"),
                ("Ours", "result/mfg_regretnet_privacy_*_checkpoint.pt", True, "MFG-RegretNet"),
            ]):
                e, Rr, Ss = sweep_neural(lab, pat, wm, pref)
                if e:
                    fig2_json["series"][lab] = {"epochs": e, "R": Rr, "W": Ss}
                    ax0.plot(e, Rr, marker="o", ms=4, label=lab, color=cmap[i % 3])
                    ax1.plot(e, Ss, marker="s", ms=4, label=lab, color=cmap[i % 3])
            xmax = max([max(s["epochs"]) for s in fig2_json["series"].values()] or [1])
            for k, v in baseline_R.items():
                ax0.hlines(v, 0, xmax, linestyles=":", linewidth=1.2, alpha=0.7)
            for k, v in baseline_SW.items():
                ax1.hlines(v, 0, xmax, linestyles=":", linewidth=1.2, alpha=0.7)
            ax1.axhline(0, color="k", lw=0.8)
            ax0.set_ylabel("Mean revenue $R$")
            ax0.set_title(
                "RQ3-Fig2: time-avg R̄, W̄ per seed vs epoch (1 seed); dotted = baselines"
            )
            ax0.legend(loc="best", fontsize=8)
            ax0.grid(True, alpha=0.3)
            ax1.set_ylabel(r"$\bar{\mathcal{W}}_s$ (1 seed, T rounds)")
            ax1.set_xlabel("Training epoch $t$")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            p2 = os.path.join(args.out_dir, "figure_rq3_R_W_vs_epoch.png")
            fig.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close()
            print("Wrote", p2)
        except Exception as ex:
            print("[WARN] Fig2:", ex)

    _jdump(os.path.join(args.out_dir, "rq3_figure2.json"), fig2_json)

    # ---------- Figure 3 ----------
    mults = [float(x.strip()) for x in args.budget_multipliers.split(",") if x.strip()]
    fig3_data = {"budgets": [], "methods": {}}
    if not args.no_figure3 and mults:
        B0 = args.budget
        try:
            for tm in trade_ls:
                key = tm[0]
                disp = _display_name(key)
                fig3_data["methods"][disp] = {"eta": [], "eta_err": [], "sw": [], "sw_err": [], "B": []}
                for m in mults:
                    B = B0 * m
                    agg_eta, agg_sw = [], []
                    for sd in seeds:
                        r = rq3_eval_once(tm, args.n_agents, args.n_items, B, args.num_profiles_fig3, sd)
                        if r:
                            agg_eta.append(r["eta_rev"])
                            agg_sw.append(r["mean_sw"])
                    if agg_eta:
                        fig3_data["methods"][disp]["B"].append(B)
                        fig3_data["methods"][disp]["eta"].append(float(np.mean(agg_eta)))
                        fig3_data["methods"][disp]["eta_err"].append(
                            float(np.std(agg_eta, ddof=1)) if len(agg_eta) > 1 else 0.0
                        )
                        fig3_data["methods"][disp]["sw"].append(float(np.mean(agg_sw)))
                        fig3_data["methods"][disp]["sw_err"].append(
                            float(np.std(agg_sw, ddof=1)) if len(agg_sw) > 1 else 0.0
                        )
                fig3_data["budgets"] = [B0 * m for m in mults]

            import matplotlib.pyplot as plt

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
            cols = plt.cm.tab10(np.linspace(0, 0.9, max(8, len(fig3_data["methods"]))))
            for disp, col in zip(fig3_data["methods"].keys(), cols):
                d = fig3_data["methods"][disp]
                if not d["B"]:
                    continue
                ax0.errorbar(
                    d["B"], d["eta"], yerr=d.get("eta_err", [0] * len(d["B"])),
                    marker="o", label=disp, linewidth=1.2, capsize=2, color=col,
                )
                ax1.errorbar(
                    d["B"], d["sw"], yerr=d.get("sw_err", [0] * len(d["B"])),
                    marker="s", label=disp, linewidth=1.2, capsize=2, color=col,
                )
            ax1.axhline(0, color="k", lw=0.8)
            ax0.set_xlabel("Budget $B$")
            ax0.set_ylabel(r"$\eta_{\mathrm{rev}}$")
            ax0.set_title("(a) Revenue efficiency vs budget")
            ax0.grid(True, alpha=0.3)
            ax1.set_xlabel("Budget $B$")
            ax1.set_ylabel(r"$\bar{\mathcal{W}}$ (mean±std over seeds)")
            ax1.set_title("(b) Time-avg welfare vs budget")
            ax1.grid(True, alpha=0.3)
            ax0.legend(fontsize=7, loc="best")
            ax1.legend(fontsize=7, loc="best")
            fig.suptitle("RQ3-Fig3: Budget sensitivity", y=1.02)
            fig.tight_layout()
            p3 = os.path.join(args.out_dir, "figure_rq3_budget_sensitivity.png")
            fig.savefig(p3, dpi=150, bbox_inches="tight")
            plt.close()
            print("Wrote", p3)
        except Exception as ex:
            print("[WARN] Fig3:", ex)

    _jdump(os.path.join(args.out_dir, "rq3_figure3.json"), fig3_data)

    # Markdown table
    md = os.path.join(args.out_dir, "table_rq3_paper.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# RQ3: Revenue & social welfare\n\n")
        f.write("| Method | R̄ (mean±std) | BF | η_rev (mean±std) | W̄ (mean±std) | T | seeds |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        def _fmt(v):
            if isinstance(v, float) and (v != v):
                return "—"
            return "{:.4f}".format(v)

        def pm(m, s):
            if isinstance(m, float) and (m != m):
                return "—"
            if not s or (isinstance(s, float) and s == 0):
                return _fmt(m)
            return "{:.4f} ± {:.4f}".format(m, s)

        for x in fig1_data:
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                x["display"],
                pm(x["mean_revenue"], x.get("std_revenue")),
                _fmt(x["bf_rate"]),
                pm(x["revenue_efficiency"], x.get("std_revenue_efficiency")),
                pm(x["mean_social_welfare"], x.get("std_social_welfare")),
                x.get("num_rounds_T", ""),
                x.get("num_seeds", ""),
            ))
    print("Wrote", md)
    print("RQ3 complete ->", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

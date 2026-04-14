#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 论文主表 + 主图 A（遗憾柱）+ 主图 B（IR% 柱）。
行：PAC, VCG, CSRA, RegretNet, DM-RegretNet, MFG-Pricing（均值场按ε比例定价）, Ours=MFG-RegretNet
列：Avg ex-post regret, IR violation (%), Truthful rate (%), Bid stability (CV)
5 seeds、paired t-test（Ours vs 各法）见 table 脚注与 rq1_paper.json。
"""
from __future__ import division, print_function

import argparse
import csv
import json
import os
import sys

import numpy as np

# 论文图中的显示顺序与标签
METHOD_ROWS = [
    ("PAC", "PAC", "baseline"),
    ("VCG", "VCG", "baseline"),
    ("CSRA", "CSRA", "baseline"),
    ("RegretNet", "RegretNet", "neural"),
    ("DM-RegretNet", "DM-RegretNet", "neural"),
    ("MFG-Pricing", "MFG-Pricing", "baseline"),
    ("MFG-RegretNet", "Ours", "neural"),
]


def _parse_seeds(s):
    out = []
    for x in s.split(","):
        x = x.strip()
        if x.isdigit():
            out.append(int(x))
    return out or [42]


def _baseline_seed_metrics(mech_name, n_agents, n_items, budget, num_profiles, seed, batch_size, v_grid, eps_grid):
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import DEVICE
    from exp_rq.guarantees_eval_baselines import guarantees_eval_procurement_baseline
    from datasets import Dataloader
    import torch

    reports, _, val_type = build_privacy_paper_batch(
        num_profiles, n_agents, n_items, budget, seed, DEVICE
    )
    loader = Dataloader(torch.cat([reports, val_type], dim=2), batch_size=batch_size, shuffle=False)
    regs, irs, ir_pct_l, truth_l, cv_l = [], [], [], [], []
    for batch in loader:
        rep = batch[:, :, :-2].to(DEVICE)
        b = budget * torch.ones(rep.shape[0], 1, device=DEVICE)
        r, irn = guarantees_eval_procurement_baseline(
            rep, b, mech_name, v_grid_n=v_grid, eps_grid_n=eps_grid
        )
        from baselines.pac import pac_batch
        from baselines.vcg import vcg_procurement_batch
        from baselines.csra import csra_qms_batch
        from baselines.mfg_pricing import mfg_pricing_batch
        fd = {
            "PAC": pac_batch,
            "VCG": vcg_procurement_batch,
            "CSRA": csra_qms_batch,
            "MFG-Pricing": mfg_pricing_batch,
        }[mech_name]
        pl0, pay0 = fd(rep, b)
        util0 = pay0 - rep[:, :, 0] * pl0
        ir_pct_l.append((util0 < 0).float().mean().item() * 100.0)
        rv = r.detach().flatten()
        truth_l.append((rv < 0.02).float().mean().item() * 100.0)
        pay = pay0
        cv_row = pay.std(dim=1) / (pay.abs().mean(dim=1) + 1e-6)
        cv_l.append(float(cv_row.mean().item()))
        regs.append(rv.cpu().numpy())
        irs.append(irn.detach().flatten().cpu().numpy())
    return {
        "mean_regret": float(np.concatenate(regs).mean()),
        "mean_ir_norm": float(np.concatenate(irs).mean()),
        "ir_violation_pct": float(np.mean(ir_pct_l)),
        "truthful_pct": float(np.mean(truth_l)),
        "bid_cv": float(np.mean(cv_l)),
    }


def _neural_seed_metrics(name, ckpt_path, n_agents, n_items, budget, num_profiles, seed, batch_size, misreport_iter, regret_tol):
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import load_auc_model, DEVICE
    from regretnet import MFGRegretNet
    from utils import optimize_misreports, tiled_misreport_util, calc_agent_util, allocs_to_plosses
    from datasets import Dataloader
    import torch

    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    try:
        model = load_auc_model(ckpt_path).to(DEVICE)
    except Exception as e:
        print("[WARN] {} load failed: {}".format(name, e))
        return None
    model.eval()
    cost_from_plosses = isinstance(model, MFGRegretNet) and n_items == 1

    reports, _, val_type = build_privacy_paper_batch(
        num_profiles, n_agents, n_items, budget, seed, DEVICE
    )
    loader = Dataloader(torch.cat([reports, val_type], dim=2), batch_size=batch_size, shuffle=False)

    regs, ir_pct_l, truth_l, cv_l = [], [], [], []
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
        regrets = torch.clamp(untruthful_util - truthful_util, min=0)
        costs_safe = costs.clamp(min=1e-10)
        rnorm = regrets / costs_safe
        ir_pct_l.append((truthful_util < 0).float().mean().item() * 100.0)
        truth_l.append((rnorm < regret_tol).float().mean().item() * 100.0)
        msv = misreports[:, :, 0]
        cv_l.append(float((msv.std(dim=1) / (msv.abs().mean(dim=1) + 1e-6)).mean().item()))
        regs.append(rnorm.detach().flatten().cpu().numpy())

    return {
        "mean_regret": float(np.concatenate(regs).mean()),
        "mean_ir_norm": 0.0,
        "ir_violation_pct": float(np.mean(ir_pct_l)),
        "truthful_pct": float(np.mean(truth_l)),
        "bid_cv": float(np.mean(cv_l)),
    }


def _aggregate(per_seed_dicts):
    """per_seed_dicts: list of metric dicts (one per seed)"""
    keys = ["mean_regret", "ir_violation_pct", "truthful_pct", "bid_cv"]
    out = {}
    for k in keys:
        xs = [d[k] for d in per_seed_dicts]
        out[k + "_mean"] = float(np.mean(xs))
        out[k + "_std"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return out


def _fmt_pm(m, s):
    return "{:.4f} ± {:.4f}".format(m, s) if s > 0 else "{:.4f}".format(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-items", type=int, default=1)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--num-profiles", type=int, default=1000)
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--baseline-v-grid", type=int, default=15)
    ap.add_argument("--baseline-eps-grid", type=int, default=7)
    ap.add_argument("--misreport-iter", type=int, default=25)
    ap.add_argument("--regret-tol-truthful", type=float, default=0.02, help="normalized regret below → truthful")
    ap.add_argument("--regretnet-ckpt", type=str, default="")
    ap.add_argument("--dm-regretnet-ckpt", type=str, default="")
    ap.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/rq1")
    ap.add_argument("--ir-log-scale", action="store_true", help="主图 B 纵轴对数")
    ap.add_argument("--regret-log-scale", action="store_true", help="主图 A 纵轴对数（矮柱仍建议看标注/小图）")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    _rc = (args.regretnet_ckpt or "").strip()
    if not _rc or not os.path.isfile(_rc):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt
            _r = resolve_regretnet_ckpt(args.n_agents, args.n_items)
            if _r:
                args.regretnet_ckpt = _r
                print("[INFO] RegretNet ckpt (auto):", _r)
        except Exception:
            pass

    _dc = (args.dm_regretnet_ckpt or "").strip()
    if not _dc or not os.path.isfile(_dc):
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt
            _d = resolve_dm_regretnet_ckpt(args.n_agents, args.n_items)
            if _d:
                args.dm_regretnet_ckpt = _d
                print("[INFO] DM-RegretNet ckpt (auto):", _d)
        except Exception:
            pass

    seeds = _parse_seeds(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = {
        "RegretNet": args.regretnet_ckpt.strip(),
        "DM-RegretNet": args.dm_regretnet_ckpt.strip(),
        "MFG-RegretNet": args.mfg_regretnet_ckpt.strip(),
    }

    rows_out = []
    per_method_seeds = {}  # key -> {seed: mean_regret} for paired tests

    for key, display, kind in METHOD_ROWS:
        per_seed = []
        if kind == "baseline":
            for seed in seeds:
                try:
                    m = _baseline_seed_metrics(
                        key, args.n_agents, args.n_items, args.budget,
                        args.num_profiles, seed, args.batch_size,
                        args.baseline_v_grid, args.baseline_eps_grid,
                    )
                    per_seed.append(m)
                except Exception as e:
                    print("[ERROR] {} seed {}: {}".format(key, seed, e))
                    per_seed = None
                    break
        else:
            path = ckpts.get(key, "")
            if not path or not os.path.isfile(path):
                rows_out.append({
                    "key": key, "display": display, "kind": "neural_skip",
                    "regret": "—", "ir_pct": "—", "truthful_pct": "—", "bid_cv": "—",
                    "note": "无 checkpoint 或架构不匹配",
                })
                continue
            for seed in seeds:
                try:
                    m = _neural_seed_metrics(
                        key, path, args.n_agents, args.n_items, args.budget,
                        args.num_profiles, seed, args.batch_size,
                        args.misreport_iter, args.regret_tol_truthful,
                    )
                    if m is None:
                        per_seed = None
                        break
                    per_seed.append(m)
                except Exception as e:
                    print("[ERROR] {} seed {}: {}".format(key, seed, e))
                    per_seed = None
                    break

        if not per_seed:
            if not any(r.get("key") == key for r in rows_out):
                rows_out.append({
                    "key": key, "display": display, "kind": "error",
                    "regret": "—", "ir_pct": "—", "truthful_pct": "—", "bid_cv": "—",
                    "note": "eval failed",
                })
            continue

        agg = _aggregate(per_seed)
        rows_out.append({
            "key": key,
            "display": display,
            "kind": kind,
            "regret": _fmt_pm(agg["mean_regret_mean"], agg["mean_regret_std"]),
            "ir_pct": _fmt_pm(agg["ir_violation_pct_mean"], agg["ir_violation_pct_std"]),
            "truthful_pct": _fmt_pm(agg["truthful_pct_mean"], agg["truthful_pct_std"]),
            "bid_cv": _fmt_pm(agg["bid_cv_mean"], agg["bid_cv_std"]),
            "mean_regret_mean": agg["mean_regret_mean"],
            "mean_regret_std": agg["mean_regret_std"],
            "ir_violation_pct_mean": agg["ir_violation_pct_mean"],
            "ir_violation_pct_std": agg["ir_violation_pct_std"],
            "truthful_pct_mean": agg["truthful_pct_mean"],
            "truthful_pct_std": agg["truthful_pct_std"],
            "bid_cv_mean": agg["bid_cv_mean"],
            "bid_cv_std": agg["bid_cv_std"],
            "per_seed": [{"seed": seeds[i], **per_seed[i]} for i in range(len(seeds))],
        })
        per_method_seeds[key] = {seeds[i]: per_seed[i]["mean_regret"] for i in range(len(seeds))}

    # paired t-test: Ours vs others
    tests = {}
    ours_key = "MFG-RegretNet"
    if ours_key in per_method_seeds and len(per_method_seeds[ours_key]) >= 2:
        o = per_method_seeds[ours_key]
        for key, _, knd in METHOD_ROWS:
            if key == ours_key or knd in ("na",):
                continue
            if key not in per_method_seeds:
                continue
            b = per_method_seeds[key]
            aligned = [(o[s], b[s]) for s in seeds if s in o and s in b]
            if len(aligned) < 2:
                continue
            a0 = np.array([x[0] for x in aligned])
            a1 = np.array([x[1] for x in aligned])
            try:
                from scipy import stats
                t, p = stats.ttest_rel(a0, a1)
                tests[key] = {"statistic": float(t), "pvalue": float(p), "significant_005": bool(p < 0.05)}
            except ImportError:
                tests[key] = {"error": "install scipy for paired t-test"}
            except Exception as e:
                tests[key] = {"error": str(e)}

    payload = {
        "seeds": seeds,
        "n_profiles": args.num_profiles,
        "paired_ttest_ours_vs": tests,
        "footnote": "MFG-Pricing: posted budget split pay_i=B·ε_i/Σε_j, pl_i=ε_i (mean-field style). "
        "IR violation % = fraction of agent-slots with u_i=p_i−v_i·ε_alloc<0. "
                    "Truthful % = normalized regret < {:.2f} (neural after PGA; baselines grid). "
                    "Bid CV = CV of strategic v (neural) or payments (baselines) across agents per profile.".format(
                        args.regret_tol_truthful),
        "rows": rows_out,
    }
    with open(os.path.join(args.out_dir, "rq1_paper.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.out_dir, "table_rq1_paper.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Avg_expost_regret", "IR_violation_pct", "Truthful_pct", "Bid_stability_CV", "note"])
        for r in rows_out:
            w.writerow([
                r["display"],
                r.get("regret", ""),
                r.get("ir_pct", ""),
                r.get("truthful_pct", ""),
                r.get("bid_cv", ""),
                r.get("note", ""),
            ])

    md_path = os.path.join(args.out_dir, "table_rq1_paper.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RQ1 Main table\n\n")
        f.write("| Method | $\\bar{\\mathrm{rgt}}$ | $\\mathcal{V}_{\\mathrm{IR}}$ (%) | Truthful (%) | Bid CV |\n")
        f.write("|--------|----------------------|-----------------------------------|--------------|--------|\n")
        for r in rows_out:
            f.write("| {} | {} | {} | {} | {} |\n".format(
                r["display"],
                r.get("regret", "—"),
                r.get("ir_pct", "—"),
                r.get("truthful_pct", "—"),
                r.get("bid_cv", "—"),
            ))
        f.write("\n*{} seeds: {}. Paired t-test (Ours vs method) on mean regret per seed:*\n\n".format(
            len(seeds), ",".join(map(str, seeds))))
        for k, v in sorted(tests.items()):
            if "pvalue" in v:
                star = " *" if v.get("significant_005") else ""
                f.write("- **Ours vs {}**: p = {:.4f}{}\n".format(k, v["pvalue"], star))
            else:
                f.write("- **Ours vs {}**: {}\n".format(k, v.get("error", v)))
        f.write("\n* p<0.05.\n\n")
        f.write(payload["footnote"] + "\n")

    # Figures: 7 bars（MFG-Pricing 为可评测基线）
    if not args.no_figure:
        try:
            import matplotlib.pyplot as plt

            def _series(field_mean, field_std):
                lm, ls = [], []
                row_by_key = {r["key"]: r for r in rows_out if "key" in r}
                for key, disp, knd in METHOD_ROWS:
                    r = row_by_key.get(key)
                    if r and field_mean in r:
                        lm.append(r[field_mean])
                        ls.append(r[field_std])
                    else:
                        lm.append(0.0)
                        ls.append(0.0)
                return lm, ls

            from matplotlib.ticker import MaxNLocator

            labels = [disp for _, disp, _ in METHOD_ROWS]
            x = np.arange(len(labels))
            reg_m, reg_s = _series("mean_regret_mean", "mean_regret_std")
            reg_m_arr = np.maximum(np.array(reg_m, dtype=float), 0.0)
            reg_s_arr = np.array(reg_s, dtype=float)
            colors = ["#4472c4"] * len(labels)
            for i, (_, _, knd) in enumerate(METHOD_ROWS):
                if knd == "na":
                    colors[i] = "#d9d9d9"
            fig, ax = plt.subplots(figsize=(10, 4.6))
            if args.regret_log_scale:
                reg_plot = np.maximum(reg_m_arr, 1e-14)
                bars = ax.bar(x, reg_plot, yerr=reg_s_arr, capsize=3, color=colors, edgecolor="k", linewidth=0.5)
                ax.set_yscale("log")
            else:
                bars = ax.bar(x, reg_m_arr, yerr=reg_s_arr, capsize=3, color=colors, edgecolor="k", linewidth=0.5)
            for i, (_, _, knd) in enumerate(METHOD_ROWS):
                if knd == "na":
                    bars[i].set_hatch("//")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(r"$\bar{\mathrm{rgt}}$ (avg. ex-post regret)")
            ax.set_title(
                "RQ1-A: Regret (mean $\\pm$ std, {} seeds){}".format(
                    len(seeds), ", log-y" if args.regret_log_scale else "",
                )
            )
            ax.grid(True, axis="y", alpha=0.3)
            if not args.regret_log_scale:
                ytop = float(np.nanmax(reg_m_arr + reg_s_arr))
                if not np.isfinite(ytop) or ytop <= 0:
                    ytop = max(float(np.nanmax(reg_m_arr)), 1e-12) * 1.25
                ytop = max(ytop * 1.16, float(np.nanmax(reg_m_arr)) * 1.08 + 1e-15)
                ytop = max(ytop, 1e-10)
                ax.set_ylim(0.0, ytop)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=12, min_n_ticks=6))
                # 矮柱 + 大尺度：柱顶/近轴标注数值
                ylim = ax.get_ylim()[1]

                def _fmt_rgt(v):
                    if abs(v) < 1e-8:
                        return "0"
                    if ylim > 1e3 or (ylim > 0 and abs(v) < ylim * 1e-3):
                        return "{:.2e}".format(v)
                    return "{:.4g}".format(v)

                for i, (_, _, knd) in enumerate(METHOD_ROWS):
                    v, se = float(reg_m[i]), float(reg_s[i])
                    if knd == "na":
                        ax.text(
                            x[i], ylim * 0.015, "N/A", ha="center", va="bottom", fontsize=9, color="#444",
                        )
                    elif v + se < ylim * 0.045:
                        ax.text(
                            x[i],
                            ylim * 0.038,
                            _fmt_rgt(v),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#e8f0fc", edgecolor="#4472c4", alpha=0.95),
                        )
                    else:
                        ax.text(
                            x[i],
                            min(v + se + ylim * 0.02, ylim * 0.985),
                            _fmt_rgt(v),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                # 与 IR 图类似：最大 regret 远大于部分方法时，左上角小图放大低 regret
                mx_r = max(
                    (float(reg_m[j]) for j in range(len(METHOD_ROWS)) if METHOD_ROWS[j][2] != "na"),
                    default=0.0,
                )
                if mx_r > 1e-12:
                    small_idx = [
                        j
                        for j in range(len(METHOD_ROWS))
                        if METHOD_ROWS[j][2] != "na"
                        and float(reg_m[j]) < 0.35 * mx_r
                        and not (METHOD_ROWS[j][0] == "DM-RegretNet" and float(reg_m[j]) < 1e-12)
                    ]
                    if len(small_idx) >= 2:
                        try:
                            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                            axins = inset_axes(
                                ax, width="40%", height="44%", loc="upper left",
                                bbox_to_anchor=(0.01, 0.99, 1, 1), bbox_transform=ax.transAxes, borderpad=0,
                            )
                            xs2 = np.arange(len(small_idx))
                            vv = [float(reg_m[j]) for j in small_idx]
                            ss = [float(reg_s[j]) for j in small_idx]
                            axins.bar(xs2, vv, yerr=ss, capsize=2, color="#4472c4", edgecolor="k", linewidth=0.4)
                            axins.set_xticks(xs2)
                            axins.set_xticklabels(
                                [METHOD_ROWS[j][1][:9] for j in small_idx], rotation=30, ha="right", fontsize=7,
                            )
                            ztop = max((max(vv) + max(ss)) if vv else 1e-12, 1e-12) * 1.45
                            ztop = min(ztop, 0.55 * mx_r)
                            axins.set_ylim(0, max(ztop, max(vv) * 1.15 + max(ss) + 1e-12, 1e-10))
                            axins.set_ylabel(r"$\bar{\mathrm{rgt}}$", fontsize=7)
                            axins.yaxis.set_major_locator(MaxNLocator(nbins=6))
                            axins.grid(True, axis="y", alpha=0.3)
                            axins.set_title("Low regret (same scale as main)", fontsize=8)
                            for ji, xi in enumerate(xs2):
                                yu = axins.get_ylim()[1]
                                axins.text(
                                    xi,
                                    min(vv[ji] + ss[ji] + yu * 0.04, yu * 0.92),
                                    _fmt_rgt(vv[ji]),
                                    ha="center",
                                    fontsize=6,
                                )
                        except Exception:
                            pass
            fig.tight_layout()
            p1 = os.path.join(args.out_dir, "figure_rq1_paper_regret.png")
            fig.savefig(p1, dpi=150, bbox_inches="tight")
            plt.close()
            print("Wrote", p1)

            ir_m, ir_s = _series("ir_violation_pct_mean", "ir_violation_pct_std")
            ir_plot = np.maximum(np.array(ir_m, dtype=float), 1e-6) if args.ir_log_scale else np.array(ir_m, dtype=float)
            fig, ax = plt.subplots(figsize=(10, 4.8))
            bars = ax.bar(x, ir_plot, yerr=ir_s, capsize=3, color="#c55a11", edgecolor="k", linewidth=0.5)
            for i, (_, _, knd) in enumerate(METHOD_ROWS):
                if knd == "na":
                    bars[i].set_color("#d9d9d9")
                    bars[i].set_hatch("//")
            if args.ir_log_scale:
                ax.set_yscale("log")
            else:
                from matplotlib.ticker import MaxNLocator

                ytop = float(np.max(np.array(ir_m) + np.array(ir_s))) * 1.18
                ytop = max(ytop, 0.5)
                ax.set_ylim(0.0, ytop)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=12, min_n_ticks=6))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(r"$\mathcal{V}_{\mathrm{IR}}$ (%)")
            ax.set_title("RQ1-B: IR violation rate (mean $\\pm$ std{})".format(", log-y" if args.ir_log_scale else ""))
            ax.grid(True, axis="y", alpha=0.3)
            # 柱太矮时主图上看不见：柱顶/近轴标注数值
            if not args.ir_log_scale:
                ylim = ax.get_ylim()[1]
                for i, (_, _, knd) in enumerate(METHOD_ROWS):
                    v, se = float(ir_m[i]), float(ir_s[i])
                    if knd == "na":
                        ax.text(
                            x[i], ylim * 0.02, "N/A", ha="center", va="bottom", fontsize=9, color="#444",
                        )
                    elif v + se < ylim * 0.04:
                        ax.text(
                            x[i],
                            ylim * 0.035,
                            "{:.3f}%".format(v),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff8e6", edgecolor="#c55a11", alpha=0.95),
                        )
                    else:
                        ax.text(
                            x[i],
                            min(v + se + ylim * 0.025, ylim * 0.99),
                            "{:.2f}%".format(v),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                # 量级差大时加局部放大（PAC/VCG/RN/Ours 等小 IR）
                mx = max(ir_m)
                if mx > 5.0:
                    small_idx = [
                        i for i in range(len(METHOD_ROWS))
                        if METHOD_ROWS[i][2] != "na"
                        and float(ir_m[i]) < 0.35 * mx
                        and not (
                            METHOD_ROWS[i][0] == "DM-RegretNet" and float(ir_m[i]) < 1e-6
                        )
                    ]
                    if len(small_idx) >= 2:
                        try:
                            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                            axins = inset_axes(
                                ax, width="38%", height="42%", loc="upper left",
                                bbox_to_anchor=(0.01, 0.99, 1, 1), bbox_transform=ax.transAxes, borderpad=0,
                            )
                            xs = np.arange(len(small_idx))
                            vv = [float(ir_m[j]) for j in small_idx]
                            ss = [float(ir_s[j]) for j in small_idx]
                            axins.bar(xs, vv, yerr=ss, capsize=2, color="#c55a11", edgecolor="k", linewidth=0.4)
                            axins.set_xticks(xs)
                            axins.set_xticklabels(
                                [METHOD_ROWS[j][1][:9] for j in small_idx], rotation=30, ha="right", fontsize=7,
                            )
                            ztop = max((max(vv) + max(ss)) if vv else 0.01, 0.01) * 1.4
                            ztop = min(ztop, 0.5 * mx)
                            axins.set_ylim(0, max(ztop, 0.6))
                            axins.set_ylabel(r"IR %", fontsize=7)
                            axins.yaxis.set_major_locator(MaxNLocator(nbins=6))
                            axins.grid(True, axis="y", alpha=0.3)
                            axins.set_title("Low IR (same unit %)", fontsize=8)
                            for j, xi in enumerate(xs):
                                axins.text(
                                    xi, min(vv[j] + ss[j] + 0.05, axins.get_ylim()[1] * 0.92),
                                    "{:.2f}".format(vv[j]), ha="center", fontsize=6,
                                )
                        except Exception:
                            pass
            fig.tight_layout()
            p2 = os.path.join(args.out_dir, "figure_rq1_paper_ir.png")
            fig.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close()
            print("Wrote", p2)
        except ImportError:
            print("matplotlib missing, skip figures")

    print("RQ1 paper table/figures done ->", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

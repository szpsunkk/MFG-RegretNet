#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2 论文三张图：log-log 时间+参考线、通信开销、堆叠延迟。

输入: run/privacy_paper/rq2/rq2_paper_data.json
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

import numpy as np

_COLORS = {
    "Ours":        "#1b5e20",
    "RegretNet":   "#e65100",
    "CSRA":        "#1565c0",
    "PAC":         "#546e7a",
    "VCG":         "#78909c",
    "MFG-Pricing": "#6a1b9a",
}
_MARKERS = {
    "Ours": "o", "RegretNet": "s", "CSRA": "^",
    "PAC": "D", "VCG": "v", "MFG-Pricing": "P",
}
# 论文中各方法的理论/实际复杂度标注
_COMPLEXITY = {
    "PAC": "O(N log N)",
    "VCG": "O(N log N)",
    "CSRA": "O(N log N)",
    "MFG-Pricing": "O(1)",
    "Ours": "O(N)",
    "RegretNet": "O(N)",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Fig 1: log-log time vs N
# ─────────────────────────────────────────────
def fig1_loglog_time(data, out_path):
    import matplotlib.pyplot as plt

    rows = data.get("rq2_time_rows") or []

    # 按方法收集 (N, t) 列表
    mech_data = {}
    for r in rows:
        m = r["mechanism"]
        n = int(r["n_agents"])
        t = max(float(r["mean_time_sec"]), 1e-9)
        mech_data.setdefault(m, []).append((n, t))
    for m in mech_data:
        mech_data[m].sort()

    fig, ax = plt.subplots(figsize=(7, 5.2))

    # 绘制顺序：先画基线，再画神经方法（在上层）
    draw_order = ["MFG-Pricing", "PAC", "VCG", "CSRA", "RegretNet", "Ours"]
    plotted = []
    for m in draw_order:
        if m not in mech_data or m in plotted:
            continue
        pts = mech_data[m]
        ns  = [p[0] for p in pts]
        ts  = [p[1] for p in pts]
        color  = _COLORS.get(m, "#333333")
        marker = _MARKERS.get(m, "o")
        cplx   = _COMPLEXITY.get(m, "")
        label  = f"{m}  [{cplx}]"

        if len(pts) == 1:
            # 单点：只画点，不画线，加 * 注释
            ax.loglog(ns, ts, marker=marker, linestyle="None",
                      color=color, ms=10, alpha=0.9,
                      label=label + " *", zorder=5)
        else:
            ax.loglog(ns, ts, marker=marker, linestyle="-",
                      color=color, lw=2.0, ms=7, alpha=0.9,
                      label=label, zorder=4)
        plotted.append(m)

    # --- 参考线（锚在 PAC N=10 那个点上）---
    ref_m = next((m for m in ("PAC", "VCG", "CSRA") if m in mech_data
                  and len(mech_data[m]) >= 2), None)
    if ref_m:
        ref_n0 = mech_data[ref_m][0][0]
        ref_t0 = mech_data[ref_m][0][1]
        n_max  = max(max(p[0] for p in pts) for pts in mech_data.values())
        nn = np.logspace(np.log10(ref_n0), np.log10(n_max * 1.1), 80)
        ax.loglog(nn, ref_t0 * (nn / ref_n0),
                  "k--", lw=1.2, alpha=0.55, label=r"$O(N)$ ref", zorder=1)
        ax.loglog(nn, ref_t0 * (nn / ref_n0) ** 2,
                  "k:",  lw=1.2, alpha=0.55, label=r"$O(N^2)$ ref", zorder=1)

    ax.set_xlabel("Number of clients $N$", fontsize=12)
    ax.set_ylabel("Wall-clock time per round (s)", fontsize=12)
    ax.set_title("RQ2 — Scalability: Auction Time vs $N$ (log–log)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92, ncol=1)
    ax.grid(True, which="both", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=10)
    # 脚注说明单点
    single = [m for m in mech_data if len(mech_data[m]) == 1]
    if single:
        ax.annotate(
            f"* Only N={mech_data[single[0]][0][0]} checkpoint available for: "
            + ", ".join(single),
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=7, color="#555", va="bottom"
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


# ─────────────────────────────────────────────
# Fig 2: communication overhead (+ memory if available)
# ─────────────────────────────────────────────
def fig2_memory_comm(data, out_path):
    import matplotlib.pyplot as plt

    detail = data.get("per_n_detail") or {}
    Ns = sorted(int(k) for k in detail)

    comm_neural, comm_base = [], []
    mem_any = []  # collect GPU mem if non-zero

    for N in Ns:
        d = detail[str(N)]
        comm_neural.append(float((d.get("Ours") or d.get("RegretNet") or {}).get("comm_bytes_est") or 0))
        comm_base.append(float((d.get("PAC") or {}).get("comm_bytes_est") or 0))
        # GPU 内存（如果机器有 GPU 会记录）
        mem_val = max(
            float((d.get("PAC") or {}).get("peak_gpu_gb") or 0),
            float((d.get("CSRA") or {}).get("peak_gpu_gb") or 0),
        )
        mem_any.append(mem_val)

    has_mem = any(v > 0 for v in mem_any)

    if has_mem:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(6, 4.5))
        ax1 = None

    # ── 子图 a: 峰值内存 (仅在有数据时) ──
    if ax1 is not None:
        x = np.arange(len(Ns))
        for method, color in [("PAC", _COLORS["PAC"]), ("CSRA", _COLORS["CSRA"]),
                               ("Ours", _COLORS["Ours"])]:
            vals = [float(detail[str(N)].get(method, {}).get("peak_gpu_gb") or 0) for N in Ns]
            if any(v > 0 for v in vals):
                ax1.plot(Ns, vals, marker=_MARKERS.get(method, "o"), linestyle="-",
                         color=color, lw=1.8, ms=7, label=method)
        ax1.set_xlabel("$N$", fontsize=11)
        ax1.set_ylabel("Peak GPU memory (GB)", fontsize=11)
        ax1.set_title("(a) Peak GPU memory", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

    # ── 子图 b: 通信量 ──
    ax = ax2
    if any(comm_base):
        ax.plot(Ns, np.array(comm_base) / 1e3, marker=_MARKERS["PAC"], linestyle="-",
                color=_COLORS["PAC"], lw=2, ms=7, label="Baseline (PAC/VCG/CSRA)")
    if any(comm_neural):
        neural_label = "Neural (Ours/RegretNet)"
        ax.plot(Ns, np.array(comm_neural) / 1e3, marker=_MARKERS["Ours"], linestyle="-",
                color=_COLORS["Ours"], lw=2, ms=8, label=neural_label)

    ax.set_xlabel("Number of clients $N$", fontsize=11)
    ax.set_ylabel("Comm. volume estimate (KB / round)", fontsize=11)
    ax.set_title("(b) Bid + allocation payload (float32 est.)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("RQ2 — Memory & Communication Overhead", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


# ─────────────────────────────────────────────
# Fig 3: stacked per-round latency breakdown
# ─────────────────────────────────────────────
def fig3_stacked(data, out_path):
    import matplotlib.pyplot as plt

    detail = data.get("per_n_detail") or {}
    Ns = sorted(int(k) for k in detail)

    # 优先用 Ours；否则 CSRA；否则 PAC
    key_m = None
    for cand in ("Ours", "CSRA", "PAC"):
        if any(cand in detail.get(str(N), {}) for N in Ns):
            key_m = cand
            break
    if key_m is None:
        print("  [WARN] fig3: no detail data found, skipping")
        return

    loc_vals, srv_vals, auc_vals = [], [], []
    for N in Ns:
        nd = detail[str(N)].get(key_m) or {}
        loc_vals.append(float(nd.get("t_local_train_proxy", 0)))
        srv_vals.append(float(nd.get("t_server_grad_agg", 0)))
        auc_vals.append(float(nd.get("t_auction_solve", 0)) +
                        float(nd.get("t_aggr_fl_weights", 0)))

    loc = np.array(loc_vals)
    srv = np.array(srv_vals)
    auc = np.array(auc_vals)
    total = loc + srv + auc

    fig, ax = plt.subplots(figsize=(8, 5))
    ind   = np.arange(len(Ns))
    width = 0.55

    ax.bar(ind,       loc, width, label="Local training (1 client, parallel)",
           color="#66bb6a", alpha=0.9)
    ax.bar(ind, srv, width, bottom=loc,
           label="Server grad aggregation (serial O(N))",
           color="#42a5f5", alpha=0.9)
    ax.bar(ind, auc, width, bottom=loc + srv,
           label="Auction solve + mech. aggregation",
           color="#ffa726", alpha=0.9)

    ax.set_xticks(ind)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of clients $N$", fontsize=12)
    ax.set_ylabel("Time per FL round (s)", fontsize=12)
    ax.set_title(
        f"RQ2 — End-to-end Round Latency ({key_m})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # 百分比标签（仅标注 auction 占比）
    for i, N in enumerate(Ns):
        if total[i] > 0:
            pct = auc[i] / total[i] * 100
            ax.text(i, total[i] + total[i] * 0.01,
                    f"auc={pct:.0f}%\n{total[i]*1000:.1f}ms",
                    ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   type=str, default="run/privacy_paper/rq2/rq2_paper_data.json")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ Missing {args.input}")
        print(f"   Run: python exp_rq/rq2_paper_benchmark.py")
        return 1

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.input), "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("RQ2 Figure Generation")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {out_dir}/")

    data  = load_json(args.input)
    rows  = data.get("rq2_time_rows") or []
    mechs = sorted(set(r["mechanism"] for r in rows))
    n_list = data.get("meta", {}).get("n_list", [])
    print(f"\nMechanisms : {', '.join(mechs)}")
    print(f"N values   : {n_list}")
    print(f"Data points: {len(rows)}")
    print("=" * 60)

    fig1_loglog_time(data, os.path.join(out_dir, "figure_rq2_1_time_vs_N_loglog.png"))
    fig2_memory_comm(data, os.path.join(out_dir, "figure_rq2_2_memory_comm.png"))
    fig3_stacked    (data, os.path.join(out_dir, "figure_rq2_3_stacked_latency.png"))

    print("\n✓ All figures generated!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

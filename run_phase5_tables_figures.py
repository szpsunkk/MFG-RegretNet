#!/usr/bin/env python3
"""
Phase 5: Tables and figures for the privacy paper (RQ1–RQ4).

Reads phase4_summary.json (from run_phase4_eval.py) and optionally an accuracy JSON for RQ4.
Outputs:
  - Tables: RQ1 (regret, IR), RQ2 (N vs time), RQ3 (revenue, BF), RQ4 (accuracy) as CSV and Markdown.
  - Figures: RQ2 (N vs wall-clock time), optional RQ4 (rounds vs test accuracy).
"""
from __future__ import division, print_function

import argparse
import csv
import json
import os
import sys

import numpy as np


def _safe_float(x):
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("phase4_summary.json root must be a JSON object")
    return data


def write_table_rq1(summary, out_dir):
    """Table: mechanism | mean_regret | mean_ir_violation (RQ1)."""
    if "rq1" not in summary or not isinstance(summary["rq1"], list):
        print("Phase5: no rq1 in summary or invalid format, skip table RQ1")
        return
    rows = summary["rq1"]
    csv_path = os.path.join(out_dir, "table_rq1.csv")
    md_path = os.path.join(out_dir, "table_rq1.md")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mechanism", "mean_regret", "mean_ir_violation"])
        for r in rows:
            mech = r.get("mechanism", "")
            reg = _safe_float(r.get("mean_regret"))
            ir = _safe_float(r.get("mean_ir_violation"))
            w.writerow([mech, reg, ir])
    with open(md_path, "w") as f:
        f.write("| Mechanism | Mean Regret | Mean IR Violation |\n")
        f.write("|-----------|-------------|--------------------|\n")
        for r in rows:
            mech = r.get("mechanism", "")
            reg = r.get("mean_regret")
            ir = r.get("mean_ir_violation")
            reg_s = "{:.6f}".format(_safe_float(reg)) if reg is not None else "—"
            ir_s = "{:.6f}".format(_safe_float(ir)) if ir is not None else "—"
            f.write("| {} | {} | {} |\n".format(mech, reg_s, ir_s))
    print("Wrote", csv_path, "and", md_path)


def write_table_rq2(summary, out_dir):
    """Table: n_agents | mechanism | mean_time_sec (RQ2)."""
    if "rq2" not in summary or not isinstance(summary["rq2"], list):
        print("Phase5: no rq2 in summary or invalid format, skip table RQ2")
        return
    rows = summary["rq2"]
    csv_path = os.path.join(out_dir, "table_rq2.csv")
    md_path = os.path.join(out_dir, "table_rq2.md")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_agents", "mechanism", "mean_time_sec"])
        for r in rows:
            n = r.get("n_agents", "")
            mech = r.get("mechanism", "")
            t = _safe_float(r.get("mean_time_sec"))
            w.writerow([n, mech, t])
    # Markdown: pivot-style rows per N, then mechanisms (sort N numerically)
    n_vals = sorted(set(r.get("n_agents") for r in rows), key=lambda x: (x if isinstance(x, (int, float)) else float("inf")))
    mechs = []
    for r in rows:
        m = r.get("mechanism", "")
        if m not in mechs:
            mechs.append(m)
    with open(md_path, "w") as f:
        f.write("| N (agents) | " + " | ".join(str(m) for m in mechs) + " |\n")
        f.write("|" + "---|" * (len(mechs) + 1) + "\n")
        for n in n_vals:
            cells = [str(n)]
            for m in mechs:
                t = None
                for r in rows:
                    if r.get("n_agents") == n and r.get("mechanism") == m:
                        t = r.get("mean_time_sec")
                        break
                cells.append("{:.4f}".format(_safe_float(t)) if t is not None else "—")
            f.write("| " + " | ".join(cells) + " |\n")
    print("Wrote", csv_path, "and", md_path)


def write_table_rq3(summary, out_dir):
    """RQ3: W̄ = time-avg welfare per seed over T rounds; ± std over seeds (见 exp_rq/RQ3_PROCESS.md)."""
    if "rq3" not in summary or not isinstance(summary["rq3"], list):
        print("Phase5: no rq3 in summary or invalid format, skip table RQ3")
        return
    rows = summary["rq3"]
    csv_path = os.path.join(out_dir, "table_rq3.csv")
    md_path = os.path.join(out_dir, "table_rq3.md")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mechanism", "mean_revenue", "std_revenue", "bf_rate",
            "revenue_efficiency", "std_revenue_efficiency",
            "mean_social_welfare", "std_social_welfare", "num_rounds_T", "num_seeds",
        ])
        for r in rows:
            mech = r.get("mechanism", "")
            if isinstance(r, dict) and "mean_social_welfare" in r:
                w.writerow([
                    mech,
                    _safe_float(r.get("mean_revenue")),
                    _safe_float(r.get("std_revenue")),
                    _safe_float(r.get("bf_rate")),
                    _safe_float(r.get("revenue_efficiency")),
                    _safe_float(r.get("std_revenue_efficiency")),
                    _safe_float(r.get("mean_social_welfare")),
                    _safe_float(r.get("std_social_welfare")),
                    r.get("num_rounds_T", ""),
                    r.get("num_seeds", ""),
                ])
            else:
                rev = _safe_float(r.get("mean_revenue"))
                bf = _safe_float(r.get("bf_rate"))
                eta = _safe_float(r.get("revenue_efficiency"))
                sw = _safe_float(r.get("mean_social_welfare"))
                w.writerow([mech, rev, "", bf, eta, "", sw, "", "", ""])
    with open(md_path, "w") as f:
        f.write(
            "| Mechanism | R̄ (mean±std) | BF | η_rev (mean±std) | W̄ (mean±std) | T rounds | seeds |\n"
        )
        f.write("|-----------|-------------|----|------------------|--------------|---------|-------|\n")

        def pm(m, s):
            if m is None or (isinstance(m, float) and m != m):
                return "—"
            if s is None or (isinstance(s, float) and s != s) or s == 0:
                return "{:.4f}".format(_safe_float(m))
            return "{:.4f} ± {:.4f}".format(_safe_float(m), _safe_float(s))

        for r in rows:
            mech = r.get("mechanism", "")
            if not isinstance(r, dict):
                continue
            rev_s = pm(r.get("mean_revenue"), r.get("std_revenue"))
            bf_s = "{:.4f}".format(_safe_float(r.get("bf_rate"))) if r.get("bf_rate") is not None else "—"
            eta_s = pm(r.get("revenue_efficiency"), r.get("std_revenue_efficiency"))
            sw_s = pm(r.get("mean_social_welfare"), r.get("std_social_welfare"))
            T = r.get("num_rounds_T", "")
            ns = r.get("num_seeds", "")
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                mech, rev_s, bf_s, eta_s, sw_s, T, ns))
    print("Wrote", csv_path, "and", md_path)


def write_figure_rq2(summary, out_dir, log_scale=False):
    """Figure: N vs wall-clock time (one curve per mechanism). Optional log-log scale (实验思路 RQ2)."""
    if "rq2" not in summary or not isinstance(summary.get("rq2"), list):
        print("Phase5: no rq2 in summary or invalid format, skip figure RQ2")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Phase5: matplotlib not found, skip figure RQ2")
        return
    rows = summary["rq2"]
    mech_to_n = {}
    mech_to_t = {}
    for r in rows:
        m = r.get("mechanism", "")
        n = r.get("n_agents")
        t = _safe_float(r.get("mean_time_sec"))
        if m not in mech_to_n:
            mech_to_n[m] = []
            mech_to_t[m] = []
        mech_to_n[m].append(n)
        mech_to_t[m].append(max(t, 1e-6))  # avoid log(0)
    for m in mech_to_n:
        n_list = mech_to_n[m]
        t_list = mech_to_t[m]
        order = np.argsort([float(x) if isinstance(x, (int, float)) else float("inf") for x in n_list])
        mech_to_n[m] = [n_list[i] for i in order]
        mech_to_t[m] = [t_list[i] for i in order]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots()
    for i, m in enumerate(mech_to_n):
        c = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        ns, ts = mech_to_n[m], mech_to_t[m]
        ms = 10 if len(ns) == 1 else 6
        if log_scale:
            ax.loglog(ns, ts, label=m, color=c, marker=mk, linestyle="-", linewidth=1.5, markersize=ms)
        else:
            ax.plot(ns, ts, label=m, color=c, marker=mk, linestyle="-", linewidth=1.5, markersize=ms)
    ax.set_xlabel("Number of agents (N)")
    ax.set_ylabel("Time per round (s)")
    ax.set_title("RQ2: Scalability — Time vs N" + (" (log-log)" if log_scale else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure_rq2_time_vs_n" + ("_loglog" if log_scale else "") + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote", path)


def write_figure_rq4(accuracy_json_path, out_dir):
    """Figure: training round vs test accuracy (one curve per method). Optional."""
    if not accuracy_json_path or not os.path.isfile(accuracy_json_path):
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Phase5: matplotlib not found, skip figure RQ4")
        return
    with open(accuracy_json_path, "r") as f:
        data = json.load(f)
    # Expected: { "rounds": [1, 10, 20, ...], "methods": { "PAC": [acc1, ...], "VCG": [...], ... } }
    # or list of { "method", "rounds", "accuracies" }
    if "methods" in data and "rounds" in data:
        rounds = data["rounds"]
        methods = data["methods"]
    elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values() if isinstance(v, list)):
        methods = {k: v for k, v in data.items() if isinstance(v, list) and k != "rounds"}
        rounds = data.get("rounds", list(range(1, 1 + max(len(v) for v in methods.values()) if methods else 0)))
    else:
        print("Phase5: unknown accuracy JSON format, skip figure RQ4")
        return
    if not methods:
        return
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots()
    for i, (name, accs) in enumerate(methods.items()):
        r = rounds if len(rounds) == len(accs) else list(range(1, len(accs) + 1))
        c = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        ax.plot(r[: len(accs)], accs, label=name, color=c, marker=mk, linestyle="-", linewidth=1.5, markersize=5)
    ax.set_xlabel("Training round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("RQ4: FL accuracy vs round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure_rq4_accuracy_vs_round.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote", path)


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Tables and figures from Phase 4 summary")
    parser.add_argument("--input", type=str, default="run/privacy_paper/phase4_summary.json", help="path to phase4_summary.json")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper", help="base output dir; tables/ and figures/ under it")
    parser.add_argument("--accuracy-json", type=str, default="", help="optional JSON for RQ4 accuracy curve (rounds + methods)")
    parser.add_argument("--no-figures", action="store_true", help="skip generating figures")
    parser.add_argument("--log-scale", action="store_true", help="RQ2 figure: use log-log scale (log(time) vs log(N))")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Error: input not found:", args.input)
        print("Run run_phase4_eval.py first to generate phase4_summary.json")
        sys.exit(1)

    try:
        summary = load_summary(args.input)
    except json.JSONDecodeError as e:
        print("Error: invalid JSON in", args.input, ":", e)
        sys.exit(1)
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)
    tables_dir = os.path.join(args.out_dir, "tables")
    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    write_table_rq1(summary, tables_dir)
    write_table_rq2(summary, tables_dir)
    write_table_rq3(summary, tables_dir)

    if not args.no_figures:
        write_figure_rq2(summary, figures_dir, log_scale=args.log_scale)
        if args.accuracy_json:
            write_figure_rq4(args.accuracy_json, figures_dir)

    print("Phase 5 done. Tables in {}/, figures in {}/".format(tables_dir, figures_dir))


if __name__ == "__main__":
    main()

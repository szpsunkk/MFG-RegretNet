#!/usr/bin/env python3
"""
RQ1：激励相容性（实验思路 3.1）

数据：v_i ~ Uniform[0,1]，ε_i ~ Uniform[0.1, 5]（与隐私论文一致，无需 MNIST/CIFAR 即可跑机制评估）。
基线：PAC、VCG、CSRA、MFG-Pricing（按 ε 比例分摊预算的均值场定价）；神经：RegretNet、DM-RegretNet、MFG-RegretNet（需 checkpoint）。

指标：
  - 平均归一化事后遗憾、IR 违反率（与神经 guarantees_eval 同一效用 u=p−v·ε_alloc）
  - 多 seed 下 mean±std
  - 可选：RegretNet vs MFG-RegretNet 的 t 检验（基于各 seed 的平均遗憾）

输出：run/privacy_paper/rq1/ 下 table_rq1.csv、table_rq1.md、rq1_statistics.json、figure_rq1_regret_bar.png（柱图仅神经方法）
"""
from __future__ import division, print_function

import argparse
import csv
import json
import os
import sys

import numpy as np


def _parse_int_list(s):
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def build_trade_mech_ls(args):
    ls = [
        ["PAC", "ConvlAggr", "", args.n_items],
        ["VCG", "ConvlAggr", "", args.n_items],
        ["CSRA", "ConvlAggr", "", args.n_items],
        ["MFG-Pricing", "ConvlAggr", "", args.n_items],
    ]
    if args.regretnet_ckpt and os.path.isfile(args.regretnet_ckpt):
        ls.append(["RegretNet", "ConvlAggr", args.regretnet_ckpt, args.n_items])
    if getattr(args, "dm_regretnet_ckpt", "").strip() and os.path.isfile(args.dm_regretnet_ckpt.strip()):
        ls.append(["DM-RegretNet", "ConvlAggr", args.dm_regretnet_ckpt.strip(), args.n_items])
    mfg_ok = False
    if args.mfg_regretnet_ckpt_by_n.strip():
        ckpt_by_n = {}
        for part in args.mfg_regretnet_ckpt_by_n.split(","):
            if ":" not in part.strip():
                continue
            n_str, path = part.split(":", 1)
            try:
                n_val = int(n_str.strip())
                path = path.strip()
                if path and os.path.isfile(path):
                    ckpt_by_n[n_val] = path
            except ValueError:
                pass
        if ckpt_by_n and args.n_agents in ckpt_by_n:
            ls.append(["MFG-RegretNet", "ConvlAggr", ckpt_by_n, args.n_items])
            mfg_ok = True
    if not mfg_ok and args.mfg_regretnet_ckpt and os.path.isfile(args.mfg_regretnet_ckpt):
        ls.append(["MFG-RegretNet", "ConvlAggr", args.mfg_regretnet_ckpt, args.n_items])
    return ls


def run_rq1_baseline_per_seed(
    mech_name, n_agents, n_items, budget, num_profiles, seeds, batch_size,
    v_grid_n, eps_grid_n,
):
    """PAC/VCG/CSRA/MFG-Pricing：网格近似单代理最优偏离后的 regret / IR。"""
    from run_phase4_eval import build_privacy_paper_batch
    from experiments import DEVICE as EXP_DEVICE
    from exp_rq.guarantees_eval_baselines import guarantees_eval_procurement_baseline
    from datasets import Dataloader
    import torch

    per_seed = []
    for seed in seeds:
        reports, bud, val_type = build_privacy_paper_batch(
            num_profiles, n_agents, n_items, budget, seed, EXP_DEVICE
        )
        loader = Dataloader(
            torch.cat([reports, val_type], dim=2),
            batch_size=batch_size,
            shuffle=False,
        )
        rs, irs = [], []
        for batch in loader:
            rep = batch[:, :, :-2].to(EXP_DEVICE)
            b = budget * torch.ones(rep.shape[0], 1, device=EXP_DEVICE)
            try:
                r, ir = guarantees_eval_procurement_baseline(
                    rep, b, mech_name,
                    v_grid_n=v_grid_n, eps_grid_n=eps_grid_n,
                )
                rs.append(r.detach().cpu().numpy().ravel())
                irs.append(ir.detach().cpu().numpy().ravel())
            except Exception as e:
                print("RQ1 baseline {} seed {} error: {}".format(mech_name, seed, e))
                return None
        mean_r = float(np.concatenate(rs).mean())
        mean_ir = float(np.concatenate(irs).mean())
        per_seed.append({"seed": seed, "mean_regret": mean_r, "mean_ir_violation": mean_ir})
    return per_seed


def run_rq1_neural_per_seed(trade_mech, n_agents, n_items, budget, num_profiles, seeds, batch_size, device):
    from run_phase4_eval import build_privacy_paper_batch, get_ckpt_path
    from experiments import guarantees_eval, load_auc_model, DEVICE as EXP_DEVICE
    from datasets import Dataloader
    import torch

    name = trade_mech[0]
    path = get_ckpt_path(trade_mech, n_agents)
    if not path or not os.path.isfile(path):
        return None
    mech = (name, trade_mech[1], path, n_items)
    per_seed = []
    for seed in seeds:
        reports, bud, val_type = build_privacy_paper_batch(
            num_profiles, n_agents, n_items, budget, seed, EXP_DEVICE
        )
        loader = Dataloader(
            torch.cat([reports, val_type], dim=2),
            batch_size=batch_size,
            shuffle=False,
        )
        rs, irs = [], []
        for batch in loader:
            rep = batch[:, :, :-2].to(EXP_DEVICE)
            vt = batch[:, :, -2:].to(EXP_DEVICE)
            b = budget * torch.ones(rep.shape[0], 1, device=EXP_DEVICE)
            try:
                r, ir = guarantees_eval(rep, b, vt, mech, misreport_iter=25, lr=1e-1)
                rs.append(r.detach().cpu().numpy().ravel())
                irs.append(ir.detach().cpu().numpy().ravel())
            except Exception as e:
                print("RQ1 seed {} error: {}".format(seed, e))
                return None
        mean_r = float(np.concatenate(rs).mean())
        mean_ir = float(np.concatenate(irs).mean())
        per_seed.append({"seed": seed, "mean_regret": mean_r, "mean_ir_violation": mean_ir})
    return per_seed


def main():
    parser = argparse.ArgumentParser(description="RQ1 incentive compatibility (实验思路 3.1)")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-items", type=int, default=1)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--num-profiles", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--regretnet-ckpt", type=str, default="")
    parser.add_argument("--dm-regretnet-ckpt", type=str, default="")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    parser.add_argument("--mfg-regretnet-ckpt-by-n", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper/rq1")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--convergence-curve", action="store_true",
                        help="also run rq1_convergence_curve.py (regret vs PGA rounds)")
    parser.add_argument("--baseline-v-grid", type=int, default=17,
                        help="PAC/VCG/CSRA/MFG-Pricing: grid v; MFG-Pricing 还与 eps 联搜")
    parser.add_argument("--baseline-eps-grid", type=int, default=8,
                        help="CSRA / MFG-Pricing: grid points for misreported eps")
    parser.add_argument("--baseline-truthful-only", action="store_true",
                        help="PAC/VCG/CSRA: force regret/IR to 0 (old behavior;不推荐)")
    args = parser.parse_args()

    try:
        from exp_rq.rq1_ckpt_resolve import resolve_regretnet_ckpt
        _rc = (args.regretnet_ckpt or "").strip()
        if not _rc or not os.path.isfile(_rc):
            _r = resolve_regretnet_ckpt(args.n_agents, args.n_items)
            if _r:
                args.regretnet_ckpt = _r
                print("[INFO] RegretNet ckpt (auto):", _r)
    except Exception:
        pass

    _dm = (args.dm_regretnet_ckpt or "").strip()
    if _dm and not os.path.isfile(_dm):
        args.dm_regretnet_ckpt = ""
        _dm = ""
    if not _dm:
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_dm_regretnet_ckpt
            _d = resolve_dm_regretnet_ckpt(args.n_agents, args.n_items)
            if _d:
                args.dm_regretnet_ckpt = _d
                print("[INFO] DM-RegretNet ckpt (auto):", _d)
        except Exception:
            pass

    _mfg = (args.mfg_regretnet_ckpt or "").strip()
    if _mfg and not os.path.isfile(_mfg):
        args.mfg_regretnet_ckpt = ""
        _mfg = ""
    if not _mfg:
        try:
            from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt
            _m = resolve_mfg_regretnet_ckpt(args.n_agents, args.n_items)
            if _m:
                args.mfg_regretnet_ckpt = _m
                print("[INFO] MFG-RegretNet ckpt (auto):", _m)
        except Exception:
            pass

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        seeds = [42]

    os.makedirs(args.out_dir, exist_ok=True)

    trade_mech_ls = build_trade_mech_ls(args)
    _names = [t[0] for t in trade_mech_ls]
    if "DM-RegretNet" not in _names:
        print(
            "[WARN] 柱图/表不含 DM-RegretNet：缺少 result/dm_regretnet_privacy_*_checkpoint.pt，"
            "请运行 python3 train_dm_regretnet_privacy.py 或一键脚本（含 DM 自动训练）"
        )
    if "MFG-RegretNet" not in _names:
        print(
            "[WARN] 柱图/表不含 MFG-RegretNet：缺少 result/mfg_regretnet_privacy_*_checkpoint.pt，"
            "请运行 python3 train_mfg_regretnet.py 或设置 --mfg-regretnet-ckpt"
        )
    rows = []
    neural_per_mech = {}

    for trade_mech in trade_mech_ls:
        name = trade_mech[0]
        if name in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
            if args.baseline_truthful_only:
                rows.append({
                    "mechanism": name,
                    "mean_regret": 0.0,
                    "std_regret": 0.0,
                    "mean_ir_violation": 0.0,
                    "std_ir_violation": 0.0,
                    "honesty_proxy": 1.0,
                    "note": "truthful-only placeholder",
                })
                continue
            ps = run_rq1_baseline_per_seed(
                name, args.n_agents, args.n_items, args.budget,
                args.num_profiles, seeds, args.batch_size,
                args.baseline_v_grid, args.baseline_eps_grid,
            )
            if ps is None:
                rows.append({
                    "mechanism": name,
                    "mean_regret": None,
                    "std_regret": None,
                    "mean_ir_violation": None,
                    "std_ir_violation": None,
                    "honesty_proxy": None,
                    "note": "baseline eval error",
                })
                continue
            mr = [p["mean_regret"] for p in ps]
            mir = [p["mean_ir_violation"] for p in ps]
            mean_r = float(np.mean(mr))
            rows.append({
                "mechanism": name,
                "mean_regret": mean_r,
                "std_regret": float(np.std(mr, ddof=1)) if len(mr) > 1 else 0.0,
                "mean_ir_violation": float(np.mean(mir)),
                "std_ir_violation": float(np.std(mir, ddof=1)) if len(mir) > 1 else 0.0,
                "honesty_proxy": float(1.0 / (1.0 + max(mean_r, 0.0))),
                "note": "grid best single-agent dev",
            })
            continue
        ps = run_rq1_neural_per_seed(
            trade_mech, args.n_agents, args.n_items, args.budget,
            args.num_profiles, seeds, args.batch_size, None,
        )
        if ps is None:
            rows.append({
                "mechanism": name,
                "mean_regret": None,
                "std_regret": None,
                "mean_ir_violation": None,
                "std_ir_violation": None,
                "honesty_proxy": None,
                "note": "skipped (no checkpoint or error)",
            })
            continue
        neural_per_mech[name] = ps
        mr = [p["mean_regret"] for p in ps]
        mir = [p["mean_ir_violation"] for p in ps]
        mean_r = float(np.mean(mr))
        # 诚实报价率近似：遗憾越小越接近 1（1/(1+r̄gt)，非论文正式定义，仅作对比用）
        honesty_proxy = float(1.0 / (1.0 + max(mean_r, 0.0)))
        rows.append({
            "mechanism": name,
            "mean_regret": mean_r,
            "std_regret": float(np.std(mr, ddof=1)) if len(mr) > 1 else 0.0,
            "mean_ir_violation": float(np.mean(mir)),
            "std_ir_violation": float(np.std(mir, ddof=1)) if len(mir) > 1 else 0.0,
            "honesty_proxy": honesty_proxy,
            "note": "strategy_stability=std(regret) across seeds",
        })

    # t-test: RegretNet vs MFG-RegretNet (per-seed mean regret)
    ttest = None
    if "RegretNet" in neural_per_mech and "MFG-RegretNet" in neural_per_mech:
        r_reg = [p["mean_regret"] for p in neural_per_mech["RegretNet"]]
        r_mfg = [p["mean_regret"] for p in neural_per_mech["MFG-RegretNet"]]
        if len(r_reg) >= 2 and len(r_mfg) >= 2:
            try:
                from scipy import stats
                stat, pval = stats.ttest_ind(r_reg, r_mfg, equal_var=False)
                ttest = {
                    "test": "Welch t-test on per-seed mean regret",
                    "RegretNet_seeds": r_reg,
                    "MFG-RegretNet_seeds": r_mfg,
                    "statistic": float(stat),
                    "pvalue": float(pval),
                }
            except Exception as e:
                ttest = {"error": str(e)}

    out_json = {
        "description": "RQ1 v~U[0,1], eps~U[0.1,5], budget B",
        "n_agents": args.n_agents,
        "budget": args.budget,
        "num_profiles": args.num_profiles,
        "seeds": seeds,
        "rows": rows,
        "per_seed_neural": neural_per_mech,
        "ttest_regretnet_vs_mfg": ttest,
    }
    with open(os.path.join(args.out_dir, "rq1_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.out_dir, "table_rq1.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mechanism", "mean_regret", "std_regret", "mean_ir_violation", "std_ir_violation",
                    "honesty_proxy", "note"])
        for r in rows:
            hp = r.get("honesty_proxy")
            w.writerow([
                r["mechanism"],
                r["mean_regret"] if r["mean_regret"] is not None else "",
                r["std_regret"] if r["std_regret"] is not None else "",
                r["mean_ir_violation"] if r["mean_ir_violation"] is not None else "",
                r["std_ir_violation"] if r["std_ir_violation"] is not None else "",
                hp if hp is not None else "",
                r.get("note", ""),
            ])

    md_path = os.path.join(args.out_dir, "table_rq1.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RQ1: Incentive compatibility\n\n")
        f.write("| Mechanism | Mean Regret | Std | Mean IR | Std IR | Honesty proxy | Note |\n")
        f.write("|-----------|-------------|-----|---------|--------|---------------|------|\n")
        for r in rows:
            def fmt(x):
                return "{:.6f}".format(x) if x is not None else "—"
            hp = r.get("honesty_proxy")
            f.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["mechanism"],
                fmt(r["mean_regret"]), fmt(r["std_regret"]),
                fmt(r["mean_ir_violation"]), fmt(r["std_ir_violation"]),
                fmt(hp) if hp is not None else "—",
                (r.get("note") or "").replace("|", "\\|")[:40],
            ))
        if ttest and "pvalue" in ttest:
            f.write("\n**RegretNet vs MFG-RegretNet (Welch t-test on per-seed mean regret):** p = {:.4f}\n".format(ttest["pvalue"]))

    if not args.no_figure:
        try:
            import matplotlib.pyplot as plt
            # 柱图仅展示神经机制（解析基线见 table_rq1 / 论文主表）
            _non_neural_bar = frozenset(("PAC", "VCG", "CSRA", "MFG-Pricing"))
            plot_rows = [
                r for r in rows
                if r["mean_regret"] is not None and r["mechanism"] not in _non_neural_bar
            ]
            if not plot_rows:
                plot_rows = [r for r in rows if r["mean_regret"] is not None]
            names = [r["mechanism"] for r in plot_rows]
            means = [max(0.0, float(r["mean_regret"])) for r in plot_rows]
            stds = [r["std_regret"] or 0.0 for r in plot_rows]
            cmap = ["#1f77b4", "#ff7f0e", "#9467bd", "#d62728", "#17becf"]
            colors = [cmap[i % len(cmap)] for i in range(len(names))]
            fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(names)), 4))
            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="k", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=15, ha="right")
            ax.set_ylabel("Mean regret (normalized)")
            # 量级差 >20 倍时用 symlog，避免一条柱占满纵轴、其余看不见
            pos = [m for m in means if m > 1e-12]
            if pos and max(means) / (min(pos) + 1e-12) > 20.0:
                ax.set_yscale("symlog", linthresh=max(1e-4, 0.05 * min(pos)))
            ax.set_title(
                "RQ1: Neural mechanisms — mean regret (± std); "
                "norm: v·eps_alloc, else v·eps_bid if ploss≈0"
            )
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "figure_rq1_regret_bar.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("Wrote", os.path.join(args.out_dir, "figure_rq1_regret_bar.png"))
        except ImportError:
            print("matplotlib not installed, skip figure")

    print("RQ1 done. Outputs in:", args.out_dir)
    print("  ", csv_path)
    print("  ", md_path)
    print("  ", os.path.join(args.out_dir, "rq1_statistics.json"))

    if args.convergence_curve:
        import subprocess
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "rq1_convergence_curve.py"),
            "--out-dir", args.out_dir,
            "--num-profiles", str(min(args.num_profiles, 500)),
            "--regretnet-ckpt", args.regretnet_ckpt,
            "--mfg-regretnet-ckpt", args.mfg_regretnet_ckpt,
            "--n-agents", str(args.n_agents),
            "--budget", str(args.budget),
        ]
        subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

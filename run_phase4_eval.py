#!/usr/bin/env python3
"""
Phase 4: System evaluation and RQ (Research Questions) for the privacy paper.

RQ1: Incentive compatibility — mean ex-post regret and IR violation rate.
RQ2: Scalability — wall-clock time per round (auction + aggregation) vs N.
RQ3: Revenue — total payment sum (with BF check) per mechanism.
RQ4: FL accuracy — final test accuracy when using each mechanism (optional; needs dataset + checkpoints).

Uses privacy-paper setting: v~U[0,1], eps~U[0.1,5], c=v*eps, fixed budget B.
"""
from __future__ import division, print_function

import argparse
import os
import sys
import time
import json

import numpy as np
import torch

# Optional: avoid loading FL/datasets if only RQ1–RQ3
try:
    from experiments import (
        auction,
        load_auc_model,
        guarantees_eval,
        DEVICE,
    )
    from aggregation import aggr_batch
except Exception as e:
    print("Phase 4 requires experiments/aggregation. Error:", e)
    sys.exit(1)

from datasets_fl_benchmark import generate_privacy_paper_bids
from datasets import Dataloader


def get_ckpt_path(trade_mech, n_agents):
    """Resolve checkpoint path: trade_mech[2] can be str (single path) or dict {n: path}."""
    spec = trade_mech[2]
    if isinstance(spec, dict):
        return spec.get(n_agents)
    return spec if n_agents else spec


def build_privacy_paper_batch(num_profiles, n_agents, n_items, budget_val, seed, device):
    """Build (reports, budget, val_type) for privacy paper. reports: (P, N, n_items+2)."""
    reports = generate_privacy_paper_bids(
        n_agents=n_agents,
        n_items=n_items,
        num_profiles=num_profiles,
        v_min=0.0,
        v_max=1.0,
        eps_min=0.1,
        eps_max=5.0,
        seed=seed,
    )
    reports = reports.to(device)
    val_type = torch.zeros((num_profiles, n_agents, 2), device=device)
    val_type[:, :, 1] = 1.0
    budget = budget_val * torch.ones(num_profiles, 1, device=device)
    return reports, budget, val_type


def rq1_guarantees_privacy_paper(trade_mech_ls, n_agents, n_items, budget, num_profiles, seeds, batch_size=256):
    """
    RQ1: Average normalized regret and IR violation rate per mechanism.
    PAC/VCG/CSRA get (0, 0). Neural mechs use guarantees_eval on generated profiles.
    Returns: list of (mech_name, mean_regret, mean_ir).
    """
    results = []
    for trade_mech in trade_mech_ls:
        name = trade_mech[0]
        if name in ("PAC", "VCG", "CSRA"):
            results.append((name, 0.0, 0.0))
            continue
        model_path = get_ckpt_path(trade_mech, n_agents)
        if not model_path or not os.path.isfile(model_path):
            print("RQ1: skip {} (no checkpoint {})".format(name, model_path))
            results.append((name, float("nan"), float("nan")))
            continue
        mech = (trade_mech[0], trade_mech[1], model_path, n_items)
        all_regrets = []
        all_irs = []
        for seed in seeds:
            reports, bud, val_type = build_privacy_paper_batch(
                num_profiles, n_agents, n_items, budget, seed, DEVICE
            )
            loader = Dataloader(
                torch.cat([reports, val_type], dim=2),
                batch_size=batch_size,
                shuffle=False,
            )
            for batch in loader:
                rep = batch[:, :, :-2].to(DEVICE)
                vt = batch[:, :, -2:].to(DEVICE)
                b = budget * torch.ones(rep.shape[0], 1, device=DEVICE)
                try:
                    r, ir = guarantees_eval(rep, b, vt, mech, misreport_iter=25, lr=1e-1)
                    all_regrets.append(r.detach().cpu().numpy())
                    all_irs.append(ir.detach().cpu().numpy())
                except Exception as e:
                    print("RQ1 guarantees_eval error:", e)
                    break
        if all_regrets:
            mean_r = np.concatenate(all_regrets).mean()
            mean_ir = np.concatenate(all_irs).mean()
            results.append((name, float(mean_r), float(mean_ir)))
        else:
            results.append((name, float("nan"), float("nan")))
    return results


def rq2_time_vs_n(trade_mech_ls, n_agents_list, n_items, budget, n_agents_for_ckpt, num_profiles_per_n=100, warmup=2, repeat=5):
    """
    RQ2: Wall-clock time (auction + aggregation) vs N.
    Neural mechanisms (RegretNet/MFG) are only timed for n_agents == n_agents_for_ckpt (checkpoint arch).
    Returns: list of (n_agents, mech_name, mean_time_sec).
    """
    results = []
    for n_agents in n_agents_list:
        reports, bud, _ = build_privacy_paper_batch(
            num_profiles_per_n, n_agents, n_items, budget, seed=42, device=DEVICE
        )
        sizes = reports[:, :, -1].view(-1, n_agents)
        for trade_mech in trade_mech_ls:
            name = trade_mech[0]
            if name in ("PAC", "VCG", "CSRA"):
                model = None
            else:
                path = get_ckpt_path(trade_mech, n_agents)
                if not path or not os.path.isfile(path):
                    continue
                model = load_auc_model(path).to(DEVICE)
            times = []
            for _ in range(warmup + repeat):
                if model is not None:
                    model.eval()
                t0 = time.perf_counter()
                out = auction(reports, bud, trade_mech, model=model, return_payments=False)
                plosses, weights = out[0], out[1]
                weights = aggr_batch(plosses, sizes, method=trade_mech[1])
                t1 = time.perf_counter()
                times.append(t1 - t0)
            mean_t = np.mean(times[warmup:])
            results.append((n_agents, name, mean_t))
    return results


def rq3_revenue_privacy_paper(trade_mech_ls, n_agents, n_items, budget, num_profiles, seeds):
    """
    RQ3（社会福利定义，与 FL 轮次对齐的仿真）：

    - 将每个合成 profile 视为一轮 FL 拍卖轮次 t∈{1,…,T}（T=num_profiles）。
    - 第 t 轮：客户端 i 获得支付 p_i^(t)，隐私产出 ε_i^{(t),out}，成本 c=v_i·ε。
    - 单轮社会福利：W^(t)=∑_i ( p_i^(t) - v_i·ε_i^{(t),out} )。
    - 对随机种子 s：时间平均 W̄_s = (1/T)∑_t W^(t)；同理 R̄_s = (1/T)∑_t ∑_i p_i^(t)。
    - 报告：跨种子的 mean(W̄_s) ± std(W̄_s)，以及 R、η_rev= R̄/B 的 mean ± std。
    - BF 率：每种子上「满足 ∑_i p_i^(t)≤B 的轮次 t」占比，再对种子取平均。

    返回：list[dict]，键含 mechanism, mean_revenue, std_revenue, mean_social_welfare,
    std_social_welfare, revenue_efficiency, std_revenue_efficiency, bf_rate, num_rounds_T, num_seeds。
    """
    T = int(num_profiles)
    budget_val = float(budget)
    results = []
    for trade_mech in trade_mech_ls:
        name = trade_mech[0]
        if name in ("All-in", "FairQuery"):
            model = None
        elif name in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
            model = None
        else:
            path = get_ckpt_path(trade_mech, n_agents)
            if not path or not os.path.isfile(path):
                results.append({
                    "mechanism": name,
                    "mean_revenue": float("nan"),
                    "std_revenue": float("nan"),
                    "bf_rate": float("nan"),
                    "revenue_efficiency": float("nan"),
                    "std_revenue_efficiency": float("nan"),
                    "mean_social_welfare": float("nan"),
                    "std_social_welfare": float("nan"),
                    "num_rounds_T": T,
                    "num_seeds": 0,
                })
                continue
            model = load_auc_model(path).to(DEVICE)
        R_bar_per_seed = []
        W_bar_per_seed = []
        eta_per_seed = []
        bf_per_seed = []
        for seed in seeds:
            reports, bud, _ = build_privacy_paper_batch(
                T, n_agents, n_items, budget, seed, DEVICE
            )
            out = auction(reports, bud, trade_mech, model=model, return_payments=True, expected=True)
            plosses, weights, payments = out[0], out[1], out[2]
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
            if pay.ndim == 2:
                cost = (v * eps_out).sum(axis=1)
                W_t = pay.sum(axis=1) - cost
                R_t = pay.sum(axis=1)
            else:
                cost = (v * eps_out).sum(axis=-1)
                W_t = pay.sum(axis=-1) - cost
                R_t = pay.sum(axis=-1)
            R_t = np.asarray(R_t, dtype=np.float64)
            W_t = np.asarray(W_t, dtype=np.float64)
            R_bar_s = float(R_t.mean())
            W_bar_s = float(W_t.mean())
            R_bar_per_seed.append(R_bar_s)
            W_bar_per_seed.append(W_bar_s)
            if budget_val > 0:
                eta_per_seed.append(R_bar_s / budget_val)
            bf_per_seed.append(float((rev.cpu() <= b.cpu()).float().mean().item()))
        R_bar_per_seed = np.array(R_bar_per_seed, dtype=np.float64)
        W_bar_per_seed = np.array(W_bar_per_seed, dtype=np.float64)
        eta_per_seed = np.array(eta_per_seed, dtype=np.float64) if eta_per_seed else np.array([float("nan")])
        n_s = len(R_bar_per_seed)
        std_r = float(R_bar_per_seed.std(ddof=1)) if n_s > 1 else 0.0
        std_w = float(W_bar_per_seed.std(ddof=1)) if n_s > 1 else 0.0
        std_eta = float(eta_per_seed.std(ddof=1)) if n_s > 1 and not np.any(np.isnan(eta_per_seed)) else 0.0
        results.append({
            "mechanism": name,
            "mean_revenue": float(R_bar_per_seed.mean()),
            "std_revenue": std_r,
            "bf_rate": float(np.mean(bf_per_seed)),
            "revenue_efficiency": float(np.nanmean(eta_per_seed)) if budget_val > 0 else float("nan"),
            "std_revenue_efficiency": std_eta,
            "mean_social_welfare": float(W_bar_per_seed.mean()),
            "std_social_welfare": std_w,
            "num_rounds_T": T,
            "num_seeds": n_s,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: System evaluation and RQ")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-items", type=int, default=1)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--num-profiles", type=int, default=1000)
    parser.add_argument("--seeds", type=str, default="42,43,44", help="comma-separated")
    parser.add_argument("--skip-rq1", action="store_true", help="skip RQ1 (incentive compatibility)")
    parser.add_argument("--skip-rq2", action="store_true", help="skip RQ2 (scalability)")
    parser.add_argument("--skip-rq3", action="store_true", help="skip RQ3 (revenue)")
    parser.add_argument("--run-rq4", action="store_true", help="run RQ4 (FL accuracy; needs dataset + checkpoints)")
    parser.add_argument("--n-list", type=str, default="10,50,100", help="comma-separated N for RQ2")
    parser.add_argument("--regretnet-ckpt", type=str, default="", help="path to RegretNet checkpoint for RQ1/RQ2/RQ3")
    parser.add_argument("--dm-regretnet-ckpt", type=str, default="", help="path to DM-RegretNet checkpoint for RQ1/RQ3")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="", help="path to MFG-RegretNet checkpoint (single N)")
    parser.add_argument("--mfg-regretnet-ckpt-by-n", type=str, default="", help="for RQ2 multi-N: comma-separated N:path, e.g. 10:ckpt10.pt,50:ckpt50.pt,100:ckpt100.pt")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper")
    args = parser.parse_args()

    def _parse_int_list(s, name):
        out = []
        for x in s.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                out.append(int(x))
            except ValueError:
                print("Warning: skip invalid {} value {!r}".format(name, x))
        return out

    seeds = _parse_int_list(args.seeds, "seed")
    n_list = _parse_int_list(args.n_list, "n-list")
    if not seeds:
        seeds = [42]
        print("Warning: no seeds provided, using seeds=[42]")
    if not n_list and not args.skip_rq2:
        n_list = [args.n_agents]
        print("Warning: empty --n-list for RQ2, using n_list=[--n-agents]")
    run_rq1 = not args.skip_rq1
    run_rq2 = not args.skip_rq2
    run_rq3 = not args.skip_rq3

    # Build mechanism list: PAC, VCG, CSRA, optionally RegretNet, MFG-RegretNet
    trade_mech_ls = [
        ["PAC", "ConvlAggr", "", args.n_items],
        ["VCG", "ConvlAggr", "", args.n_items],
        ["CSRA", "ConvlAggr", "", args.n_items],
    ]
    if args.regretnet_ckpt and os.path.isfile(args.regretnet_ckpt):
        trade_mech_ls.append(["RegretNet", "ConvlAggr", args.regretnet_ckpt, args.n_items])
    dm_ck = getattr(args, "dm_regretnet_ckpt", "") or ""
    if dm_ck.strip() and os.path.isfile(dm_ck.strip()):
        trade_mech_ls.append(["DM-RegretNet", "ConvlAggr", dm_ck.strip(), args.n_items])
    # MFG-RegretNet: either per-N checkpoints (RQ2 gets multiple points) or single checkpoint
    if args.mfg_regretnet_ckpt_by_n.strip():
        ckpt_by_n = {}
        for part in args.mfg_regretnet_ckpt_by_n.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            n_str, path = part.split(":", 1)
            n_str, path = n_str.strip(), path.strip()
            try:
                n_val = int(n_str)
                if path and os.path.isfile(path):
                    ckpt_by_n[n_val] = path
            except ValueError:
                pass
        if ckpt_by_n:
            trade_mech_ls.append(["MFG-RegretNet", "ConvlAggr", ckpt_by_n, args.n_items])
    elif args.mfg_regretnet_ckpt and os.path.isfile(args.mfg_regretnet_ckpt):
        trade_mech_ls.append(["MFG-RegretNet", "ConvlAggr", args.mfg_regretnet_ckpt, args.n_items])

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"n_agents": args.n_agents, "budget": args.budget, "seeds": seeds}

    # RQ1
    if run_rq1:
        print("Running RQ1 (incentive compatibility: regret, IR)...")
        rq1 = rq1_guarantees_privacy_paper(
            trade_mech_ls, args.n_agents, args.n_items, args.budget,
            num_profiles=args.num_profiles, seeds=seeds,
        )
        summary["rq1"] = [{"mechanism": n, "mean_regret": r, "mean_ir_violation": ir} for n, r, ir in rq1]
        print("RQ1 results:")
        for n, r, ir in rq1:
            print("  {}: regret={:.6f}, IR={:.6f}".format(n, r, ir))

    # RQ2
    if run_rq2:
        print("Running RQ2 (scalability: time vs N)...")
        rq2 = rq2_time_vs_n(
            trade_mech_ls, n_list, args.n_items, args.budget,
            n_agents_for_ckpt=args.n_agents,
            num_profiles_per_n=min(200, args.num_profiles), warmup=1, repeat=3,
        )
        summary["rq2"] = [{"n_agents": n, "mechanism": m, "mean_time_sec": t} for n, m, t in rq2]
        print("RQ2 results (N, mechanism, mean_time_sec):")
        for n, m, t in rq2:
            print("  N={}, {}: {:.4f}s".format(n, m, t))

    # RQ3
    if run_rq3:
        print("Running RQ3 (revenue and BF)...")
        rq3 = rq3_revenue_privacy_paper(
            trade_mech_ls, args.n_agents, args.n_items, args.budget,
            num_profiles=args.num_profiles, seeds=seeds,
        )
        summary["rq3"] = rq3
        print("RQ3 results (R̄, W̄ = time-avg per seed over T rounds; ± std over seeds):")
        for row in rq3:
            n = row["mechanism"]
            print(
                "  {}: R={:.4f}±{:.4f}, BF={:.4f}, η={:.4f}±{:.4f}, W̄={:.4f}±{:.4f} (T={}, seeds={})".format(
                    n, row["mean_revenue"], row["std_revenue"], row["bf_rate"],
                    row["revenue_efficiency"], row["std_revenue_efficiency"],
                    row["mean_social_welfare"], row["std_social_welfare"],
                    row["num_rounds_T"], row["num_seeds"],
                )
            )

    # RQ4: optional, would call acc_eval_mechs; skip here unless user provides dataset + ckpts
    if args.run_rq4:
        print("RQ4 (FL accuracy) is optional. Run experiments.acc_eval_mechs_parallel with your trade_mech_ls and dataset.")
        summary["rq4"] = "skipped (use acc_eval_mechs_parallel with dataset and checkpoints)"

    def _nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(x) for x in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    out_path = os.path.join(args.out_dir, "phase4_summary.json")
    with open(out_path, "w") as f:
        json.dump(_nan_to_none(summary), f, indent=2)
    print("Summary written to", out_path)


if __name__ == "__main__":
    main()

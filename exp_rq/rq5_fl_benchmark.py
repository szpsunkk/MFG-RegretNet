#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5: Privacy–utility on FL (MNIST/CIFAR): mean ε̄ vs final accuracy, per-client ε distribution,
Gini, optional ||Δw||_2 per round. Fixed budget_rate B per run (same bids each round).

Outputs: run/privacy_paper/rq5/raw/{DATASET}_a{alpha}_s{seed}.json
Plot: exp_rq/rq5_plot_paper_figures.py

Run from repo root:
  python exp_rq/rq5_fl_benchmark.py --dataset MNIST --alpha 0.5 --seed 0
  python exp_rq/rq5_fl_benchmark.py --budget-rates 0.3,0.6,1.0,1.4 --rounds 50
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets_fl_benchmark import dirichlet_split, generate_privacy_paper_bids, load_cifar10, load_mnist
from exp_rq.rq4_fl_benchmark import (
    _device,
    _fed_round,
    _fed_round_pag_alg2,
    _load_auc,
    _local_xy_tensors,
    _run_auction_round,
)
from FL import Arguments, CIFAR10Net, Net, test
from utils import generate_max_cost


def gini_coefficient(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 1e-9]
    if len(x) < 2:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum() + 1e-12) - (n + 1.0) / n)


def _build_mech_list(mfg_ckpt, regretnet_ckpt, uniform_eps, skip_regretnet, include_pac):
    mechs = []
    mfg = _load_auc(mfg_ckpt)
    reg = _load_auc(regretnet_ckpt) if not skip_regretnet else None
    if mfg is not None:
        mechs.append(("Ours", ["MFG-RegretNet", "ConvlAggr", mfg_ckpt, 1], mfg))
    else:
        print("[RQ5] Warning: no MFG ckpt, skip Ours")
    mechs.append(("CSRA", ["CSRA", "ConvlAggr", "", 1], None))
    mechs.append(("MFG-Pricing", ["MFG-Pricing", "ConvlAggr", "", 1], None))
    if include_pac:
        mechs.append(("PAC", ["PAC", "ConvlAggr", "", 1], None))
    if reg is not None:
        mechs.append(("RegretNet", ["RegretNet", "ConvlAggr", regretnet_ckpt, 1], reg))
    else:
        print("[RQ5] Warning: no RegretNet ckpt, skip RegretNet")
    mechs.append(("Uniform-DP", ["Uniform-DP", "ConvlAggr", "", 1, uniform_eps], None))
    mechs.append(("No-DP (upper)", ["No-DP", "ConvlAggr", "", 1], None))
    return mechs


def _rounds_data_fixed_b(n_rounds, n_agents, budget_rate, base_seed, device):
    out = []
    for r in range(n_rounds):
        rep = generate_privacy_paper_bids(n_agents, 1, 1, seed=base_seed * 100000 + r)
        if rep.dim() == 2:
            rep = rep.unsqueeze(0)
        rep = rep.to(device)
        mc = generate_max_cost(rep)
        budget = mc * float(budget_rate)
        out.append((rep, budget))
    return out


def run_one_fl_rq5(
    label,
    tm,
    auc_m,
    rounds_data,
    n_rounds,
    client_active,
    local_xy,
    model_factory,
    test_data,
    fl_args,
    device,
    pag_fl_alg2=False,
    delta_dp=0.01,
    eps_min_alg2=0.1,
):
    """Returns dict: eps_bar_time_avg, final_acc, eps_per_client_last, v_times_eps_last, gini_eps, update_l2."""
    model = model_factory()
    tm = list(tm)
    auc_copy = auc_m
    eps_means = []
    update_norms = []
    plosses_last = None
    v_last = None

    for rnd in range(n_rounds):
        rep, budget = rounds_data[rnd]
        prev = torch.cat([p.detach().float().flatten() for p in model.parameters()])
        plosses, weights = _run_auction_round(rep, budget, tm, auc_copy, device, client_active)
        ep = plosses.detach().cpu().numpy()
        m = client_active
        vals = [ep[i] for i in range(len(ep)) if m[i] and ep[i] > 1e-9]
        eps_means.append(float(np.mean(vals)) if vals else 0.0)
        if pag_fl_alg2:
            model, _ = _fed_round_pag_alg2(
                model,
                fl_args,
                plosses,
                weights,
                local_xy,
                device,
                delta_dp=delta_dp,
                eps_min=eps_min_alg2,
            )
        else:
            model, _ = _fed_round(model, fl_args, plosses, weights, local_xy, device)
        newv = torch.cat([p.detach().float().flatten() for p in model.parameters()])
        update_norms.append(float((newv - prev).norm().item()))
        if rnd == n_rounds - 1:
            plosses_last = ep.copy()
            v_last = rep[0, :, 0].detach().cpu().numpy()

    final_acc = float(test(model, test_data, fl_args, n_rounds - 1))
    g_eps = gini_coefficient(plosses_last) if plosses_last is not None else 0.0
    vte = (v_last * plosses_last).tolist() if v_last is not None else []

    return {
        "method": label,
        "eps_bar_time_avg": float(np.mean(eps_means)) if eps_means else 0.0,
        "eps_bar_final_round_participants": float(np.mean([plosses_last[i] for i in range(len(plosses_last)) if client_active[i] and plosses_last[i] > 1e-9])) if plosses_last is not None else 0.0,
        "final_test_acc": final_acc,
        "per_client_eps_out": plosses_last.tolist() if plosses_last is not None else [],
        "per_client_v_times_eps": vte,
        "gini_eps_out": g_eps,
        "update_l2_norms": update_norms,
        "note_eps_bar": "Mean over rounds of (mean ε_i among clients with ε_i>0 and active data).",
        "fl_deploy": (
            "pag_alg2_gaussian_weights_eps_fedavg"
            if pag_fl_alg2
            else "laplace_gradient_convlaggr"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="MNIST")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--budget-rates", type=str, default="0.4,0.8,1.2", help="comma-separated B multipliers")
    ap.add_argument("--uniform-eps", type=float, default=2.555)
    ap.add_argument("--regretnet-ckpt", type=str, default="")
    ap.add_argument("--mfg-ckpt", type=str, default="")
    ap.add_argument("--skip-regretnet", action="store_true")
    ap.add_argument("--pac", action="store_true")
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/rq5")
    ap.add_argument("--quick", action="store_true", help="15 rounds, 2 budget rates")
    ap.add_argument(
        "--fl-lr",
        type=float,
        default=None,
        help="FL SGD lr (default: 0.01 MNIST / 0.05 CIFAR10)",
    )
    ap.add_argument(
        "--pag-fl-alg2",
        action="store_true",
        help="Paper Alg.2 FL (Gaussian on weights + ε-FedAvg)",
    )
    ap.add_argument("--delta-dp", type=float, default=0.01)
    ap.add_argument("--eps-min-alg2", type=float, default=0.1)
    ap.add_argument(
        "--local-epochs",
        type=int,
        default=2,
        help="local passes per FL round (align with RQ4; default 2)",
    )
    ap.add_argument("--local-batch-size", type=int, default=64)
    args = ap.parse_args()

    if args.quick:
        args.rounds = min(args.rounds, 15)
        brs = [0.6, 1.0]
    else:
        brs = [float(x.strip()) for x in args.budget_rates.split(",") if x.strip()]

    try:
        from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt, resolve_regretnet_ckpt
    except ImportError:
        resolve_regretnet_ckpt = lambda **k: ""
        resolve_mfg_regretnet_ckpt = lambda **k: ""

    rn = args.regretnet_ckpt or resolve_regretnet_ckpt(n_agents=args.n_agents, n_items=1)
    mfg = args.mfg_ckpt or resolve_mfg_regretnet_ckpt(n_agents=args.n_agents, n_items=1)

    device = _device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = args.dataset.upper()
    if ds == "MNIST":
        train_full, test_data = load_mnist()
        model_factory = lambda: Net().to(device)
    else:
        train_full, test_data = load_cifar10()
        model_factory = lambda: CIFAR10Net().to(device)
    if args.fl_lr is not None:
        lr = float(args.fl_lr)
    elif args.local_epochs >= 2:
        lr = 0.02 if ds == "MNIST" else 0.045
    else:
        lr = 0.01 if ds == "MNIST" else 0.05

    idxs = dirichlet_split(train_full, args.n_agents, 10, alpha=args.alpha, min_size=10, seed=args.seed)
    client_active = [len(idxs[i]) > 0 for i in range(args.n_agents)]
    local_xy = _local_xy_tensors(train_full, idxs, args.n_agents, device)

    fl_args = Arguments()
    fl_args.rounds = args.rounds
    fl_args.lr = lr
    fl_args.local_epochs = int(args.local_epochs)
    fl_args.local_batch_size = int(args.local_batch_size)
    fl_args.L = 1.0
    fl_args.sensi = 2.0
    fl_args.device = device
    fl_args.no_cuda = not torch.cuda.is_available()

    mechs = _build_mech_list(mfg, rn, args.uniform_eps, args.skip_regretnet, args.pac)

    all_runs = []
    for B in brs:
        # Same bid sequence for every B; only budget multiplier changes (fair Pareto).
        rounds_data = _rounds_data_fixed_b(args.rounds, args.n_agents, B, args.seed, device)
        for label, tm, auc_m in mechs:
            row = run_one_fl_rq5(
                label,
                tm,
                auc_m,
                rounds_data,
                args.rounds,
                client_active,
                local_xy,
                model_factory,
                test_data,
                fl_args,
                device,
                pag_fl_alg2=args.pag_fl_alg2,
                delta_dp=args.delta_dp,
                eps_min_alg2=args.eps_min_alg2,
            )
            row["budget_rate"] = B
            # trim large arrays for JSON if needed
            row["update_l2_norms"] = row["update_l2_norms"][:200]
            all_runs.append(row)
            print(
                "  B={} {}  eps_bar={:.4f}  acc={:.4f}  gini={:.3f}".format(
                    B, label, row["eps_bar_time_avg"], row["final_test_acc"], row["gini_eps_out"]
                )
            )

    meta = {
        "dataset": args.dataset,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_rounds": args.rounds,
        "budget_rates": brs,
        "fl_deploy": (
            "pag_alg2_gaussian_weights_eps_fedavg"
            if args.pag_fl_alg2
            else "laplace_gradient_convlaggr"
        ),
    }
    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    suffix = "_pagalg2" if args.pag_fl_alg2 else ""
    if "CIFAR" in ds:
        fn = "CIFAR10_a{}_s{}{}.json".format(args.alpha, args.seed, suffix)
    else:
        fn = "MNIST_a{}_s{}{}.json".format(args.alpha, args.seed, suffix)
    out_path = os.path.join(raw_dir, fn)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "runs": all_runs}, f, indent=2)
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

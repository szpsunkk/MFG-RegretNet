#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4: FL test accuracy / train loss vs global round on MNIST & CIFAR-10 (Dirichlet non-IID).

- Same auction inputs per round across all methods (fair comparison).
- FL deploy: default Laplace-on-gradient + ConvlAggr; use --pag-fl-alg2 for paper Alg.2
  (Gaussian on full local weights + ε-weighted FedAvg).
- Methods: Ours (MFG-RegretNet), CSRA, MFG-Pricing, RegretNet, Uniform-DP, No-DP (upper bound).
- Outputs one JSON per (dataset, alpha, seed); aggregate + plot via rq4_plot_paper_figures.py.

Run from repo root:
  python exp_rq/rq4_fl_benchmark.py --dataset MNIST --alpha 0.5 --seed 0 --out-dir run/privacy_paper/rq4
"""
from __future__ import division, print_function

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets_fl_benchmark import (
    dirichlet_split,
    generate_privacy_paper_bids,
    load_cifar10,
    load_mnist,
)
from FL import Arguments, CIFAR10Net, Net, laplace_noise_like, pag_fl_alg2_round, test
from utils import generate_max_cost


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _local_xy_tensors(full_train, idxs_client_dict, n_agents, device):
    """One (X, Y) tensor pair per client (full local data, same as client.Client.data)."""
    out = []
    for i in range(n_agents):
        idxs = sorted(idxs_client_dict[i])
        if len(idxs) == 0:
            # pad with dummy — should not train; mechanism should give plosses 0
            idxs = [0]
        sub = torch.utils.data.Subset(full_train, idxs)
        loader = Data.DataLoader(sub, batch_size=len(sub), shuffle=False)
        X, Y = next(iter(loader))
        X = X.to(device).float() if X.dtype in (torch.float32, torch.float64) else X.to(device).long()
        Y = Y.to(device).long()
        out.append((X, Y))
    return out


def _fed_round(model, fl_args, plosses, weights, local_xy, device):
    """
    One FL round: proper FedAvg with Laplace DP noise on gradients.
    Each client starts from the global model, computes noisy gradient updates,
    and the server aggregates them via weighted sum.
    """
    plosses = plosses.view(-1).float()
    weights = weights.view(-1).float()
    n_agents = plosses.shape[0]
    if (plosses <= 0).all():
        return model, 0.0

    le = max(1, int(getattr(fl_args, "local_epochs", 1)))
    lbs = max(1, int(getattr(fl_args, "local_batch_size", 64)))
    max_batches = int(getattr(fl_args, "max_batches_per_client", 2))
    sensi = float(fl_args.sensi)
    flr = float(fl_args.lr)
    clip = float(getattr(fl_args, "L", 1.0))

    global_sd = {k: v.clone().detach() for k, v in model.state_dict().items()}
    # Accumulate weighted update: Σ w_i * (local_params_i - global_params)
    aggregated_update = {k: torch.zeros_like(v) for k, v in global_sd.items()}
    total_weight = 0.0
    losses = []

    for i in range(n_agents):
        if plosses[i] <= 0:
            continue
        epsi = float(plosses[i].clamp(min=1e-6).item())
        w_i = float(weights[i].item())
        if w_i <= 0:
            continue

        # Each client starts from global model
        local_model = copy.deepcopy(model).to(device)
        local_model.train()
        X, Y = local_xy[i]
        n_samples = X.size(0)
        bs = max(1, min(lbs, n_samples))
        loader = Data.DataLoader(
            Data.TensorDataset(X, Y), batch_size=bs, shuffle=True, drop_last=False
        )
        optimizer = optim.SGD(local_model.parameters(), lr=flr, momentum=0.9)

        for _ep in range(le):
            for batch_idx, (xb, yb) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                optimizer.zero_grad()
                pred_Y = local_model(xb)
                loss = nn.CrossEntropyLoss()(pred_Y, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(local_model.parameters(), clip)
                # Add Laplace noise to gradients BEFORE step
                with torch.no_grad():
                    for param in local_model.parameters():
                        if param.grad is not None:
                            param.grad.add_(laplace_noise_like(param.grad, sensi / (epsi * bs)))
                optimizer.step()
                losses.append(loss.item())

        # Compute update = local_params - global_params
        local_sd = local_model.state_dict()
        for k in global_sd:
            if global_sd[k].dtype in (torch.float32, torch.float64):
                aggregated_update[k] += w_i * (local_sd[k].float() - global_sd[k].float())
        total_weight += w_i

    if total_weight <= 0:
        return model, 0.0

    # Apply aggregated update
    new_sd = {}
    for k in global_sd:
        if global_sd[k].dtype in (torch.float32, torch.float64):
            new_sd[k] = global_sd[k] + aggregated_update[k] / max(total_weight, 1e-6)
        else:
            new_sd[k] = global_sd[k]
    model.load_state_dict(new_sd)
    return model, float(np.mean(losses)) if losses else 0.0


def _fed_round_pag_alg2(
    model, fl_args, plosses, weights, local_xy, device, delta_dp=0.01, eps_min=0.1
):
    """
    Paper Algorithm 2: Gaussian on full local weights + ε-weighted FedAvg.
    `weights` (ConvlAggr) are ignored for aggregation; only ε_out from plosses.
    """
    m = model.to(device)
    return pag_fl_alg2_round(
        m, fl_args, plosses.to(device), local_xy, delta=delta_dp, eps_min=eps_min
    )


def _run_auction_round(reports_1row, budget_1, trade_mech, auc_model, device, client_active_mask):
    """reports_1row: (1, n_agents, 3). client_active_mask: list[bool]."""
    from experiments import auction

    reports = reports_1row.to(device).float()
    budget = budget_1.to(device).float()
    n_agents = reports.shape[1]
    mask = torch.tensor(client_active_mask, device=device, dtype=torch.bool)

    if trade_mech[0] == "Uniform-DP":
        eps_u = float(trade_mech[4]) if len(trade_mech) > 4 else 2.555
        plosses = torch.full((1, n_agents), eps_u, device=device)
        plosses[0, ~mask] = 0.0
        from aggregation import aggr_batch

        weights = aggr_batch(plosses, reports[:, :, -1], method="ConvlAggr")
        return plosses[0].detach(), weights[0].detach()
    if trade_mech[0] == "No-DP":
        plosses = torch.full((1, n_agents), 1e6, device=device)
        plosses[0, ~mask] = 0.0
        from aggregation import aggr_batch

        weights = aggr_batch(plosses, reports[:, :, -1], method="ConvlAggr")
        return plosses[0].detach(), weights[0].detach()

    tm = [trade_mech[0], trade_mech[1], trade_mech[2], trade_mech[3]]
    auc = auc_model.to(device) if auc_model is not None else None
    if tm[0] in ("PAC", "VCG", "CSRA", "MFG-Pricing", "All-in", "FairQuery"):
        auc = None
    plosses, weights = auction(reports, budget, tm, model=auc)
    p = plosses.clone()
    p[0, ~mask] = 0.0
    return p[0].detach(), weights[0].detach()


def _load_auc(path):
    if not path or not os.path.isfile(path):
        return None
    from experiments import load_auc_model

    return load_auc_model(path)


def run_one_setting(
    dataset_name,
    alpha,
    seed,
    n_agents,
    n_rounds,
    rnd_step,
    min_budget_rate,
    max_budget_rate,
    out_path,
    regretnet_ckpt,
    dm_regretnet_ckpt="",
    mfg_ckpt="",
    uniform_eps=2.555,
    skip_regretnet=False,
    include_pac=True,
    include_vcg=True,
    pag_fl_alg2=False,
    delta_dp=0.01,
    eps_min_alg2=0.1,
    local_epochs=2,
    local_batch_size=64,
    fl_lr=None,
    fixed_budget=50.0,
):
    device = _device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if dataset_name.upper() == "MNIST":
        train_full, test_data = load_mnist()
        model_factory = lambda: Net().to(device)
    elif dataset_name.upper() == "CIFAR10":
        train_full, test_data = load_cifar10()
        model_factory = lambda: CIFAR10Net().to(device)
    else:
        raise ValueError("dataset must be MNIST or CIFAR10")

    n_classes = 10
    idxs = dirichlet_split(train_full, n_agents, n_classes, alpha=alpha, min_size=10, seed=seed)
    client_active = [len(idxs[i]) > 0 for i in range(n_agents)]
    local_xy = _local_xy_tensors(train_full, idxs, n_agents, device)

    fl_args = Arguments()
    fl_args.rounds = n_rounds
    dsu = dataset_name.upper()
    fl_args.local_epochs = int(local_epochs)
    fl_args.local_batch_size = int(local_batch_size)
    fl_args.max_batches_per_client = 2  # limit batches per client for speed; enough signal
    if fl_lr is not None:
        fl_args.lr = float(fl_lr)
    elif fl_args.local_epochs >= 2:
        # gentler step with multi-pass local training (smoother acc vs rounds)
        fl_args.lr = 0.02 if dsu == "MNIST" else 0.045
    else:
        fl_args.lr = 0.01 if dsu == "MNIST" else 0.05
    fl_args.L = 1.0
    fl_args.sensi = 2.0 * fl_args.L
    fl_args.device = device
    fl_args.no_cuda = not torch.cuda.is_available()

    # Optional: match experiments.py random budget per round (same draw for all mechs)
    rng_b = np.random.RandomState(seed + 999)
    budget_rates = torch.tensor(
        [
            rng_b.uniform(min_budget_rate, max_budget_rate)
            for _ in range(n_rounds)
        ],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1)

    rounds_data = []
    for r in range(n_rounds):
        # Order is (n_agents, n_items, num_profiles), not (num_profiles, n_agents, n_items).
        rep = generate_privacy_paper_bids(n_agents, 1, 1, seed=seed * 100000 + r)
        if rep.dim() == 2:
            rep = rep.unsqueeze(0)
        rep = rep.to(device)
        if fixed_budget > 0:
            budget = fixed_budget * torch.ones(1, 1, device=device)
        else:
            mc = generate_max_cost(rep)
            budget = mc * budget_rates[r]
        rounds_data.append((rep, budget))

    mechs = []
    mfg = _load_auc(mfg_ckpt)
    reg = _load_auc(regretnet_ckpt) if not skip_regretnet else None
    dm_reg = _load_auc(dm_regretnet_ckpt) if dm_regretnet_ckpt else None

    if mfg is not None:
        mechs.append(
            ("Ours", ["MFG-RegretNet", "ConvlAggr", mfg_ckpt, 1], mfg)
        )
    else:
        print("[RQ4] Warning: MFG-RegretNet ckpt missing, skip Ours")

    mechs.append(("CSRA", ["CSRA", "ConvlAggr", "", 1], None))
    mechs.append(("MFG-Pricing", ["MFG-Pricing", "ConvlAggr", "", 1], None))
    if include_pac:
        mechs.append(("PAC", ["PAC", "ConvlAggr", "", 1], None))
    if include_vcg:
        mechs.append(("VCG", ["VCG", "ConvlAggr", "", 1], None))
    if reg is not None:
        mechs.append(("RegretNet", ["RegretNet", "ConvlAggr", regretnet_ckpt, 1], reg))
    else:
        print("[RQ4] Warning: RegretNet ckpt missing, skip RegretNet")
    if dm_reg is not None:
        mechs.append(("DM-RegretNet", ["DM-RegretNet", "ConvlAggr", dm_regretnet_ckpt, 1], dm_reg))

    uni = ["Uniform-DP", "ConvlAggr", "", 1, uniform_eps]
    mechs.append(("Uniform-DP", uni, None))
    nodp = ["No-DP", "ConvlAggr", "", 1]
    mechs.append(("No-DP (upper)", nodp, None))

    results = {}
    round_log_idx = []
    for rnd in range(n_rounds):
        if rnd == 0 or (rnd + 1) % rnd_step == 0 or rnd == n_rounds - 1:
            round_log_idx.append(rnd + 1)

    for label, tm_base, auc_m in mechs:
        print("[RQ4] method: {}  ({} rounds)".format(label, n_rounds), flush=True)
        tm = list(tm_base)
        model = model_factory()
        accs = []
        losses = []
        eps_bars = []  # track mean eps_out per round for matched-ε column
        auc_copy = auc_m
        if auc_m is not None and label not in ("Ours", "RegretNet", "DM-RegretNet"):
            auc_copy = auc_m
        for rnd in range(n_rounds):
            if rnd == 0 or (rnd + 1) % max(1, n_rounds // 10) == 0 or rnd == n_rounds - 1:
                print("  round {}/{}".format(rnd + 1, n_rounds), flush=True)
            rep, budget = rounds_data[rnd]
            plosses, weights = _run_auction_round(
                rep, budget, tm, auc_copy, device, client_active
            )
            # Track mean eps_out for matched-ε analysis
            active_eps = plosses[plosses > 0]
            if len(active_eps) > 0:
                eps_bars.append(float(active_eps.mean().item()))
            if pag_fl_alg2:
                model, tr_loss = _fed_round_pag_alg2(
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
                model, tr_loss = _fed_round(
                    model, fl_args, plosses, weights, local_xy, device
                )
            if rnd + 1 in round_log_idx:
                acc = test(model, test_data, fl_args, rnd)
                accs.append(acc)
                losses.append(tr_loss)
        results[label] = {"test_acc": accs, "train_loss": losses, "mean_eps_bar": float(np.mean(eps_bars)) if eps_bars else 0.0}

    meta = {
        "dataset": dataset_name,
        "alpha": alpha,
        "seed": seed,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "rnd_step": rnd_step,
        "budget_rate_range": [min_budget_rate, max_budget_rate],
        "fixed_budget": fixed_budget,
        "note": "Per-round budget = fixed_budget if >0 else max_cost(reports) * U[min,max]; same draws for all methods.",
        "rounds_logged": round_log_idx,
        "fl_deploy": (
            "pag_alg2_gaussian_weights_eps_fedavg"
            if pag_fl_alg2
            else "laplace_gradient_convlaggr"
        ),
        "delta_dp_gaussian": float(delta_dp) if pag_fl_alg2 else None,
        "eps_min_alg2": float(eps_min_alg2) if pag_fl_alg2 else None,
        "local_epochs": int(local_epochs),
        "local_batch_size": int(local_batch_size),
        "fl_lr": float(fl_args.lr),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "methods": results}, f, indent=2)
    print("Wrote", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="MNIST", help="MNIST | CIFAR10")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--rnd-step", type=int, default=5)
    ap.add_argument("--min-budget-rate", type=float, default=0.1)
    ap.add_argument("--max-budget-rate", type=float, default=2.0)
    ap.add_argument("--budget-rate", type=float, default=None, help="if set, fixed B = rate * max_cost (overrides random)")
    ap.add_argument("--uniform-eps", type=float, default=2.555, help="epsilon for Uniform-DP baseline")
    ap.add_argument("--regretnet-ckpt", type=str, default="")
    ap.add_argument("--dm-regretnet-ckpt", type=str, default="")
    ap.add_argument("--mfg-ckpt", type=str, default="")
    ap.add_argument("--skip-regretnet", action="store_true")
    ap.add_argument("--fixed-budget", type=float, default=50.0, help="fixed per-round budget (0=random from max_cost)")
    ap.add_argument("--max-batches-per-client", type=int, default=2,
                    help="max mini-batches per client per local epoch (limit for speed)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="run/privacy_paper/rq4",
        help="raw JSON: {out-dir}/raw/{dataset}_a{alpha}_s{seed}.json",
    )
    ap.add_argument("--quick", action="store_true", help="20 rounds, rnd_step 4")
    ap.add_argument("--pac", action="store_true", help="include PAC baseline")
    ap.add_argument(
        "--pag-fl-alg2",
        action="store_true",
        help="Paper Alg.2: Gaussian noise on local weights + ε-weighted FedAvg (else Laplace grad + ConvlAggr)",
    )
    ap.add_argument(
        "--delta-dp",
        type=float,
        default=0.01,
        help="δ for Gaussian σ in Alg.2 (default 0.01)",
    )
    ap.add_argument(
        "--eps-min-alg2",
        type=float,
        default=0.1,
        help="floor on ε_out in Alg.2 noise scale (default 0.1)",
    )
    ap.add_argument(
        "--local-epochs",
        type=int,
        default=2,
        help="local passes over each client's data per FL round (>=1; default 2 for smoother curves)",
    )
    ap.add_argument(
        "--local-batch-size",
        type=int,
        default=64,
        help="mini-batch size for local training (default 64)",
    )
    ap.add_argument(
        "--fl-lr",
        type=float,
        default=None,
        help="FL SGD lr (default: auto from dataset & local-epochs)",
    )
    args = ap.parse_args()

    if args.quick:
        args.rounds = min(args.rounds, 20)
        args.rnd_step = min(args.rnd_step, 4)

    try:
        from exp_rq.rq1_ckpt_resolve import resolve_mfg_regretnet_ckpt, resolve_regretnet_ckpt, resolve_dm_regretnet_ckpt
    except ImportError:
        resolve_regretnet_ckpt = lambda **k: ""
        resolve_mfg_regretnet_ckpt = lambda **k: ""
        resolve_dm_regretnet_ckpt = lambda **k: ""

    rn = args.regretnet_ckpt or resolve_regretnet_ckpt(
        n_agents=args.n_agents, n_items=1
    )
    dm_rn = args.dm_regretnet_ckpt or resolve_dm_regretnet_ckpt(
        n_agents=args.n_agents, n_items=1
    )
    mfg = args.mfg_ckpt or resolve_mfg_regretnet_ckpt(
        n_agents=args.n_agents, n_items=1
    )

    br_min, br_max = args.min_budget_rate, args.max_budget_rate
    if args.budget_rate is not None:
        br_min = br_max = float(args.budget_rate)

    ds = args.dataset.upper().replace("-", "")
    suf = "_pagalg2" if args.pag_fl_alg2 else ""
    fn = "{}_a{}_s{}{}.json".format(ds, args.alpha, args.seed, suf)
    raw_dir = os.path.join(args.out_dir, "raw")
    out_path = os.path.join(raw_dir, fn)

    run_one_setting(
        dataset_name=args.dataset,
        alpha=args.alpha,
        seed=args.seed,
        n_agents=args.n_agents,
        n_rounds=args.rounds,
        rnd_step=args.rnd_step,
        min_budget_rate=br_min,
        max_budget_rate=br_max,
        out_path=out_path,
        regretnet_ckpt=rn,
        dm_regretnet_ckpt=dm_rn,
        mfg_ckpt=mfg,
        uniform_eps=args.uniform_eps,
        skip_regretnet=args.skip_regretnet,
        include_pac=args.pac,
        include_vcg=True,
        pag_fl_alg2=args.pag_fl_alg2,
        delta_dp=args.delta_dp,
        eps_min_alg2=args.eps_min_alg2,
        local_epochs=args.local_epochs,
        local_batch_size=args.local_batch_size,
        fl_lr=args.fl_lr,
        fixed_budget=args.fixed_budget,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

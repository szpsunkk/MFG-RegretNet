#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Figure 9 from FL-market.pdf:
  "Model accuracy over FL training rounds"

Setup (from paper Section VI-A):
  - Dataset: NSL-KDD, logistic regression, 5-class classification
  - 1000 data owners (IID or Dirichlet non-IID), 10 selected per round
  - 100 FL rounds, budget factor ~ U[0.1, 2.0] per round
  - Mechanisms: RegretNet+ConvlAggr, RegretNet+OptAggr,
                M-RegretNet+ConvlAggr, M-RegretNet+OptAggr,
                DM-RegretNet+ConvlAggr, DM-RegretNet+OptAggr
  - Evaluation logged every 10 rounds (rnd_step=10)

Run from repo root:
  python reproduce_fig9.py
"""
from __future__ import division, print_function

import copy
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from aggregation import aggr_batch
from client import Clients
from datasets import load_nslkdd
from experiments import (
    Exp_Args,
    acc_eval,
    auction,
    load_auc_model,
    map_data_dir,
    map_labels,
)
from FL import Arguments, Logistic
from utils import (
    DM_CONVL_NSLKDD_IID,
    DM_CONVL_NSLKDD_NIID,
    DM_OPT_NSLKDD_IID,
    DM_OPT_NSLKDD_NIID,
    MREG_CONVL_NSLKDD_IID,
    MREG_CONVL_NSLKDD_NIID,
    MREG_OPT_NSLKDD_IID,
    MREG_OPT_NSLKDD_NIID,
    REG_CONVL_NSLKDD_IID,
    REG_CONVL_NSLKDD_NIID,
    REG_OPT_NSLKDD_IID,
    REG_OPT_NSLKDD_NIID,
    generate_max_cost,
)

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_RUNS = 10          # average over N_RUNS independent experiments
N_ROUNDS = 100       # FL training rounds per run
N_AGENTS = 10        # bidders per round
RND_STEP = 10        # accuracy logging interval
MIN_B = 0.1
MAX_B = 2.0
OUT_DIR = "run/fig9_reproduction"
FIG_DIR = "run/fig9_reproduction"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def _run_one(trade_mech_ls, train_data, test_data, clients, fl_args, exp_args, run_idx):
    """Run one independent trial; returns list[list[float]] — acc curves per mech."""
    torch.manual_seed(run_idx * 7 + 13)
    np.random.seed(run_idx * 7 + 13)

    # Always run on CPU in worker processes to avoid multi-GPU issues
    device = torch.device("cpu")
    fl_args.device = device
    fl_args.use_cuda = False
    fl_args.no_cuda = True

    local_sets = clients.return_local_sets_run(train_data, exp_args.n_agents, 0)
    del train_data

    # Per-round budget rate drawn uniformly at random (same for all mechs in this trial)
    budget_rate = torch.rand((fl_args.rounds, 1)) * (
        exp_args.max_budget_rate - exp_args.min_budget_rate
    ) + exp_args.min_budget_rate

    fl_model_base = Logistic(fl_args.input_size, fl_args.output_size).to(device)

    acc_mech_ls = []
    for trade_mech in trade_mech_ls:
        n_items = trade_mech[3]
        if trade_mech[0] in ("All-in", "PAC", "VCG", "CSRA"):
            auc_model = None
        else:
            auc_model = load_auc_model(trade_mech[2]).to(device)

        reports = clients.return_bids_run(n_items, 0)
        reports = torch.tensor(reports).float().to(device).reshape(
            -1, exp_args.n_agents, n_items + 4
        )[:, :, :-2]
        fl_args.device = device

        model = copy.deepcopy(fl_model_base)
        max_cost = generate_max_cost(reports)
        budget = max_cost * budget_rate
        plosses, weights = auction(reports, budget, trade_mech, model=auc_model)
        accs = acc_eval(plosses, weights, model, local_sets, test_data, fl_args,
                        multirnd=exp_args.rnd_step)
        acc_mech_ls.append(accs)

    return acc_mech_ls


def run_parallel(trade_mech_ls, exp_args):
    """Run N_RUNS trials in parallel (up to 4 workers) and average."""
    fl_args = Arguments()
    fl_args.rounds = exp_args.n_rounds
    fl_args.input_size = 122
    fl_args.output_size = 5
    fl_args.shape = (-1, 122)
    fl_args.device = DEVICE

    clients_master = Clients()
    clients_master.dirs = map_data_dir(exp_args.dataset, exp_args.iid)
    clients_master.filename = "test_profiles_2mp.json"
    clients_master.load_json()

    train_data, test_data = load_nslkdd()

    n_workers = min(4, exp_args.n_runs)
    n_pools = math.ceil(exp_args.n_runs / n_workers)

    all_accs = []   # list of (n_mechs, n_log_rounds) arrays
    run_global = 0
    for pool_idx in range(n_pools):
        pool = mp.Pool(n_workers)
        workers = []
        for _ in range(n_workers):
            if run_global >= exp_args.n_runs:
                break
            sub_clients = clients_master.return_clients_by_run(run_global % clients_master.n_runs)
            w = pool.apply_async(
                _run_one,
                args=(trade_mech_ls, train_data, test_data, sub_clients, fl_args, exp_args, run_global),
            )
            workers.append(w)
            run_global += 1

        pool.close()
        pool.join()

        for w in workers:
            result = w.get()   # list of n_mechs acc curves
            all_accs.append(result)

    # shape: (n_runs, n_mechs, n_log_pts)
    accs_np = np.array(all_accs)
    mean_accs = accs_np.mean(axis=0).tolist()  # (n_mechs, n_log_pts)

    rnd_ls = [1] + [(r + 1) * exp_args.rnd_step
                    for r in range(exp_args.n_rounds // exp_args.rnd_step)]
    return rnd_ls, mean_accs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STYLE_CYCLE = [
    dict(marker='s', color='r',  linestyle='solid'),
    dict(marker='h', color='g',  linestyle='solid'),
    dict(marker='d', color='b',  linestyle='solid'),
    dict(marker='o', color='k',  linestyle='solid'),
    dict(marker='^', color='m',  linestyle='dashed'),
    dict(marker='v', color='c',  linestyle='dashed'),
]

LEGEND_FONT = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
LABEL_FONT  = {'family': 'Times New Roman', 'weight': 'black',  'size': 14}
TITLE_FONT  = {'family': 'Times New Roman', 'weight': 'black',  'size': 14}


def _plot_acc_rnd(rnd_ls, acc_ls, labels, title, fpath):
    plt.figure(figsize=(7, 4.5))
    for i, (rnds, accs) in enumerate(zip(rnd_ls, acc_ls)):
        st = STYLE_CYCLE[i % len(STYLE_CYCLE)]
        plt.plot(rnds, accs, label=labels[i],
                 marker=st['marker'], color=st['color'],
                 linestyle=st['linestyle'])
    plt.xlabel('Training Round', LABEL_FONT)
    plt.ylabel('Model Accuracy', LABEL_FONT)
    plt.title(title, TITLE_FONT)
    plt.legend(prop=LEGEND_FONT)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 300
    _make_dir(os.path.dirname(fpath))
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _make_dir(OUT_DIR)

    # --- IID ---
    trade_mechs_iid = [
        REG_CONVL_NSLKDD_IID,
        MREG_CONVL_NSLKDD_IID,
        DM_CONVL_NSLKDD_IID,
        REG_OPT_NSLKDD_IID,
        MREG_OPT_NSLKDD_IID,
        DM_OPT_NSLKDD_IID,
    ]
    labels_iid = map_labels(trade_mechs_iid)

    exp_iid = Exp_Args()
    exp_iid.n_runs = N_RUNS
    exp_iid.n_rounds = N_ROUNDS
    exp_iid.n_agents = N_AGENTS
    exp_iid.rnd_step = RND_STEP
    exp_iid.dataset = "NSL-KDD"
    exp_iid.iid = True
    exp_iid.min_budget_rate = MIN_B
    exp_iid.max_budget_rate = MAX_B
    exp_iid.vary_budget = False

    print("=" * 60)
    print("Reproducing Figure 9 — NSL-KDD IID")
    print("=" * 60)
    rnd_iid, accs_iid = run_parallel(trade_mechs_iid, exp_iid)
    np.save(os.path.join(OUT_DIR, "fig9_iid_accs.npy"), np.array(accs_iid))
    np.save(os.path.join(OUT_DIR, "fig9_iid_rnds.npy"), np.array(rnd_iid))

    rnd_ls_iid = [rnd_iid] * len(trade_mechs_iid)
    _plot_acc_rnd(
        rnd_ls_iid, accs_iid, labels_iid,
        "NSL-KDD (IID) — Model Accuracy vs Training Round",
        os.path.join(FIG_DIR, "fig9_nslkdd_iid.png"),
    )

    # --- Non-IID ---
    _make_dir("data/nslkdd/niid")
    trade_mechs_niid = [
        REG_CONVL_NSLKDD_NIID,
        MREG_CONVL_NSLKDD_NIID,
        DM_CONVL_NSLKDD_NIID,
        REG_OPT_NSLKDD_NIID,
        MREG_OPT_NSLKDD_NIID,
        DM_OPT_NSLKDD_NIID,
    ]
    labels_niid = map_labels(trade_mechs_niid)

    exp_niid = Exp_Args()
    exp_niid.n_runs = N_RUNS
    exp_niid.n_rounds = N_ROUNDS
    exp_niid.n_agents = N_AGENTS
    exp_niid.rnd_step = RND_STEP
    exp_niid.dataset = "NSL-KDD"
    exp_niid.iid = False
    exp_niid.min_budget_rate = MIN_B
    exp_niid.max_budget_rate = MAX_B
    exp_niid.vary_budget = False

    # Generate non-IID client profiles if not present
    niid_profile = "data/nslkdd/niid/test_profiles_2mp.json"
    if not os.path.exists(niid_profile):
        print("\nGenerating non-IID client profiles…")
        clients_niid = Clients()
        clients_niid.dirs = "data/nslkdd/niid/"
        clients_niid.min_pbudget = 0.5
        clients_niid.max_pbudget = 2.0
        clients_niid.filename = "test_profiles_2mp.json"
        clients_niid.generate_clients_mulruns("NSL-KDD", 100, 10, N_RUNS, iid=False, overlap=True)

    print("\n" + "=" * 60)
    print("Reproducing Figure 9 — NSL-KDD Non-IID")
    print("=" * 60)
    rnd_niid, accs_niid = run_parallel(trade_mechs_niid, exp_niid)
    np.save(os.path.join(OUT_DIR, "fig9_niid_accs.npy"), np.array(accs_niid))
    np.save(os.path.join(OUT_DIR, "fig9_niid_rnds.npy"), np.array(rnd_niid))

    rnd_ls_niid = [rnd_niid] * len(trade_mechs_niid)
    _plot_acc_rnd(
        rnd_ls_niid, accs_niid, labels_niid,
        "NSL-KDD (Non-IID) — Model Accuracy vs Training Round",
        os.path.join(FIG_DIR, "fig9_nslkdd_niid.png"),
    )

    # --- Combined two-panel figure (matching paper layout) ---
    print("\nGenerating combined 2-panel figure (Fig 9)…")
    _make_dir(FIG_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("Figure 9: Model Accuracy over FL Training Rounds",
                 fontsize=14, fontweight='bold')

    for panel_idx, (rnd_ls, acc_ls, labels, subtitle) in enumerate([
        (rnd_ls_iid,  accs_iid,  labels_iid,  "NSL-KDD (IID)"),
        (rnd_ls_niid, accs_niid, labels_niid, "NSL-KDD (Non-IID)"),
    ]):
        ax = axes[panel_idx]
        for i, (rnds, accs) in enumerate(zip(rnd_ls, acc_ls)):
            st = STYLE_CYCLE[i % len(STYLE_CYCLE)]
            ax.plot(rnds, accs, label=labels[i],
                    marker=st['marker'], color=st['color'],
                    linestyle=st['linestyle'])
        ax.set_xlabel('Training Round', fontsize=13)
        ax.set_ylabel('Model Accuracy', fontsize=13)
        ax.set_title(subtitle, fontsize=13)
        ax.legend(prop={'size': 9})
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = os.path.join(FIG_DIR, "fig9_combined.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined figure: {combined_path}")

    print("\n✅ Figure 9 reproduction complete!")
    print(f"   IID final accs  : {[round(a[-1], 4) for a in accs_iid]}")
    print(f"   NIID final accs : {[round(a[-1], 4) for a in accs_niid]}")
    print(f"   Figures saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()

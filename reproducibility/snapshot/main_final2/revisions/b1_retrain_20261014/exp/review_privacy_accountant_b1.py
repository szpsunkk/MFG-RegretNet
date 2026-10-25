#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review extension: cumulative privacy accounting for independent-round PAG.

B1 variant (2026-10-15): (i) the learned mechanisms are the step-5b MFG-RegretNet and the B1
RegretNet (input scaling), given as "<kind>:<path>"; (ii) with --pre-release-check, a client is
admitted to a round only if its release keeps its cumulative epsilon within the largest cap:
after the auction on the active set, clients whose release would exceed the cap are removed and
the auction is re-run on the remaining set (neural mechanisms: their release is zeroed), so the
cap is never exceeded. Otherwise identical to review_privacy_accountant.py.

This script attaches a lightweight accountant to existing auction
outputs. Gaussian RDP is only used for the paper Algorithm 2 deployment
path. The default Laplace-gradient path is tracked with basic and
advanced composition bounds instead of being reported as RDP.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_phase4_eval import build_privacy_paper_batch, get_ckpt_path, DEVICE
from experiments import auction, load_auc_model

# B1 (2026-10-15): step-5b / B1 checkpoints are given as "<kind>:<path>" (kind in mfg, regretnet_s, dm_s)
_STEP5B_EXP = os.path.join(_ROOT, "main_final2", "revisions", "step5b_theory_bounds_20261013", "exp")
if _STEP5B_EXP not in sys.path:
    sys.path.insert(0, _STEP5B_EXP)
from models_scaled import MFGRegretNetScaled, RegretNetScaled  # noqa: E402


def _load_new(spec, n_agents):
    kind, path = spec.split(":", 1)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "mfg":
        m = MFGRegretNetScaled(n_agents, 1, hidden_layer_size=100, n_hidden_layers=3, activation="tanh", separate=True)
    else:
        m = RegretNetScaled(n_agents, 1, activation="tanh", hidden_layer_size=100, n_hidden_layers=3,
                            p_activation="softmax", a_activation="deterministic" if kind == "dm_s" else "softmax",
                            separate=True, smoothing=0.125, normalized_input=-1)
    m.load_state_dict(blob["state_dict"])
    m = m.to(DEVICE).eval()
    if kind == "dm_s":
        m.deter_train = False
    return m


def rdp_gaussian_rho(alpha: float, eps_ldp: np.ndarray, delta: float) -> np.ndarray:
    """
    Approximate per-round RDP cost for a Gaussian mechanism calibrated by
    sigma = sqrt(2 log(1.25/delta)) / eps_ldp with unit sensitivity.
    RDP(alpha) = alpha / (2 sigma^2).
    """
    eps = np.maximum(eps_ldp.astype(np.float64), 1e-12)
    coef = 2.0 * math.log(1.25 / max(min(delta, 1.24), 1e-12))
    sigma2 = coef / (eps ** 2)
    return alpha / (2.0 * sigma2)


def rdp_to_eps_min(rho_by_alpha: np.ndarray, alphas: np.ndarray, delta: float):
    eps = rho_by_alpha + math.log(1.0 / delta) / (alphas.reshape(-1, 1) - 1.0)
    best_idx = np.argmin(eps, axis=0)
    return eps[best_idx, np.arange(eps.shape[1])], alphas[best_idx]


def laplace_compose_eps(
    eps_sum: np.ndarray, eps_sq_sum: np.ndarray, adv_term_sum: np.ndarray, delta: float
) -> np.ndarray:
    if delta <= 0:
        return eps_sum.copy()
    adv = adv_term_sum + np.sqrt(2.0 * math.log(1.0 / delta) * eps_sq_sum)
    return np.minimum(eps_sum, adv)


def _load_model_for_mech(mech, n_agents=10):
    if mech[0] in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
        return None
    path = mech[2]
    if ":" in path:
        return _load_new(path, n_agents)
    if not path or not os.path.isfile(path):
        return None
    return load_auc_model(path).to(DEVICE)


def _is_neural_mech(mech):
    return mech[0] not in ("PAC", "VCG", "CSRA", "MFG-Pricing")


def _auction_with_active_mask(rep_t, bud_t, mech, model, active_mask, expected):
    n_agents = rep_t.shape[1]
    if not np.any(active_mask):
        return np.zeros(n_agents, dtype=np.float64)

    mask_t = torch.tensor(active_mask, device=rep_t.device, dtype=torch.bool)
    if _is_neural_mech(mech):
        # Neural checkpoints are trained for a fixed n_agents input dimension.
        # Run the full profile and zero inactive clients afterwards.
        plosses, _ = auction(rep_t, bud_t, mech, model=model, expected=expected, return_payments=False)
        eps_out = plosses.detach().cpu().numpy().reshape(-1)
        return np.where(active_mask, eps_out, 0.0)

    rep_active = rep_t[:, mask_t, :]
    plosses, _ = auction(rep_active, bud_t, mech, model=None, expected=expected, return_payments=False)
    eps_active = plosses.detach().cpu().numpy().reshape(-1)
    eps_out = np.zeros(n_agents, dtype=np.float64)
    eps_out[active_mask] = eps_active
    return eps_out


def run_mechanism(args, mech, seed):
    model = _load_model_for_mech(mech, args.n_agents)
    reports, budget, _ = build_privacy_paper_batch(
        args.rounds, args.n_agents, 1, args.budget, seed, DEVICE
    )
    excluded_by_check = 0

    alphas = np.array(args.alpha_orders, dtype=np.float64)
    rho = np.zeros((len(alphas), args.n_agents), dtype=np.float64)
    eps_sum = np.zeros(args.n_agents, dtype=np.float64)
    eps_sq_sum = np.zeros(args.n_agents, dtype=np.float64)
    adv_term_sum = np.zeros(args.n_agents, dtype=np.float64)
    exhausted = {str(cap): [None for _ in range(args.n_agents)] for cap in args.total_budgets}
    curves = []
    active_mask = np.ones(args.n_agents, dtype=bool)

    for t in range(args.rounds):
        rep_t = reports[t : t + 1].clone()
        bud_t = budget[t : t + 1].clone()
        if args.clip_when_exhausted:
            # Exclude only after the largest cap so all requested cap curves remain observable.
            clip_cap = max(args.total_budgets)
            if args.noise_mechanism == "gaussian":
                eps_total, _ = rdp_to_eps_min(rho, alphas, args.delta)
            else:
                eps_total = laplace_compose_eps(eps_sum, eps_sq_sum, adv_term_sum, args.delta)
            active_mask = eps_total < clip_cap

        eps_out = _auction_with_active_mask(rep_t, bud_t, mech, model, active_mask, args.expected)
        if args.pre_release_check:
            clip_cap = max(args.total_budgets)
            for _ in range(args.n_agents + 1):
                if args.noise_mechanism == "gaussian":
                    rho_h = rho + np.stack([rdp_gaussian_rho(al, eps_out, args.delta) if np.any(eps_out > 0)
                                            else np.zeros(args.n_agents) for al in alphas])
                    rho_h = np.where(eps_out[None, :] > 0, rho_h, rho)
                    hyp, _ = rdp_to_eps_min(rho_h, alphas, args.delta)
                else:
                    hyp = laplace_compose_eps(eps_sum + eps_out, eps_sq_sum + eps_out ** 2,
                                              adv_term_sum + eps_out * np.expm1(np.minimum(eps_out, 50.0)), args.delta)
                viol = (eps_out > 0) & (hyp > clip_cap + 1e-9)
                if not np.any(viol):
                    break
                excluded_by_check += int(np.sum(viol))
                active_mask = active_mask & ~viol
                eps_out = _auction_with_active_mask(rep_t, bud_t, mech, model, active_mask, args.expected)

        if args.noise_mechanism == "gaussian":
            for ai, alpha in enumerate(alphas):
                rho[ai] += rdp_gaussian_rho(alpha, eps_out, args.delta)
            eps_total, best_alpha = rdp_to_eps_min(rho, alphas, args.delta)
        else:
            eps_sum += eps_out
            eps_sq_sum += eps_out ** 2
            assert np.max(eps_out) < 50.0, (
                f"eps_out={np.max(eps_out)} >= 50: expm1 overflow cap was triggered, "
                "advanced composition bound becomes incorrect. Check eps_max in bid generation."
            )
            adv_term_sum += eps_out * np.expm1(np.minimum(eps_out, 50.0))
            eps_total = laplace_compose_eps(eps_sum, eps_sq_sum, adv_term_sum, args.delta)
            best_alpha = np.full(args.n_agents, np.nan, dtype=np.float64)

        for cap in args.total_budgets:
            key = str(cap)
            for i, val in enumerate(eps_total):
                if exhausted[key][i] is None and val >= cap:
                    exhausted[key][i] = t + 1

        if t == 0 or (t + 1) % args.rnd_step == 0 or t == args.rounds - 1:
            curves.append(
                {
                    "round": t + 1,
                    "mean_eps_total": float(np.mean(eps_total)),
                    "p90_eps_total": float(np.percentile(eps_total, 90)),
                    "max_eps_total": float(np.max(eps_total)),
                    "mean_eps_out": float(np.mean(eps_out[eps_out > 0])) if np.any(eps_out > 0) else 0.0,
                    "active_clients": int(np.sum(active_mask)),
                    "best_alpha_median": float(np.nanmedian(best_alpha)) if args.noise_mechanism == "gaussian" else None,
                    "exhausted_count_by_cap": {
                        str(cap): int(np.sum(eps_total >= cap)) for cap in args.total_budgets
                    },
                }
            )

    return {
        "mechanism": mech[0],
        "seed": seed,
        "curve": curves,
        "exhaustion_round_by_cap": exhausted,
        "excluded_by_pre_release_check": excluded_by_check,
        "masking_note": (
            "Analytical baselines are run on the active subprofile before scattering outputs back. "
            "Fixed-N neural checkpoints are evaluated on the full profile and inactive clients are zeroed afterwards."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="run/privacy_paper/review_extensions/privacy_accountant.json")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--rnd-step", type=int, default=10)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--total-budgets", default="10,30,60")
    ap.add_argument("--alpha-orders", default="2,4,8,16,32,64")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--noise-mechanism", choices=["gaussian", "laplace"], default="gaussian")
    ap.add_argument("--expected", action="store_true", default=True)
    ap.add_argument("--instantiate", dest="expected", action="store_false")
    ap.add_argument("--mfg-regretnet-ckpt", default="")
    ap.add_argument("--regretnet-ckpt", default="")
    ap.add_argument("--include-mfg-pricing", action="store_true")
    ap.add_argument("--clip-when-exhausted", action="store_true")
    ap.add_argument("--pre-release-check", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.rounds = min(args.rounds, 5)
        args.rnd_step = 1
        args.n_agents = min(args.n_agents, 5)

    args.total_budgets = [float(x) for x in args.total_budgets.split(",") if x.strip()]
    args.alpha_orders = [float(x) for x in args.alpha_orders.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    mechs = [
        ["PAC", "ConvlAggr", "", 1],
        ["VCG", "ConvlAggr", "", 1],
        ["CSRA", "ConvlAggr", "", 1],
    ]
    if args.include_mfg_pricing:
        mechs.append(["MFG-Pricing", "ConvlAggr", "", 1])
    if args.regretnet_ckpt and (":" in args.regretnet_ckpt or os.path.isfile(args.regretnet_ckpt)):
        mechs.append(["RegretNet", "ConvlAggr", args.regretnet_ckpt, 1])
    if args.mfg_regretnet_ckpt and (":" in args.mfg_regretnet_ckpt or os.path.isfile(args.mfg_regretnet_ckpt)):
        mechs.append(["MFG-RegretNet", "ConvlAggr", args.mfg_regretnet_ckpt, 1])

    rows = []
    for seed in seeds:
        for mech in mechs:
            print("[accountant] mech={} seed={}".format(mech[0], seed), flush=True)
            rows.append(run_mechanism(args, mech, seed))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": vars(args), "results": rows}, f, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()

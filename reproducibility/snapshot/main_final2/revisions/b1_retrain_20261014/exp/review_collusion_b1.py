#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review extension: coordinated collusion robustness.

B1 variant (2026-10-15): (i) the learned mechanisms are the step-5b MFG-RegretNet and the B1
RegretNet (input scaling), given as "<kind>:<path>"; (ii) besides winner manipulation, it reports
the coalition's utility gain under the true valuations,
G_C = sum_{i in C} [u_i(attacked) - u_i(clean)], u_i = p_i - v_i * eps_out_i,
per profile (mean, mean per attacker, share of profiles with G_C > 0) and the coalition's
truthful utility. Otherwise identical to review_collusion_robustness.py.

This complements RQ6 false-bid evaluation by perturbing a coordinated
subset of clients and measuring revenue, welfare, winner manipulation,
and attacker win rate under several collusion patterns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_phase4_eval import build_privacy_paper_batch, DEVICE
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

BF_TOLERANCE = 1e-3


def _load_model(mech, n_agents=10):
    if mech[0] in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
        return None
    path = mech[2]
    if ":" in path:
        return _load_new(path, n_agents)
    if path and os.path.isfile(path):
        return load_auc_model(path).to(DEVICE)
    return None


def _value_high(orig, magnitude):
    if magnitude == "mild":
        return torch.clamp(orig * 1.2, max=1.0)
    if magnitude == "moderate":
        return torch.clamp(orig * 1.5, max=1.0)
    return torch.ones_like(orig)


def _value_low(orig, magnitude):
    if magnitude == "mild":
        return torch.clamp(orig * 0.8, min=0.0)
    if magnitude == "moderate":
        return torch.clamp(orig * 0.5, min=0.0)
    return torch.zeros_like(orig)


def _eps_high(orig, magnitude):
    factor = {"mild": 1.2, "moderate": 1.5, "extreme": 2.0}[magnitude]
    return torch.clamp(orig * factor, max=5.0)


def _eps_low(orig, magnitude):
    factor = {"mild": 0.8, "moderate": 0.5, "extreme": 0.2}[magnitude]
    return torch.clamp(orig * factor, min=0.1)


def _attack_reports(reports, ratio, mode, magnitude, seed):
    rng = np.random.RandomState(seed)
    out = reports.clone()
    n_agents = reports.shape[1]
    k = max(1, int(round(n_agents * ratio)))
    attackers = rng.choice(n_agents, size=k, replace=False)
    if mode == "overbid":
        out[:, attackers, 0] = _value_high(out[:, attackers, 0], magnitude)
    elif mode == "underbid":
        out[:, attackers, 0] = _value_low(out[:, attackers, 0], magnitude)
    elif mode == "epsilon_inflate":
        out[:, attackers, -2] = _eps_high(out[:, attackers, -2], magnitude)
    elif mode == "epsilon_deflate":
        out[:, attackers, -2] = _eps_low(out[:, attackers, -2], magnitude)
    elif mode == "underbid_epsilon_deflate":
        out[:, attackers, 0] = _value_low(out[:, attackers, 0], magnitude)
        out[:, attackers, -2] = _eps_low(out[:, attackers, -2], magnitude)
    elif mode == "underbid_epsilon_inflate":
        out[:, attackers, 0] = _value_low(out[:, attackers, 0], magnitude)
        out[:, attackers, -2] = _eps_high(out[:, attackers, -2], magnitude)
    else:
        raise ValueError("unknown attack mode {}".format(mode))
    return out, attackers.tolist()


def _eval_once(reports, budget, mech, model, true_reports, winner_threshold, expected):
    plosses, weights, payments = auction(reports, budget, mech, model=model, return_payments=True, expected=expected)
    rev_t = payments.sum(dim=1).detach().cpu().numpy()
    winners = (plosses.detach().cpu().numpy() > winner_threshold)
    v_true = true_reports[:, :, 0].detach().cpu().numpy()
    eps_out = plosses.detach().cpu().numpy()
    pay = payments.detach().cpu().numpy()
    welfare_t = pay.sum(axis=1) - (v_true * eps_out).sum(axis=1)
    return {
        "revenue_mean": float(np.mean(rev_t)),
        "welfare_mean": float(np.mean(welfare_t)),
        "utility": pay - v_true * eps_out,
        "winners": winners,
        "bf_rate": float(np.mean(rev_t <= budget.detach().cpu().numpy().reshape(-1) + BF_TOLERANCE)),
    }


def run(args, mech, seed, ratio, mode, magnitude):
    model = _load_model(mech, args.n_agents)
    reports, budget, _ = build_privacy_paper_batch(
        args.num_profiles, args.n_agents, 1, args.budget, seed, DEVICE
    )
    attacked, attackers = _attack_reports(reports, ratio, mode, magnitude, seed + 999)
    clean = _eval_once(reports, budget, mech, model, reports, args.winner_threshold, args.expected)
    robust = _eval_once(attacked, budget, mech, model, reports, args.winner_threshold, args.expected)
    clean_w = clean["winners"]
    rob_w = robust["winners"]
    changed = np.any(clean_w != rob_w, axis=1)
    attackers_arr = np.array(attackers, dtype=int)
    attacker_win_clean = clean_w[:, attackers_arr].mean() if len(attackers_arr) else 0.0
    attacker_win_attack = rob_w[:, attackers_arr].mean() if len(attackers_arr) else 0.0
    gain = (robust["utility"][:, attackers_arr] - clean["utility"][:, attackers_arr]).sum(axis=1)
    coal_truth = clean["utility"][:, attackers_arr].sum(axis=1)
    rev_loss = 1.0 - robust["revenue_mean"] / clean["revenue_mean"] if clean["revenue_mean"] > 0 else 0.0
    welfare_loss = 1.0 - robust["welfare_mean"] / clean["welfare_mean"] if clean["welfare_mean"] > 0 else 0.0
    return {
        "mechanism": mech[0],
        "seed": seed,
        "attack_mode": mode,
        "attack_magnitude": magnitude,
        "collusion_ratio": float(ratio),
        "num_attackers": len(attackers),
        "winner_threshold": float(args.winner_threshold),
        "revenue_clean": clean["revenue_mean"],
        "revenue_attack": robust["revenue_mean"],
        "revenue_loss_ratio": float(rev_loss),
        "welfare_clean": clean["welfare_mean"],
        "welfare_attack": robust["welfare_mean"],
        "welfare_loss_ratio": float(welfare_loss),
        "winner_manipulation_rate": float(np.mean(changed)),
        "coalition_gain_mean": float(np.mean(gain)),
        "coalition_gain_per_attacker": float(np.mean(gain) / max(1, len(attackers))),
        "coalition_gain_positive_share": float(np.mean(gain > 1e-6)),
        "coalition_truthful_utility_mean": float(np.mean(coal_truth)),
        "attacker_win_rate_clean": float(attacker_win_clean),
        "attacker_win_rate_attack": float(attacker_win_attack),
        "bf_clean": clean["bf_rate"],
        "bf_attack": robust["bf_rate"],
        "interpretation_note": (
            "PAC/VCG/CSRA are reverse-procurement auctions; overbidding can exclude attackers, "
            "so revenue_loss_ratio may be negative. Winner manipulation and attacker win-rate delta "
            "are the primary manipulation signals for those cases."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="run/privacy_paper/review_extensions/collusion_robustness.json")
    ap.add_argument("--num-profiles", type=int, default=1000)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ratios", default="0.1,0.3,0.5")
    ap.add_argument(
        "--modes",
        default="overbid,underbid,epsilon_inflate,epsilon_deflate,underbid_epsilon_deflate,underbid_epsilon_inflate",
    )
    ap.add_argument("--magnitudes", default="mild,moderate,extreme")
    ap.add_argument("--winner-threshold", type=float, default=0.1)
    ap.add_argument("--expected", action="store_true", default=True)
    ap.add_argument("--instantiate", dest="expected", action="store_false")
    ap.add_argument("--mfg-regretnet-ckpt", default="")
    ap.add_argument("--regretnet-ckpt", default="")
    ap.add_argument("--include-mfg-pricing", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.num_profiles = min(args.num_profiles, 50)
        args.seeds = "42"
        args.ratios = "0.1"
        args.modes = "overbid"
        args.magnitudes = "extreme"

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    magnitudes = [x.strip() for x in args.magnitudes.split(",") if x.strip()]
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
    for mech in mechs:
        for seed in seeds:
            for ratio in ratios:
                for mode in modes:
                    for magnitude in magnitudes:
                        print(
                            "[collusion] mech={} seed={} ratio={} mode={} magnitude={}".format(
                                mech[0], seed, ratio, mode, magnitude
                            ),
                            flush=True,
                        )
                        rows.append(run(args, mech, seed, ratio, mode, magnitude))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": vars(args),
                "notes": [
                    "Smoke mode only covers one mechanism grid point and is not hypothesis evidence.",
                    "PAC/VCG/CSRA are reverse auctions; negative revenue loss under overbid is interpretable.",
                ],
                "results": rows,
            },
            f,
            indent=2,
        )
    print("Wrote", args.out)


if __name__ == "__main__":
    main()

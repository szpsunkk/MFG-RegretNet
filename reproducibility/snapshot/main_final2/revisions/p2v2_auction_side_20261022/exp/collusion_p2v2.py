"""Coordinated attacks on MFG-RegretNet (P2-v2): review_collusion_b1.py with this mechanism.

Unchanged from the original (batch-3 configuration): B = 50, 1,000 profiles per seed (generate_privacy_paper_bids),
seeds 42-44, collusion ratios 0.1/0.3/0.5, six attack modes, three magnitudes, attackers drawn with
RandomState(seed + 999), the same report transformations, winner threshold 0.1, coalition gain
G_C = sum_{i in C} [u_i(attacked) - u_i(clean)] with u_i = p_i - v_i e_i under the true valuations.
Added: kappa in {3, 30}; the mechanism with and without the dividend (p + h vs p).
Usage: python collusion_p2v2.py --family mfg_p2v2 --out result.json [--seeds 42,43,44]
"""
import argparse
import time

import numpy as np
import torch

import common_p2v2 as C
import p2v2_mech as M

BF_TOLERANCE = 1e-3
MODES = ["overbid", "underbid", "epsilon_inflate", "epsilon_deflate", "underbid_epsilon_deflate",
         "underbid_epsilon_inflate"]
MAGNITUDES = ["mild", "moderate", "extreme"]
RATIOS = [0.1, 0.3, 0.5]


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


def attack_reports(reports, ratio, mode, magnitude, seed):
    """review_collusion_b1._attack_reports, unchanged."""
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
        raise ValueError(mode)
    return out, attackers.tolist()


def evaluate(model, reports, true_reports, budget, kappa):
    P = len(reports)
    b = torch.full((P, 1), budget, dtype=torch.float64)
    k = torch.full((P, 1), kappa, dtype=torch.float64)
    e, p, h = M.mechanism(model, reports, b, k)
    v_true = true_reports[..., 0]
    out = {}
    for variant, pay in (("with_dividend", p + h), ("without_dividend", p)):
        rev = pay.sum(1)
        out[variant] = {"revenue_mean": float(rev.mean()), "welfare_mean": float((pay - v_true * e).sum(1).mean()),
                        "utility": (pay - v_true * e).numpy(), "winners": (e > 0.1).numpy(),
                        "bf_rate": float((rev <= budget + BF_TOLERANCE).double().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=M.FAMILIES, default="mfg_p2v2")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--kappas", default="3,30")
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--num-profiles", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = M.load(a.family)
    rows = []
    began = time.time()
    for kappa in C.floats(a.kappas):
        for seed in (int(s) for s in a.seeds.split(",")):
            reports = C.generate_privacy_paper_bids(C.N_AGENTS, a.num_profiles, seed).double()
            clean = evaluate(model, reports, reports, a.budget, kappa)
            for ratio in RATIOS:
                for mode in MODES:
                    for magnitude in MAGNITUDES:
                        attacked, attackers = attack_reports(reports, ratio, mode, magnitude, seed + 999)
                        robust = evaluate(model, attacked, reports, a.budget, kappa)
                        ix = np.array(attackers, dtype=int)
                        for variant in ("with_dividend", "without_dividend"):
                            cl, rb = clean[variant], robust[variant]
                            gain = (rb["utility"][:, ix] - cl["utility"][:, ix]).sum(1)
                            truth = cl["utility"][:, ix].sum(1)
                            rows.append({
                                "mechanism": "MFG-RegretNet (P2-v2)", "family": a.family, "variant": variant,
                                "kappa": kappa, "seed": seed, "attack_mode": mode, "attack_magnitude": magnitude,
                                "collusion_ratio": ratio, "num_attackers": len(attackers),
                                "revenue_clean": cl["revenue_mean"], "revenue_attack": rb["revenue_mean"],
                                "welfare_clean": cl["welfare_mean"], "welfare_attack": rb["welfare_mean"],
                                "winner_manipulation_rate": float(np.mean(np.any(cl["winners"] != rb["winners"], 1))),
                                "coalition_gain_mean": float(gain.mean()),
                                "coalition_gain_per_attacker": float(gain.mean() / max(1, len(attackers))),
                                "coalition_gain_positive_share": float(np.mean(gain > 1e-6)),
                                "coalition_truthful_utility_mean": float(truth.mean()),
                                "attacker_win_rate_clean": float(cl["winners"][:, ix].mean()),
                                "attacker_win_rate_attack": float(rb["winners"][:, ix].mean()),
                                "bf_clean": cl["bf_rate"], "bf_attack": rb["bf_rate"]})
            C.atomic_json(a.out, {"meta": {**vars(a), "host": C.host_info(), "provenance": M.provenance()},
                                  "results": rows})
            print({"kappa": kappa, "seed": seed, "rows": len(rows), "seconds": round(time.time() - began, 1)},
                  flush=True)
    C.atomic_json(a.out, {"meta": {**vars(a), "seconds": time.time() - began, "host": C.host_info(),
                                   "provenance": M.provenance()}, "results": rows})


if __name__ == "__main__":
    main()

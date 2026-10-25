"""Long-horizon privacy accounting for MFG-RegretNet (P2-v2): review_privacy_accountant_b1.py with this mechanism.

Unchanged from the original (batch-3 configuration): 200 rounds, a fresh profile each round
(generate_privacy_paper_bids, seeds 42-44), B = 50, caps 10/30/60, RDP orders 2..64, delta = 1e-5, Gaussian RDP
(curve every round) and Laplace basic/advanced composition (curve every 10 rounds), clip-when-exhausted and the
pre-release check at the largest cap 60. The accounting functions below are copied from the original.
One difference, stated in the output: a client excluded by the check withdraws from the round, i.e. reports cap 0,
and the auction is re-run (as the original does for the analytical mechanisms on the active subprofile). The
original zeroed the release of fixed-N neural checkpoints instead, because they could not run on fewer clients.
Usage: python accountant_p2v2.py --family mfg_p2v2 --noise gaussian --out result.json [--kappas 3,30]
"""
import argparse
import math
import time

import numpy as np
import torch

import common_p2v2 as C
import p2v2_mech as M


def rdp_gaussian_rho(alpha, eps_ldp, delta):
    eps = np.maximum(eps_ldp.astype(np.float64), 1e-12)
    coef = 2.0 * math.log(1.25 / max(min(delta, 1.24), 1e-12))
    sigma2 = coef / (eps ** 2)
    return alpha / (2.0 * sigma2)


def rdp_to_eps_min(rho_by_alpha, alphas, delta):
    eps = rho_by_alpha + math.log(1.0 / delta) / (alphas.reshape(-1, 1) - 1.0)
    best_idx = np.argmin(eps, axis=0)
    return eps[best_idx, np.arange(eps.shape[1])], alphas[best_idx]


def laplace_compose_eps(eps_sum, eps_sq_sum, adv_term_sum, delta):
    if delta <= 0:
        return eps_sum.copy()
    adv = adv_term_sum + np.sqrt(2.0 * math.log(1.0 / delta) * eps_sq_sum)
    return np.minimum(eps_sum, adv)


def release(model, rep, budget, kappa, active):
    r = rep.clone()
    r[0, ~torch.as_tensor(active), 1] = 0.0          # withdrawn clients report cap 0
    b = torch.full((1, 1), budget, dtype=torch.float64)
    k = torch.full((1, 1), kappa, dtype=torch.float64)
    with torch.no_grad():
        e = model.allocate(r, b, k)
    return np.where(active, e[0].numpy(), 0.0)


def run(model, args, kappa, seed):
    reports = C.generate_privacy_paper_bids(args.n_agents, args.rounds, seed).double()
    n = args.n_agents
    alphas = np.array(args.alpha_orders, dtype=np.float64)
    rho = np.zeros((len(alphas), n))
    eps_sum, eps_sq_sum, adv_term_sum = np.zeros(n), np.zeros(n), np.zeros(n)
    exhausted = {str(cap): [None] * n for cap in args.total_budgets}
    curves, excluded = [], 0
    active = np.ones(n, dtype=bool)
    clip_cap = max(args.total_budgets)
    for t in range(args.rounds):
        rep = reports[t:t + 1]
        if args.noise == "gaussian":
            eps_total, _ = rdp_to_eps_min(rho, alphas, args.delta)
        else:
            eps_total = laplace_compose_eps(eps_sum, eps_sq_sum, adv_term_sum, args.delta)
        active = eps_total < clip_cap
        eps_out = release(model, rep, args.budget, kappa, active)
        for _ in range(n + 1):
            if args.noise == "gaussian":
                rho_h = rho + np.stack([rdp_gaussian_rho(al, eps_out, args.delta) if np.any(eps_out > 0)
                                        else np.zeros(n) for al in alphas])
                rho_h = np.where(eps_out[None, :] > 0, rho_h, rho)
                hyp, _ = rdp_to_eps_min(rho_h, alphas, args.delta)
            else:
                hyp = laplace_compose_eps(eps_sum + eps_out, eps_sq_sum + eps_out ** 2,
                                          adv_term_sum + eps_out * np.expm1(np.minimum(eps_out, 50.0)), args.delta)
            viol = (eps_out > 0) & (hyp > clip_cap + 1e-9)
            if not np.any(viol):
                break
            excluded += int(viol.sum())
            active = active & ~viol
            eps_out = release(model, rep, args.budget, kappa, active)
        if args.noise == "gaussian":
            for ai, alpha in enumerate(alphas):
                rho[ai] += rdp_gaussian_rho(alpha, eps_out, args.delta)
            eps_total, best_alpha = rdp_to_eps_min(rho, alphas, args.delta)
        else:
            eps_sum += eps_out
            eps_sq_sum += eps_out ** 2
            adv_term_sum += eps_out * np.expm1(np.minimum(eps_out, 50.0))
            eps_total = laplace_compose_eps(eps_sum, eps_sq_sum, adv_term_sum, args.delta)
            best_alpha = np.full(n, np.nan)
        for cap in args.total_budgets:
            for i, val in enumerate(eps_total):
                if exhausted[str(cap)][i] is None and val >= cap:
                    exhausted[str(cap)][i] = t + 1
        if t == 0 or (t + 1) % args.rnd_step == 0 or t == args.rounds - 1:
            curves.append({"round": t + 1, "mean_eps_total": float(np.mean(eps_total)),
                           "p90_eps_total": float(np.percentile(eps_total, 90)),
                           "max_eps_total": float(np.max(eps_total)),
                           "mean_eps_out": float(np.mean(eps_out[eps_out > 0])) if np.any(eps_out > 0) else 0.0,
                           "active_clients": int(np.sum(active)),
                           "best_alpha_median": float(np.nanmedian(best_alpha)) if args.noise == "gaussian" else None,
                           "exhausted_count_by_cap": {str(c): int(np.sum(eps_total >= c)) for c in args.total_budgets}})
    first10 = [r for r in exhausted["10.0"] if r is not None]
    return {"mechanism": "MFG-RegretNet (P2-v2)", "family": model.family, "kappa": kappa, "seed": seed,
            "curve": curves, "exhaustion_round_by_cap": exhausted, "excluded_by_pre_release_check": excluded,
            "first_client_reaches_10": min(first10) if first10 else None,
            "masking_note": "excluded clients withdraw (cap 0) and the auction is re-run"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=M.FAMILIES, default="mfg_p2v2")
    ap.add_argument("--noise", choices=("gaussian", "laplace"), required=True)
    ap.add_argument("--kappas", default="3,30")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--total-budgets", default="10,30,60")
    ap.add_argument("--alpha-orders", default="2,4,8,16,32,64")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.rnd_step = 1 if a.noise == "gaussian" else 10
    a.total_budgets = C.floats(a.total_budgets)
    a.alpha_orders = C.floats(a.alpha_orders)
    model = M.load(a.family)
    began = time.time()
    rows = [run(model, a, kappa, seed) for kappa in C.floats(a.kappas) for seed in (int(s) for s in a.seeds.split(","))]
    C.atomic_json(a.out, {"meta": {**vars(a), "seconds": time.time() - began, "host": C.host_info(),
                                   "provenance": M.provenance()}, "results": rows})
    print({"noise": a.noise, "rows": len(rows), "seconds": round(time.time() - began, 1)}, flush=True)


if __name__ == "__main__":
    main()

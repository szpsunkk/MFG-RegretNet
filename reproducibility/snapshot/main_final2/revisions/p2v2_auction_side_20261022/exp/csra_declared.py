"""CSRA (one-shot CSRA-QMS) with two release rules, for Supplementary Table csra-declared.

The original code (main_final2/code/final_20e327e/baselines/csra.py) prices each winner against its declared budget
(quality q_i = eps_i/sum eps * d_i/sum d, bid b_i = v_i eps_i, payment r_i = (b_{k+1}/q_{k+1}) q_i) but releases
1/(N-k) to every winner, as PAC and VCG do ("adapted"). This file repeats that rule in vectorized form and checks it
against the original function, then evaluates it and the variant whose winners release their declared budgets
("declared") with the RQ1 metrics of the analytical baselines (eval_rq1.baseline_regret): grid 21 x 21 x 3 over
(v', eps' fraction, d' fraction), truthful utility under the true valuation, share of offered contracts with
negative utility, budget used, on the paper's 10,000 RQ1 profiles.
Also records the RQ3 decomposition (truthful reports) at B in {25, 50, 75, 100} for both rules.
Usage: python csra_declared.py --out result.json [--budgets 25,50] [--rq3-budgets 25,50,75,100] [--check-only]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common_p2v2 as C

ROOT = Path(__file__).resolve().parents[4]


def csra(reports, budget, release="adapted"):
    """reports (P, N, 3) = (v, eps, d), budget (P,). Returns release e (P, N) and payment p (P, N)."""
    v, eps, d = reports[..., 0], reports[..., 1], reports[..., 2]
    P, N = v.shape
    sum_eps = eps.sum(-1, keepdim=True)
    sum_d = d.sum(-1, keepdim=True)
    ok = (sum_eps > 0) & (sum_d > 0)
    q = torch.where(ok, (eps / sum_eps.clamp_min(1e-300)) * (d / sum_d.clamp_min(1e-300)), torch.zeros_like(eps))
    b = v * eps
    ratio = torch.where(b > 0, q / b.clamp_min(1e-300), torch.zeros_like(q))
    order = torch.argsort(-ratio, dim=-1, stable=True)
    qs = torch.gather(q, -1, order)
    bs = torch.gather(b, -1, order)
    prefix = torch.cumsum(qs, -1) - qs                      # sum of q over the first k_cand entries
    scale = torch.where(qs > 0, bs / qs.clamp_min(1e-300), torch.zeros_like(qs))
    kk = torch.arange(N, device=v.device).expand(P, N)
    cond = (kk >= 1) & (qs > 0) & (prefix > 0) & (scale * prefix <= budget[:, None])
    k = torch.where(cond, kk, torch.zeros_like(kk)).max(-1).values      # largest admissible k_cand (0: none)
    k = torch.where(ok[:, 0], k, torch.zeros_like(k))
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, kk)
    winner = rank < k[:, None]
    thr = torch.gather(scale, -1, k.clamp_min(0)[:, None])[:, 0]           # b_k / q_k at the first loser
    pay = torch.where(winner, thr[:, None] * q, torch.zeros_like(q))
    if release == "adapted":
        rel = torch.where(winner, (1.0 / (N - k).clamp_min(1).to(v.dtype))[:, None].expand(P, N), torch.zeros_like(q))
    else:
        rel = torch.where(winner, eps, torch.zeros_like(q))
    return rel, pay


def check_against_original(rng, n=400):
    sys.path.insert(0, str(ROOT / "main_final2/code/final_20e327e"))
    from baselines.csra import csra_qms_batch  # original implementation
    worst = 0.0
    for B in (5.0, 25.0, 50.0, 100.0):
        r = C.generate_privacy_paper_bids(10, n, int(rng.integers(1, 10**6))).double()
        bud = torch.full((n, 1), B, dtype=torch.float64)
        # numpy inputs keep the original in float64 (torch inputs return float32 tensors)
        e0, p0 = csra_qms_batch(r.numpy(), bud.numpy())
        e1, p1 = csra(r, bud[:, 0], "adapted")
        worst = max(worst, float(np.abs(e0 - e1.numpy()).max()), float(np.abs(p0 - p1.numpy()).max()))
    return worst


def existing_paper_rows(B):
    """CSRA row of the manuscript's RQ1 baselines (float32 run of the original function), if present."""
    f = ROOT / f"main_final2/revisions/method_improvement_20261015/eval/rq1test_base2_B{int(B)}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())["results"]["CSRA"]
    return {"regret_mean_x1e3": 1e3 * r["regret_mean"], "regret_max": r["regret_max"],
            "ir_violation_rate": r["ir_violation_rate"], "budget_used": r["budget_used"],
            "released_eps_sum": r["released_eps_sum"], "file": str(f.relative_to(ROOT))}


def rq1_metrics(reports, B, release, chunk=40):
    """eval_rq1.baseline_regret for one rule: grid 21 x 21 x 3, utilities under the true valuation."""
    vg = torch.linspace(0, 1, 21, dtype=torch.float64)
    cand = torch.cartesian_prod(vg, torch.linspace(0, 1, 21, dtype=torch.float64),
                                torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
    G = len(cand)
    P, N, _ = reports.shape
    lb, u0s, pays, rels = [], [], [], []
    for s in range(0, P, chunk):
        R = reports[s:s + chunk]
        Pc = len(R)
        bud = torch.full((Pc,), B, dtype=torch.float64)
        e0, p0 = csra(R, bud, release)
        v = R[..., 0]
        u0 = p0 - v * e0
        best = u0.clone()
        for i in range(N):
            Rx = R[:, None].expand(Pc, G, N, 3).clone()
            Rx[:, :, i, 0] = cand[None, :, 0]
            Rx[:, :, i, 1] = cand[None, :, 1] * R[:, None, i, 1]
            Rx[:, :, i, 2] = 1 + cand[None, :, 2] * (R[:, None, i, 2] - 1)
            e, p = csra(Rx.reshape(Pc * G, N, 3), bud.repeat_interleave(G), release)
            ui = p.reshape(Pc, G, N)[:, :, i] - v[:, i, None] * e.reshape(Pc, G, N)[:, :, i]
            best[:, i] = torch.maximum(best[:, i], ui.max(1).values)
        lb.append((best - u0).clamp_min(0)); u0s.append(u0); pays.append(p0); rels.append(e0)
    lb, u0, pay, rel = torch.cat(lb), torch.cat(u0s), torch.cat(pays), torch.cat(rels)
    return {"regret_mean_x1e3": float(1e3 * lb.mean()), "regret_max": float(lb.max()),
            "ir_violation_rate": float((u0 < -1e-6).double().mean()),
            "budget_used": float(pay.sum(1).mean() / B), "released_eps_sum": float(rel.sum(1).mean()),
            "profiles": int(P)}


def rq3_metrics(reports, B, release, chunk=2000):
    """eval_rq3 metrics under truthful reports: budget used, revenue, welfare sum_i (p_i - v_i e_i), released eps,
    privacy cost sum_i v_i e_i, IR violation rate, share of markets without a purchase."""
    rows = []
    for s in range(0, len(reports), chunk):
        R = reports[s:s + chunk]
        e, p = csra(R, torch.full((len(R),), B, dtype=torch.float64), release)
        rows.append((e, p, R[..., 0]))
    e = torch.cat([r[0] for r in rows]); p = torch.cat([r[1] for r in rows]); v = torch.cat([r[2] for r in rows])
    u = p - v * e
    return {"budget_used": float(p.sum(1).mean() / B), "revenue": float(p.sum(1).mean()),
            "welfare": float(u.sum(1).mean()), "privacy_cost": float((v * e).sum(1).mean()),
            "released_eps_sum": float(e.sum(1).mean()), "ir_violation_rate": float((u < -1e-6).double().mean()),
            "no_purchase_share": float((e.sum(1) <= 0).double().mean()),
            "budget_violation_max": float((p.sum(1) - B).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--rq3-budgets", default="25,50,75,100")
    ap.add_argument("--per-seed", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    worst = check_against_original(np.random.default_rng(1))
    print({"max |difference| to the original csra_qms_batch": worst}, flush=True)
    if worst > 1e-9:
        raise SystemExit("vectorized CSRA differs from the original")
    if a.check_only:
        return
    reports = C.test_profiles(a.per_seed)
    out = {"profiles": len(reports), "check_vs_original": worst, "host": C.host_info(), "results": {}}
    began = time.time()
    for B in C.floats(a.budgets):
        for release in ("adapted", "declared"):
            out["results"][f"B{B:g}_{release}"] = rq1_metrics(reports, B, release)
            if release == "adapted":
                out["results"][f"B{B:g}_manuscript_row"] = existing_paper_rows(B)
            print(B, release, out["results"][f"B{B:g}_{release}"], round(time.time() - began, 1), flush=True)
            C.atomic_json(a.out, out)
    out["rq3"] = {f"B{B:g}_{release}": rq3_metrics(reports, B, release)
                  for B in C.floats(a.rq3_budgets) for release in ("adapted", "declared")}
    out["seconds"] = time.time() - began
    C.atomic_json(a.out, out)


if __name__ == "__main__":
    main()

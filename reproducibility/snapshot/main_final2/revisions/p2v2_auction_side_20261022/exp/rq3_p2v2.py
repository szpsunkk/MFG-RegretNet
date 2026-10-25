"""RQ3 budget sweep for MFG-RegretNet (P2-v2): truthful reports, the paper's 10,000 profiles (eval_rq3.py metrics).

For B in {25, 50, 75, 100}, kappa in {3, 30}, each family, with and without the dividend:
  budget used sum(p)/B, revenue sum(p), welfare SW = sum_i (p_i - v_i e_i), released eps sum(e), IR violation
  rate (u < -1e-6), budget-feasibility rate (sum p <= B + 1e-9) and the largest excess.
Also records, for the dividend (data sizes verified; Sbar_i = purchase with price 0, budget 5, own size): how often
Sbar_i is below the actual purchase (must be never), and, for reference, how often a size report of 500 would give a
purchase below the actual one and how often sum(p + h) would then exceed B.
Usage: python rq3_p2v2.py --family F --out result.json [--per-seed 2000] [--threads 4]
"""
import argparse
import time

import torch

import common_p2v2 as C
import p2v2_mech as M

BUDGETS = [25.0, 50.0, 75.0, 100.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=M.FAMILIES, required=True)
    ap.add_argument("--kappas", default="3,30")
    ap.add_argument("--per-seed", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    reports = C.test_profiles(a.per_seed)
    model = M.load(a.family)
    P, N, _ = reports.shape
    v = reports[..., 0]
    res = {"family": a.family, "profiles": P, "budgets": BUDGETS, "kappas": C.floats(a.kappas),
           "host": C.host_info(), "provenance": M.provenance(), "results": {}}
    began = time.time()
    for kap in C.floats(a.kappas):
        for B in BUDGETS:
            es, ps, hs, sb, sbf, stot = [], [], [], [], [], []
            with torch.no_grad():
                for lo in range(0, P, a.chunk):
                    r = reports[lo:lo + a.chunk]
                    b = torch.full((len(r), 1), B, dtype=torch.float64)
                    k = torch.full((len(r), 1), kap, dtype=torch.float64)
                    e, p = model.mechanism(r, b, k)
                    sizes = r[..., 2].pow(model.gamma)
                    share = sizes / sizes.sum(-1, keepdim=True)
                    h = torch.zeros_like(e)
                    sbar = torch.zeros_like(e)
                    sfix = torch.zeros_like(e)
                    for c in range(N):
                        ctx = M.ClientContext(model, r, c)
                        sbar[:, c] = ctx.sbar(b, k)
                        sfix[:, c] = ctx.sbar_fixed(b, k)
                        h[:, c] = share[:, c] * (B - M.VBAR * sbar[:, c])
                    caps_sel = torch.where(M.selection(share, model.threshold(k, b, "budget")), r[..., 1],
                                           torch.zeros_like(e)).sum(-1, keepdim=True)
                    s_actual = torch.minimum(torch.minimum(model.total(r, b, k), b), caps_sel)
                    es.append(e), ps.append(p), hs.append(h), sb.append(sbar), sbf.append(sfix), stot.append(s_actual)
            e, p, h = torch.cat(es), torch.cat(ps), torch.cat(hs)
            sbar, sfix, s_act = torch.cat(sb), torch.cat(sbf), torch.cat(stot)
            sizes = reports[..., 2]
            share = sizes / sizes.sum(-1, keepdim=True)
            h_fixed = share * (B - M.VBAR * sfix)
            row = {}
            for name, pay in (("with_dividend", p + h), ("without_dividend", p)):
                u = pay - v * e
                row[name] = {"budget_used": float(pay.sum(1).mean() / B), "revenue": float(pay.sum(1).mean()),
                             "welfare": float(u.sum(1).mean()), "released_eps_sum": float(e.sum(1).mean()),
                             "ir_violation_rate": float((u < -1e-6).double().mean()),
                             "bf_rate": float((pay.sum(1) <= B + 1e-9).double().mean()),
                             "budget_excess_max": float((pay.sum(1) - B).max())}
            row["dividend_reference"] = {
                "sbar_below_purchase_share": float((sbar < s_act - 1e-12).double().mean()),
                "fixed_reference_below_purchase_share": float((sfix < s_act - 1e-12).double().mean()),
                "fixed_reference_budget_excess_share": float(((p + h_fixed).sum(1) > B + 1e-9).double().mean()),
                "fixed_reference_budget_excess_max": float(((p + h_fixed).sum(1) - B).max()),
                "sbar_minus_fixed_mean": float((sbar - sfix).mean()),
                "purchase_mean": float(s_act.mean()),
                "selected_clients_mean": float(M.selection(share, model.threshold(
                    torch.full((P, 1), kap, dtype=torch.float64), torch.full((P, 1), B, dtype=torch.float64),
                    "budget")).double().sum(-1).mean()),
            }
            res["results"][f"kappa{kap:g}_B{B:g}"] = row
            print(a.family, kap, B, {k: round(v_, 4) for k, v_ in row["with_dividend"].items()}, flush=True)
            C.atomic_json(a.out, res)
    res["seconds"] = time.time() - began
    C.atomic_json(a.out, res)


if __name__ == "__main__":
    main()

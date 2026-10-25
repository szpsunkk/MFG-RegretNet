"""Inputs of the RQ3 welfare decomposition that rq3_p2v2.py does not record (Claude, 2026-10-24).

On the paper's 10,000 RQ3 profiles (truthful reports) and B in {25, 50, 75, 100}:
  * MFG-RegretNet (kappa 3 and 30): share of markets without a purchase (allocation only, Codex's Selector);
  * analytical baselines (the implementations of the experiments: PAC, VCG, CSRA with release 1/(N-k) and with
    declared release, MFG-Pricing): total payment, privacy cost sum_i v_i e_i under the true valuations, client
    surplus sum_i (p_i - v_i e_i), released privacy, IR violation rate and share of markets without a purchase.
The learned baselines need their trained checkpoints and are not recomputed here.
Usage: python rq3_decomposition_extra.py --out rq3_decomposition_extra.json
"""
import argparse
import sys
import time
from pathlib import Path

import torch

import common_p2v2 as C
import p2v2_mech as M
from csra_declared import csra

ROOT = Path(__file__).resolve().parents[4]
BUDGETS = (25.0, 50.0, 75.0, 100.0)


def summary(e, p, v, B):
    u = p - v * e
    return {"revenue": float(p.sum(1).mean()), "privacy_cost": float((v * e).sum(1).mean()),
            "welfare": float(u.sum(1).mean()), "released_eps_sum": float(e.sum(1).mean()),
            "budget_used": float(p.sum(1).mean() / B), "ir_violation_rate": float((u < -1e-6).double().mean()),
            "no_purchase_share": float((e.sum(1) <= 0).double().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-seed", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(ROOT / "main_final2/code/final_20e327e"))
    from baselines.pac import pac_batch
    from baselines.vcg import vcg_procurement_batch
    from baselines.mfg_pricing import mfg_pricing_batch
    reports = C.test_profiles(a.per_seed)
    v = reports[..., 0]
    P = len(reports)
    out = {"profiles": P, "host": C.host_info(), "ours_no_purchase_share": {}, "baselines": {}}
    began = time.time()
    model = M.load("mfg_p2v2")
    with torch.no_grad():
        for kap in (3.0, 30.0):
            for B in BUDGETS:
                zero = 0
                for lo in range(0, P, 1000):
                    r = reports[lo:lo + 1000]
                    e = model.allocate(r, torch.full((len(r), 1), B, dtype=torch.float64),
                                       torch.full((len(r), 1), kap, dtype=torch.float64))
                    zero += int((e.sum(1) <= 0).sum())
                out["ours_no_purchase_share"][f"kappa{kap:g}_B{B:g}"] = zero / P
    r32 = reports.float()
    for B in BUDGETS:
        b32 = torch.full((P, 1), B)
        rows = {}
        for name, fn in (("PAC", pac_batch), ("VCG", vcg_procurement_batch), ("MFG-Pricing", mfg_pricing_batch)):
            e, p = fn(r32, b32)
            rows[name] = summary(torch.as_tensor(e).double(), torch.as_tensor(p).double(), v, B)
        for rule, label in (("adapted", "CSRA"), ("declared", "CSRA-declared")):
            e, p = csra(reports, torch.full((P,), B, dtype=torch.float64), rule)
            rows[label] = summary(e, p, v, B)
        out["baselines"][f"B{B:g}"] = rows
        print(B, {k: round(x["welfare"], 2) for k, x in rows.items()}, flush=True)
    out["seconds"] = time.time() - began
    C.atomic_json(a.out, out)
    print("ours no-purchase shares:", out["ours_no_purchase_share"])


if __name__ == "__main__":
    main()

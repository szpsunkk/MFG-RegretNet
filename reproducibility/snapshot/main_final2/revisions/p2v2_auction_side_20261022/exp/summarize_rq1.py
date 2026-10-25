"""Merge rq1_p2v2.py shards into the RQ1 metrics (same definitions as eval_rq1.summarize).

Per family, scenario, setting (B, kappa) and mechanism variant (with / without the dividend):
  regret mean, std, p95, max (over profiles x clients), in utility units and x1e-3 as in the RQ1 table;
  the per-client bound vbar * eps_i / 64 of Corollary 2 (largest excess of the measured regret over the bound);
  IR violation rate (truthful utility < -1e-6) and mean violation; budget used sum(p + h) / B;
  largest budget excess max(sum(p + h) - B); released eps per round; share of clients with positive regret.
Usage: python summarize_rq1.py <shard_dir> [<shard_dir> ...] --out summary.json
"""
import argparse
import json
from pathlib import Path

import torch

import common_p2v2 as C


def load_dir(path):
    shards = sorted(Path(path).glob("shard_*_of_*.pt"))
    if not shards:
        raise SystemExit(f"no shards in {path}")
    blobs = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
    metas = [b["meta"] for b in blobs]
    K = int(shards[0].stem.split("_of_")[1])
    ranges = sorted(tuple(m["profile_range"]) for m in metas)
    total = metas[0]["profiles_total"]
    complete = len(shards) == K and ranges[0][0] == 0 and ranges[-1][1] == total and \
        all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    order = sorted(range(len(blobs)), key=lambda i: metas[i]["profile_range"][0])
    merged = {}
    for s in blobs[0]["results"]:
        merged[s] = {k: torch.cat([blobs[i]["results"][s][k] for i in order]) for k in blobs[0]["results"][s]}
    return metas[order[0]], merged, complete, ranges, sum(m["seconds"] for m in metas)


def summarize(r, eps, budget):
    out = {}
    for variant, h_on in (("with_dividend", True), ("without_dividend", False)):
        key = "div" if h_on else "nodiv"
        reg = r[f"regret_{key}"]
        ut = r[f"u_truth_{key}"]
        pay = r["p"] + (r["h"] if h_on else 0.)
        bound = 1.0 * eps / 64.0
        out[variant] = {
            "regret_mean": float(reg.mean()), "regret_mean_x1e3": float(1e3 * reg.mean()),
            "regret_std": float(reg.std()), "regret_p95": float(reg.flatten().quantile(0.95)),
            "regret_max": float(reg.max()), "positive_regret_share": float((reg > 1e-12).double().mean()),
            "regret_over_bound_max": float((reg - bound).max()),
            "bound_mean": float(bound.mean()), "bound_max_vbar_epsbar_over_n": 5.0 / 64.0,
            "ir_violation_rate": float((ut < -1e-6).double().mean()),
            "ir_violation_mean": float((-ut).clamp_min(0).mean()),
            "regret_size_only_mean": float(r[f"regret_size_{key}"].mean()),
            "regret_size_only_max": float(r[f"regret_size_{key}"].max()),
            "regret_size_only_positive_share": float((r[f"regret_size_{key}"] > 1e-12).double().mean()),
            "regret_verified_sizes_grid_mean": float(r[f"regret_verified_{key}"].mean()),
            "regret_verified_sizes_grid_max": float(r[f"regret_verified_{key}"].max()),
            "budget_used": float(pay.sum(1).mean() / budget),
            "budget_excess_max": float((pay.sum(1) - budget).max()),
            "released_eps_sum": float(r["e"].sum(1).mean()),
            "revenue": float(pay.sum(1).mean()),
            "welfare": float(ut.sum(1).mean()),
        }
        # requested by the external review: markets without a purchase, payment per unit of released privacy
        # (payment for the release only, without the dividend), and the share of contracts paid at the cap price
        rel = r["e"] > 0
        unit = r["p"][rel] / r["e"][rel]
        out[variant].update({
            "zero_purchase_share": float((r["e"].sum(1) <= 0).double().mean()),
            "pay_per_eps_mean": float(unit.mean()) if rel.any() else None,
            "pay_per_eps_min": float(unit.min()) if rel.any() else None,
            "pay_per_eps_at_cap_share": float((unit >= 0.999).double().mean()) if rel.any() else None,
            "released_contracts_per_market": float(rel.double().sum(1).mean()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summary = {"definitions": __doc__, "runs": []}
    for d in a.dirs:
        meta, merged, complete, ranges, seconds = load_dir(d)
        if meta["scenario"] == "test":
            reports = C.test_profiles()
        else:
            reports = C.hetero_profiles(meta["scenario"])
        lo, hi = ranges[0][0], ranges[-1][1]
        eps = reports[lo:hi, :, 1]
        n = merged[0]["e"].shape[0]
        eps = eps[:n]
        row = {"dir": str(d), "family": meta["family"], "scenario": meta["scenario"], "complete": complete,
               "profiles": n, "shard_ranges": ranges, "cpu_seconds": seconds, "grid": meta["grid"],
               "pga_steps": meta["pga_steps"], "provenance": meta["provenance"], "settings": []}
        for s, st in enumerate(meta["settings"]):
            row["settings"].append({**st, **summarize(merged[s], eps, st["budget_effective"])})
        summary["runs"].append(row)
        print(meta["family"], meta["scenario"], "complete" if complete else "INCOMPLETE", n, "profiles", flush=True)
    C.atomic_json(a.out, summary)


if __name__ == "__main__":
    main()

"""RQ1 regret search (and the heterogeneous-valuation variant) for MFG-RegretNet (P2-v2), one shard of profiles.

Same search as eval_rq1.py uses for the neural rows of the RQ1 table, for every client of every profile:
  * grid 21 x 21 x 5 over (v', eps', d') in [0, 1] x [0, eps_i] x [1, d_i] (regret_tools.candidate_grid);
    the truthful report competes;
  * 25 projected gradient-ascent steps (lr 0.01) from the best grid point, on the same box;
  * regret = max(u_grid, u_pga, u_truth) - u_truth, with u = p + h - v_true * e (true valuation).
The search runs twice per setting: for the paper's mechanism (with the dividend h) and without the dividend.

Data-size crossings (added 2026-10-22): with the selection threshold tau(kappa, B), the release is not monotone in
the own data size where another client's share crosses tau, and a 5-point size grid can miss these points. Every
client therefore also tries, with its true price and budget, the sizes just below and just above each crossing in
[1, d_i] (own share = tau, another client's share = tau, for the tau of every setting), plus the 5 grid sizes.
Recorded separately: regret_size_* (only these size deviations) and regret_verified_* (grid points with d' = d_i,
i.e. data sizes verified; no PGA). The headline regret_* keeps the full search including the crossing candidates.

Profiles: --scenario test = the paper's 10,000 RQ1/RQ3 profiles (seeds 1001-1005 x 2,000);
bimodal / realistic = the heterogeneous scenarios of hetero_eval.py (seeds 7001-7005 x 1,000). For realistic the
valuation is in units of its 99th percentile, so the budget is divided by the same scale (as hetero_eval.py).

Output: <out>/shard_<k>_of_<K>.pt (per profile and client: regrets, truthful utilities, e, p, h, best deviations)
and shard_<k>_of_<K>.json (settings, timing, provenance). summarize_rq1.py merges the shards.
"""
import argparse
import time
from pathlib import Path

import torch

import common_p2v2 as C
import p2v2_mech as M


def size_candidates(R, c, taus, gd=5):
    """(P, K, 3) reports (v_c, eps_c, d') with d' at both sides of every threshold crossing and on the size grid."""
    P = len(R)
    d = R[..., 2]
    d_c = d[:, c]
    rest = torch.cat((d[:, :c], d[:, c + 1:]), 1)
    D = rest.sum(1, keepdim=True)
    pts = []
    for t in taus:
        pts.append(t * D / (1 - t))
        pts.append(rest / t - D)
    pts = torch.cat(pts, 1)
    pts = torch.cat((pts * (1 - 1e-9), pts * (1 + 1e-9)), 1)
    grid = 1.0 + torch.linspace(0.0, 1.0, gd, dtype=R.dtype)[None] * (d_c[:, None] - 1.0)
    dd = torch.cat((pts, grid), 1)
    dd = torch.minimum(dd.clamp_min(1.0), d_c[:, None])
    K = dd.shape[1]
    return torch.stack((R[:, c, 0:1].expand(P, K), R[:, c, 1:2].expand(P, K), dd), -1)


def utilities(e, p, h, v_true):
    u_nodiv = p - v_true[:, None] * e
    return u_nodiv + h, u_nodiv


def project(mis, truth):
    with torch.no_grad():
        mis[..., 0].clamp_(0.0, 1.0)
        mis[..., 1] = torch.minimum(mis[..., 1].clamp_min(0.0), truth[..., 1])
        mis[..., 2] = torch.minimum(mis[..., 2].clamp_min(1.0), truth[..., 2])


def pga(ctx, budget, kappa, start, truth, v_true, with_dividend, steps, lr):
    mis = start.clone().requires_grad_(True)
    for _ in range(steps):
        e, p, h = ctx.outcomes(budget, kappa, mis, dividend=with_dividend)
        u = p - v_true[:, None] * e + (h if with_dividend else 0.)
        (g,) = torch.autograd.grad(u.sum(), mis)
        with torch.no_grad():
            mis += lr * torch.nan_to_num(g)
        project(mis, truth)
    with torch.no_grad():
        e, p, h = ctx.outcomes(budget, kappa, mis.detach(), dividend=with_dividend)
        u = p - v_true[:, None] * e + (h if with_dividend else 0.)
    return u[:, 0], mis.detach()[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=M.FAMILIES, required=True)
    ap.add_argument("--scenario", choices=("test", "bimodal", "realistic"), default="test")
    ap.add_argument("--budgets", default="25,50", help="in table units")
    ap.add_argument("--kappas", default="3,30")
    ap.add_argument("--per-seed", type=int, default=None, help="profiles per seed (2000 test, 1000 hetero)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--grid", default="21,21,5")
    ap.add_argument("--pga-steps", type=int, default=25)
    ap.add_argument("--pga-lr", type=float, default=0.01)
    ap.add_argument("--chunk-profiles", type=int, default=4)
    ap.add_argument("--chunk-cands", type=int, default=441)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: only the first profiles of the shard")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    gv, ge, gd = (int(x) for x in a.grid.split(","))
    if a.scenario == "test":
        reports = C.test_profiles(a.per_seed or 2000)
    else:
        reports = C.hetero_profiles(a.scenario, a.per_seed or 1000)
    scale = C.HETERO_SCALE if a.scenario == "realistic" else 1.0
    budgets_table, kappas = C.floats(a.budgets), C.floats(a.kappas)
    settings = [(B, k) for B in budgets_table for k in kappas]
    total = len(reports)
    lo = total * a.shard // a.shards
    hi = total * (a.shard + 1) // a.shards
    if a.limit is not None:
        hi = min(hi, lo + a.limit)
    reports = reports[lo:hi]
    P, N, _ = reports.shape
    model = M.load(a.family)
    taus = sorted({float(model.threshold(torch.tensor([[k]], dtype=torch.float64),
                                         torch.tensor([[B / scale]], dtype=torch.float64), "budget"))
                   for B, k in settings})
    taus = [t for t in taus if t > 0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"shard_{a.shard:03d}_of_{a.shards:03d}"
    meta = {"family": a.family, "scenario": a.scenario, "budgets_table": budgets_table, "budget_scale": scale,
            "kappas": kappas, "profile_range": [lo, hi], "profiles_total": total, "grid": [gv, ge, gd],
            "pga_steps": a.pga_steps, "pga_lr": a.pga_lr, "host": C.host_info(), "provenance": M.provenance(),
            "crossing_taus": taus}
    keys = ["regret_div", "regret_nodiv", "u_truth_div", "u_truth_nodiv", "e", "p", "h",
            "u_grid_div", "u_pga_div", "u_grid_nodiv", "u_pga_nodiv",
            "regret_size_div", "regret_size_nodiv", "regret_verified_div", "regret_verified_nodiv"]
    res = {s: {k: torch.zeros(P, N, dtype=torch.float64) for k in keys} for s in range(len(settings))}
    for s in res:
        res[s]["best_div"] = torch.zeros(P, N, 3, dtype=torch.float64)
        res[s]["best_nodiv"] = torch.zeros(P, N, 3, dtype=torch.float64)
    began = time.time()
    for p0 in range(0, P, a.chunk_profiles):
        R = reports[p0:p0 + a.chunk_profiles]
        Pc = len(R)
        sets = [(torch.full((Pc, 1), B / scale, dtype=torch.float64), torch.full((Pc, 1), k, dtype=torch.float64))
                for B, k in settings]
        for c in range(N):
            ctx = M.ClientContext(model, R, c)
            truth = R[:, c:c + 1].clone()
            v_true = R[:, c, 0]
            grid = M.candidate_grid(R, c, gv, ge, gd)
            outs_t = ctx.outcomes_multi(sets, truth)
            best = []
            for s, (e, p, h) in enumerate(outs_t):
                ud, un = utilities(e, p, h, v_true)
                best.append({"div": [ud[:, 0].clone(), truth[:, 0].clone()], "nodiv": [un[:, 0].clone(), truth[:, 0].clone()]})
                r = res[s]
                r["u_truth_div"][p0:p0 + Pc, c], r["u_truth_nodiv"][p0:p0 + Pc, c] = ud[:, 0], un[:, 0]
                r["e"][p0:p0 + Pc, c], r["p"][p0:p0 + Pc, c], r["h"][p0:p0 + Pc, c] = e[:, 0], p[:, 0], h[:, 0]
            for g0 in range(0, grid.shape[1], a.chunk_cands):
                cand = grid[:, g0:g0 + a.chunk_cands]
                for s, (e, p, h) in enumerate(ctx.outcomes_multi(sets, cand)):
                    ud, un = utilities(e, p, h, v_true)
                    for name, u in (("div", ud), ("nodiv", un)):
                        val, idx = u.max(1)
                        better = val > best[s][name][0]
                        best[s][name][0] = torch.where(better, val, best[s][name][0])
                        pick = cand[torch.arange(Pc), idx]
                        best[s][name][1] = torch.where(better[:, None], pick, best[s][name][1])
            # data-size deviations at the threshold crossings (true price and budget)
            sc = size_candidates(R, c, taus, gd)
            for s, (e, p, h) in enumerate(ctx.outcomes_multi(sets, sc)):
                ud, un = utilities(e, p, h, v_true)
                r = res[s]
                for name, u in (("div", ud), ("nodiv", un)):
                    val, idx = u.max(1)
                    u_t = r[f"u_truth_{name}"][p0:p0 + Pc, c]
                    r[f"regret_size_{name}"][p0:p0 + Pc, c] = torch.maximum(val, u_t) - u_t
                    better = val > best[s][name][0]
                    best[s][name][0] = torch.where(better, val, best[s][name][0])
                    best[s][name][1] = torch.where(better[:, None], sc[torch.arange(Pc), idx], best[s][name][1])
            # verified sizes: grid points with d' = d_i only (no PGA)
            ver = grid[:, (torch.arange(grid.shape[1]) % gd) == gd - 1]
            for s, (e, p, h) in enumerate(ctx.outcomes_multi(sets, ver)):
                ud, un = utilities(e, p, h, v_true)
                r = res[s]
                for name, u in (("div", ud), ("nodiv", un)):
                    u_t = r[f"u_truth_{name}"][p0:p0 + Pc, c]
                    r[f"regret_verified_{name}"][p0:p0 + Pc, c] = torch.maximum(u.max(1).values, u_t) - u_t
            for s, (b, k) in enumerate(sets):
                r = res[s]
                for name, with_div in (("div", True), ("nodiv", False)):
                    u_grid, start = best[s][name]
                    u_pga, end = pga(ctx, b, k, start[:, None], truth, v_true, with_div, a.pga_steps, a.pga_lr)
                    u_t = r[f"u_truth_{name}"][p0:p0 + Pc, c]
                    top = torch.maximum(torch.maximum(u_grid, u_pga), u_t)
                    r[f"regret_{name}"][p0:p0 + Pc, c] = top - u_t
                    r[f"u_grid_{name}"][p0:p0 + Pc, c] = u_grid
                    r[f"u_pga_{name}"][p0:p0 + Pc, c] = u_pga
                    r[f"best_{name}"][p0:p0 + Pc, c] = torch.where((u_pga > u_grid)[:, None], end, start)
        done = p0 + Pc
        elapsed = time.time() - began
        C.atomic_json(str(stem) + ".progress.json", {"profiles_done": done, "profiles": P, "seconds": elapsed,
                                                     "eta_seconds": elapsed / done * (P - done)})
    meta["seconds"] = time.time() - began
    meta["settings"] = [{"index": s, "budget_table": B, "budget_effective": B / scale, "kappa": k}
                        for s, (B, k) in enumerate(settings)]
    torch.save({"meta": meta, "results": res}, str(stem) + ".pt")
    C.atomic_json(str(stem) + ".json", meta)
    print({"shard": a.shard, "profiles": P, "seconds": round(meta["seconds"], 1)}, flush=True)


if __name__ == "__main__":
    main()

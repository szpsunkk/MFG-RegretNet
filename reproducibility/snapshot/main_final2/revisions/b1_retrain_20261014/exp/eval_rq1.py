"""Unified RQ1 evaluation for the retrained mechanisms (step 5b).

Every mechanism is evaluated on the same test profiles (B = 50, N = 10), with
the same utility u_i = p_i - v_i * eps_out_i (true v_i) and the same one-sided
report set v' in [0, 1], eps' in [0, eps_i], d' in [1, d_i].

Neural mechanisms (expected allocations):
  * regret_lb : grid 21 x 21 x 5 over (v', eps', d') plus 25 PGA steps (eta 0.01)
                from the best grid point; a lower bound on the exact regret;
  * regret_pga: 25 PGA steps (eta 0.1) from the truthful report, as in the
                original RQ1 evaluation, to show the gap;
  * regret_ub : MFG-RegretNet only, certified upper bound (cert_eval.certify).
Analytical baselines (PAC, VCG, CSRA, MFG-Pricing): grid 21 x 21 x 3.

Usage: python eval_rq1.py <code_dir> <runs_dir> <out_json> <profiles_per_seed> [kinds] [cert 0|1]
"""
import json
import math
import os
import sys
import time

CODE_DIR, RUNS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PPS = int(sys.argv[4])
KINDS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["mfg", "mfgalign", "regretnet", "dm", "PAC", "VCG", "CSRA", "MFG-Pricing"]
CERT = int(sys.argv[6]) if len(sys.argv) > 6 else 1
CERT_N = int(os.environ.get("CERT_PROFILES", "10000"))
CERT_TOL = float(os.environ.get("CERT_TOL", "0.02"))
# per-seed certified sample sizes, e.g. "42:10000,43:1000" (seeds not listed use CERT_PROFILES)
CERT_PLAN = {int(a): int(b) for a, b in (x.split(":") for x in os.environ.get("CERT_PLAN", "").split(",") if x)}
CKPT_EPOCH = int(os.environ.get("CKPT_EPOCH", "200"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import regret_tools as rt  # noqa: E402
import crown  # noqa: E402
from models_scaled import MFGRegretNetScaled, MFGRegretNetNoMF, RegretNetScaled  # noqa: E402
from regretnet import RegretNet  # noqa: E402
from datasets_fl_benchmark import generate_privacy_paper_bids  # noqa: E402
from utils import calc_agent_util, tiled_misreport_util, allocs_to_plosses  # noqa: E402
from exp_rq.guarantees_eval_baselines import _dispatch  # noqa: E402

dev = os.environ.get("EVAL_DEVICE", "cuda")
N, VBAR, EPS_BAR = 10, 1.0, 5.0
B = float(os.environ.get("EVAL_B", "50"))  # RQ1 budget (50 by default; 25 for the binding-budget check)
U_BAR = B + VBAR * EPS_BAR
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]
TRAIN_SEEDS = [42, 43, 44, 45, 46]
SUB = 100  # profiles per sub-batch
ANALYTIC = ("PAC", "VCG", "CSRA", "MFG-Pricing")


def test_set():
    reps = torch.cat([generate_privacy_paper_bids(N, 1, PPS, seed=s) for s in TEST_SEEDS]).to(dev)
    vt = torch.zeros(reps.shape[0], N, 2, device=dev)
    vt[:, :, 1] = 1.0
    return reps, B * torch.ones(reps.shape[0], 1, device=dev), vt


def load(kind, seed):
    blob = torch.load(os.path.join(RUNS, f"{kind}_seed{seed}_ep{CKPT_EPOCH}.pt"), map_location="cpu", weights_only=False)
    if kind.startswith("mfg"):
        cls = MFGRegretNetNoMF if kind.startswith("mfg_nob") else MFGRegretNetScaled
        m = cls(N, 1, hidden_layer_size=100, n_hidden_layers=3, activation="tanh", separate=True)
    else:
        cls = RegretNetScaled if kind.endswith("_s") else RegretNet
        m = cls(N, 1, activation="tanh", hidden_layer_size=100, n_hidden_layers=3, p_activation="softmax",
                a_activation="deterministic" if kind.startswith("dm") else "softmax", separate=True,
                smoothing=0.125, normalized_input=-1)
    m.load_state_dict(blob["state_dict"])
    m = m.to(dev).eval()
    if kind.startswith("dm"):
        m.deter_train = False  # one-hot allocation at test time, as in experiments.load_auc_model
    return m


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, x):
        return self.module(x)


def neural_regret(model, reps, bud, vt):
    w = Wrap(model)
    out = {k: [] for k in ("u_truth", "lb", "pga", "cost", "pay", "eps_out")}
    for s in range(0, reps.shape[0], SUB):
        R, Bd, V = reps[s:s + SUB], bud[s:s + SUB], vt[s:s + SUB]
        with torch.no_grad():
            al, pay = w((R, Bd))
            ut = calc_agent_util(R, al, pay, cost_from_plosses=True)
            pl = allocs_to_plosses(al, R[:, :, -2])
        init = rt.grid_init_misreports(w, R, Bd, V, True, gv=21, ge=21, gd=5, chunk=32)
        with torch.no_grad():
            u_grid = tiled_misreport_util(init, R, w, Bd, V, cost_from_plosses=True)
        mis = init.clone()
        rt.pga_box(w, R, mis, Bd, V, 25, 0.01, cost_from_plosses=True)
        with torch.no_grad():
            u_pga = tiled_misreport_util(mis.detach(), R, w, Bd, V, cost_from_plosses=True)
        mis2 = R.clone()
        rt.pga_box(w, R, mis2, Bd, V, 25, 0.1, cost_from_plosses=True)
        with torch.no_grad():
            u_orig = tiled_misreport_util(mis2.detach(), R, w, Bd, V, cost_from_plosses=True)
        out["u_truth"].append(ut)
        out["lb"].append(torch.maximum(torch.maximum(u_grid, u_pga), ut) - ut)
        out["pga"].append((u_orig - ut).clamp_min(0))
        out["cost"].append(R[:, :, 0] * pl)
        out["pay"].append(pay)
        out["eps_out"].append(pl)
    return {k: torch.cat(v) for k, v in out.items()}


def baseline_regret(name, reps, bud):
    fn = _dispatch(name)
    out = {k: [] for k in ("u_truth", "lb", "cost", "pay", "eps_out")}
    # search only the report fields each rule reads (PAC and VCG ignore eps and size)
    vg = torch.linspace(0, 1, 21, device=dev)
    one = torch.ones(1, device=dev)
    if name in ("PAC", "VCG"):
        cand = torch.cartesian_prod(vg, one, one)
    elif name == "MFG-Pricing":
        cand = torch.cartesian_prod(vg, torch.linspace(0, 1, 21, device=dev), one)
    else:
        cand = torch.cartesian_prod(vg, torch.linspace(0, 1, 21, device=dev), torch.tensor([0.0, 0.5, 1.0], device=dev))
    G = cand.shape[0]
    for s in range(0, reps.shape[0], SUB):
        R, Bd = reps[s:s + SUB], bud[s:s + SUB]
        P = R.shape[0]
        pl0, pay0 = fn(R, Bd)
        v = R[:, :, 0]
        u0 = pay0 - v * pl0
        best = u0.clone()
        for i in range(N):
            Rx = R.unsqueeze(1).repeat(1, G, 1, 1)
            Rx[:, :, i, 0] = cand[None, :, 0]
            Rx[:, :, i, 1] = cand[None, :, 1] * R[:, None, i, 1]
            Rx[:, :, i, 2] = 1 + cand[None, :, 2] * (R[:, None, i, 2] - 1)
            pl, pay = fn(Rx.view(P * G, N, 3), Bd.repeat_interleave(G, 0))
            ui = pay.view(P, G, N)[:, :, i] - v[:, i, None] * pl.view(P, G, N)[:, :, i]
            best[:, i] = torch.maximum(best[:, i], ui.max(1)[0])
        out["u_truth"].append(u0)
        out["lb"].append((best - u0).clamp_min(0))
        out["cost"].append(v * pl0)
        out["pay"].append(pay0)
        out["eps_out"].append(pl0)
    return {k: torch.cat(v) for k, v in out.items()}


def summarize(r):
    lb = r["lb"]
    cost = r["cost"].clamp(min=1e-10)
    s = {"regret_mean": float(lb.mean()), "regret_std": float(lb.std()),
         "regret_p95": float(lb.flatten().quantile(0.95)), "regret_max": float(lb.max()),
         "regret_over_per_capita_budget": float(lb.mean() / (B / N)),
         "regret_over_cost_mean": float((lb / cost).mean()),
         "ir_violation_rate": float((r["u_truth"] < -1e-6).float().mean()),
         "ir_violation_mean": float((-r["u_truth"]).clamp_min(0).mean()),
         "budget_used": float(r["pay"].sum(1).mean() / B), "released_eps_sum": float(r["eps_out"].sum(1).mean()),
         "budget_violation_max": float((r["pay"].sum(1) - B).max())}
    if "pga" in r:
        s["regret_pga_orig_mean"] = float(r["pga"].mean())
    if "ub" in r:
        ub = r["ub"]
        n = ub.shape[0]
        per_client = ub.mean(0)
        var = ub.var(0)
        l = math.log(2 * N / 0.05)
        s.update({"regret_ub_mean": float(ub.mean()), "cert_closed_share": float(r["closed"]),
                  "cert_profiles": int(n), "cert_tol": CERT_TOL,
                  "cert_lb_mean": float(r["cert_lb"].mean()),
                  "cert_hoeffding": float(per_client.max() + U_BAR * math.sqrt(math.log(N / 0.05) / (2 * n))),
                  "cert_bernstein": float((per_client + torch.sqrt(2 * var * l / n) + 7 * U_BAR * l / (3 * (n - 1))).max())})
    return s


def main():
    torch.manual_seed(0)
    reps, bud, vt = test_set()
    res = {"profiles": int(reps.shape[0]), "budget": B, "test_seeds": TEST_SEEDS, "results": {}}
    for kind in KINDS:
        t0 = time.time()
        if kind not in ANALYTIC:
            per_seed = {}
            for sd in TRAIN_SEEDS:
                path = os.path.join(RUNS, f"{kind}_seed{sd}_ep{CKPT_EPOCH}.pt")
                if not os.path.exists(path):
                    continue
                model = load(kind, sd)
                r = neural_regret(model, reps, bud, vt)
                n_cert = CERT_PLAN.get(sd, CERT_N)
                if kind in ("mfg", "mfgalign") and CERT and n_cert > 0:
                    c = crown.certify_crown(model, reps[:n_cert], bud[:n_cert], tol=CERT_TOL)
                    r["ub"] = c["regret_ub"]
                    r["closed"] = (c["open"] == 0).float().mean()
                    r["cert_lb"] = c["regret_lb"]
                    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in c.items()}, os.path.join(
                        os.path.dirname(OUT), f"cert_{kind}_seed{sd}.pt"))
                per_seed[sd] = summarize(r)
                print(kind, sd, json.dumps(per_seed[sd]), flush=True)
            res["results"][kind] = per_seed
        else:
            r = baseline_regret(kind, reps, bud)
            res["results"][kind] = summarize(r)
            print(kind, json.dumps(res["results"][kind]), flush=True)
        res.setdefault("seconds", {})[kind] = time.time() - t0
        with open(OUT, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""RQ2 cost of clearing a batch of auction profiles, N in {10, 50, 100, 200, 500} (Claude, 2026-10-24).

Mechanisms, all on the same device and in the same process, timed in interleaved random order (median of the
repeats; warm-up calls are not timed):
  * MFG-RegretNet (trained checkpoint, kappa = 3): allocation, payment layer (n = 64 intervals, trapezoid rule) and
    dividend for every client, in the cached form of Sec. VI (each client's counterfactual valuations change only
    its own embedding, so the sum of the other embeddings is computed once; the selection and the data shares are
    fixed by the verified sizes; every grid point costs one purchase evaluation and one cap projection). It is
    checked against Codex's implementation (model.mechanism, which re-evaluates the whole market for every grid
    point) and the dividend of p2v2_mech.mechanism before any timing; the run stops if they differ;
  * PAC, VCG, CSRA (release 1/(N-k)), MFG-Pricing: the implementations of main_final2/code/final_20e327e/baselines
    used in the experiments (PAC, VCG and CSRA loop over profiles in numpy);
  * RegretNet and DM-RegretNet: the architecture of the experiments (RegretNetScaled, 3 hidden layers of 100,
    softmax or deterministic allocation) with random weights, sized for N clients (the trained checkpoints exist
    for N = 10 only; the forward cost does not depend on the weights).
Memory: the increase of the peak allocated device memory over the memory held before the call (CUDA only).
The machine may run other jobs; load averages before and after are recorded, and all mechanisms share the
conditions because the calls are interleaved.
Usage: python rq2_timing_p2v2.py --out rq2_timing.json [--n-list 10,50,100,200,500] [--profiles 100] [--repeats 7]
       [--device cuda|cpu] [--check-only]
"""
import sys
sys.dont_write_bytecode = True   # the baseline code may live in another tool's read-only directory

import argparse
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common_p2v2 as C
import p2v2_mech as M

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "main_final2/code/final_20e327e"
B1 = ROOT / "main_final2/revisions/b1_retrain_20261014/exp"


@torch.no_grad()
def ours_cached(model, reports, budget, kappa, chunk=8):
    """Release e, payment p (release part) and dividend h of every client, cached form; reports (P, N, 3)."""
    P, N, _ = reports.shape
    dt, dev = reports.dtype, reports.device
    f = M.features(reports)
    emb = model.phi(f)                                                         # (P, N, H)
    emb_sum = emb.sum(1)                                                       # (P, H)
    pub = torch.cat((budget.reshape(-1, 1) / 110., M.V2.kappa_feature(kappa.reshape(-1, 1), .003, 300.)), -1)
    out = model.public(pub)
    coef, bias = F.softplus(out[:, :-1]), out[:, -1]
    sizes, caps = reports[..., 2], reports[..., 1]
    shares = sizes / sizes.sum(-1, keepdim=True)
    tau = model.threshold(kappa, budget, "budget")
    mask = shares >= tau[:, None]
    fallback = F.one_hot(shares.argmax(-1), num_classes=N).bool()
    mask = torch.where(mask.any(-1, keepdim=True), mask, fallback)
    caps_s = torch.where(mask, caps, 0.)
    shares_s = torch.where(mask, shares, 0.)
    shares_s = shares_s / shares_s.sum(-1, keepdim=True)
    cap_sum = caps_s.sum(-1)
    b = budget.reshape(-1)

    def purchase(z, lead, caps_total=None):
        """S for embedding means z (P, *lead, H); caps_total (P, *lead) replaces the selected caps' sum if given."""
        ex = (slice(None),) + (None,) * len(lead)
        score = bias[ex] + (coef[ex] * z).sum(-1)
        s = M.S_MAX * torch.sigmoid(score)
        return torch.minimum(torch.minimum(s, b[ex]), cap_sum[ex] if caps_total is None else caps_total)

    def release(S, rows):
        """Releases of all clients for purchases S (rows of the profile index); floor 0.01."""
        e = M.V2.cap_projection(S.reshape(-1, 1), shares_s[rows], caps_s[rows])
        return torch.where(e >= M.MIN_EVENT, e, 0.)

    e = release(purchase(emb_sum / N, ()), torch.arange(P, device=dev))
    fraction = torch.linspace(0., 1., M.INTERVALS + 1, dtype=dt, device=dev)
    p = torch.zeros_like(e)
    h = torch.zeros_like(e)
    G = M.INTERVALS + 1
    for lo in range(0, N, chunk):
        cs = torch.arange(lo, min(N, lo + chunk), device=dev)
        K = len(cs)
        v = reports[:, cs, 0]                                                  # (P, K)
        grid = v[..., None] + (1 - v[..., None]) * fraction                    # (P, K, G)
        fc = f[:, cs][:, :, None, :].expand(P, K, G, 3).clone()
        fc[..., 0] = 1 - grid
        z = (emb_sum[:, None, None, :] - emb[:, cs][:, :, None, :] + model.phi(fc)) / N
        S = purchase(z, (K, G))                                                # (P, K, G)
        rows = torch.arange(P, device=dev)[:, None, None].expand(P, K, G).reshape(-1)
        alloc = release(S, rows).reshape(P, K, G, N)
        own = alloc[torch.arange(P, device=dev)[:, None, None], torch.arange(K, device=dev)[None, :, None],
                    torch.arange(G, device=dev)[None, None, :], cs[None, :, None]]   # (P, K, G)
        integral = ((own[..., 1:] + own[..., :-1]) * .5).sum(-1) * (1 - v) / M.INTERVALS
        p[:, cs] = v * e[:, cs] + integral
        fb = f[:, cs].clone()                                                  # dividend: price 0, budget eps_bar
        fb[..., 0] = 1.0
        fb[..., 1] = M.EPSBAR / 5.
        cap_bar = cap_sum[:, None] - caps_s[:, cs] + torch.where(mask[:, cs], M.EPSBAR, 0.)    # own cap = eps_bar
        sbar = purchase((emb_sum[:, None, :] - emb[:, cs] + model.phi(fb)) / N, (K,), cap_bar)
        h[:, cs] = shares[:, cs] * (b[:, None] - M.VBAR * sbar)
    return e, p, h


def parity(model, device, profiles=64):
    reps = C.generate_privacy_paper_bids(10, profiles, 4242).double().to(device)
    bud = torch.full((profiles, 1), 50.0, dtype=torch.float64, device=device)
    kap = torch.full((profiles, 1), 3.0, dtype=torch.float64, device=device)
    e0, p0, h0 = M.mechanism(model, reps, bud, kap)
    e1, p1, h1 = ours_cached(model, reps, bud, kap)
    return {"release": float((e0 - e1).abs().max()), "payment": float((p0 - p1).abs().max()),
            "dividend": float((h0 - h1).abs().max())}


def baselines(code_dir, models_dir):
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(models_dir))
    from baselines.pac import pac_batch
    from baselines.vcg import vcg_procurement_batch
    from baselines.csra import csra_qms_batch
    from baselines.mfg_pricing import mfg_pricing_batch
    from models_scaled import RegretNetScaled
    return pac_batch, vcg_procurement_batch, csra_qms_batch, mfg_pricing_batch, RegretNetScaled


def neural(cls, N, kind, device):
    torch.manual_seed(0)
    m = cls(N, 1, activation="tanh", hidden_layer_size=100, n_hidden_layers=3, p_activation="softmax",
            a_activation="deterministic" if kind == "dm" else "softmax", separate=True, smoothing=.125,
            normalized_input=-1)
    return m.to(device).eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-list", default="10,50,100,200,500")
    ap.add_argument("--profiles", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--code-dir", default=str(CODE), help="directory with baselines/ and regretnet.py")
    ap.add_argument("--models-dir", default=str(B1), help="directory with models_scaled.py")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    model = M.load("mfg_p2v2").to(device)
    par = parity(model, device)
    print({"parity_max_abs_difference": par}, flush=True)
    if max(par.values()) > 1e-9:
        raise SystemExit("cached implementation differs from Codex's mechanism")
    if a.check_only:
        return
    pac, vcg, csra, pricing, RN = baselines(a.code_dir, a.models_dir)
    import hashlib
    code_sha = {f: hashlib.sha256((Path(a.code_dir) / f).read_bytes()).hexdigest()
                for f in ("baselines/pac.py", "baselines/vcg.py", "baselines/csra.py", "baselines/mfg_pricing.py",
                          "regretnet.py")}
    code_sha["models_scaled.py"] = hashlib.sha256((Path(a.models_dir) / "models_scaled.py").read_bytes()).hexdigest()
    out = {"code_dirs": [a.code_dir, a.models_dir], "code_sha256": code_sha,
           "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "host": C.host_info(), "load_before": os.getloadavg(), "profiles": a.profiles, "budget": a.budget,
           "repeats": a.repeats, "parity": par, "times_s": {}, "median_s": {}, "memory_mib": {}}
    rng = random.Random(0)
    for N in [int(x) for x in a.n_list.split(",")]:
        reps = C.generate_privacy_paper_bids(N, a.profiles, 1001).double().to(device)
        bud = torch.full((a.profiles, 1), a.budget, dtype=torch.float64, device=device)
        kap = torch.full((a.profiles, 1), 3.0, dtype=torch.float64, device=device)
        r32, b32 = reps.float(), bud.float()
        nets = {"RegretNet": neural(RN, N, "rn", device), "DM-RegretNet": neural(RN, N, "dm", device)}
        calls = {
            "MFG-RegretNet": lambda: ours_cached(model, reps, bud, kap),
            "PAC": lambda: pac(r32, b32), "VCG": lambda: vcg(r32, b32), "CSRA": lambda: csra(r32, b32),
            "MFG-Pricing": lambda: pricing(r32, b32),
            "RegretNet": lambda: nets["RegretNet"]((r32, b32)), "DM-RegretNet": lambda: nets["DM-RegretNet"]((r32, b32)),
        }
        times = {k: [] for k in calls}
        with torch.no_grad():
            for k, fn in calls.items():                                        # warm-up, not timed
                fn(); sync(device)
            for _ in range(a.repeats):
                order = list(calls)
                rng.shuffle(order)
                for k in order:
                    sync(device)
                    t0 = time.perf_counter()
                    calls[k]()
                    sync(device)
                    times[k].append(time.perf_counter() - t0)
            mem = {}
            if device.type == "cuda":
                for k, fn in calls.items():
                    sync(device)
                    base = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    fn(); sync(device)
                    mem[k] = (torch.cuda.max_memory_allocated(device) - base) / 2 ** 20
        out["times_s"][str(N)] = times
        out["median_s"][str(N)] = {k: st.median(v) for k, v in times.items()}
        out["memory_mib"][str(N)] = mem
        print(N, {k: round(v, 4) for k, v in out["median_s"][str(N)].items()}, flush=True)
        C.atomic_json(a.out, out)
    out["load_after"] = os.getloadavg()
    C.atomic_json(a.out, out)


if __name__ == "__main__":
    main()

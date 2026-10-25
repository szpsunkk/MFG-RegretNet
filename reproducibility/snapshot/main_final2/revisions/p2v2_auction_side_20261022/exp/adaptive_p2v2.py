"""Adaptive (learning) attacker against MFG-RegretNet (P2-v2): adaptive_attack_b1.py with this mechanism.

Unchanged from the original: client 0 attacks, the others report truthfully, a fresh market every round
(generate_privacy_paper_bids, seed 50000 + 101 k for trajectory k), bandit feedback, 70 actions (valuation x
{0.25..2} clipped to [0, 1], cap x {0, .25, .5, .75, 1}, size 1 + {0.5, 1} (d - 1)), 8 contexts, explore-then-commit
with M = 10, T = 20,000 rounds, 10 trajectories, gains per round over the truthful action.
Added: kappa in {3, 30}, and the learner runs separately on the mechanism with and without the dividend.
Usage: python adaptive_p2v2.py --family mfg_p2v2 --traj k --out dir [--T 20000] [--budgets 25,50] [--kappas 3,30]
One trajectory per process; summarize_adaptive.py merges them.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common_p2v2 as C
import p2v2_mech as M

KV = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
KE = [0.0, 0.25, 0.5, 0.75, 1.0]
KD = [0.5, 1.0]
ACTIONS = [(a, b, c) for a in KV for b in KE for c in KD]
TRUTH = ACTIONS.index((1.0, 1.0, 1.0))
M_EXPLORE = 10


def contexts(reps):
    v, e = reps[:, 0, 0].numpy(), reps[:, 0, 1].numpy()
    return np.minimum((v * 4).astype(int), 3) * 2 + (e >= 2.55).astype(int)


def explore_then_commit(U, ctx, rng):
    """adaptive_attack_b1.explore_then_commit, unchanged."""
    Tn, A = U.shape
    n = np.zeros((8, A))
    s = np.zeros((8, A))
    order = [rng.permutation(A) for _ in range(8)]
    seen = np.zeros(8, dtype=int)
    played = np.empty(Tn, dtype=int)
    commit = np.zeros(Tn, dtype=bool)
    for t in range(Tn):
        c = ctx[t]
        if seen[c] < M_EXPLORE * A:
            a = order[c][seen[c] % A]
            seen[c] += 1
        else:
            a = int(np.argmax(s[c] / n[c]))
            commit[t] = True
        played[t] = a
        n[c, a] += 1
        s[c, a] += U[t, a]
    return played, commit


def utility_tables(model, reps, settings, chunk):
    """U[setting][variant] (T, 70): client 0's utility (true valuation) for every action in every round."""
    act = torch.tensor(ACTIONS, dtype=torch.float64)
    T = len(reps)
    out = [{"with_dividend": np.empty((T, len(ACTIONS))), "without_dividend": np.empty((T, len(ACTIONS)))}
           for _ in settings]
    for lo in range(0, T, chunk):
        R = reps[lo:lo + chunk]
        P = len(R)
        cand = torch.stack((torch.clamp(R[:, None, 0, 0] * act[None, :, 0], 0.0, 1.0),
                            R[:, None, 0, 1] * act[None, :, 1],
                            1.0 + act[None, :, 2] * (R[:, None, 0, 2] - 1.0)), -1)
        ctx = M.ClientContext(model, R, 0)
        sets = [(torch.full((P, 1), B, dtype=torch.float64), torch.full((P, 1), k, dtype=torch.float64))
                for B, k in settings]
        v = R[:, 0, 0]
        for s, (e, p, h) in enumerate(ctx.outcomes_multi(sets, cand)):
            u = p - v[:, None] * e
            out[s]["without_dividend"][lo:lo + P] = u.numpy()
            out[s]["with_dividend"][lo:lo + P] = (u + h).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=M.FAMILIES, default="mfg_p2v2")
    ap.add_argument("--traj", type=int, required=True)
    ap.add_argument("--T", type=int, default=20000)
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--kappas", default="3,30")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = M.load(a.family)
    settings = [(B, k) for B in C.floats(a.budgets) for k in C.floats(a.kappas)]
    k = a.traj
    began = time.time()
    reps = C.generate_privacy_paper_bids(C.N_AGENTS, a.T, seed=50000 + 101 * k).double()
    tables = utility_tables(model, reps, settings, a.chunk)
    ctx = contexts(reps)
    res = {"family": a.family, "trajectory": k, "T": a.T, "actions": ACTIONS, "truthful_action": TRUTH,
           "attacker": "client 0, bandit feedback, contextual explore-then-commit (8 contexts, M=10)",
           "host": C.host_info(), "provenance": M.provenance(), "results": {}}
    for s, (B, kap) in enumerate(settings):
        for variant, U in tables[s].items():
            rng = np.random.default_rng(k)
            played, commit = explore_then_commit(U, ctx, rng)
            gain = U[np.arange(a.T), played] - U[:, TRUTH]
            D = U - U[:, [TRUTH]]
            best_policy = sum(max(D[ctx == c].sum(0).max(), 0.0) for c in range(8)) / a.T
            res["results"][f"B{B:g}_kappa{kap:g}_{variant}"] = {
                "budget": B, "kappa": kap, "variant": variant,
                "gain_commit": float(gain[commit].mean()), "gain_learner": float(gain.mean()),
                "gain_best_policy": float(best_policy),
                "gain_best_response": float((U.max(1) - U[:, TRUTH]).mean()),
                "truthful_utility": float(U[:, TRUTH].mean()),
                "commit_share": float(commit.mean()),
                "commit_truthful_share": float((played[commit] == TRUTH).mean())}
    res["seconds"] = time.time() - began
    C.atomic_json(Path(a.out) / f"adaptive_{a.family}_traj{k}.json", res)
    print({"traj": k, "seconds": round(res["seconds"], 1)}, flush=True)


if __name__ == "__main__":
    main()

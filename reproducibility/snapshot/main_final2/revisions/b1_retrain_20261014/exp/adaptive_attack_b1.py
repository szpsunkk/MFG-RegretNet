"""Step 10: adaptive (learning) attacker against the RQ1 mechanisms.

B1 variant (2026-10-15): same attacker, markets, actions and seeds as step 10; only the list of
mechanisms changes, given by the environment variable ADAPT_KINDS ("kind:seed,..."), e.g. the B1
baselines "regretnet_s:42,dm_s:42" loaded from the B1 runs directory.

Client 0 is the attacker; the other N-1 clients report truthfully; every round
draws a fresh i.i.d. market. The attacker knows its own type and only observes
its own outcome (bandit feedback). It learns a misreport policy online with a
contextual explore-then-commit rule: its type is binned into 8 contexts (4
valuation quartiles x 2 cap halves); in each context it first plays every
misreport action M times (round-robin), then commits to the action with the
highest empirical mean utility. Actions scale the valuation, withhold part or
all of the privacy cap, and withhold data size; the truthful action is included.
Because the other clients' behaviour does not depend on the attacker's history,
the attacker's utility of every action in every round is computed in one batch.

Reported per mechanism (and per MFG-RegretNet training seed):
  * gain_commit      : mean per-round gain over the truthful report during the
                       commit phase (after learning);
  * gain_learner     : the same over the whole horizon (includes exploration);
  * gain_best_policy : best context-dependent action in hindsight;
  * gain_best_response: mean per-round gain of the best action in each round
                       (an upper bound for any strategy over this action set).

Usage: python adaptive_attack.py <code_dir> <runs_dir> <out_json> <T> <trajectories> <budgets>
"""
import json
import math
import os
import sys

CODE_DIR, RUNS, OUT, T, TRAJ, BUDGETS = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                         int(sys.argv[5]), [float(b) for b in sys.argv[6].split(",")])
sys.argv = [sys.argv[0], CODE_DIR, RUNS, "/dev/null", "20", "mfg", "0"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import eval_rq1 as ev  # noqa: E402
from datasets_fl_benchmark import generate_privacy_paper_bids  # noqa: E402
from utils import allocs_to_plosses  # noqa: E402
from exp_rq.guarantees_eval_baselines import _dispatch  # noqa: E402

dev = ev.dev
N = ev.N
KV = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
KE = [0.0, 0.25, 0.5, 0.75, 1.0]
KD = [0.5, 1.0]
ACTIONS = [(a, b, c) for a in KV for b in KE for c in KD]
TRUTH = ACTIONS.index((1.0, 1.0, 1.0))


def mechanism_fn(kind, seed=None):
    if kind in ("PAC", "VCG", "CSRA", "MFG-Pricing"):
        f = _dispatch(kind)
        return lambda R, Bd: f(R, Bd)
    w = ev.Wrap(ev.load(kind, seed))

    def g(R, Bd):
        with torch.no_grad():
            al, pay = w((R, Bd))
        return allocs_to_plosses(al, R[:, :, -2]), pay
    return g


@torch.no_grad()
def utility_table(fn, reps, B, chunk=4000):
    """U[t, a]: attacker utility (true valuation) when client 0 plays action a in round t."""
    Tn = reps.shape[0]
    A = len(ACTIONS)
    act = torch.tensor(ACTIONS, device=dev)
    U = torch.empty(Tn, A, device=dev)
    v_true = reps[:, 0, 0]
    for s in range(0, Tn, chunk // A + 1):
        R = reps[s:s + chunk // A + 1]
        P = R.shape[0]
        Rx = R.unsqueeze(1).repeat(1, A, 1, 1)
        Rx[:, :, 0, 0] = torch.clamp(R[:, None, 0, 0] * act[None, :, 0], 0.0, 1.0)
        Rx[:, :, 0, 1] = R[:, None, 0, 1] * act[None, :, 1]
        Rx[:, :, 0, 2] = 1.0 + act[None, :, 2] * (R[:, None, 0, 2] - 1.0)
        Rf = Rx.view(P * A, N, 3)
        Bd = B * torch.ones(P * A, 1, device=dev)
        pl, pay = fn(Rf, Bd)
        u = pay.view(P, A, N)[:, :, 0] - v_true[s:s + P, None] * pl.view(P, A, N)[:, :, 0]
        U[s:s + P] = u
    return U.cpu().numpy()


M_EXPLORE = 10


def contexts(reps):
    v, e = reps[:, 0, 0].cpu().numpy(), reps[:, 0, 1].cpu().numpy()
    return np.minimum((v * 4).astype(int), 3) * 2 + (e >= 2.55).astype(int)


def explore_then_commit(U, ctx, rng):
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


def main():
    kinds = [(x.split(":")[0], int(x.split(":")[1])) for x in os.environ["ADAPT_KINDS"].split(",")]
    res = {"setting": {"T": T, "trajectories": TRAJ, "actions": ACTIONS, "truthful_action": TRUTH,
                       "attacker": "client 0, bandit feedback (own utility only), contextual explore-then-commit (8 contexts, M=10)",
                       "opponents": "truthful, fresh i.i.d. market every round"}, "results": {}}
    for B in BUDGETS:
        res["results"][str(B)] = {}
        for kind, sd in kinds:
            fn = mechanism_fn(kind, sd)
            rows = []
            for k in range(TRAJ):
                reps = generate_privacy_paper_bids(N, 1, T, seed=50000 + 101 * k).to(dev)
                U = utility_table(fn, reps, B)
                ctx = contexts(reps)
                rng = np.random.default_rng(k)
                played, commit = explore_then_commit(U, ctx, rng)
                gain = U[np.arange(T), played] - U[:, TRUTH]
                D = U - U[:, [TRUTH]]
                best_policy = sum(max(D[ctx == c].sum(0).max(), 0.0) for c in range(8)) / T
                rows.append({"gain_commit": float(gain[commit].mean()), "gain_learner": float(gain.mean()),
                             "gain_best_policy": float(best_policy),
                             "gain_best_response": float((U.max(1) - U[:, TRUTH]).mean()),
                             "truthful_utility": float(U[:, TRUTH].mean()),
                             "commit_share": float(commit.mean()),
                             "commit_truthful_share": float((played[commit] == TRUTH).mean())})
            name = f"{kind}_seed{sd}"
            agg = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            agg["gain_commit_sd"] = float(np.std([r["gain_commit"] for r in rows]))
            res["results"][str(B)][name] = {"mean": agg, "per_trajectory": rows}
            print(B, name, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in agg.items()}), flush=True)
            with open(OUT, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

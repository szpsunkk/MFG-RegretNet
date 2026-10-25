"""Is the trained selector's release monotone in the client's own data-size report?

Proposition 2(a) needs e_i non-decreasing in d_i. With the data-share threshold, a larger d_i can drop another client
j from the selection; the shares are renormalized and j's mass is redistributed, which can lower e_i when j was
capped (an argument by hand gives such cases). This script measures it on the trained checkpoints:
for each profile and client, e_i at 400 sizes d' in [1, d_i] (other reports and the client's price and cap fixed),
and the largest drop max_{d' < d''} (e_i(d') - e_i(d'')).
Profiles: the paper's RQ1 test profiles (first --profiles) and training-style profiles with sizes 10 + 490 U^2.
Prints counts and the largest drops only.
"""
import argparse

import numpy as np
import torch

import common_p2v2 as C
import p2v2_mech as M


def drops(model, reports, B, kappa, grid=400):
    P, N, _ = reports.shape
    out = torch.zeros(P, N, dtype=torch.float64)
    b = torch.full((P * grid, 1), B, dtype=torch.float64)
    k = torch.full((P * grid, 1), kappa, dtype=torch.float64)
    frac = torch.linspace(0., 1., grid, dtype=torch.float64)
    with torch.no_grad():
        for c in range(N):
            r = reports[:, None].expand(P, grid, N, 3).clone()
            r[:, :, c, 2] = 1.0 + frac[None] * (reports[:, None, c, 2] - 1.0)
            e = model.allocate(r.reshape(-1, N, 3), b, k).reshape(P, grid, N)[:, :, c]
            run_max = torch.cummax(e, dim=1).values
            out[:, c] = (run_max - e).max(1).values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=int, default=400)
    ap.add_argument("--family", default="mfg_p2v2")
    a = ap.parse_args()
    torch.set_num_threads(2)
    model = M.load(a.family)
    rng = np.random.default_rng(20261022)
    test = C.test_profiles()[:a.profiles]
    train = test.clone()
    train[..., 2] = torch.tensor(10 + 490 * rng.random((a.profiles, 10)) ** 2)
    for name, reps in (("test profiles", test), ("training-style sizes", train)):
        for B in (25., 50., 100.):
            for kap in (3., 30.):
                d = drops(model, reps, B, kap)
                bad = d > 1e-9
                print(f"{name:22s} B={B:5.0f} kappa={kap:4.0f}: client-profiles with a drop {int(bad.sum()):5d} of {bad.numel()}"
                      f" | largest drop {float(d.max()):.3e}", flush=True)


if __name__ == "__main__":
    main()

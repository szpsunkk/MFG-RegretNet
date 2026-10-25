"""Checks for p2_reference.py. Run: /home/skk/anaconda3/bin/python test_p2_reference.py"""
import math

import numpy as np
import torch

import p2_reference as P

RNG = np.random.default_rng(20261018)


def random_market(n=10):
    value = RNG.uniform(0, 1, n)
    cap = RNG.uniform(.1, 5, n)
    size = 10 + 490 * RNG.uniform(0, 1, n) ** 2
    return value, cap, size / size.sum()


def brute_projection(total, target, cap, cost=None, budget=None):
    import cvxpy as cp
    e = cp.Variable(len(cap))
    cons = [cp.sum(e) == total, e >= 0, e <= cap]
    if cost is not None:
        cons.append(cost @ e <= budget)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(e - target)), cons)
    problem.solve(solver=cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None)
    return None if e.value is None else np.asarray(e.value)


def test_projection_matches_brute_force():
    worst = 0.0
    for _ in range(200):
        value, cap, share = random_market()
        total = RNG.uniform(.05, .95) * cap.sum()
        budget = RNG.uniform(.3, 1.2) * P.min_cost_for_total(total, cap, value) + 1e-6
        for cost, b in ((None, None), (value, budget)):
            e = P.capped_simplex_projection(total, total * share, cap, cost, b)
            ref = brute_projection(total, total * share, cap, cost, b)
            if cost is not None and P.min_cost_for_total(total, cap, value) > budget + 1e-9:
                assert e is None
                continue
            assert e is not None and ref is not None
            assert abs(e.sum() - total) < 1e-7 and (e >= -1e-12).all() and (e <= cap + 1e-12).all()
            if cost is not None:
                assert cost @ e <= b + 1e-7
            obj, obj_ref = np.square(e - total * share).sum(), np.square(ref - total * share).sum()
            assert obj <= obj_ref + 1e-6 * max(1.0, obj_ref), (obj, obj_ref)
            worst = max(worst, obj - obj_ref)
    print("projection: max objective excess over cvxpy", worst)


def test_uncapped_alignment_is_exact():
    share = np.array([.1, .2, .3, .4])
    e = P.aligned_allocation(2.0, share, np.full(4, 10.0))
    assert np.allclose(e, 2.0 * share)
    assert P.misalignment(e, share)[0] < 1e-9


def test_j_kappa_hand_example():
    eps = np.array([1.0, 1.0, 0.0, 0.005])  # the last one is below the minimum release and does not upload
    share = np.array([.5, .25, .25, 0.0])
    # alpha = (.5, .5, 0, 0); ||alpha - pi||^2 = 0 + .0625 + .0625 + 0 = .125; |I| = 2; S = 2
    assert math.isclose(P.j_kappa(eps, share, 4.0), 4 * .125 + 2 / 4)


def test_oracle_limits():
    for _ in range(30):
        value, cap, share = random_market()
        e0, pay0, info0 = P.p2_oracle(cap, value, share, 1e6, 0.0)
        assert np.allclose(e0, cap, atol=1e-6), "kappa = 0 with a slack budget must buy every cap"
        e_hi, _, _ = P.p2_oracle(cap, value, share, 1e6, 1e6)
        assert P.misalignment(e_hi, share)[0] <= P.misalignment(cap, share)[0] + 1e-9
        e_b, pay_b, _ = P.p2_oracle(cap, value, share, 3.0, 1.0)
        assert pay_b.sum() <= 3.0 + 1e-6 and np.allclose(pay_b, value * e_b)
        assert ((e_b == 0) | (e_b >= P.MIN_EVENT_EPSILON)).all() and (e_b <= cap + 1e-9).all()


def test_oracle_monotone_in_kappa():
    value, cap, share = random_market()
    totals = [P.p2_oracle(cap, value, share, 1e6, k)[2]["total"] for k in (0.0, .1, 1.0, 10.0, 100.0)]
    assert all(a >= b - 1e-6 for a, b in zip(totals, totals[1:])), totals


def test_aligned_at_total():
    value, cap, share = random_market()
    e, pay = P.aligned_at_total(cap, value, share, 8.0, 30.0)
    assert abs(e.sum() - 8.0) < 0.05 and pay.sum() <= 30 + 1e-6


def test_torch_matches_numpy():
    for _ in range(50):
        value, cap, share = random_market()
        eps = cap * RNG.uniform(0, 1, cap.shape)
        eps[RNG.uniform(0, 1, cap.shape) < .2] = 0.0
        eps = P.drop_below_min_event(eps)
        if eps.sum() == 0:
            continue
        j_np = P.j_kappa(eps, share, 3.0)
        j_t = P.j_kappa_torch(torch.tensor(eps)[None], torch.tensor(share)[None], torch.tensor([3.0]), smooth=False)
        assert math.isclose(j_np, float(j_t[0]), rel_tol=1e-9)


def test_training_term():
    g = torch.Generator().manual_seed(0)
    reports = torch.rand(64, 10, 3, generator=g, dtype=torch.float64)
    reports[..., 1] = .1 + 4.9 * reports[..., 1]
    reports[..., 2] = 10 + 490 * reports[..., 2].square()
    kappa = P.sample_log_uniform(64, .01, 100., g).to(torch.float64)
    caps = reports[..., 1].clone().requires_grad_(True)
    term = P.p2_training_term(caps, reports, kappa)
    assert torch.allclose(term, torch.ones_like(term))
    eps = (reports[..., 1] * torch.rand(64, 10, generator=g, dtype=torch.float64)).requires_grad_(True)
    P.p2_training_term(eps, reports, kappa).mean().backward()
    assert torch.isfinite(eps.grad).all()
    f = P.kappa_feature(kappa, .01, 100.)
    assert (f >= 0).all() and (f <= 1).all()


def test_noise_energy_and_kappa():
    # one coordinate, lr 1, clip 1, q * n = 4, sigma 2, T = 3 -> 3 * (2 / 4)^2 = 0.75
    assert math.isclose(P.dp_noise_energy(2.0, .25, 3, 16, 1.0, 1.0, 1), .75)
    zeta2, n, d = 0.3, 10, 20000
    updates = RNG.normal(0, math.sqrt(zeta2 / d), (n, d)) + RNG.normal(0, 1, (1, d))
    out = P.kappa_from_updates(updates, np.full(n, 0.1))
    assert abs(out["zeta2"] - zeta2) / zeta2 < .05 and math.isclose(out["c_dp"], .1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

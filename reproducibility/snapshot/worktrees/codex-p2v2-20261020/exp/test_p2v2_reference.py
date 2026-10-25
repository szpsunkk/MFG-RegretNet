"""Tests for p2v2_reference.py (amendment A4). Run: python test_p2v2_reference.py  (about 10 minutes on one CPU core)."""
import numpy as np

import p2_reference as R
import p2v2_reference as M

RNG = np.random.default_rng(20261020)


def profiles(n):
    return np.stack([RNG.random((n, 10)), .1 + 4.9 * RNG.random((n, 10)), 10 + 490 * RNG.random((n, 10)) ** 2], -1)


def monotone_net(seed):
    """Random mean-field net, nonincreasing in every v_j and nondecreasing in every cap_j and d_j."""
    g = np.random.default_rng(seed)
    w1, b1 = np.abs(g.normal(size=(16, 3))), g.normal(size=16)
    w2, b2 = np.abs(g.normal(size=(16, 16))) / 4, g.normal(size=16)
    a, b = np.abs(g.normal(size=16)), g.normal() - 2

    def s_net(r):
        f = np.stack([1 - r[:, 0], r[:, 1] / 5, np.log(r[:, 2] / 10) / np.log(50)], -1)
        h = np.tanh(np.tanh(f @ w1.T + b1) @ w2.T + b2).mean(0)
        return 50 / (1 + np.exp(-(h @ a + b)))
    return s_net


NETS = [monotone_net(s) for s in (1, 2, 3)] + [lambda r: 6.0, lambda r: np.inf]


def test_projection():
    for r in profiles(200):
        share = r[:, 2] / r[:, 2].sum()
        for S in (0.5, 5.0, 12.0, r[:, 1].sum()):
            e = M.projection(S, share, r[:, 1])
            assert abs(e.sum() - min(S, r[:, 1].sum())) < 1e-9 and (e >= -1e-12).all() and (e <= r[:, 1] + 1e-12).all()


def test_monotone_allocation():
    worst = 0.0
    for r in profiles(150):
        for B in (10., 30., 100.):
            for net in NETS:
                for fixed in (None, 8.0):
                    x = M.allocation(r, B, net, fixed)
                    for i in range(10):
                        for c in range(3):
                            ch = r.copy()
                            if c == 0:
                                ch[i, 0] = min(ch[i, 0] + .1, 1.)
                            elif c == 1:
                                ch[i, 1] *= .9
                            else:
                                ch[i, 2] = max(ch[i, 2] * .9, 10.)
                            worst = max(worst, M.allocation(ch, B, net, fixed)[i] - x[i])
    assert worst <= 1e-9, worst


def test_ir_bf_payment_bounds():
    for r in profiles(40):
        for B in (10., 30., 100.):
            for net in NETS[:2]:
                for fixed in (None, 5.0):
                    x, p = M.mechanism(r, B, net, fixed)
                    assert (p - r[:, 0] * x >= -1e-12).all()          # IR at the reported (true) cost
                    assert (p <= x + 1e-12).all()                    # at most V_MAX = 1 per unit
                    assert p.sum() <= B + 1e-9                       # budget feasibility


def test_dsic_bruteforce():
    worst = 0.0
    for r in profiles(12):
        for B in (10., 100.):
            net = NETS[0]
            x, p = M.mechanism(r, B, net)
            for i in range(10):
                u_true = p[i] - r[i, 0] * x[i]
                for v2 in np.linspace(0, 1, 11):
                    for cf in (1., .75, .5):
                        for df in (1., .5):
                            ch = r.copy()
                            ch[i, 0], ch[i, 1], ch[i, 2] = v2, r[i, 1] * cf, max(r[i, 2] * df, 10.)
                            x2, p2 = M.mechanism(ch, B, net)
                            worst = max(worst, (p2[i] - r[i, 0] * x2[i]) - u_true)
    assert worst <= 1e-3, worst


def test_fixed_total_mode():
    for r in profiles(30):
        x, p = M.mechanism(r, 110., fixed_total=8.0)
        assert abs(x.sum() - min(8.0, r[:, 1].sum())) < 0.02            # only min-event drops can remove mass
        assert np.allclose(p, x, atol=1e-12)                            # allocation ignores costs: pays V_MAX
        r2 = r.copy(); r2[:, 0] = RNG.random(10)
        assert np.allclose(M.allocation(r2, 110., fixed_total=8.0), x)


def test_family_matches_p2_oracle_when_budget_slack():
    gaps = []
    for r in profiles(40):
        share = r[:, 2] / r[:, 2].sum()
        _, j_family = M.family_oracle_total(r, 30., 30.)
        j_oracle = R.p2_oracle(r[:, 1], r[:, 0], share, 30., 30.)[2]['j']
        gaps.append((j_family - j_oracle) / j_oracle)
    assert np.mean(gaps) < 5e-3, np.mean(gaps)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name, flush=True)

"""Tests for qv_reference.py (amendment A5). Run: python test_qv_reference.py  (a few seconds)."""
import numpy as np

import p2_reference as R
import qv_reference as Q

RNG = np.random.default_rng(20261021)


def profiles(n):
    return np.stack([RNG.random((n, 10)), .1 + 4.9 * RNG.random((n, 10)), 10 + 490 * RNG.random((n, 10)) ** 2], -1)


def test_tempered_share():
    d = np.array([10., 40., 90., 160.])
    assert np.allclose(Q.tempered_share(d, 1), d / d.sum())
    assert np.allclose(Q.tempered_share(d, 0), np.full(4, .25))
    assert np.allclose(Q.tempered_share(d, .5), np.sqrt(d) / np.sqrt(d).sum())


def test_proxy_feasible_and_ir():
    for r in profiles(60):
        for B in (10., 30.):
            for g in (1., .5, 0.):
                for fixed in (None, 8.0):
                    x, p = Q.qv_proxy(r, B, 3., g, fixed)
                    S = x.sum()
                    assert (x <= r[:, 1] + 1e-12).all() and (x >= 0).all()
                    assert p.sum() <= (B if fixed is None else 110.) + 1e-9
                    assert (p - r[:, 0] * x >= -1e-12).all()
                    if fixed is None:
                        assert S <= min(B, r[:, 1].sum()) + 1e-9


def test_tempered_projection_monotone_for_fixed_total():
    """For a fixed total the tempered projection is monotone in own cap and own size (gamma >= 0)."""
    worst = 0.0
    for r in profiles(150):
        for g in (1., .5, 0.):
            for S in (3., 8., 15.):
                x = Q.executed(S, Q.tempered_share(r[:, 2], g), r[:, 1])
                for i in range(10):
                    for c in (1, 2):
                        ch = r.copy()
                        if c == 1:
                            ch[i, 1] *= .9
                        else:
                            ch[i, 2] = max(ch[i, 2] * .9, 10.)
                        x2 = Q.executed(S, Q.tempered_share(ch[:, 2], g), ch[:, 1])
                        worst = max(worst, x2[i] - x[i])
    assert worst <= 1e-9, worst


def test_gamma_one_matches_a4_family_oracle_closely():
    import p2v2_reference as M
    gaps = []
    for r in profiles(25):
        share = r[:, 2] / r[:, 2].sum()
        S = Q.proxy_total(r, 30., 30., 1.)
        j_q = R.j_kappa(Q.executed(S, share, r[:, 1]), share, 30.)
        _, j_m = M.family_oracle_total(r, 30., 30.)
        gaps.append(j_q - j_m)
    assert np.mean(gaps) < 2e-3, np.mean(gaps)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name, flush=True)

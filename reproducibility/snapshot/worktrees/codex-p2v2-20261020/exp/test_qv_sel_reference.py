"""Tests for qv_sel_reference.py (amendment A6). Run: python test_qv_sel_reference.py  (a few seconds)."""
import numpy as np

import p2_reference as R
import qv_reference as Q
import qv_sel_reference as S

RNG = np.random.default_rng(20261022)


def profiles(n):
    return np.stack([RNG.random((n, 10)), .1 + 4.9 * RNG.random((n, 10)), 10 + 490 * RNG.random((n, 10)) ** 2], -1)


def test_k_all_equals_g1_projection():
    for r in profiles(50):
        for T in (3., 8.):
            a = S.topk_executed(T, r[:, 2], r[:, 1], 10)
            b = Q.executed(T, Q.tempered_share(r[:, 2], 1.), r[:, 1])
            assert np.allclose(a, b, atol=1e-12)


def test_feasible_ir_and_not_worse_than_all_clients():
    for r in profiles(30):
        pi = r[:, 2] / r[:, 2].sum()
        for kappa in (3., 30.):
            for T in (5., 8.):
                x, p = S.qv_sel(r, 110., kappa, fixed_total=T)
                assert (x <= r[:, 1] + 1e-12).all() and (x >= 0).all() and x.sum() <= T + 1e-9
                assert (p - r[:, 0] * x >= -1e-12).all()
                j_all = R.j_kappa(S.topk_executed(T, r[:, 2], r[:, 1], 10), pi, kappa)
                assert R.j_kappa(x, pi, kappa) <= j_all + 1e-12


def test_budget_mode_respects_budget():
    for r in profiles(8):
        x, p = S.qv_sel(r, 10., 3.)
        assert p.sum() <= 10. + 1e-9 and x.sum() <= min(10., r[:, 1].sum()) + 1e-9


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name, flush=True)

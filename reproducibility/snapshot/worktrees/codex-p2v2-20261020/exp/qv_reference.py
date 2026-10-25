"""Reference for amendment A5: quick-validation (QV) allocation proxies and the tempered target of p2-v2.

Tempered target: pi^gamma_j = d_j^gamma / sum_k d_k^gamma, gamma in {1, 0.5, 0}; gamma = 1 is the A4 design.

QV proxy for one round (no training):
  budget cells   S* = argmin over the fixed absolute grid {0.05 k} with S <= min(B, sum cap) of
                 J^gamma(x(S)) = kappa * ||w - pi^gamma||^2 + |I| / S^2,
                 x(S) = executed cap-only projection of S * pi^gamma (releases below 0.01 dropped)
  privacy cells  S = min(T, sum cap), T calibrated per cell as in A4.6
  payments       p = x (unit price V_MAX = 1): IR holds and sum p = sum x <= S <= B; payments do not enter the FL run.
The grid is absolute, so it does not move with the reports (the p2-v1 oracle's 200-point grid did; see the review).
"""
import numpy as np

import p2_reference as R

GRID_STEP = 0.05


def tempered_share(sizes, gamma):
    w = np.asarray(sizes, float) ** float(gamma)
    return w / w.sum()


def executed(S, target, cap):
    e = R._fill_for_mu(S * target, cap, min(float(S), float(cap.sum())))
    return R.drop_below_min_event(e, R.MIN_EVENT_EPSILON)


def proxy_total(reports, budget, kappa, gamma):
    reports = np.asarray(reports, float)
    cap, target = reports[:, 1], tempered_share(reports[:, 2], gamma)
    top = min(float(budget), float(cap.sum()))
    best = None
    for s in np.arange(GRID_STEP, top + 1e-12, GRID_STEP):
        j = R.j_kappa(executed(s, target, cap), target, kappa, R.MIN_EVENT_EPSILON)
        if best is None or j < best[0] - 1e-15:
            best = (j, s)
    return best[1]


def qv_proxy(reports, budget, kappa, gamma, fixed_total=None):
    """Executed allocation and payments of the QV proxy for one round."""
    reports = np.asarray(reports, float)
    S = proxy_total(reports, budget, kappa, gamma) if fixed_total is None else min(float(fixed_total), float(reports[:, 1].sum()))
    x = executed(S, tempered_share(reports[:, 2], gamma), reports[:, 1])
    return x, x.copy()

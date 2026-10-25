"""Reference for amendment A6: the QV-sel proxy, the exact P2 optimum over 'aligned among the k largest clients'.

For one round with reports (v, cap, d), shares pi = d / sum d and the dataset's deployment kappa:
  for k = 1..N, participants I_k = the k clients with the largest reported size (stable order);
  target t^k = pi restricted to I_k and renormalised; x(S, k) = executed cap-only projection of S * t^k within I_k;
  budget cells: (S, k) = argmin of J_kappa(x(S, k), pi) over S on the absolute grid {0.05 j} with S <= min(B, sum cap);
  privacy cells: S = min(T, sum cap) fixed, k = argmin of J_kappa(x(S, k), pi).
J_kappa uses the true data shares pi, exactly as in the paper's P2 objective and the Stage 1 screen.
Payments p = x (unit price 1); they do not enter the FL run.
"""
import numpy as np

import p2_reference as R
from qv_reference import GRID_STEP


def topk_executed(S, sizes, cap, k):
    sizes = np.asarray(sizes, float)
    idx = np.argsort(-sizes, kind='stable')[:k]
    t = np.zeros_like(sizes)
    t[idx] = sizes[idx] / sizes[idx].sum()
    c = np.where(t > 0, np.asarray(cap, float), 0.)
    e = R._fill_for_mu(S * t, c, min(float(S), float(c.sum())))
    return R.drop_below_min_event(e, R.MIN_EVENT_EPSILON)


def qv_sel(reports, budget, kappa, fixed_total=None):
    reports = np.asarray(reports, float)
    cap, sizes = reports[:, 1], reports[:, 2]
    pi = sizes / sizes.sum()
    if fixed_total is None:
        totals = np.arange(GRID_STEP, min(float(budget), float(cap.sum())) + 1e-12, GRID_STEP)
    else:
        totals = [min(float(fixed_total), float(cap.sum()))]
    best = None
    for S in totals:
        for k in range(1, len(sizes) + 1):
            x = topk_executed(S, sizes, cap, k)
            if x.sum() <= 0:
                continue
            j = R.j_kappa(x, pi, kappa, R.MIN_EVENT_EPSILON)
            if best is None or j < best[0] - 1e-15:
                best = (j, x)
    x = best[1]
    return x, x.copy()

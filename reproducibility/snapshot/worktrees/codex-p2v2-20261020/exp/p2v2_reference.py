"""Reference implementation of the P2-v2 mechanism (protocol p2-v1, amendment A4).

One market: N clients with reports r_j = (v_j, cap_j, d_j), public budget B.

  share   pi_j = d_j / sum_k d_k
  total   budget mode:       S = min(s_net(r), B, sum cap)
          fixed-total mode:  S = min(fixed_total, sum cap)        (privacy cells)
  alloc   e(S) = argmin ||e - S pi||^2  s.t.  sum e = S, 0 <= e <= cap      (no cost term)
  exec    x_j = e_j if e_j >= MIN_EVENT_EPSILON else 0
  pay     p_i = v_i x_i(v_i) + integral_{v_i}^{V_MAX} x_i(t) dt               (Myerson, trapezoid rule)

x_i(t) is the executed allocation of client i when it reports cost t and nothing else changes. When s_net is
nonincreasing in every v_j and nondecreasing in every cap_j and d_j, x_i is nonincreasing in the own cost and
nondecreasing in the own cap and size. The payment rule then gives truthful reporting as a dominant strategy over
costs in [0, V_MAX] and downward cap and size reports, utility = integral >= 0 (IR), and
p_i <= v_i x_i + (V_MAX - v_i) x_i <= x_i, so sum p <= S <= B when V_MAX = 1 (budget feasibility).
"""
import numpy as np

import p2_reference as R

V_MAX = 1.0
INTERVALS = 64


def total(reports, budget, s_net=None, fixed_total=None):
    cap = reports[:, 1]
    if fixed_total is not None:
        return float(min(fixed_total, cap.sum()))
    return float(min(s_net(reports), budget, cap.sum()))


def projection(S, share, cap):
    """Cap-only aligned projection of S * share (exact breakpoint solution)."""
    return R._fill_for_mu(S * share, cap, min(float(S), float(cap.sum())))


def allocation(reports, budget, s_net=None, fixed_total=None):
    reports = np.asarray(reports, float)
    share = reports[:, 2] / reports[:, 2].sum()
    e = projection(total(reports, budget, s_net, fixed_total), share, reports[:, 1])
    return R.drop_below_min_event(e, R.MIN_EVENT_EPSILON)


def own_curve(reports, i, budget, s_net=None, fixed_total=None, intervals=INTERVALS):
    """Grid t_0 = v_i < ... < t_K = V_MAX and client i's executed allocation at each own cost report."""
    ts = np.linspace(reports[i, 0], V_MAX, intervals + 1)
    xs = np.empty_like(ts)
    for k, t in enumerate(ts):
        r = np.array(reports, float)
        r[i, 0] = t
        xs[k] = allocation(r, budget, s_net, fixed_total)[i]
    return ts, xs


def mechanism(reports, budget, s_net=None, fixed_total=None, intervals=INTERVALS):
    """Executed allocation x and payments p for one market."""
    reports = np.asarray(reports, float)
    x = allocation(reports, budget, s_net, fixed_total)
    p = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] <= 0:
            continue
        ts, xs = own_curve(reports, i, budget, s_net, fixed_total, intervals)
        p[i] = reports[i, 0] * x[i] + float(np.sum(0.5 * (xs[1:] + xs[:-1]) * np.diff(ts)))
    return x, p


def family_oracle_total(reports, budget, kappa, grid=400):
    """Best total of the P2-v2 allocation family for one market (reference for J, not a mechanism)."""
    reports = np.asarray(reports, float)
    share = reports[:, 2] / reports[:, 2].sum()
    s_hi = min(float(budget), float(reports[:, 1].sum()))
    best = None
    for s in np.linspace(s_hi / grid, s_hi, grid):
        x = R.drop_below_min_event(projection(s, share, reports[:, 1]), R.MIN_EVENT_EPSILON)
        j = R.j_kappa(x, share, kappa, R.MIN_EVENT_EPSILON)
        if best is None or j < best[0]:
            best = (j, s)
    return best[1], best[0]

"""Reference implementations for the P2-aligned procurement experiment (protocol p2-v1, 2026-10-18).

Codex imports or copies these functions unchanged. The review recomputes every allocation, loss and
kappa estimate with this same file, so any deviation must be recorded as a protocol amendment.
All functions are pure: no file I/O, no global random state.

Notation (design file RQ4_P2_MECHANISM_REDESIGN_20261018.md, Sec. 1):
  share pi_i = d_i / sum_j d_j; Algorithm-2 weights alpha_i = eps_i / sum_{eps_j >= min_event} eps_j;
  J_kappa(eps) = kappa * ||alpha - pi||^2 + |I| / S^2, with I the uploading clients and S their total epsilon.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MIN_EVENT_EPSILON = 0.01  # equals contracts.MIN_EVENT_EPSILON of the task-aware execution layer


# ---------------------------------------------------------------- allocation geometry (numpy, one market round)

def _fill_for_mu(base, cap, total):
    """Exact solution of sum_i clip(base_i + mu, 0, cap_i) = total (piecewise linear in mu, breakpoint search)."""
    points = np.unique(np.concatenate([-base, cap - base]))
    values = np.clip(base[None, :] + points[:, None], 0.0, cap[None, :]).sum(1)
    k = int(np.searchsorted(values, total - 1e-12))
    if k <= 0:
        mu = points[0]
    elif k >= len(points):
        mu = points[-1]
    else:
        lo, hi = values[k - 1], values[k]
        mu = points[k - 1] + (total - lo) * (points[k] - points[k - 1]) / (hi - lo) if hi > lo else points[k]
    return np.clip(base + mu, 0.0, cap)


def min_cost_for_total(total, cap, cost):
    """Cheapest way to buy `total` epsilon within caps at unit costs `cost` (fractional knapsack)."""
    need, spent = float(total), 0.0
    for i in np.argsort(cost, kind="stable"):
        take = min(float(cap[i]), need)
        spent += take * float(cost[i])
        need -= take
        if need <= 1e-12:
            return spent
    return math.inf


def max_total_for_budget(cap, cost, budget):
    """Largest total epsilon affordable at unit costs `cost` within caps (fractional knapsack)."""
    left, total = float(budget), 0.0
    for i in np.argsort(cost, kind="stable"):
        if cost[i] <= 0:
            total += float(cap[i])
            continue
        take = min(float(cap[i]), left / float(cost[i]))
        total += take
        left -= take * float(cost[i])
        if left <= 1e-12:
            break
    return min(total, float(cap.sum()))


def capped_simplex_projection(total, target, cap, cost=None, budget=None, iters=100):
    """argmin_e sum_i (e_i - target_i)^2  s.t.  sum e = total, 0 <= e <= cap, and cost.e <= budget if given.

    KKT form: e_i = clip(target_i + mu - lam * cost_i, 0, cap_i) with lam >= 0; mu is solved exactly, lam by
    bisection on the nonincreasing spend cost.e(lam). Returns None when infeasible.
    """
    target = np.asarray(target, float)
    cap = np.asarray(cap, float)
    total = float(total)
    if total < 0 or total > cap.sum() * (1 + 1e-12) + 1e-12:
        return None
    total = min(total, float(cap.sum()))
    e = _fill_for_mu(target, cap, total)
    if cost is None or budget is None or float(np.dot(cost, e)) <= budget + 1e-9:
        return e
    cost = np.asarray(cost, float)
    if min_cost_for_total(total, cap, cost) > budget + 1e-9:
        return None
    lo, hi = 0.0, 1.0
    while float(np.dot(cost, _fill_for_mu(target - hi * cost, cap, total))) > budget + 1e-9:
        hi *= 2.0
        if hi > 1e12:
            return None
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        if float(np.dot(cost, _fill_for_mu(target - lam * cost, cap, total))) > budget + 1e-9:
            lo = lam
        else:
            hi = lam
    return _fill_for_mu(target - hi * cost, cap, total)


def aligned_allocation(total, share, cap, cost=None, budget=None):
    """Bias-optimal allocation of a fixed total: projection of total * share onto the capped simplex."""
    return capped_simplex_projection(total, float(total) * np.asarray(share, float), cap, cost, budget)


def drop_below_min_event(eps, min_event=MIN_EVENT_EPSILON):
    eps = np.asarray(eps, float)
    return np.where(eps >= min_event, eps, 0.0)


def j_kappa(eps, share, kappa, min_event=MIN_EVENT_EPSILON):
    """P2 error per unit C_DP under Algorithm 2; clients below min_event do not upload."""
    e = drop_below_min_event(eps, min_event)
    total = float(e.sum())
    if total <= 0:
        return math.inf
    alpha = e / total
    return float(kappa) * float(np.square(alpha - np.asarray(share, float)).sum()) + float((e > 0).sum()) / total ** 2


def misalignment(eps, share, min_event=MIN_EVENT_EPSILON):
    """(TV, squared L2) between Algorithm-2 weights and data shares."""
    e = drop_below_min_event(eps, min_event)
    total = float(e.sum())
    w = e / total if total > 0 else np.zeros_like(e)
    share = np.asarray(share, float)
    return 0.5 * float(np.abs(w - share).sum()), float(np.square(w - share).sum())


def p2_oracle(cap, value, share, budget, kappa, n_grid=200, min_event=MIN_EVENT_EPSILON):
    """Non-strategic reference: best member of the aligned family under cost-only payments value.e <= budget.

    Scans the purchased total S on a uniform grid (S_max / n_grid, ..., S_max], with S_max the largest affordable
    total, projects S * share onto the capped simplex with the budget constraint, drops releases below min_event
    and keeps the allocation with the smallest J_kappa (ties: larger S). Returns (eps, payment, info).
    """
    cap, value, share = (np.asarray(x, float) for x in (cap, value, share))
    s_max = max_total_for_budget(cap, value, budget)
    best = None
    for s in np.linspace(s_max / n_grid, s_max, n_grid):
        e = aligned_allocation(s, share, cap, value, budget)
        if e is None:
            continue
        e = drop_below_min_event(e, min_event)
        j = j_kappa(e, share, kappa, min_event)
        if best is None or j < best[0] - 1e-15 or (abs(j - best[0]) <= 1e-15 and e.sum() > best[1].sum()):
            best = (j, e)
    if best is None:
        zero = np.zeros_like(cap)
        return zero, zero, dict(j=math.inf, total=0.0, s_max=s_max)
    j, e = best
    return e, value * e, dict(j=j, total=float(e.sum()), s_max=s_max)


def aligned_at_total(cap, value, share, total, budget, min_event=MIN_EVENT_EPSILON):
    """Equal-privacy reference: the same total as a given mechanism, distributed by data share (cost-only pay)."""
    cap, value, share = (np.asarray(x, float) for x in (cap, value, share))
    total = min(float(total), max_total_for_budget(cap, value, budget))
    e = aligned_allocation(total, share, cap, value, budget)
    if e is None:
        e = np.zeros_like(cap)
    e = drop_below_min_event(e, min_event)
    return e, value * e


# ---------------------------------------------------------------- training objective (torch, batched)

def j_kappa_torch(eps, share, kappa, min_event=MIN_EVENT_EPSILON, smooth=True):
    """Differentiable J_kappa for a batch (B, N). The winner count is smoothed as sum e / (e + min_event)."""
    total = eps.sum(-1).clamp_min(1e-8)
    alpha = eps / total.unsqueeze(-1)
    bias = (alpha - share).square().sum(-1)
    count = (eps / (eps + min_event)).sum(-1) if smooth else (eps >= min_event).to(eps.dtype).sum(-1)
    return kappa.reshape(-1) * bias + count / total.square()


def p2_training_term(executed_eps, reports, kappa, min_event=MIN_EVENT_EPSILON):
    """Buyer-side term of the augmented Lagrangian: J_kappa(executed) / J_kappa(all caps), reference detached.

    reports[..., 1] are caps and reports[..., 2] the (verified) sizes of the truthful profile.
    """
    share = reports[..., 2] / reports[..., 2].sum(-1, keepdim=True)
    reference = j_kappa_torch(reports[..., 1], share, kappa, min_event).detach()
    return j_kappa_torch(executed_eps, share, kappa, min_event) / reference


def sample_log_uniform(count, lo, hi, generator, device="cpu"):
    u = torch.rand(count, 1, generator=generator, device=device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def kappa_feature(kappa, lo, hi):
    """Network input for kappa: log-scale position in the training range, in [0, 1]."""
    return ((torch.log(kappa) - math.log(lo)) / (math.log(hi) - math.log(lo))).clamp(0.0, 1.0)


# ---------------------------------------------------------------- kappa-hat estimator (Stage -1)

def dp_noise_energy(noise_multiplier, sample_rate, optimizer_steps, dataset_size, lr, clip, n_params):
    """Expected squared norm of the DP-SGD noise in one local update.

    Opacus adds N(0, (sigma * clip)^2) per coordinate to the summed clipped gradients and divides by the
    expected batch size q * n, so each step contributes lr^2 (sigma * clip / (q * n))^2 per coordinate.
    """
    expected_batch = float(sample_rate) * float(dataset_size)
    return float(n_params) * float(lr) ** 2 * int(optimizer_steps) * (float(noise_multiplier) * float(clip)
                                                                       / expected_batch) ** 2


def kappa_from_updates(updates, noise_energy):
    """kappa-hat = zeta^2 / C with zeta^2 = sum_i ||u_i - mean u||^2 / (N - 1) over clean clipped updates
    taken from the same global model, and C the mean DP noise energy at event epsilon 1."""
    u = np.asarray(updates, float)
    centred = u - u.mean(0, keepdims=True)
    zeta2 = float(np.square(centred).sum() / (u.shape[0] - 1))
    c_dp = float(np.mean(noise_energy))
    return dict(zeta2=zeta2, c_dp=c_dp, kappa=zeta2 / c_dp)

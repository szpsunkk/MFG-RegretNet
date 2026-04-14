"""
CSRA-QMS (Client Selection with Reverse Auction — Quality Maximization Selection) baseline.
Ref: baselines/CSRA.md and [1] IEEE TIFS 2024.

One-shot version: p(H_i)=1 (no history). Quality q_i = (ε_i / Σε_m) * (|D_i| / Σ|D_i|).
Bid/cost b_i = v_i * ε_i. Sort by q_i/b_i descending; select largest k s.t.
(b_{k+1}/q_{k+1}) * Σ_{i≤k} q_i ≤ B. Payments r_i = (b_{k+1}/q_{k+1}) * q_i for winners.
Plosses: winners get ε_alloc = 1/(n-k) (consistent with PAC/VCG).
"""
import torch
import numpy as np


def _csra_single(v, eps, sizes, B, cost_fn=None):
    """
    Single profile. v, eps, sizes: (n_agents,). B scalar.
    Returns plosses (n_agents,), payments (n_agents).
    """
    if cost_fn is None:
        cost_fn = lambda v, e: v * e
    n = v.shape[0]
    if n == 0:
        return np.zeros(n), np.zeros(n)
    v_np = np.asarray(v)
    eps_np = np.asarray(eps)
    sizes_np = np.asarray(sizes, dtype=np.float64)
    B = float(B)
    sum_eps = eps_np.sum()
    sum_sizes = sizes_np.sum()
    if sum_eps <= 0 or sum_sizes <= 0:
        return np.zeros(n), np.zeros(n)
    q = (eps_np / sum_eps) * (sizes_np / sum_sizes)
    b = np.array([cost_fn(v_np[i], eps_np[i]) for i in range(n)])
    # Sort by q_i/b_i descending（避免 np.where 仍先算 q/b 触发除零警告）
    ratio = np.zeros_like(q, dtype=np.float64)
    np.divide(q, b, out=ratio, where=b > 0)
    idx_sorted = np.argsort(-ratio)
    q_sorted = q[idx_sorted]
    b_sorted = b[idx_sorted]
    # Largest k s.t. (b_sorted[k]/q_sorted[k]) * sum(q_sorted[0:k]) <= B
    k = 0
    for k_cand in range(1, n):
        if q_sorted[k_cand] <= 0:
            continue
        total_q = q_sorted[:k_cand].sum()
        if total_q <= 0:
            continue
        scale = b_sorted[k_cand] / q_sorted[k_cand]
        if scale * total_q <= B:
            k = k_cand
    plosses_np = np.zeros(n)
    payments_np = np.zeros(n)
    if k > 0:
        eps_alloc = 1.0 / (n - k)
        plosses_np[idx_sorted[:k]] = eps_alloc
        scale = b_sorted[k] / q_sorted[k] if q_sorted[k] > 0 else 0
        for i in range(k):
            payments_np[idx_sorted[i]] = scale * q_sorted[i]
    return plosses_np, payments_np


def csra_qms_batch(reports, budget, cost_fn=None):
    """
    Batch CSRA-QMS. reports: (batch, n_agents, n_items+2).
    reports[:,:,0] = valuation v, [..., -2] = epsilon, [..., -1] = size.
    budget: (batch, 1).
    Returns plosses (batch, n_agents), payments (batch, n_agents).
    """
    if cost_fn is None:
        cost_fn = lambda v, eps: v * eps
    device = reports.device if isinstance(reports, torch.Tensor) else None
    batch_size = reports.shape[0]
    n_agents = reports.shape[1]
    v = reports[:, :, 0]
    eps = reports[:, :, -2]
    sizes = reports[:, :, -1]
    if isinstance(reports, torch.Tensor):
        v_np = v.detach().cpu().numpy()
        eps_np = eps.detach().cpu().numpy()
        sizes_np = sizes.detach().cpu().numpy()
        budget_np = budget.detach().cpu().numpy().reshape(-1)
    else:
        v_np = np.asarray(v)
        eps_np = np.asarray(eps)
        sizes_np = np.asarray(sizes)
        budget_np = np.asarray(budget).reshape(-1)
    plosses_list = []
    payments_list = []
    for b in range(batch_size):
        pl, pay = _csra_single(v_np[b], eps_np[b], sizes_np[b], budget_np[b], cost_fn=cost_fn)
        plosses_list.append(pl)
        payments_list.append(pay)
    plosses = np.stack(plosses_list, axis=0)
    payments = np.stack(payments_list, axis=0)
    if device is not None:
        plosses = torch.tensor(plosses, dtype=torch.float32, device=device)
        payments = torch.tensor(payments, dtype=torch.float32, device=device)
    return plosses, payments

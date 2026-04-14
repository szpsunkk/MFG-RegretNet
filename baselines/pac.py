"""
PAC mechanism (Algorithm 1) from privacy paper.
Threshold-based procurement: sort v ascending, select largest k s.t. c(v_k, 1/(n-k)) <= B/k,
payments p_i = min(B/k, c(v_{k+1}, 1/(n-k))) for i<=k, else 0. Cost c(v, eps) = v * eps.
"""
import torch
import numpy as np


def _pac_single(v, B, cost_fn=None):
    """
    Single profile: v (n_agents,), B scalar.
    Returns plosses (n_agents,), payments (n_agents).
    """
    if cost_fn is None:
        cost_fn = lambda v, eps: v * eps
    n = v.shape[0]
    if n == 0:
        return np.zeros(n), np.zeros(n)
    v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    B = float(B)
    idx_sorted = np.argsort(v_np)
    v_sorted = v_np[idx_sorted]
    # k = largest in {1, ..., n-1} s.t. c(v_sorted[k-1], 1/(n-k)) <= B/k  =>  v_sorted[k-1]/(n-k) <= B/k
    k = 0
    for k_cand in range(1, n):
        if (n - k_cand) <= 0:
            continue
        if cost_fn(v_sorted[k_cand - 1], 1.0 / (n - k_cand)) <= B / k_cand:
            k = k_cand
    plosses_np = np.zeros(n)
    payments_np = np.zeros(n)
    if k > 0:
        eps_alloc = 1.0 / (n - k)
        plosses_np[idx_sorted[:k]] = eps_alloc
        if k < n:
            pay_val = min(B / k, cost_fn(v_sorted[k], eps_alloc))
        else:
            pay_val = B / k
        payments_np[idx_sorted[:k]] = pay_val
    return plosses_np, payments_np


def pac_batch(reports, budget, cost_fn=None):
    """
    Batch PAC. reports: (batch, n_agents, n_items+2), valuation = reports[:,:,0].
    budget: (batch, 1).
    Returns plosses (batch, n_agents), payments (batch, n_agents).
    """
    if cost_fn is None:
        cost_fn = lambda v, eps: v * eps
    device = reports.device if isinstance(reports, torch.Tensor) else None
    batch_size = reports.shape[0]
    n_agents = reports.shape[1]
    v = reports[:, :, 0]
    if isinstance(reports, torch.Tensor):
        v_np = v.detach().cpu().numpy()
        budget_np = budget.detach().cpu().numpy().reshape(-1)
    else:
        v_np = np.asarray(v)
        budget_np = np.asarray(budget).reshape(-1)
    plosses_list = []
    payments_list = []
    for b in range(batch_size):
        pl, pay = _pac_single(v_np[b], budget_np[b], cost_fn=cost_fn)
        plosses_list.append(pl)
        payments_list.append(pay)
    plosses = np.stack(plosses_list, axis=0)
    payments = np.stack(payments_list, axis=0)
    if device is not None:
        plosses = torch.tensor(plosses, dtype=torch.float32, device=device)
        payments = torch.tensor(payments, dtype=torch.float32, device=device)
    return plosses, payments

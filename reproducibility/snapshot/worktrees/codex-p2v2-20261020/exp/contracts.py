"""Shared contract semantics for training, attacks and actual execution.

No partial delivery at the original payment: a contract is either accepted as
offered or refused. Reported types and privately checked true types are distinct.
DP calibration is an additional exact gate supplied by the deployment runner.
"""
import torch

MIN_EVENT_EPSILON = .01


def settlement(true_types, exposure, payment, calibration_ok=None):
    utility = payment - true_types[..., 0] * exposure
    feasible = ((exposure >= MIN_EVENT_EPSILON)
                & (exposure <= true_types[..., 1] + 1e-7)
                & (utility >= -1e-9) & (payment >= 0)
                & torch.isfinite(exposure) & torch.isfinite(payment))
    if calibration_ok is not None:
        feasible = feasible & calibration_ok
    return (torch.where(feasible, exposure, 0.),
            torch.where(feasible, payment, 0.), feasible)


def utility(true_types, exposure, payment, calibration_ok=None):
    executed, paid, _ = settlement(true_types, exposure, payment, calibration_ok)
    return paid - true_types[..., 0] * executed


def balanced_target(reports, budget, ceiling=1., exponent=.5):
    """Public teacher/control: cap-aware water filling, unit payment per epsilon.

    This is a training reference, not the deployed neural payment formula.
    Public size weights are fixed under deviations. No raw labels are inputs.
    """
    cap = reports[..., 1]
    weights = reports[..., 2].clamp_min(1).pow(exponent)
    weights = weights / weights.mean(-1, keepdim=True)
    total = torch.minimum(budget.reshape(-1, 1), torch.full_like(budget.reshape(-1, 1), cap.shape[-1] * ceiling))
    total = torch.minimum(total, cap.sum(-1, keepdim=True))
    lo = torch.zeros_like(total)
    hi = (cap / weights).max(-1, keepdim=True).values
    for _ in range(45):
        mid = (lo + hi) / 2
        below = torch.minimum(cap, mid * weights).sum(-1, keepdim=True) < total
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    exposure = torch.minimum(cap, (lo + hi) / 2 * weights)
    return exposure, exposure.clone()

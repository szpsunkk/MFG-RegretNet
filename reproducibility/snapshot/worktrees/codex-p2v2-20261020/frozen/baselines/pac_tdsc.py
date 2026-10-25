"""Single-parameter PAC reference mechanism used by the TDSC revision.

This module is intentionally separate from the legacy adaptive-privacy PAC
implementation.  The privacy contract, budget, capacity, and public tie order
are fixed before unit-cost reports are submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


TensorOrNumber = Union[torch.Tensor, float, int]


@dataclass(frozen=True)
class PACOutcome:
    winner: torch.Tensor
    actual_epsilon: torch.Tensor
    payment: torch.Tensor
    critical_unit_price: torch.Tensor
    reserve_unit_price: torch.Tensor


def _batch_vector(
    value: TensorOrNumber,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    elif tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    return tensor


def single_parameter_pac(
    unit_cost_report: torch.Tensor,
    budget: TensorOrNumber,
    capacity: TensorOrNumber,
    epsilon_bar: TensorOrNumber,
    priority_key: torch.Tensor,
    *,
    eligible: torch.Tensor | None = None,
) -> PACOutcome:
    """Run homogeneous-contract PAC with report-independent tie breaking.

    Reports are unit costs.  Each winner receives the public privacy loss
    ``epsilon_bar`` and its critical total payment.  Ineligible clients are
    excluded before reports are ranked and cannot set the critical price.
    """

    reports = torch.as_tensor(unit_cost_report)
    if reports.ndim == 1:
        reports = reports.unsqueeze(0)
        squeeze = True
    elif reports.ndim == 2:
        squeeze = False
    else:
        raise ValueError("unit_cost_report must have shape (N,) or (batch,N)")
    if not reports.is_floating_point():
        reports = reports.to(torch.float64)
    if bool((~torch.isfinite(reports)).any()) or bool((reports < 0.0).any()):
        raise ValueError("unit-cost reports must be finite and nonnegative")

    batch_size, n_clients = reports.shape
    device, dtype = reports.device, reports.dtype
    monetary_budget = _batch_vector(
        budget, batch_size, device=device, dtype=dtype, name="budget"
    )
    contract_loss = _batch_vector(
        epsilon_bar,
        batch_size,
        device=device,
        dtype=dtype,
        name="epsilon_bar",
    )
    slots = _batch_vector(
        capacity,
        batch_size,
        device=device,
        dtype=torch.long,
        name="capacity",
    )
    if bool((monetary_budget < 0.0).any()):
        raise ValueError("budget must be nonnegative")
    if bool((contract_loss <= 0.0).any()):
        raise ValueError("epsilon_bar must be positive")
    if bool((slots < 0).any()) or bool((slots > n_clients).any()):
        raise ValueError("capacity must lie in [0,N]")

    keys = torch.as_tensor(priority_key, device=device)
    if keys.ndim == 1:
        keys = keys.unsqueeze(0).expand(batch_size, -1)
    if keys.shape != reports.shape:
        raise ValueError("priority_key must have shape (N,) or (batch,N)")
    sorted_keys, _ = torch.sort(keys, dim=1)
    if n_clients > 1 and bool((sorted_keys[:, 1:] == sorted_keys[:, :-1]).any()):
        raise ValueError("priority keys must be pairwise distinct per profile")

    if eligible is None:
        eligibility = torch.ones_like(reports, dtype=torch.bool)
    else:
        eligibility = torch.as_tensor(eligible, device=device, dtype=torch.bool)
        if eligibility.shape != reports.shape:
            raise ValueError("eligible must have shape (batch,N)")

    winner = torch.zeros_like(reports, dtype=torch.bool)
    actual_epsilon = torch.zeros_like(reports)
    payment = torch.zeros_like(reports)
    critical = torch.zeros(batch_size, device=device, dtype=dtype)
    reserve = torch.where(
        slots > 0,
        monetary_budget / (slots.to(dtype) * contract_loss),
        torch.zeros_like(monetary_budget),
    )

    # The reference path is deliberately explicit.  It doubles as an
    # independently inspectable oracle for boundary and tie tests.
    for profile in range(batch_size):
        k = int(slots[profile].item())
        if k == 0:
            continue
        candidates = [
            client
            for client in range(n_clients)
            if bool(eligibility[profile, client])
        ]
        candidates.sort(
            key=lambda client: (
                float(reports[profile, client].item()),
                int(keys[profile, client].item()),
            )
        )
        kth_loser = (
            reports[profile, candidates[k]]
            if len(candidates) > k
            else torch.tensor(float("inf"), device=device, dtype=dtype)
        )
        threshold = torch.minimum(reserve[profile], kth_loser)
        critical[profile] = threshold
        selected = [
            client
            for client in candidates[:k]
            if reports[profile, client] <= reserve[profile]
        ]
        if selected:
            selected_tensor = torch.tensor(selected, device=device, dtype=torch.long)
            winner[profile, selected_tensor] = True
            actual_epsilon[profile, selected_tensor] = contract_loss[profile]
            payment[profile, selected_tensor] = contract_loss[profile] * threshold

    outcome = PACOutcome(
        winner=winner,
        actual_epsilon=actual_epsilon,
        payment=payment,
        critical_unit_price=critical,
        reserve_unit_price=reserve,
    )
    if not squeeze:
        return outcome
    return PACOutcome(
        winner=outcome.winner[0],
        actual_epsilon=outcome.actual_epsilon[0],
        payment=outcome.payment[0],
        critical_unit_price=outcome.critical_unit_price[0],
        reserve_unit_price=outcome.reserve_unit_price[0],
    )


def pac_utility(
    true_unit_cost: torch.Tensor,
    outcome: PACOutcome,
) -> torch.Tensor:
    cost = torch.as_tensor(
        true_unit_cost,
        device=outcome.payment.device,
        dtype=outcome.payment.dtype,
    )
    return outcome.payment - cost * outcome.actual_epsilon

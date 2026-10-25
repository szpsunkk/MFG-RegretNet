"""Report-independent posted-price fallback for privacy procurement.

The server computes recipients, unit prices, and offered privacy losses from
public inputs only.  A client's private unit cost and privacy cap determine
only whether it accepts and how much of the public offer can be released.
Refused offers are not backfilled, so no client's private report changes
another client's opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class PublicPostedPriceParameters:
    budget: float
    capacity: int
    unit_price_floor: float
    unit_price_ceiling: float
    epsilon_floor: float
    epsilon_ceiling: float
    quality_bounds: Tuple[float, float] = (0.0, 1.0)
    data_size_bounds: Tuple[float, float] = (1.0, 1000.0)
    price_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    epsilon_weights: Tuple[float, float, float] = (0.5, 0.2, 0.3)
    rank_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    tolerance: float = 1.0e-10

    @property
    def worst_case_payment(self) -> float:
        return self.capacity * self.unit_price_ceiling * self.epsilon_ceiling

    def validate(self, n_clients: int) -> None:
        scalar_values = (
            self.budget,
            self.unit_price_floor,
            self.unit_price_ceiling,
            self.epsilon_floor,
            self.epsilon_ceiling,
            self.tolerance,
            *self.quality_bounds,
            *self.data_size_bounds,
            *self.price_weights,
            *self.epsilon_weights,
            *self.rank_weights,
        )
        if not all(torch.isfinite(torch.tensor(value)).item() for value in scalar_values):
            raise ValueError("posted-price parameters must be finite")
        if n_clients <= 0:
            raise ValueError("n_clients must be positive")
        if self.capacity < 0 or self.capacity > n_clients:
            raise ValueError("capacity must lie in [0,N]")
        if self.budget < 0.0 or self.tolerance < 0.0:
            raise ValueError("budget and tolerance must be nonnegative")
        if not 0.0 <= self.unit_price_floor <= self.unit_price_ceiling:
            raise ValueError("unit-price bounds must be ordered and nonnegative")
        if not 0.0 <= self.epsilon_floor <= self.epsilon_ceiling:
            raise ValueError("epsilon bounds must be ordered and nonnegative")
        if self.quality_bounds[0] >= self.quality_bounds[1]:
            raise ValueError("quality bounds must have positive width")
        if self.data_size_bounds[0] >= self.data_size_bounds[1]:
            raise ValueError("data-size bounds must have positive width")
        for name, weights in (
            ("price_weights", self.price_weights),
            ("epsilon_weights", self.epsilon_weights),
            ("rank_weights", self.rank_weights),
        ):
            if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
                raise ValueError(f"{name} must be nonnegative with positive sum")
        if self.worst_case_payment > self.budget + self.tolerance:
            raise ValueError(
                "parameters violate worst-case budget feasibility: "
                f"K*p_max*epsilon_max={self.worst_case_payment} > B={self.budget}"
            )


@dataclass(frozen=True)
class PublicPostedPriceOutcome:
    selected_for_offer: torch.Tensor
    accepted: torch.Tensor
    offered_unit_price: torch.Tensor
    offered_epsilon: torch.Tensor
    actual_epsilon: torch.Tensor
    payment: torch.Tensor
    public_rank_score: torch.Tensor
    remaining: torch.Tensor


@dataclass(frozen=True)
class GridDeviationCertificate:
    truthful_utility: torch.Tensor
    best_deviation_utility: torch.Tensor
    regret: torch.Tensor
    deviations_per_client: int
    cap_deviation_domain: str = "reported_cap in [0, true_cap]"


def _profile_matrix(value: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    elif tensor.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"{name} must have shape (N,) or (batch,N)")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float64)
    return tensor, squeeze


def _same_shape(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=dtype)
    if tensor.ndim == 1 and reference.shape[0] > 1:
        tensor = tensor.unsqueeze(0).expand(reference.shape[0], -1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must match public_quality shape")
    return tensor


def _normalized(
    value: torch.Tensor, bounds: Tuple[float, float], name: str
) -> torch.Tensor:
    lower, upper = bounds
    if bool((value < lower).any()) or bool((value > upper).any()):
        raise ValueError(f"{name} lies outside its public bounds")
    return (value - lower) / (upper - lower)


def _weighted_score(
    quality: torch.Tensor,
    data_size: torch.Tensor,
    remaining: torch.Tensor,
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    denominator = float(sum(weights))
    return (
        weights[0] * quality + weights[1] * data_size + weights[2] * remaining
    ) / denominator


def public_posted_price(
    public_quality: torch.Tensor,
    public_data_size: torch.Tensor,
    public_remaining: torch.Tensor,
    availability: torch.Tensor,
    priority_key: torch.Tensor,
    reported_unit_cost: torch.Tensor,
    reported_cap: torch.Tensor,
    parameters: PublicPostedPriceParameters,
) -> PublicPostedPriceOutcome:
    """Evaluate the public posted-price mechanism.

    ``reported_unit_cost`` and ``reported_cap`` are intentionally excluded
    from offer construction and public ranking.  A cap above the client's true
    privacy cap is not an admissible deviation; the mechanism cannot verify
    that physical constraint by itself.
    """

    quality, squeeze = _profile_matrix(public_quality, "public_quality")
    if not quality.is_floating_point():
        quality = quality.to(torch.float64)
    batch_size, n_clients = quality.shape
    parameters.validate(n_clients)
    dtype, device = quality.dtype, quality.device
    data_size = _same_shape(
        public_data_size, quality, "public_data_size", dtype=dtype
    )
    remaining = _same_shape(
        public_remaining, quality, "public_remaining", dtype=dtype
    )
    available = _same_shape(
        availability, quality, "availability", dtype=torch.bool
    )
    keys = _same_shape(priority_key, quality, "priority_key")
    unit_cost = _same_shape(
        reported_unit_cost, quality, "reported_unit_cost", dtype=dtype
    )
    cap = _same_shape(reported_cap, quality, "reported_cap", dtype=dtype)

    for name, value in (
        ("public_quality", quality),
        ("public_data_size", data_size),
        ("public_remaining", remaining),
        ("priority_key", keys),
        ("reported_unit_cost", unit_cost),
        ("reported_cap", cap),
    ):
        if bool((~torch.isfinite(value)).any()):
            raise ValueError(f"{name} must be finite")
    if bool((remaining < 0.0).any()):
        raise ValueError("public_remaining must be nonnegative")
    if bool((unit_cost < 0.0).any()) or bool((cap < 0.0).any()):
        raise ValueError("private reports must be nonnegative")
    sorted_keys, _ = torch.sort(keys, dim=1)
    if n_clients > 1 and bool((sorted_keys[:, 1:] == sorted_keys[:, :-1]).any()):
        raise ValueError("priority keys must be pairwise distinct per profile")

    quality_norm = _normalized(quality, parameters.quality_bounds, "public_quality")
    data_size_norm = _normalized(
        data_size, parameters.data_size_bounds, "public_data_size"
    )
    remaining_norm = torch.clamp(
        remaining / max(parameters.epsilon_ceiling, torch.finfo(dtype).eps),
        min=0.0,
        max=1.0,
    )
    price_score = _weighted_score(
        quality_norm, data_size_norm, remaining_norm, parameters.price_weights
    )
    epsilon_score = _weighted_score(
        quality_norm, data_size_norm, remaining_norm, parameters.epsilon_weights
    )
    rank_score = _weighted_score(
        quality_norm, data_size_norm, remaining_norm, parameters.rank_weights
    )
    unit_price = parameters.unit_price_floor + (
        parameters.unit_price_ceiling - parameters.unit_price_floor
    ) * price_score
    offered_epsilon = parameters.epsilon_floor + (
        parameters.epsilon_ceiling - parameters.epsilon_floor
    ) * epsilon_score

    selected = torch.zeros_like(available)
    for profile in range(batch_size):
        candidates = [
            client
            for client in range(n_clients)
            if bool(available[profile, client]) and remaining[profile, client] > 0.0
        ]
        candidates.sort(
            key=lambda client: (
                -float(rank_score[profile, client].item()),
                float(keys[profile, client].item()),
            )
        )
        if parameters.capacity:
            selected[profile, candidates[: parameters.capacity]] = True

    accepted = (
        selected
        & available
        & (remaining > 0.0)
        & (cap > 0.0)
        & (unit_price >= unit_cost)
    )
    feasible_epsilon = torch.minimum(
        torch.minimum(offered_epsilon, cap), remaining
    )
    actual_epsilon = torch.where(accepted, feasible_epsilon, 0.0)
    payment = unit_price * actual_epsilon
    if bool(
        (payment.sum(dim=1) > parameters.budget + parameters.tolerance).any()
    ):
        raise AssertionError("worst-case budget-feasibility invariant failed")

    outcome = PublicPostedPriceOutcome(
        selected_for_offer=selected,
        accepted=accepted,
        offered_unit_price=unit_price,
        offered_epsilon=offered_epsilon,
        actual_epsilon=actual_epsilon,
        payment=payment,
        public_rank_score=rank_score,
        remaining=remaining,
    )
    if not squeeze:
        return outcome
    return PublicPostedPriceOutcome(
        **{name: value[0] for name, value in outcome.__dict__.items()}
    )


def posted_price_utility(
    true_unit_cost: torch.Tensor, outcome: PublicPostedPriceOutcome
) -> torch.Tensor:
    cost = torch.as_tensor(
        true_unit_cost, device=outcome.payment.device, dtype=outcome.payment.dtype
    )
    return outcome.payment - cost * outcome.actual_epsilon


def one_sided_grid_deviation_certificate(
    true_unit_cost: torch.Tensor,
    true_cap: torch.Tensor,
    truthful_outcome: PublicPostedPriceOutcome,
    *,
    unit_cost_steps: int = 31,
    cap_steps: int = 21,
    unit_cost_grid_max: float | None = None,
) -> GridDeviationCertificate:
    """Search unit-cost reports and downward-only privacy-cap reports.

    The cap grid is one-sided because reports above the true privacy cap are
    physically infeasible rather than strategic actions.  Offers and selected
    recipients stay fixed throughout the search by mechanism construction.
    """

    if unit_cost_steps < 2 or cap_steps < 2:
        raise ValueError("both deviation grids require at least two points")
    squeeze = truthful_outcome.payment.ndim == 1

    def matrix(value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(
            value,
            device=truthful_outcome.payment.device,
            dtype=truthful_outcome.payment.dtype,
        )
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    true_cost = matrix(true_unit_cost)
    cap = matrix(true_cap)
    price = matrix(truthful_outcome.offered_unit_price)
    offer = matrix(truthful_outcome.offered_epsilon)
    remaining = matrix(truthful_outcome.remaining)
    selected = matrix(truthful_outcome.selected_for_offer).bool()
    if true_cost.shape != price.shape or cap.shape != price.shape:
        raise ValueError("true types must match the outcome shape")
    if bool((true_cost < 0.0).any()) or bool((cap < 0.0).any()):
        raise ValueError("true types must be nonnegative")

    maximum = unit_cost_grid_max
    if maximum is None:
        maximum = 1.25 * max(
            float(true_cost.max().item()), float(price.max().item()), 1.0e-6
        ) + 1.0e-6
    if maximum <= float(price.max().item()):
        raise ValueError("unit_cost_grid_max must exceed every offered unit price")
    reports = torch.linspace(
        0.0,
        float(maximum),
        unit_cost_steps,
        device=price.device,
        dtype=price.dtype,
    )
    cap_fractions = torch.linspace(
        0.0, 1.0, cap_steps, device=price.device, dtype=price.dtype
    )
    deviation_caps = cap[..., None] * cap_fractions
    deviation_epsilon = torch.minimum(
        torch.minimum(offer[..., None], deviation_caps), remaining[..., None]
    )
    accepts = (
        selected[..., None, None]
        & (remaining[..., None, None] > 0.0)
        & (deviation_caps[..., None, :] > 0.0)
        & (price[..., None, None] >= reports[None, None, :, None])
    )
    deviation_utility = torch.where(
        accepts,
        (price - true_cost)[..., None, None]
        * deviation_epsilon[..., None, :],
        0.0,
    )
    best = deviation_utility.amax(dim=(-2, -1))
    truthful = matrix(posted_price_utility(true_cost, truthful_outcome))
    regret = torch.clamp(best - truthful, min=0.0)
    if squeeze:
        truthful, best, regret = truthful[0], best[0], regret[0]
    return GridDeviationCertificate(
        truthful_utility=truthful,
        best_deviation_utility=best,
        regret=regret,
        deviations_per_client=unit_cost_steps * cap_steps,
    )

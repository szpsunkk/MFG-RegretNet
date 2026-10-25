"""Lin Sec. V/Alg. 2 menu with explicit common-event procurement adaptation.

The native solver is unchanged. This module preserves its menu and payment
recursion, scales both coordinates before reports, and checks finite choices.
It does not implement Lin's original local-model perturbation FL algorithm.
"""
from dataclasses import dataclass

import numpy as np

from .mechanism import ContractMenu, solve_discrete_contracts


@dataclass(frozen=True)
class CommonMenu:
    native: ContractMenu
    epsilon: np.ndarray
    payment: np.ndarray
    scale: float
    active: np.ndarray
    budget_limited: bool


@dataclass(frozen=True)
class MenuOutcome:
    item: np.ndarray
    offered: np.ndarray
    epsilon: np.ndarray
    payment: np.ndarray
    best_utility: np.ndarray


def build_common_menu(protocol, *, candidate_ceiling, budget, capacity):
    if not np.isfinite([candidate_ceiling, budget]).all() or candidate_ceiling <= 0 or budget < 0:
        raise ValueError("finite positive candidate and nonnegative budget required")
    if int(capacity) != capacity or capacity < 0:
        raise ValueError("capacity must be a nonnegative integer")
    floor = float(protocol["minimum_event_loss"])
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("positive finite event floor required")
    if float(protocol["alpha"]) != 1.0:
        raise ValueError("common monetary utility requires registered alpha=1")
    native = solve_discrete_contracts(
        protocol["types"], protocol["probabilities"], protocol["alpha"],
        protocol["z1"], protocol["z2"])
    privacy_scale = float(candidate_ceiling) / float(native.epsilon.max())
    budget_scale = float(budget) / (capacity * float(native.payment.max())) if capacity else 0.0
    scale = min(privacy_scale, budget_scale)
    epsilon, payment = native.epsilon * scale, native.payment * scale
    return CommonMenu(native, epsilon, payment, scale, epsilon >= floor,
                      budget_scale < privacy_scale)


def choose_menu(menu, *, costs, caps, available, priority, public_feasible, capacity):
    """Public lottery first; private maximisation only changes one's own item.

    Exit has utility zero. Tie keys/availability/feasible rows are held fixed in
    single-round deviations. No report-dependent repricing or replacement occurs.
    """
    costs, caps = np.asarray(costs, float), np.asarray(caps, float)
    n, m = len(costs), len(menu.epsilon)
    available, priority = np.asarray(available, bool), np.asarray(priority)
    public_feasible = np.asarray(public_feasible, bool)
    if costs.shape != (n,) or caps.shape != (n,) or available.shape != (n,) or priority.shape != (n,):
        raise ValueError("client vectors must be one dimensional and match")
    if not np.isfinite(costs).all() or not np.isfinite(caps).all() or np.any(costs < 0) or np.any(caps < 0):
        raise ValueError("finite nonnegative costs and caps required")
    if public_feasible.shape != (n, m) or len(np.unique(priority)) != n:
        raise ValueError("feasible matrix or unique public priorities invalid")
    if capacity < 0 or int(capacity) != capacity:
        raise ValueError("invalid capacity")
    public_feasible = public_feasible & menu.active[None, :]
    public_eligible = available & public_feasible.any(axis=1)
    candidates = np.flatnonzero(public_eligible)
    offered_ids = candidates[np.argsort(priority[candidates], kind="stable")[:int(capacity)]]
    offered = np.zeros(n, bool)
    offered[offered_ids] = True
    utilities = menu.payment[None, :] - costs[:, None] * menu.epsilon[None, :]
    permitted = public_feasible & (menu.epsilon[None, :] <= caps[:, None])
    # The source menu is in descending epsilon order, so first argmax implements
    # the registered exact-tie rule without a near-tie tolerance.
    feasible_utilities = np.where(permitted, utilities, -np.inf)
    item = np.argmax(feasible_utilities, axis=1)
    best = feasible_utilities[np.arange(n), item]
    accepted = offered & (best >= 0)
    item = np.where(accepted, item, -1)
    return MenuOutcome(
        item, offered, np.where(accepted, menu.epsilon[np.maximum(item, 0)], 0.0),
        np.where(accepted, menu.payment[np.maximum(item, 0)], 0.0),
        np.where(accepted, best, 0.0))


def epsilon_from_states(states, orders, delta):
    """Vectorised pinned Opacus conversion; zero-event states retain epsilon=0."""
    states, orders = np.asarray(states, float), np.asarray(orders, float)
    if states.shape[-1] != len(orders) or not 0 < delta < 1:
        raise ValueError("invalid RDP shape or delta")
    if not np.isfinite(states).all() or np.any(states < 0) or np.any(orders <= 1):
        raise ValueError("invalid RDP state/orders")
    values = states - (np.log(delta) + np.log(orders)) / (orders - 1)
    values += np.log((orders - 1) / orders)
    epsilon = np.maximum(0, np.min(values, axis=-1))
    return np.where(np.any(states != 0, axis=-1), epsilon, 0.0)


def event_vectors(menu, *, sample_rate, steps, delta, orders):
    from tdsc_dpsgd_runner import allocated_event_rdp
    vectors = np.zeros((len(menu.epsilon), len(orders)))
    events = []
    cache = {}
    for index, loss in enumerate(menu.epsilon):
        if not menu.active[index]:
            events.append(None)
            continue
        key = float(loss)
        if key not in cache:
            cache[key] = allocated_event_rdp(
                allocated_event_epsilon=key, sample_rate=sample_rate,
                optimizer_steps=steps, event_delta=delta, orders=orders)
        _, calibration, vectors[index] = cache[key]
        events.append(calibration)
    return vectors, events


def replay(menu, vectors, *, assets, config, target, retain=False):
    """Full-horizon, training-free replay; retain all finite outcomes if requested."""
    n, order_count = len(assets.unit_cost), vectors.shape[1]
    states = np.zeros((n, order_count))
    privacy, schedule = config["privacy_contract"], config["shared_schedule"]
    rounds = int(config["experiment"]["rounds"])
    decisions, history = [], []
    releases, payment, cost, max_payment, min_surplus = 0, 0.0, 0.0, 0.0, 0.0
    for t in range(rounds):
        feasible = epsilon_from_states(states[:, None, :] + vectors[None, :, :],
                                       privacy["orders"], privacy["delta"]) <= target + config["acceptance"]["cap_tolerance"]
        outcome = choose_menu(
            menu, costs=assets.unit_cost, caps=assets.maximum_acceptable_event_loss,
            available=assets.availability[t], priority=assets.priority_key[t],
            public_feasible=feasible, capacity=schedule["capacity"])
        ids = np.flatnonzero(outcome.item >= 0)
        states[ids] += vectors[outcome.item[ids]]
        round_payment = float(outcome.payment.sum())
        releases += len(ids)
        payment += round_payment
        cost += float(np.asarray(assets.unit_cost) @ outcome.epsilon)
        max_payment = max(max_payment, round_payment)
        min_surplus = min(min_surplus, float(outcome.best_utility.min()))
        decisions.append(outcome.item.tolist())
        if retain:
            history.append(outcome)
    epsilon = epsilon_from_states(states, privacy["orders"], privacy["delta"])
    # Use the same conservative observed-value quantiles as the shared backend.
    summary = dict(
        epsilon_max=float(epsilon.max()),
        epsilon_p50=float(np.quantile(epsilon, .5, method="higher")),
        epsilon_p95=float(np.quantile(epsilon, .95, method="higher")),
        match_error=abs(float(epsilon.max()) - target), release_count=releases,
        payment_total=payment, privacy_cost_total=cost,
        client_surplus_total=payment-cost, max_round_payment=max_payment,
        minimum_accepted_surplus=min_surplus,
        budget_limited=menu.budget_limited,
        public_active_items=int(menu.active.sum()))
    return summary, decisions, states, history

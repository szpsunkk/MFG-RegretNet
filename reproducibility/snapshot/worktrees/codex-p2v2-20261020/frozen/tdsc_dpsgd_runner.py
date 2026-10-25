#!/usr/bin/env python3
"""Record-level DP-SGD federated-learning core for the TDSC revision.

This module is intentionally separate from :mod:`tdsc_fl_runner`.  The older
runner clips and randomises a complete local update.  Here, Opacus 1.5.x
performs Poisson-sampled DP-SGD inside each selected client: it computes
per-sample gradients, clips them, and adds Gaussian noise on every optimiser
step.  The resulting local model delta is post-processing and is therefore
aggregated without clipping it again or adding a second noise vector.

Opacus' sampled-Gaussian RDP accountant is labelled here as an
``add-or-remove-one-record`` guarantee.  A replace-one record change can be
connected by two add/remove changes, but that conservative conversion changes
both epsilon and delta.  This module never relabels an Opacus epsilon as a
same-delta replace-one guarantee.

The deterministic generators are for reproducible research experiments.  They
are not cryptographically secure RNGs and must not be cited as a production
privacy implementation.  Smoke artifacts are explicitly marked
``DIAGNOSTIC/NONPUBLICATION``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import opacus
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import PRVAccountant, RDPAccountant
from opacus.accountants.analysis import rdp as opacus_rdp
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
from scipy.special import gammaln, logsumexp
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


TensorState = Dict[str, torch.Tensor]

ADD_REMOVE_RECORD_ADJACENCY = "add-or-remove-one-record"
REPLACE_ONE_RECORD_ADJACENCY = "replace-one-record"
DIAGNOSTIC_EVIDENCE_CLASS = "DIAGNOSTIC/NONPUBLICATION"
SEEDED_RNG_SCOPE = "seeded-pseudorandom-research-only-not-production-csprng"
DEFAULT_RDP_ORDERS = tuple(float(value) for value in RDPAccountant.DEFAULT_ALPHAS)
EXPANDED_RDP_ORDERS = DEFAULT_RDP_ORDERS + (
    128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 1536.0,
    2048.0, 3072.0, 4096.0,
)


_opacus_integer_log_a = opacus_rdp._compute_log_a_for_int_alpha


def _stable_high_order_log_a(q: float, sigma: float, alpha: int) -> float:
    """Same sampled-Gaussian integer moment, avoiding binomial overflow.

    Opacus 1.5.3 evaluates binom(alpha, i) before taking its logarithm;
    for alpha >= 1536 this overflows and get_privacy_spent ignores NaNs.
    Preserve its exact low-order path, evaluate high orders in log space.
    This is numerical accounting, not outward-rounded interval arithmetic.
    """
    if alpha <= 63:
        return _opacus_integer_log_a(q, sigma, alpha)
    i = np.arange(alpha + 1, dtype=np.float64)
    log_coeff = gammaln(alpha + 1) - gammaln(i + 1) - gammaln(alpha - i + 1)
    terms = log_coeff + i * math.log(q) + (alpha - i) * math.log1p(-q)
    result = float(logsumexp(terms + (i * i - i) / (2 * sigma**2)))
    if not math.isfinite(result) or result < -1e-10:
        raise ArithmeticError('Nonfinite or negative sampled-Gaussian log moment')
    return max(0.0, result)


# Calibration's RDPAccountant and the runner call this same module function.
# Default orders <=63 therefore retain the historical code path unchanged.
opacus_rdp._compute_log_a_for_int_alpha = _stable_high_order_log_a


class _LengthOnlyDataset(Dataset):
    """Non-materialised dataset used only by the privacy preview API."""

    def __init__(self, length: int) -> None:
        if int(length) <= 0:
            raise ValueError("dataset_size must be positive")
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        raise RuntimeError("a privacy-only dataset must never be iterated")


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Return a stable digest without retaining another model-sized clone."""

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def backend_implementation_sha256() -> str:
    digest = hashlib.sha256()
    with open(Path(__file__).resolve(), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> Dict[str, object]:
    root = Path(path).resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    mapping = {
        str(item.relative_to(root)): file_sha256(item)
        for item in files
    }
    fingerprint = hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "file_count": len(mapping),
        "fingerprint_sha256": fingerprint,
        "file_sha256": mapping,
    }


def _event_seed(
    *,
    dataset_name: str,
    base_seed: int,
    target_epsilon: Optional[float],
    round_index: int,
    client_id: int,
    stream: str,
) -> int:
    """Derive a selection-order-independent seed for one client event."""

    key = json.dumps(
        [
            str(dataset_name),
            int(base_seed),
            None if target_epsilon is None else float(target_epsilon),
            int(round_index),
            int(client_id),
            str(stream),
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    # Torch accepts signed 64-bit seeds.  Keep the high bit clear.
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 63 - 1)


@dataclass(frozen=True)
class DPSGDConfig:
    """Frozen local DP-SGD and accounting parameters."""

    noise_multiplier: float
    max_grad_norm: float
    learning_rate: float
    local_epochs: int
    batch_size: int
    delta: float = 1.0e-6
    momentum: float = 0.0
    adjacency: str = ADD_REMOVE_RECORD_ADJACENCY
    poisson_sampling: bool = True
    rdp_orders: Tuple[float, ...] = DEFAULT_RDP_ORDERS
    accountant_parity_tolerance: float = 1.0e-10
    accountant_parity_relative_tolerance: float = 1.0e-12
    privacy_cap_tolerance: float = 1.0e-10
    event_epsilon_tolerance: float = 1.0e-6
    dataset_name: str = "unspecified"
    target_epsilon: Optional[float] = None
    evidence_class: str = DIAGNOSTIC_EVIDENCE_CLASS
    public_client_dataset_size: Optional[int] = None

    def __post_init__(self) -> None:
        positive = {
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "accountant_parity_tolerance": self.accountant_parity_tolerance,
            "accountant_parity_relative_tolerance": (
                self.accountant_parity_relative_tolerance
            ),
            "privacy_cap_tolerance": self.privacy_cap_tolerance,
            "event_epsilon_tolerance": self.event_epsilon_tolerance,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
            raise ValueError("DP-SGD numeric parameters must be finite and positive")
        if not math.isfinite(float(self.momentum)) or self.momentum < 0.0:
            raise ValueError("momentum must be finite and nonnegative")
        if not 0.0 < float(self.delta) < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.adjacency != ADD_REMOVE_RECORD_ADJACENCY:
            raise ValueError(
                "Opacus Poisson-sampled RDP is accounted here only as {!r}; "
                "a replace-one guarantee must not reuse the same epsilon/delta label".format(
                    ADD_REMOVE_RECORD_ADJACENCY
                )
            )
        if not self.poisson_sampling:
            raise ValueError("the audited DP-SGD path requires poisson_sampling=True")
        orders = tuple(float(value) for value in self.rdp_orders)
        if not orders or any(not math.isfinite(value) or value <= 1.0 for value in orders):
            raise ValueError("every RDP order must be finite and greater than one")
        if tuple(sorted(set(orders))) != orders:
            raise ValueError("RDP orders must be strictly increasing and unique")
        if not str(self.dataset_name).strip():
            raise ValueError("dataset_name must be nonempty for event-keyed RNG")
        if self.target_epsilon is not None and (
            not math.isfinite(float(self.target_epsilon))
            or float(self.target_epsilon) <= 0.0
        ):
            raise ValueError("target_epsilon must be positive when supplied")
        if self.public_client_dataset_size is not None and int(
            self.public_client_dataset_size
        ) <= 0:
            raise ValueError("public_client_dataset_size must be positive")


@dataclass(frozen=True)
class DPSGDClientRequest:
    """One selected client's local-training request."""

    client_id: int
    dataset: Dataset
    aggregation_weight: Optional[float] = None
    noise_multiplier: Optional[float] = None
    allocated_event_epsilon: Optional[float] = None
    event_delta: Optional[float] = None
    max_acceptable_event_epsilon: Optional[float] = None
    max_epsilon_add_remove: Optional[float] = None
    payment: float = 0.0

    def explicit_noise_multiplier(self, config: DPSGDConfig) -> float:
        value = config.noise_multiplier if self.noise_multiplier is None else self.noise_multiplier
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("noise_multiplier must be finite and positive")
        return float(value)


@dataclass(frozen=True)
class DPSGDAccountingEvent:
    """One homogeneous block of sampled-Gaussian optimiser steps."""

    sample_rate: float
    optimizer_steps: int
    noise_multiplier: float

    def __post_init__(self) -> None:
        if not 0.0 < float(self.sample_rate) <= 1.0:
            raise ValueError("sample_rate must lie in (0, 1]")
        if int(self.optimizer_steps) <= 0:
            raise ValueError("optimizer_steps must be positive")
        if not math.isfinite(float(self.noise_multiplier)) or self.noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be finite and positive")


@dataclass(frozen=True)
class RDPEventPreview:
    """Versioned preview of an arbitrary per-order RDP event vector."""

    event_id: str
    state_version: int
    event: DPSGDAccountingEvent
    event_rdp: np.ndarray
    proposed_state: np.ndarray
    proposed_epsilon_add_remove: float
    proposed_best_order: float


def _rdp_for_event(
    event: DPSGDAccountingEvent, orders: Sequence[float]
) -> np.ndarray:
    return np.asarray(
        opacus_rdp.compute_rdp(
            q=float(event.sample_rate),
            noise_multiplier=float(event.noise_multiplier),
            steps=int(event.optimizer_steps),
            orders=list(orders),
        ),
        dtype=np.float64,
    )


def _epsilon_from_rdp_state(
    state: np.ndarray, orders: Sequence[float], delta: float
) -> Tuple[float, float]:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (len(orders),):
        raise ValueError("RDP state length does not match the order grid")
    if not bool(np.any(state)):
        return 0.0, float("nan")
    epsilon, best_order = opacus_rdp.get_privacy_spent(
        orders=list(orders), rdp=state, delta=float(delta)
    )
    return max(0.0, float(epsilon)), float(best_order)


def _finite_order_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def compute_add_remove_exposure(
    events: Sequence[DPSGDAccountingEvent],
    *,
    delta: float = 1.0e-6,
    orders: Sequence[float] = DEFAULT_RDP_ORDERS,
) -> Dict[str, object]:
    """Pure Opacus-analysis composition helper for calibration code."""

    resolved_orders = tuple(float(value) for value in orders)
    state = np.zeros(len(resolved_orders), dtype=np.float64)
    total_steps = 0
    for event in events:
        state += _rdp_for_event(event, resolved_orders)
        total_steps += int(event.optimizer_steps)
    epsilon, best_order = _epsilon_from_rdp_state(state, resolved_orders, delta)
    return {
        "adjacency": ADD_REMOVE_RECORD_ADJACENCY,
        "epsilon": epsilon,
        "delta": float(delta),
        "best_order": _finite_order_or_none(best_order),
        "optimizer_steps": total_steps,
        "rdp_orders": list(resolved_orders),
        "rdp_state": state.tolist(),
    }


def repeated_event_epsilon_schedule(
    event: DPSGDAccountingEvent,
    *,
    maximum_count: int,
    delta: float = 1.0e-6,
    orders: Sequence[float] = DEFAULT_RDP_ORDERS,
) -> np.ndarray:
    """Return epsilon after 0..K identical events without repeated RDP work."""

    if int(maximum_count) < 0:
        raise ValueError("maximum_count must be nonnegative")
    resolved_orders = tuple(float(value) for value in orders)
    one_event_rdp = _rdp_for_event(event, resolved_orders)
    schedule = np.zeros(int(maximum_count) + 1, dtype=np.float64)
    for count in range(1, int(maximum_count) + 1):
        schedule[count] = _epsilon_from_rdp_state(
            count * one_event_rdp, resolved_orders, float(delta)
        )[0]
    return schedule


def calibrate_noise_multiplier_for_event_epsilon(
    *,
    target_event_epsilon: float,
    sample_rate: float,
    optimizer_steps: int,
    event_delta: float = 1.0e-6,
    orders: Sequence[float] = DEFAULT_RDP_ORDERS,
    epsilon_tolerance: float = 1.0e-6,
) -> Dict[str, object]:
    """Invert one isolated participation budget with Opacus' RDP utility.

    ``target_event_epsilon`` is a procurement quantity for one participation,
    evaluated at ``event_delta``.  It is not added across rounds.  Cross-round
    composition still occurs in per-order RDP space and is converted using the
    runner's cumulative accounting delta.
    """

    target = float(target_event_epsilon)
    delta = float(event_delta)
    tolerance = float(epsilon_tolerance)
    event_shape = DPSGDAccountingEvent(
        sample_rate=float(sample_rate),
        optimizer_steps=int(optimizer_steps),
        noise_multiplier=1.0,
    )
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_event_epsilon must be finite and positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("event_delta must lie in (0, 1)")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("epsilon_tolerance must be finite and positive")
    resolved_orders = tuple(float(value) for value in orders)
    multiplier = float(
        get_noise_multiplier(
            target_epsilon=target,
            target_delta=delta,
            sample_rate=event_shape.sample_rate,
            steps=event_shape.optimizer_steps,
            accountant="rdp",
            epsilon_tolerance=tolerance,
            alphas=list(resolved_orders),
        )
    )
    calibrated_event = DPSGDAccountingEvent(
        sample_rate=event_shape.sample_rate,
        optimizer_steps=event_shape.optimizer_steps,
        noise_multiplier=multiplier,
    )
    exposure = compute_add_remove_exposure(
        [calibrated_event], delta=delta, orders=resolved_orders
    )
    realised = float(exposure["epsilon"])
    if realised > target + tolerance:
        raise AssertionError("Opacus event-epsilon inversion exceeded its target")
    return {
        "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
        "target_event_epsilon": target,
        "realized_isolated_event_epsilon": realised,
        "event_delta": delta,
        "noise_multiplier": multiplier,
        "sample_rate": event_shape.sample_rate,
        "optimizer_steps": event_shape.optimizer_steps,
        "epsilon_tolerance": tolerance,
        "absolute_inversion_error": abs(realised - target),
        "one_sided_target_satisfied": realised <= target + tolerance,
    }


def allocated_event_rdp(
    *,
    allocated_event_epsilon: float,
    sample_rate: float,
    optimizer_steps: int,
    event_delta: float,
    orders: Sequence[float] = DEFAULT_RDP_ORDERS,
    epsilon_tolerance: float = 1.0e-6,
) -> Tuple[DPSGDAccountingEvent, Dict[str, object], np.ndarray]:
    """Calibrate one DP-SGD event and return its per-order RDP vector."""

    resolved_orders = tuple(float(value) for value in orders)
    calibration = calibrate_noise_multiplier_for_event_epsilon(
        target_event_epsilon=float(allocated_event_epsilon),
        sample_rate=float(sample_rate),
        optimizer_steps=int(optimizer_steps),
        event_delta=float(event_delta),
        orders=resolved_orders,
        epsilon_tolerance=float(epsilon_tolerance),
    )
    event = DPSGDAccountingEvent(
        sample_rate=float(sample_rate),
        optimizer_steps=int(optimizer_steps),
        noise_multiplier=float(calibration["noise_multiplier"]),
    )
    return event, calibration, _rdp_for_event(event, resolved_orders)


def maximum_feasible_allocated_event_epsilon(
    *,
    current_rdp_state: Sequence[float],
    max_epsilon_add_remove: float,
    sample_rate: float,
    optimizer_steps: int,
    event_delta: float,
    accounting_delta: float,
    minimum_event_epsilon: float,
    maximum_event_epsilon: float,
    orders: Sequence[float] = DEFAULT_RDP_ORDERS,
    event_epsilon_tolerance: float = 1.0e-6,
    cap_tolerance: float = 1.0e-10,
    search_tolerance: float = 1.0e-5,
    max_iterations: int = 60,
    search_strategy: str = "nested-event-epsilon",
    event_profile_resolver: Optional[
        Callable[[float], Tuple[DPSGDAccountingEvent, Dict[str, object], np.ndarray]]
    ] = None,
) -> Dict[str, object]:
    """Return the largest conservatively feasible isolated event epsilon.

    The search composes each candidate in per-order RDP space and converts the
    proposed state at the cumulative accounting delta.  Returning zero means
    that even ``minimum_event_epsilon`` would exceed the cumulative cap.
    """

    resolved_orders = tuple(float(value) for value in orders)
    state = np.asarray(current_rdp_state, dtype=np.float64)
    if state.shape != (len(resolved_orders),):
        raise ValueError("current_rdp_state length does not match the order grid")
    if not bool(np.isfinite(state).all()) or bool((state < 0.0).any()):
        raise ValueError("current_rdp_state must be finite and nonnegative")
    cap = float(max_epsilon_add_remove)
    lower = float(minimum_event_epsilon)
    upper = float(maximum_event_epsilon)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("max_epsilon_add_remove must be finite and positive")
    if not 0.0 < lower <= upper or not math.isfinite(upper):
        raise ValueError("event-epsilon bounds must satisfy 0 < lower <= upper")
    if not 0.0 < float(accounting_delta) < 1.0:
        raise ValueError("accounting_delta must lie in (0,1)")
    if not 0.0 < float(event_delta) < 1.0:
        raise ValueError("event_delta must lie in (0,1)")
    if cap_tolerance < 0.0 or search_tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("privacy search tolerances and iteration count are invalid")
    if search_strategy not in ("nested-event-epsilon", "direct-noise"):
        raise ValueError("unsupported feasible-event search strategy")

    current_epsilon, current_order = _epsilon_from_rdp_state(
        state, resolved_orders, float(accounting_delta)
    )
    if current_epsilon > cap + float(cap_tolerance):
        raise ValueError("current RDP state already exceeds the cumulative cap")

    evaluations = 0

    def evaluate(value: float) -> Dict[str, object]:
        nonlocal evaluations
        if event_profile_resolver is None:
            event, calibration, event_rdp = allocated_event_rdp(
                allocated_event_epsilon=value,
                sample_rate=sample_rate,
                optimizer_steps=optimizer_steps,
                event_delta=event_delta,
                orders=resolved_orders,
                epsilon_tolerance=event_epsilon_tolerance,
            )
        else:
            event, calibration, event_rdp = event_profile_resolver(float(value))
        proposed_epsilon, proposed_order = _epsilon_from_rdp_state(
            state + event_rdp, resolved_orders, float(accounting_delta)
        )
        evaluations += 1
        return {
            "allocated_event_epsilon": float(value),
            "event": event,
            "event_contract": calibration,
            "event_rdp": event_rdp,
            "proposed_epsilon_add_remove": proposed_epsilon,
            "proposed_best_order": proposed_order,
            "feasible": proposed_epsilon <= cap + float(cap_tolerance),
        }

    lower_result = evaluate(lower)
    if not bool(lower_result["feasible"]):
        return {
            "maximum_feasible_allocated_event_epsilon": 0.0,
            "status": "minimum-event-infeasible",
            "current_epsilon_add_remove": current_epsilon,
            "current_best_order": _finite_order_or_none(current_order),
            "cumulative_cap": cap,
            "minimum_event_epsilon": lower,
            "maximum_event_epsilon": upper,
            "proposed_epsilon_at_returned_bound": current_epsilon,
            "proposed_epsilon_at_next_infeasible_bound": float(
                lower_result["proposed_epsilon_add_remove"]
            ),
            "search_interval_width": lower,
            "search_evaluations": evaluations,
            "search_strategy": search_strategy,
        }

    upper_result = evaluate(upper)
    if bool(upper_result["feasible"]):
        return {
            "maximum_feasible_allocated_event_epsilon": upper,
            "status": "configured-upper-bound-feasible",
            "current_epsilon_add_remove": current_epsilon,
            "current_best_order": _finite_order_or_none(current_order),
            "cumulative_cap": cap,
            "minimum_event_epsilon": lower,
            "maximum_event_epsilon": upper,
            "proposed_epsilon_at_returned_bound": float(
                upper_result["proposed_epsilon_add_remove"]
            ),
            "proposed_epsilon_at_next_infeasible_bound": None,
            "search_interval_width": 0.0,
            "search_evaluations": evaluations,
            "search_strategy": search_strategy,
        }

    feasible_result = lower_result
    infeasible_result = upper_result
    if search_strategy == "nested-event-epsilon":
        for _ in range(int(max_iterations)):
            feasible_value = float(feasible_result["allocated_event_epsilon"])
            infeasible_value = float(infeasible_result["allocated_event_epsilon"])
            if infeasible_value - feasible_value <= float(search_tolerance):
                break
            midpoint = 0.5 * (feasible_value + infeasible_value)
            midpoint_result = evaluate(midpoint)
            if bool(midpoint_result["feasible"]):
                feasible_result = midpoint_result
            else:
                infeasible_result = midpoint_result
    else:
        feasible_noise = float(feasible_result["event"].noise_multiplier)
        infeasible_noise = float(infeasible_result["event"].noise_multiplier)
        if feasible_noise <= infeasible_noise:
            raise AssertionError("event-epsilon calibration is not noise-monotone")

        def evaluate_noise(noise_multiplier: float) -> Dict[str, object]:
            nonlocal evaluations
            event = DPSGDAccountingEvent(
                sample_rate=float(sample_rate),
                optimizer_steps=int(optimizer_steps),
                noise_multiplier=float(noise_multiplier),
            )
            event_rdp = _rdp_for_event(event, resolved_orders)
            isolated_epsilon, isolated_order = _epsilon_from_rdp_state(
                event_rdp, resolved_orders, float(event_delta)
            )
            proposed_epsilon, proposed_order = _epsilon_from_rdp_state(
                state + event_rdp, resolved_orders, float(accounting_delta)
            )
            evaluations += 1
            return {
                "allocated_event_epsilon": isolated_epsilon,
                "event": event,
                "event_contract": {
                    "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
                    "target_event_epsilon": isolated_epsilon,
                    "realized_isolated_event_epsilon": isolated_epsilon,
                    "event_delta": float(event_delta),
                    "noise_multiplier": float(noise_multiplier),
                    "sample_rate": float(sample_rate),
                    "optimizer_steps": int(optimizer_steps),
                    "epsilon_tolerance": 0.0,
                    "absolute_inversion_error": 0.0,
                    "one_sided_target_satisfied": True,
                    "best_order": _finite_order_or_none(isolated_order),
                },
                "event_rdp": event_rdp,
                "proposed_epsilon_add_remove": proposed_epsilon,
                "proposed_best_order": proposed_order,
                "feasible": proposed_epsilon <= cap + float(cap_tolerance),
            }

        for _ in range(int(max_iterations)):
            feasible_value = float(feasible_result["allocated_event_epsilon"])
            infeasible_value = float(infeasible_result["allocated_event_epsilon"])
            if infeasible_value - feasible_value <= float(search_tolerance):
                break
            midpoint_noise = 0.5 * (feasible_noise + infeasible_noise)
            if midpoint_noise in (feasible_noise, infeasible_noise):
                break
            midpoint_result = evaluate_noise(midpoint_noise)
            midpoint_value = float(midpoint_result["allocated_event_epsilon"])
            if not feasible_value - event_epsilon_tolerance <= midpoint_value:
                raise AssertionError("direct noise search left its epsilon bracket")
            if not midpoint_value <= infeasible_value + event_epsilon_tolerance:
                raise AssertionError("direct noise search left its epsilon bracket")
            if bool(midpoint_result["feasible"]):
                feasible_result = midpoint_result
                feasible_noise = midpoint_noise
            else:
                infeasible_result = midpoint_result
                infeasible_noise = midpoint_noise

    returned = float(feasible_result["allocated_event_epsilon"])
    next_infeasible = float(infeasible_result["allocated_event_epsilon"])
    return {
        "maximum_feasible_allocated_event_epsilon": returned,
        "status": "interior-bound",
        "current_epsilon_add_remove": current_epsilon,
        "current_best_order": _finite_order_or_none(current_order),
        "cumulative_cap": cap,
        "minimum_event_epsilon": lower,
        "maximum_event_epsilon": upper,
        "proposed_epsilon_at_returned_bound": float(
            feasible_result["proposed_epsilon_add_remove"]
        ),
        "proposed_epsilon_at_next_infeasible_bound": float(
            infeasible_result["proposed_epsilon_add_remove"]
        ),
        "search_interval_width": next_infeasible - returned,
        "search_evaluations": evaluations,
        "search_strategy": search_strategy,
    }


def conservative_replace_one_from_add_remove(
    epsilon_add_remove: float, delta_add_remove: float
) -> Dict[str, object]:
    """Apply two-step basic group privacy to a substitution adjacency.

    For same-size datasets differing by one substituted record, remove the old
    record and then add the new one.  An ``(eps, delta)`` add/remove guarantee
    therefore implies ``(2 eps, (1 + exp(eps)) delta)`` replace-one DP.  This is
    deliberately *not* a same-delta conversion.
    """

    epsilon = float(epsilon_add_remove)
    delta = float(delta_add_remove)
    if epsilon < 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon_add_remove must be finite and nonnegative")
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta_add_remove must lie in [0, 1)")
    if epsilon == 0.0 and delta == 0.0:
        converted_delta = 0.0
    elif epsilon > math.log(np.finfo(np.float64).max):
        converted_delta = float("inf")
    else:
        converted_delta = (1.0 + math.exp(epsilon)) * delta
    return {
        "adjacency": REPLACE_ONE_RECORD_ADJACENCY,
        "epsilon": 2.0 * epsilon,
        "delta": converted_delta,
        "derivation": "two-add/remove-steps-basic-group-privacy",
        "same_delta_as_opacus": False,
        "meaningful_delta": converted_delta < 1.0,
    }


class ClientRDPLedger:
    """External, versioned RDP ledger cross-checked against Opacus hooks."""

    def __init__(
        self,
        *,
        orders: Sequence[float] = DEFAULT_RDP_ORDERS,
        delta: float = 1.0e-6,
    ) -> None:
        self.orders = tuple(float(value) for value in orders)
        self.delta = float(delta)
        self.state = np.zeros(len(self.orders), dtype=np.float64)
        self.state_version = 0
        self.events: List[Tuple[str, DPSGDAccountingEvent]] = []
        self._committed_ids: set = set()

    def epsilon(self) -> Tuple[float, float]:
        return _epsilon_from_rdp_state(self.state, self.orders, self.delta)

    def preview_event(
        self, event_id: str, event: DPSGDAccountingEvent
    ) -> RDPEventPreview:
        return self.preview_rdp_vector(event_id, event, _rdp_for_event(event, self.orders))

    def preview_rdp_vector(
        self,
        event_id: str,
        event: DPSGDAccountingEvent,
        event_rdp: Sequence[float],
    ) -> RDPEventPreview:
        """Preview a supplied event vector without mutating the ledger."""

        if not event_id:
            raise ValueError("event_id must be nonempty")
        if event_id in self._committed_ids:
            raise RuntimeError("event_id has already been committed")
        vector = np.asarray(event_rdp, dtype=np.float64)
        if vector.shape != self.state.shape:
            raise ValueError("event_rdp shape does not match the ledger")
        if not bool(np.isfinite(vector).all()) or bool((vector < 0.0).any()):
            raise ValueError("event_rdp must be finite and nonnegative")
        proposed = self.state + vector
        epsilon, best_order = _epsilon_from_rdp_state(
            proposed, self.orders, self.delta
        )
        return RDPEventPreview(
            event_id=event_id,
            state_version=self.state_version,
            event=event,
            event_rdp=vector.copy(),
            proposed_state=proposed.copy(),
            proposed_epsilon_add_remove=epsilon,
            proposed_best_order=best_order,
        )

    def commit_preview(self, preview: RDPEventPreview) -> None:
        if preview.event_id in self._committed_ids:
            raise RuntimeError("event_id has already been committed")
        if preview.state_version != self.state_version:
            raise RuntimeError("stale RDP preview")
        self.state = preview.proposed_state.copy()
        self.state_version += 1
        self.events.append((preview.event_id, preview.event))
        self._committed_ids.add(preview.event_id)

    def export(self) -> Dict[str, object]:
        epsilon, best_order = self.epsilon()
        replace_one = conservative_replace_one_from_add_remove(
            epsilon, self.delta if self.events else 0.0
        )
        return {
            "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
            "epsilon_add_remove": epsilon,
            "delta_add_remove": self.delta,
            "best_order": _finite_order_or_none(best_order),
            "participations": len(self.events),
            "optimizer_steps": sum(event.optimizer_steps for _, event in self.events),
            "state_version": self.state_version,
            "rdp_orders": list(self.orders),
            "rdp_state": self.state.tolist(),
            "events": [
                {"event_id": event_id, **asdict(event)}
                for event_id, event in self.events
            ],
            "replace_one_conservative_conversion": replace_one,
        }


@dataclass
class DPSGDRoundResult:
    round_index: int
    selected_client_ids: List[int]
    released_client_ids: List[int]
    selected_signature: str
    released_signature: str
    participation_events: List[Dict[str, object]]
    skipped_events: List[Dict[str, object]]
    released_updates: int
    release_count: int
    selected_payment_total: float
    settled_payment_total: float
    payment_total: float
    model_sha256_before: str
    model_sha256_after: str
    update_retention: str = "digest-only-after-round-aggregation"
    evidence_class: str = DIAGNOSTIC_EVIDENCE_CLASS


def _accountant_history_steps(history: Sequence[Tuple[float, float, int]]) -> int:
    return sum(int(steps) for _, _, steps in history)


def _rdp_from_opacus_history(
    history: Sequence[Tuple[float, float, int]], orders: Sequence[float]
) -> np.ndarray:
    state = np.zeros(len(orders), dtype=np.float64)
    for noise_multiplier, sample_rate, steps in history:
        state += _rdp_for_event(
            DPSGDAccountingEvent(
                sample_rate=float(sample_rate),
                optimizer_steps=int(steps),
                noise_multiplier=float(noise_multiplier),
            ),
            orders,
        )
    return state


def _single_event_from_opacus_history_delta(
    history_before: Sequence[Tuple[float, float, int]],
    history_after: Sequence[Tuple[float, float, int]],
) -> DPSGDAccountingEvent:
    """Recover one hook-recorded event without subtracting cumulative RDP.

    ``RDPAccountant.step`` compresses adjacent steps with identical noise and
    sample-rate parameters.  A local training call therefore either increments
    the final history row or appends exactly one row.  Checking that transition
    keeps event parity independent of cancellation in large cumulative states.
    """

    before = [
        (float(noise), float(rate), int(steps))
        for noise, rate, steps in history_before
    ]
    after = [
        (float(noise), float(rate), int(steps))
        for noise, rate, steps in history_after
    ]
    if len(after) == len(before) + 1 and after[:-1] == before:
        noise, rate, steps = after[-1]
        return DPSGDAccountingEvent(
            sample_rate=rate,
            optimizer_steps=steps,
            noise_multiplier=noise,
        )
    if (
        before
        and len(after) == len(before)
        and after[:-1] == before[:-1]
        and after[-1][:2] == before[-1][:2]
        and after[-1][2] > before[-1][2]
    ):
        noise, rate, steps_after = after[-1]
        return DPSGDAccountingEvent(
            sample_rate=rate,
            optimizer_steps=steps_after - before[-1][2],
            noise_multiplier=noise,
        )
    raise AssertionError("Opacus accountant history changed outside one event")


def _rdp_parity_metrics(
    left: Sequence[float],
    right: Sequence[float],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Dict[str, object]:
    """Return scale-aware parity metrics for two finite RDP vectors."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("RDP parity vectors must have the same shape")
    if not bool(np.isfinite(left_array).all() and np.isfinite(right_array).all()):
        raise ValueError("RDP parity vectors must be finite")
    absolute_error = np.abs(left_array - right_array)
    scale = np.maximum(np.abs(left_array), np.abs(right_array))
    allowed_error = float(absolute_tolerance) + float(relative_tolerance) * scale
    relative_error = np.divide(
        absolute_error,
        scale,
        out=np.zeros_like(absolute_error),
        where=scale > 0.0,
    )
    return {
        "pass": bool(np.all(absolute_error <= allowed_error)),
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "max_rel_error": float(relative_error.max(initial=0.0)),
        "max_allowed_error": float(allowed_error.max(initial=0.0)),
    }


def _prv_epsilon_from_history(
    history: Sequence[Tuple[float, float, int]], delta: float
) -> float:
    """Cross-check RDP with Opacus' heterogeneous PRV accountant."""

    if not history:
        return 0.0
    accountant = PRVAccountant()
    accountant.history = [
        (float(noise), float(rate), int(steps))
        for noise, rate, steps in history
    ]
    return float(accountant.get_epsilon(delta=float(delta)))


def _state_delta(local_model: nn.Module, global_model: nn.Module) -> TensorState:
    local_state = local_model.state_dict()
    global_state = global_model.state_dict()
    if local_state.keys() != global_state.keys():
        raise RuntimeError("local and global state dictionaries differ")
    update: TensorState = {}
    for key, global_value in global_state.items():
        local_value = local_state[key]
        if global_value.is_floating_point():
            update[key] = local_value.detach() - global_value.detach()
        else:
            update[key] = torch.zeros_like(global_value)
    return update


def _state_l2_norm(state: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(
        sum(
            float(value.detach().double().square().sum().item())
            for value in state.values()
            if value.is_floating_point()
        )
    )


class TDSCDPSGDRunner:
    """Federated runner whose only randomisation is Opacus local DP-SGD."""

    def __init__(
        self,
        model: nn.Module,
        n_clients: int,
        config: DPSGDConfig,
        *,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if int(n_clients) <= 0:
            raise ValueError("n_clients must be positive")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("the audited seeded runner supports CPU and CUDA only")
        ModuleValidator.validate(model, strict=True)
        self.model = copy.deepcopy(model).to(self.device)
        self.n_clients = int(n_clients)
        self.config = config
        self.seed = int(seed)
        self._opacus_accountants = [RDPAccountant() for _ in range(self.n_clients)]
        self._external_ledgers = [
            ClientRDPLedger(orders=config.rdp_orders, delta=config.delta)
            for _ in range(self.n_clients)
        ]
        self._allocated_event_cache: Dict[
            Tuple[int, float, float],
            Tuple[DPSGDAccountingEvent, Dict[str, object], np.ndarray],
        ] = {}
        self._feasible_event_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
        self._committed_training_event_ids: set = set()

    def _cached_allocated_event_profile(
        self,
        *,
        dataset_size: int,
        allocated_event_epsilon: float,
        event_delta: float,
    ) -> Tuple[DPSGDAccountingEvent, Dict[str, object], np.ndarray]:
        key = (
            int(dataset_size),
            float(allocated_event_epsilon),
            float(event_delta),
        )
        cached = self._allocated_event_cache.get(key)
        if cached is None:
            shape = self._planned_accounting_event(dataset_size, 1.0)
            cached = allocated_event_rdp(
                allocated_event_epsilon=float(allocated_event_epsilon),
                sample_rate=shape.sample_rate,
                optimizer_steps=shape.optimizer_steps,
                event_delta=float(event_delta),
                orders=self.config.rdp_orders,
                epsilon_tolerance=self.config.event_epsilon_tolerance,
            )
            self._allocated_event_cache[key] = cached
        event, calibration, event_rdp = cached
        return event, dict(calibration), event_rdp.copy()

    def _planned_accounting_event(
        self, dataset_size: int, noise_multiplier: float
    ) -> DPSGDAccountingEvent:
        if int(dataset_size) <= 0:
            raise ValueError("selected client datasets must be nonempty")
        public_size = (
            int(dataset_size)
            if self.config.public_client_dataset_size is None
            else int(self.config.public_client_dataset_size)
        )
        if int(dataset_size) != public_size:
            raise ValueError(
                "selected dataset length must equal the frozen public client size"
            )
        effective_batch = min(int(self.config.batch_size), public_size)
        loader_length = int(math.ceil(float(public_size) / effective_batch))
        return DPSGDAccountingEvent(
            sample_rate=1.0 / loader_length,
            optimizer_steps=int(self.config.local_epochs) * loader_length,
            noise_multiplier=float(noise_multiplier),
        )

    def _event_id(self, round_index: int, client_id: int) -> str:
        target = (
            "none"
            if self.config.target_epsilon is None
            else "{:.12g}".format(float(self.config.target_epsilon))
        )
        return "dataset={}|seed={}|target={}|round={}|client={}".format(
            self.config.dataset_name,
            self.seed,
            target,
            int(round_index),
            int(client_id),
        )

    def _resolve_request_event(
        self, request: DPSGDClientRequest
    ) -> Tuple[DPSGDAccountingEvent, Dict[str, object]]:
        if request.noise_multiplier is not None and request.allocated_event_epsilon is not None:
            raise ValueError(
                "specify either noise_multiplier or allocated_event_epsilon, not both"
            )
        event_delta = (
            float(self.config.delta)
            if request.event_delta is None
            else float(request.event_delta)
        )
        if not 0.0 < event_delta < 1.0:
            raise ValueError("event_delta must lie in (0, 1)")
        shape = self._planned_accounting_event(
            len(request.dataset), request.explicit_noise_multiplier(self.config)
        )
        calibration: Dict[str, object]
        if request.allocated_event_epsilon is not None:
            event, calibration, _ = self._cached_allocated_event_profile(
                dataset_size=len(request.dataset),
                allocated_event_epsilon=float(request.allocated_event_epsilon),
                event_delta=event_delta,
            )
            return event, calibration
        else:
            multiplier = request.explicit_noise_multiplier(self.config)
            direct_event = DPSGDAccountingEvent(
                sample_rate=shape.sample_rate,
                optimizer_steps=shape.optimizer_steps,
                noise_multiplier=multiplier,
            )
            exposure = compute_add_remove_exposure(
                [direct_event], delta=event_delta, orders=self.config.rdp_orders
            )
            realised = float(exposure["epsilon"])
            calibration = {
                "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
                "target_event_epsilon": None,
                "realized_isolated_event_epsilon": realised,
                "event_delta": event_delta,
                "noise_multiplier": multiplier,
                "sample_rate": shape.sample_rate,
                "optimizer_steps": shape.optimizer_steps,
                "epsilon_tolerance": None,
                "absolute_inversion_error": None,
                "one_sided_target_satisfied": None,
            }
        event = DPSGDAccountingEvent(
            sample_rate=shape.sample_rate,
            optimizer_steps=shape.optimizer_steps,
            noise_multiplier=multiplier,
        )
        return event, calibration

    def preview_participation(
        self,
        *,
        client_id: int,
        dataset_size: int,
        round_index: int,
        noise_multiplier: Optional[float] = None,
        allocated_event_epsilon: Optional[float] = None,
        event_delta: Optional[float] = None,
    ) -> Dict[str, object]:
        """Preview exact planned q/steps and cumulative add/remove exposure."""

        self._validate_client_id(client_id)
        if noise_multiplier is not None and allocated_event_epsilon is not None:
            raise ValueError(
                "specify either noise_multiplier or allocated_event_epsilon, not both"
            )
        placeholder = _LengthOnlyDataset(dataset_size)
        event, calibration = self._resolve_request_event(
            DPSGDClientRequest(
                client_id=client_id,
                dataset=placeholder,
                noise_multiplier=noise_multiplier,
                allocated_event_epsilon=allocated_event_epsilon,
                event_delta=event_delta,
            )
        )
        preview = self._external_ledgers[client_id].preview_event(
            self._event_id(round_index, client_id), event
        )
        return {
            "event_id": preview.event_id,
            "adjacency": ADD_REMOVE_RECORD_ADJACENCY,
            **asdict(event),
            "epsilon_add_remove_after": preview.proposed_epsilon_add_remove,
            "delta_add_remove": self.config.delta,
            "best_order_after": preview.proposed_best_order,
            "rdp_event": preview.event_rdp.tolist(),
            "event_contract": calibration,
        }

    def maximum_feasible_allocated_event_epsilon(
        self,
        *,
        client_id: int,
        dataset_size: int,
        max_epsilon_add_remove: float,
        minimum_event_epsilon: float,
        maximum_event_epsilon: float,
        event_delta: Optional[float] = None,
        cap_tolerance: Optional[float] = None,
        search_tolerance: float = 1.0e-5,
        max_iterations: int = 60,
        search_strategy: str = "nested-event-epsilon",
    ) -> Dict[str, object]:
        """Invert the current client's RDP state into a deployable input."""

        self._validate_client_id(client_id)
        shape = self._planned_accounting_event(dataset_size, 1.0)
        resolved_event_delta = (
            self.config.delta if event_delta is None else float(event_delta)
        )
        resolved_cap_tolerance = (
            self.config.privacy_cap_tolerance
            if cap_tolerance is None
            else float(cap_tolerance)
        )
        ledger = self._external_ledgers[client_id]
        cache_key = (
            ledger.state.tobytes(),
            int(dataset_size),
            float(max_epsilon_add_remove),
            float(minimum_event_epsilon),
            float(maximum_event_epsilon),
            resolved_event_delta,
            resolved_cap_tolerance,
            float(search_tolerance),
            int(max_iterations),
            str(search_strategy),
        )
        cached = self._feasible_event_cache.get(cache_key)
        cache_hit = cached is not None
        if cached is None:
            cached = maximum_feasible_allocated_event_epsilon(
                current_rdp_state=ledger.state,
                max_epsilon_add_remove=max_epsilon_add_remove,
                sample_rate=shape.sample_rate,
                optimizer_steps=shape.optimizer_steps,
                event_delta=resolved_event_delta,
                accounting_delta=self.config.delta,
                minimum_event_epsilon=minimum_event_epsilon,
                maximum_event_epsilon=maximum_event_epsilon,
                orders=self.config.rdp_orders,
                event_epsilon_tolerance=self.config.event_epsilon_tolerance,
                cap_tolerance=resolved_cap_tolerance,
                search_tolerance=search_tolerance,
                max_iterations=max_iterations,
                search_strategy=search_strategy,
                event_profile_resolver=lambda value: self._cached_allocated_event_profile(
                    dataset_size=dataset_size,
                    allocated_event_epsilon=value,
                    event_delta=resolved_event_delta,
                ),
            )
            self._feasible_event_cache[cache_key] = cached
        result = dict(cached)
        result.update(
            {
                "client_id": int(client_id),
                "state_version": int(ledger.state_version),
                "sample_rate": shape.sample_rate,
                "optimizer_steps": shape.optimizer_steps,
                "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
                "delta_add_remove": self.config.delta,
                "search_cache_hit": cache_hit,
            }
        )
        return result

    def current_add_remove_epsilon(self) -> np.ndarray:
        """Return the current external-ledger epsilon for every client."""

        return np.asarray(
            [ledger.epsilon()[0] for ledger in self._external_ledgers],
            dtype=np.float64,
        )

    def _validate_client_id(self, client_id: int) -> None:
        if not 0 <= int(client_id) < self.n_clients:
            raise IndexError("client_id is outside the runner")

    def client_privacy_snapshot(
        self, client_id: int, *, include_prv_crosscheck: bool = True
    ) -> Dict[str, object]:
        self._validate_client_id(client_id)
        external = self._external_ledgers[client_id].export()
        internal_history = list(self._opacus_accountants[client_id].history)
        internal_state = _rdp_from_opacus_history(internal_history, self.config.rdp_orders)
        internal_epsilon, internal_order = _epsilon_from_rdp_state(
            internal_state, self.config.rdp_orders, self.config.delta
        )
        prv_epsilon = (
            _prv_epsilon_from_history(internal_history, self.config.delta)
            if include_prv_crosscheck
            else None
        )
        parity = _rdp_parity_metrics(
            internal_state,
            self._external_ledgers[client_id].state,
            absolute_tolerance=self.config.accountant_parity_tolerance,
            relative_tolerance=self.config.accountant_parity_relative_tolerance,
        )
        external.update(
            {
                "client_id": int(client_id),
                "opacus_version": opacus.__version__,
                "opacus_accountant_history": [
                    {
                        "noise_multiplier": float(noise),
                        "sample_rate": float(rate),
                        "optimizer_steps": int(steps),
                    }
                    for noise, rate, steps in internal_history
                ],
                "opacus_epsilon_add_remove": internal_epsilon,
                "opacus_best_order": _finite_order_or_none(internal_order),
                "external_opacus_rdp_max_abs_error": parity["max_abs_error"],
                "external_opacus_rdp_max_rel_error": parity["max_rel_error"],
                "external_opacus_rdp_max_allowed_error": parity[
                    "max_allowed_error"
                ],
                "accountant_parity_absolute_tolerance": (
                    self.config.accountant_parity_tolerance
                ),
                "accountant_parity_relative_tolerance": (
                    self.config.accountant_parity_relative_tolerance
                ),
                "accountant_parity_pass": parity["pass"],
                "prv_epsilon_add_remove": prv_epsilon,
                "rdp_prv_crosscheck_pass": (
                    None
                    if prv_epsilon is None
                    else internal_epsilon + 1.0e-8 >= prv_epsilon
                ),
            }
        )
        return external

    def privacy_snapshot(
        self, *, include_prv_crosscheck: bool = True
    ) -> Dict[str, object]:
        clients = [
            self.client_privacy_snapshot(
                client_id, include_prv_crosscheck=include_prv_crosscheck
            )
            for client_id in range(self.n_clients)
        ]
        epsilons = np.asarray(
            [float(row["epsilon_add_remove"]) for row in clients], dtype=np.float64
        )
        # Conservative empirical quantiles: always select an observed upper value.
        p50 = float(np.quantile(epsilons, 0.50, method="higher"))
        p95 = float(np.quantile(epsilons, 0.95, method="higher"))
        return {
            "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
            "delta_add_remove": self.config.delta,
            "quantile_method": "higher",
            "epsilon_add_remove_p50": p50,
            "epsilon_add_remove_p95": p95,
            "epsilon_add_remove_max": float(epsilons.max()),
            "total_participations": sum(int(row["participations"]) for row in clients),
            "total_optimizer_steps": sum(int(row["optimizer_steps"]) for row in clients),
            "all_accountant_parity_pass": all(
                bool(row["accountant_parity_pass"]) for row in clients
            ),
            "accountant_consistency_scope": (
                "opacus-hook-history-versus-planned-rdp-vector"
            ),
            "prv_crosscheck_included": include_prv_crosscheck,
            "all_rdp_prv_crosschecks_pass": (
                all(bool(row["rdp_prv_crosscheck_pass"]) for row in clients)
                if include_prv_crosscheck
                else None
            ),
            "clients": clients,
        }

    def _train_one(
        self,
        *,
        round_index: int,
        request: DPSGDClientRequest,
        event: DPSGDAccountingEvent,
        preview: RDPEventPreview,
        event_contract: Mapping[str, object],
    ) -> Tuple[TensorState, Dict[str, object]]:
        client_id = int(request.client_id)
        local_model = copy.deepcopy(self.model).to(self.device)
        local_model.train()
        ModuleValidator.validate(local_model, strict=True)
        base_optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=float(self.config.learning_rate),
            momentum=float(self.config.momentum),
        )
        effective_batch = min(int(self.config.batch_size), len(request.dataset))
        sampler_seed = _event_seed(
            dataset_name=self.config.dataset_name,
            base_seed=self.seed,
            target_epsilon=self.config.target_epsilon,
            round_index=round_index,
            client_id=client_id,
            stream="poisson-sampler",
        )
        noise_seed = _event_seed(
            dataset_name=self.config.dataset_name,
            base_seed=self.seed,
            target_epsilon=self.config.target_epsilon,
            round_index=round_index,
            client_id=client_id,
            stream="dp-gaussian-noise",
        )
        sampler_generator = torch.Generator(device="cpu")
        sampler_generator.manual_seed(sampler_seed)
        noise_device = "cuda" if self.device.type == "cuda" else "cpu"
        noise_generator = torch.Generator(device=noise_device)
        noise_generator.manual_seed(noise_seed)
        base_loader = DataLoader(
            request.dataset,
            batch_size=effective_batch,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            generator=sampler_generator,
        )

        accountant = self._opacus_accountants[client_id]
        history_before = list(accountant.history)
        internal_state_before = _rdp_from_opacus_history(
            history_before, self.config.rdp_orders
        )
        epsilon_before, _ = _epsilon_from_rdp_state(
            internal_state_before, self.config.rdp_orders, self.config.delta
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Secure RNG turned off.*")
            privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        # Retain one Opacus accountant per client across all FL rounds.  The
        # optimizer hook, not this runner, records the actual DP steps.
        privacy_engine.accountant = accountant
        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=local_model,
            optimizer=base_optimizer,
            data_loader=base_loader,
            noise_multiplier=float(event.noise_multiplier),
            max_grad_norm=float(self.config.max_grad_norm),
            poisson_sampling=True,
            noise_generator=noise_generator,
            clipping="flat",
            grad_sample_mode="hooks",
        )
        actual_sample_rate = 1.0 / len(private_loader)
        loader_sample_rate = float(getattr(private_loader, "sample_rate", actual_sample_rate))
        if abs(actual_sample_rate - loader_sample_rate) > 1.0e-15:
            accountant.history = history_before
            raise AssertionError("private loader sample-rate metadata is inconsistent")
        if abs(actual_sample_rate - event.sample_rate) > 1.0e-15:
            accountant.history = history_before
            raise AssertionError("planned and Opacus sample rates differ")

        criterion = nn.CrossEntropyLoss()
        sampled_examples = 0
        empty_batches = 0
        loop_steps = 0
        loss_sum = 0.0
        try:
            private_model.train()
            for _ in range(int(self.config.local_epochs)):
                for features, labels in private_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device).long()
                    private_optimizer.zero_grad(set_to_none=True)
                    logits = private_model(features)
                    if int(labels.numel()) == 0:
                        # Empty Poisson draws remain optimiser events.  This
                        # zero-valued graph lets Opacus add the step's noise.
                        loss = logits.sum() * 0.0
                        empty_batches += 1
                    else:
                        loss = criterion(logits, labels)
                        batch_examples = int(labels.shape[0])
                        sampled_examples += batch_examples
                        loss_sum += float(loss.detach().item()) * batch_examples
                    loss.backward()
                    private_optimizer.step()
                    loop_steps += 1
        except Exception:
            accountant.history = history_before
            raise

        history_after = list(accountant.history)
        hook_steps = _accountant_history_steps(history_after) - _accountant_history_steps(history_before)
        if hook_steps != loop_steps or loop_steps != event.optimizer_steps:
            accountant.history = history_before
            raise AssertionError(
                "Opacus hook steps {}, loop steps {}, planned steps {} differ".format(
                    hook_steps, loop_steps, event.optimizer_steps
                )
            )
        hook_event = _single_event_from_opacus_history_delta(
            history_before, history_after
        )
        if (
            not math.isclose(
                hook_event.sample_rate,
                event.sample_rate,
                rel_tol=1.0e-15,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                hook_event.noise_multiplier,
                event.noise_multiplier,
                rel_tol=1.0e-15,
                abs_tol=1.0e-15,
            )
            or hook_event.optimizer_steps != event.optimizer_steps
        ):
            accountant.history = history_before
            raise AssertionError("planned and Opacus hook event parameters differ")
        internal_state_after = _rdp_from_opacus_history(
            history_after, self.config.rdp_orders
        )
        internal_event_rdp = _rdp_for_event(hook_event, self.config.rdp_orders)
        event_parity = _rdp_parity_metrics(
            internal_event_rdp,
            preview.event_rdp,
            absolute_tolerance=self.config.accountant_parity_tolerance,
            relative_tolerance=self.config.accountant_parity_relative_tolerance,
        )
        cumulative_parity = _rdp_parity_metrics(
            internal_state_after,
            preview.proposed_state,
            absolute_tolerance=self.config.accountant_parity_tolerance,
            relative_tolerance=self.config.accountant_parity_relative_tolerance,
        )
        if not bool(event_parity["pass"] and cumulative_parity["pass"]):
            accountant.history = history_before
            raise AssertionError("external event ledger and Opacus accountant diverged")
        epsilon_after, best_order_after = _epsilon_from_rdp_state(
            internal_state_after, self.config.rdp_orders, self.config.delta
        )
        if not math.isclose(
            epsilon_after,
            preview.proposed_epsilon_add_remove,
            rel_tol=self.config.accountant_parity_relative_tolerance,
            abs_tol=self.config.accountant_parity_tolerance,
        ):
            accountant.history = history_before
            raise AssertionError("external and Opacus cumulative epsilon diverged")

        standard_model = private_model.to_standard_module()
        update = _state_delta(standard_model, self.model)
        if any(
            not bool(torch.isfinite(value).all())
            for value in update.values()
            if value.is_floating_point()
        ):
            accountant.history = history_before
            raise FloatingPointError("DP-SGD produced a non-finite local update")

        self._external_ledgers[client_id].commit_preview(preview)
        self._committed_training_event_ids.add(preview.event_id)
        update_norm = _state_l2_norm(update)
        event_row: Dict[str, object] = {
            "event_id": preview.event_id,
            "round": int(round_index),
            "client_id": client_id,
            "status": "dp-sgd-update-produced",
            "public_client_dataset_size": (
                len(request.dataset)
                if self.config.public_client_dataset_size is None
                else int(self.config.public_client_dataset_size)
            ),
            "poisson_sampling": True,
            "sample_rate": actual_sample_rate,
            "private_loader_length": len(private_loader),
            "local_epochs": int(self.config.local_epochs),
            "optimizer_steps": loop_steps,
            "opacus_hook_optimizer_steps": hook_steps,
            "noise_multiplier": float(event.noise_multiplier),
            "allocated_event_epsilon": event_contract["target_event_epsilon"],
            "realized_isolated_event_epsilon": event_contract[
                "realized_isolated_event_epsilon"
            ],
            "event_delta": event_contract["event_delta"],
            "event_epsilon_inversion_abs_error": event_contract[
                "absolute_inversion_error"
            ],
            "max_grad_norm": float(self.config.max_grad_norm),
            "per_sample_gradient_clipping": True,
            "gaussian_noise_added_by": "opacus.DPOptimizer",
            "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
            "delta_add_remove": float(self.config.delta),
            "epsilon_add_remove_before": epsilon_before,
            "epsilon_add_remove_after": epsilon_after,
            "best_order_after": best_order_after,
            "event_rdp_max_abs_error": event_parity["max_abs_error"],
            "event_rdp_max_rel_error": event_parity["max_rel_error"],
            "cumulative_rdp_max_abs_error": cumulative_parity["max_abs_error"],
            "cumulative_rdp_max_rel_error": cumulative_parity["max_rel_error"],
            "cumulative_rdp_max_allowed_error": cumulative_parity[
                "max_allowed_error"
            ],
            "accountant_parity_absolute_tolerance": (
                self.config.accountant_parity_tolerance
            ),
            "accountant_parity_relative_tolerance": (
                self.config.accountant_parity_relative_tolerance
            ),
            "accountant_parity_pass": True,
            "local_update_semantics": "DP-SGD-local-model-delta-postprocessing",
            "complete_update_second_clipping_applied": False,
            "complete_update_second_noise_applied": False,
            "complete_update_l2_norm": update_norm,
            "complete_update_sha256": tensor_state_sha256(update),
            "aggregation_weight": (
                float(len(request.dataset))
                if request.aggregation_weight is None
                else float(request.aggregation_weight)
            ),
            "payment": float(request.payment),
            "rng_scope": SEEDED_RNG_SCOPE,
            "secure_mode": False,
            "opacus_version": opacus.__version__,
            "private_training_telemetry_redacted": True,
            "evidence_class": self.config.evidence_class,
        }
        return update, event_row

    def _aggregate(self, releases: Sequence[Tuple[TensorState, float]]) -> None:
        if not releases:
            return
        weights = np.asarray([max(0.0, float(weight)) for _, weight in releases])
        if float(weights.sum()) <= 0.0:
            weights.fill(1.0)
        weights /= weights.sum()
        global_state = self.model.state_dict()
        merged: TensorState = {}
        for key, global_value in global_state.items():
            if global_value.is_floating_point():
                increment = torch.zeros_like(global_value)
                for index, (update, _) in enumerate(releases):
                    increment.add_(
                        update[key].to(global_value.dtype), alpha=float(weights[index])
                    )
                merged[key] = global_value + increment
            else:
                merged[key] = global_value.clone()
        self.model.load_state_dict(merged, strict=True)

    def run_round(
        self,
        round_index: int,
        requests: Iterable[DPSGDClientRequest],
    ) -> DPSGDRoundResult:
        if int(round_index) < 0:
            raise ValueError("round_index must be nonnegative")
        request_list = list(requests)
        client_ids = [int(request.client_id) for request in request_list]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("a client can appear at most once per FL round")
        for request in request_list:
            self._validate_client_id(request.client_id)
            if len(request.dataset) <= 0:
                raise ValueError("selected client datasets must be nonempty")
            if request.noise_multiplier is not None and request.allocated_event_epsilon is not None:
                raise ValueError(
                    "specify either noise_multiplier or allocated_event_epsilon, not both"
                )
            if request.noise_multiplier is not None:
                request.explicit_noise_multiplier(self.config)
            if request.allocated_event_epsilon is not None and (
                not math.isfinite(float(request.allocated_event_epsilon))
                or float(request.allocated_event_epsilon) <= 0.0
            ):
                raise ValueError("allocated_event_epsilon must be finite and positive")
            if request.event_delta is not None and not 0.0 < float(request.event_delta) < 1.0:
                raise ValueError("event_delta must lie in (0, 1)")
            if request.max_acceptable_event_epsilon is not None and (
                not math.isfinite(float(request.max_acceptable_event_epsilon))
                or float(request.max_acceptable_event_epsilon) <= 0.0
            ):
                raise ValueError("max_acceptable_event_epsilon must be positive")
            if request.aggregation_weight is not None and (
                not math.isfinite(float(request.aggregation_weight))
                or float(request.aggregation_weight) < 0.0
            ):
                raise ValueError("aggregation_weight must be finite and nonnegative")
            if not math.isfinite(float(request.payment)) or request.payment < 0.0:
                raise ValueError("payment must be finite and nonnegative")
            if request.max_epsilon_add_remove is not None and (
                not math.isfinite(float(request.max_epsilon_add_remove))
                or float(request.max_epsilon_add_remove) <= 0.0
            ):
                raise ValueError("max_epsilon_add_remove must be positive")

        model_before = tensor_state_sha256(self.model.state_dict())
        releases: List[Tuple[TensorState, float]] = []
        released_ids: List[int] = []
        participation_events: List[Dict[str, object]] = []
        skipped_events: List[Dict[str, object]] = []

        for request in request_list:
            client_id = int(request.client_id)
            event_id = self._event_id(round_index, client_id)
            if event_id in self._committed_training_event_ids:
                raise RuntimeError("a successful DP-SGD event cannot be replayed")
            event, event_contract = self._resolve_request_event(request)
            preview = self._external_ledgers[client_id].preview_event(event_id, event)
            private_limit = request.max_acceptable_event_epsilon
            isolated_epsilon = float(event_contract["realized_isolated_event_epsilon"])
            if private_limit is not None and (
                isolated_epsilon
                > float(private_limit) + self.config.event_epsilon_tolerance
            ):
                epsilon_before, _ = self._external_ledgers[client_id].epsilon()
                skipped_events.append(
                    {
                        "event_id": event_id,
                        "round": int(round_index),
                        "client_id": client_id,
                        "status": "rejected-before-training-by-private-event-loss-limit",
                        "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
                        "sample_rate": event.sample_rate,
                        "optimizer_steps": 0,
                        "planned_optimizer_steps": event.optimizer_steps,
                        "noise_multiplier": event.noise_multiplier,
                        "allocated_event_epsilon": event_contract[
                            "target_event_epsilon"
                        ],
                        "realized_isolated_event_epsilon": isolated_epsilon,
                        "event_delta": event_contract["event_delta"],
                        "epsilon_add_remove_before": epsilon_before,
                        "privacy_consumed": False,
                        "payment_settled": False,
                        "private_thresholds_redacted": True,
                        "evidence_class": self.config.evidence_class,
                    }
                )
                continue
            cap = request.max_epsilon_add_remove
            if cap is not None and (
                preview.proposed_epsilon_add_remove
                > float(cap) + self.config.privacy_cap_tolerance
            ):
                epsilon_before, _ = self._external_ledgers[client_id].epsilon()
                skipped_events.append(
                    {
                        "event_id": event_id,
                        "round": int(round_index),
                        "client_id": client_id,
                        "status": "rejected-before-training-by-add-remove-epsilon-cap",
                        "accounted_adjacency": ADD_REMOVE_RECORD_ADJACENCY,
                        "sample_rate": event.sample_rate,
                        "optimizer_steps": 0,
                        "planned_optimizer_steps": event.optimizer_steps,
                        "noise_multiplier": event.noise_multiplier,
                        "allocated_event_epsilon": event_contract[
                            "target_event_epsilon"
                        ],
                        "realized_isolated_event_epsilon": isolated_epsilon,
                        "event_delta": event_contract["event_delta"],
                        "max_grad_norm": self.config.max_grad_norm,
                        "epsilon_add_remove_before": epsilon_before,
                        "epsilon_add_remove_if_participated": preview.proposed_epsilon_add_remove,
                        "privacy_consumed": False,
                        "payment_settled": False,
                        "private_thresholds_redacted": True,
                        "evidence_class": self.config.evidence_class,
                    }
                )
                continue
            update, row = self._train_one(
                round_index=round_index,
                request=request,
                event=event,
                preview=preview,
                event_contract=event_contract,
            )
            weight = (
                float(
                    len(request.dataset)
                    if self.config.public_client_dataset_size is None
                    else self.config.public_client_dataset_size
                )
                if request.aggregation_weight is None
                else float(request.aggregation_weight)
            )
            releases.append((update, weight))
            released_ids.append(client_id)
            participation_events.append(row)

        self._aggregate(releases)
        selected_signature = ";".join(str(value) for value in client_ids)
        released_signature = ";".join(str(value) for value in released_ids)
        selected_payment_total = sum(float(request.payment) for request in request_list)
        settled_payment_total = sum(
            float(row["payment"]) for row in participation_events
        )
        return DPSGDRoundResult(
            round_index=int(round_index),
            selected_client_ids=client_ids,
            released_client_ids=released_ids,
            selected_signature=selected_signature,
            released_signature=released_signature,
            participation_events=participation_events,
            skipped_events=skipped_events,
            released_updates=len(releases),
            release_count=len(releases),
            selected_payment_total=selected_payment_total,
            settled_payment_total=settled_payment_total,
            payment_total=settled_payment_total,
            model_sha256_before=model_before,
            model_sha256_after=tensor_state_sha256(self.model.state_dict()),
            evidence_class=self.config.evidence_class,
        )


def evaluate_classifier(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> Dict[str, object]:
    """Evaluate multiclass logits without mutating the training RNG."""

    resolved_device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(resolved_device)
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(resolved_device)
            labels = labels.to(resolved_device).long()
            logits = model(features)
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            examples += int(labels.shape[0])
    if examples <= 0:
        raise ValueError("evaluation dataset must be nonempty")
    return {
        "examples": examples,
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
    }


def _make_synthetic_smoke(seed: int) -> Tuple[List[Dataset], Dataset, nn.Module]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)

    def sample(count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, 2, (count,), generator=generator)
        direction = labels.float().mul(2.0).sub(1.0)
        features = torch.randn(count, 2, generator=generator) * 0.55
        features[:, 0] += direction * 1.8
        features[:, 1] += direction * 0.7
        return features, labels

    train_features, train_labels = sample(256)
    test_features, test_labels = sample(512)
    client_indices = [torch.arange(client_id, 256, 4) for client_id in range(4)]
    train = TensorDataset(train_features, train_labels)
    clients = [Subset(train, indices.tolist()) for indices in client_indices]
    test = TensorDataset(test_features, test_labels)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + 31)
        model = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2))
    return clients, test, model


def run_synthetic_diagnostic_smoke(
    *, seed: int = 11, rounds: int = 5, device: str = "cpu"
) -> Dict[str, object]:
    """Run a small learnable diagnostic and return utility/accountant parity."""

    clients, test, model = _make_synthetic_smoke(seed)
    config = DPSGDConfig(
        noise_multiplier=1.2,
        max_grad_norm=1.0,
        learning_rate=0.18,
        local_epochs=1,
        batch_size=16,
        delta=1.0e-6,
        dataset_name="synthetic-binary-smoke",
        target_epsilon=None,
        evidence_class=DIAGNOSTIC_EVIDENCE_CLASS,
        public_client_dataset_size=64,
    )
    runner = TDSCDPSGDRunner(
        model, len(clients), config, seed=seed, device=device
    )
    initial = evaluate_classifier(runner.model, test, device=device)
    round_rows: List[Dict[str, object]] = []
    for round_index in range(int(rounds)):
        result = runner.run_round(
            round_index,
            [
                DPSGDClientRequest(client_id=client_id, dataset=dataset)
                for client_id, dataset in enumerate(clients)
            ],
        )
        round_rows.append(asdict(result))
    final = evaluate_classifier(runner.model, test, device=device)
    privacy = runner.privacy_snapshot()
    consistency_pass = bool(privacy["all_accountant_parity_pass"])
    prv_crosscheck = bool(privacy["all_rdp_prv_crosschecks_pass"])
    utility_finite = all(
        math.isfinite(float(metrics[key]))
        for metrics in (initial, final)
        for key in ("loss", "accuracy")
    )
    result = "PASS" if consistency_pass and prv_crosscheck and utility_finite else "FAIL"
    return {
        "artifact_type": "tdsc-dpsgd-smoke-attestation/v2",
        "status": result,
        "result": result,
        "evidence_class": DIAGNOSTIC_EVIDENCE_CLASS,
        "completed_experiment_scope": "synthetic-smoke-only",
        "publication_evidence": False,
        "hook_history_consistency": consistency_pass,
        "rdp_prv_crosscheck": prv_crosscheck,
        "accountant_crosscheck_scope": (
            "planned-RDP versus Opacus hook history plus heterogeneous PRV sanity check"
        ),
        "complete_update_second_noise_applied": False,
        "private_training_telemetry_redacted": True,
        "utility_finite": utility_finite,
        "backend_implementation_sha256": backend_implementation_sha256(),
        "opacus_version": opacus.__version__,
        "torch_version": torch.__version__,
        "seed": int(seed),
        "rounds": int(rounds),
        "rng_scope": SEEDED_RNG_SCOPE,
        "adjacency": ADD_REMOVE_RECORD_ADJACENCY,
        "privacy_guarantee_scope": (
            "DP-calibrated research simulation; seeded RNG is not a production CSPRNG"
        ),
        "initial_utility": initial,
        "final_utility": final,
        "accuracy_change": float(final["accuracy"]) - float(initial["accuracy"]),
        "privacy": privacy,
        "rounds_detail": round_rows,
    }


def run_mnist_diagnostic_smoke(
    *,
    data_root: str = "data",
    seed: int = 11,
    rounds: int = 1,
    train_examples: int = 256,
    test_examples: int = 256,
    device: str = "cpu",
) -> Dict[str, object]:
    """Run an opt-in, small local MNIST diagnostic using the dataset registry."""

    # Imported lazily so the DP-SGD core remains isolated from formal dataset
    # orchestration while that module evolves.
    from tdsc_fl_datasets import create_model, load_tdsc_dataset

    bundle = load_tdsc_dataset("MNIST", data_root=data_root, download=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 43)
    train_count = min(int(train_examples), len(bundle.train))
    test_count = min(int(test_examples), len(bundle.test))
    if train_count % 4 != 0:
        raise ValueError("MNIST diagnostic train_examples must be divisible by four")
    train_indices = torch.randperm(len(bundle.train), generator=generator)[:train_count]
    test_indices = torch.randperm(len(bundle.test), generator=generator)[:test_count]
    selected_train = Subset(bundle.train, train_indices.tolist())
    clients = [
        Subset(selected_train, list(range(client_id, train_count, 4)))
        for client_id in range(4)
    ]
    test = Subset(bundle.test, test_indices.tolist())
    model = create_model("MNIST", architecture="mlp", hidden_dim=32, seed=seed + 47)
    config = DPSGDConfig(
        noise_multiplier=1.5,
        max_grad_norm=1.0,
        learning_rate=0.12,
        local_epochs=1,
        batch_size=16,
        delta=1.0e-6,
        dataset_name="MNIST-diagnostic-subset",
        evidence_class=DIAGNOSTIC_EVIDENCE_CLASS,
        public_client_dataset_size=train_count // 4,
    )
    runner = TDSCDPSGDRunner(model, 4, config, seed=seed, device=device)
    initial = evaluate_classifier(runner.model, test, device=device)
    rows = []
    for round_index in range(int(rounds)):
        rows.append(
            asdict(
                runner.run_round(
                    round_index,
                    [
                        DPSGDClientRequest(client_id=client_id, dataset=dataset)
                        for client_id, dataset in enumerate(clients)
                    ],
                )
            )
        )
    final = evaluate_classifier(runner.model, test, device=device)
    privacy = runner.privacy_snapshot()
    consistency_pass = bool(privacy["all_accountant_parity_pass"])
    prv_crosscheck = bool(privacy["all_rdp_prv_crosschecks_pass"])
    utility_finite = all(
        math.isfinite(float(metrics[key]))
        for metrics in (initial, final)
        for key in ("loss", "accuracy")
    )
    result = "PASS" if consistency_pass and prv_crosscheck and utility_finite else "FAIL"
    return {
        "artifact_type": "tdsc-dpsgd-smoke-attestation/v2",
        "status": result,
        "result": result,
        "evidence_class": DIAGNOSTIC_EVIDENCE_CLASS,
        "completed_experiment_scope": "small-MNIST-smoke-only",
        "publication_evidence": False,
        "hook_history_consistency": consistency_pass,
        "rdp_prv_crosscheck": prv_crosscheck,
        "accountant_crosscheck_scope": (
            "planned-RDP versus Opacus hook history plus heterogeneous PRV sanity check"
        ),
        "complete_update_second_noise_applied": False,
        "private_training_telemetry_redacted": True,
        "utility_finite": utility_finite,
        "backend_implementation_sha256": backend_implementation_sha256(),
        "opacus_version": opacus.__version__,
        "torch_version": torch.__version__,
        "seed": int(seed),
        "rounds": int(rounds),
        "train_examples": train_count,
        "test_examples": test_count,
        "rng_scope": SEEDED_RNG_SCOPE,
        "adjacency": ADD_REMOVE_RECORD_ADJACENCY,
        "privacy_guarantee_scope": (
            "DP-calibrated research simulation; seeded RNG is not a production CSPRNG"
        ),
        "initial_utility": initial,
        "final_utility": final,
        "accuracy_change": float(final["accuracy"]) - float(initial["accuracy"]),
        "privacy": privacy,
        "rounds_detail": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", choices=("synthetic", "mnist"), default="synthetic")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--test-examples", type=int, default=256)
    args = parser.parse_args()
    if args.smoke == "synthetic":
        summary = run_synthetic_diagnostic_smoke(
            seed=args.seed,
            rounds=5 if args.rounds is None else args.rounds,
            device=args.device,
        )
    else:
        summary = run_mnist_diagnostic_smoke(
            data_root=args.data_root,
            seed=args.seed,
            rounds=1 if args.rounds is None else args.rounds,
            train_examples=args.train_examples,
            test_examples=args.test_examples,
            device=args.device,
        )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(output / "summary.json", summary)
    data_provenance: Mapping[str, object]
    if args.smoke == "mnist":
        data_provenance = directory_fingerprint(Path(args.data_root) / "mnist")
    else:
        data_provenance = {
            "scope": "generated-synthetic-data",
            "seed": int(args.seed),
        }
    manifest = {
        "artifact_type": "tdsc-dpsgd-smoke-manifest/v1",
        "summary_sha256": file_sha256(output / "summary.json"),
        "backend_implementation_sha256": backend_implementation_sha256(),
        "smoke": args.smoke,
        "seed": int(args.seed),
        "rounds": int(summary["rounds"]),
        "device": str(args.device),
        "torch_version": torch.__version__,
        "opacus_version": opacus.__version__,
        "data_provenance": data_provenance,
        "publication_evidence": False,
    }
    _atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": summary["status"],
        "evidence_class": summary["evidence_class"],
        "initial_accuracy": summary["initial_utility"]["accuracy"],
        "final_accuracy": summary["final_utility"]["accuracy"],
        "epsilon_add_remove_p95": summary["privacy"]["epsilon_add_remove_p95"],
        "accountant_parity": summary["privacy"]["all_accountant_parity_pass"],
        "output": str(output / "summary.json"),
    }, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

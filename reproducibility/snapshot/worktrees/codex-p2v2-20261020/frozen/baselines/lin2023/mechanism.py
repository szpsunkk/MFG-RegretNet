"""Lin et al. (2023), Sec. V / Alg. 2, discrete privacy-price contracts.

Our reimplementation, not author code. No hard budget, DP-SGD adapter,
or cumulative privacy guarantee is added to the native optimizer.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContractMenu:
    types: np.ndarray
    probabilities: np.ndarray
    alpha: float
    z1: float
    z2: float
    raw_epsilon: np.ndarray
    epsilon: np.ndarray
    payment: np.ndarray
    blocks: tuple


def objective_coefficients(types, probabilities, alpha=1.0, z1=1.0, z2=1.0):
    """Eq. 42: coefficients of A/e + B/e**2 + C*e."""
    types = np.asarray(types, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if types.ndim != 1 or len(types) < 2 or probabilities.shape != types.shape:
        raise ValueError("At least two types and one probability per type required")
    if not np.all(np.isfinite(types)) or np.any(types <= 0) or np.any(np.diff(types) <= 0):
        raise ValueError("Types must be finite, positive and strictly increasing")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise ValueError("Probabilities must be finite and positive; zero mass unsupported")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Probabilities must sum to one; no silent normalization")
    if not np.all(np.isfinite([alpha, z1, z2])) or min(alpha, z1, z2) <= 0:
        raise ValueError("alpha, z1, z2 must be finite and positive")
    prefix = np.r_[0.0, np.cumsum(probabilities[:-1])]
    increments = np.r_[0.0, np.diff(types)]
    return np.column_stack((probabilities * z1, probabilities * z2,
                            (probabilities * types + increments * prefix) / alpha))


def _positive_minimum(coefficients):
    """Unique e>0 minimum: C = A/e² + 2B/e³, bracketed bisection."""
    a, b, c = coefficients
    if not np.all(np.isfinite(coefficients)) or min(a, b, c) <= 0:
        raise ValueError("Finite positive objective coefficients required")
    lo = max(np.sqrt(a / c), np.cbrt(2 * b / c))
    hi = 2 * lo
    for _ in range(256):
        mid = (lo + hi) / 2
        if a / mid**2 + 2 * b / mid**3 > c:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            return (lo + hi) / 2
    raise RuntimeError("Scalar root did not reach registered tolerance")


def payments_from_epsilon(types, epsilon, alpha):
    """Eq. 41/46: binding highest-type IR and adjacent upward IC."""
    types, epsilon = np.asarray(types), np.asarray(epsilon)
    if types.shape != epsilon.shape or types.ndim != 1 or len(types) == 0:
        raise ValueError("types and epsilon must be matching nonempty vectors")
    if (not np.all(np.isfinite(epsilon)) or np.any(epsilon <= 0)
            or np.any(np.diff(epsilon) > 0) or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("Positive descending epsilon and positive alpha required")
    increments = types[:-1] * (epsilon[:-1] - epsilon[1:])
    return (types[-1] * epsilon[-1] + np.r_[np.cumsum(increments[::-1])[::-1], 0]) / alpha


def solve_discrete_contracts(types, probabilities, alpha=1.0, z1=1.0, z2=1.0):
    """Alg. 2: independent optima, then contiguous bunching/ironing."""
    coefficients = objective_coefficients(types, probabilities, alpha, z1, z2)
    raw = np.array([_positive_minimum(row) for row in coefficients])
    blocks = []
    for i, row in enumerate(coefficients):
        blocks.append((i, i + 1, row.copy(), raw[i]))
        while len(blocks) >= 2 and blocks[-2][3] < blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            pooled = left[2] + right[2]
            blocks.append((left[0], right[1], pooled, _positive_minimum(pooled)))
    epsilon = np.empty_like(raw)
    for start, end, _, value in blocks:
        epsilon[start:end] = value
    types = np.asarray(types, dtype=np.float64).copy()
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    payment = payments_from_epsilon(types, epsilon, alpha)
    return ContractMenu(types, probabilities, float(alpha), float(z1), float(z2),
                        raw, epsilon, payment,
                        tuple(tuple(range(start, end)) for start, end, _, _ in blocks))


def utility_matrix(menu):
    """M² utilities: row=true type, column=chosen item, Eq. 27."""
    return menu.alpha * menu.payment[None, :] - menu.types[:, None] * menu.epsilon[None, :]


def operator_cost(menu):
    """Eq. 28/31 per client; positive cost is minus operator utility."""
    return float(menu.probabilities @ (menu.z1 / menu.epsilon
                                     + menu.z2 / menu.epsilon**2 + menu.payment))

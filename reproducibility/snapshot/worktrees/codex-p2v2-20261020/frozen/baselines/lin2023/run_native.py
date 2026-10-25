"""Run preregistered Lin Sec. VII contract cases; no FL or contract adaptation."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, minimize

from .mechanism import operator_cost, solve_discrete_contracts, utility_matrix


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_constraint_reference(types, probabilities, alpha, z1, z2):
    """Independent P1 solver in all 2M variables with all-pair IC and IR.

    Does not call payment elimination, ironing, or warm-start from our menu.
    """
    types, probabilities = np.asarray(types), np.asarray(probabilities)
    m = len(types)
    rows = []
    for i in range(m):
        row = np.zeros(2 * m)
        row[i], row[m + i] = -types[i], alpha
        rows.append(row)
        for j in range(m):
            if i != j:
                row = np.zeros(2 * m)
                row[i], row[j] = -types[i], types[i]
                row[m + i], row[m + j] = alpha, -alpha
                rows.append(row)
    matrix = np.array(rows)

    def objective(x):
        e, p = x[:m], x[m:]
        return float(probabilities @ (z1 / e + z2 / e**2 + p))

    def gradient(x):
        e = x[:m]
        return np.r_[probabilities * (-z1 / e**2 - 2 * z2 / e**3), probabilities]

    # All types can select the same feasible contract, independently of Alg. 2.
    x0 = np.r_[np.full(m, 0.5), np.full(m, types[-1] * 0.5 / alpha)]
    result = minimize(objective, x0, jac=gradient, method="SLSQP",
                      bounds=Bounds(np.r_[np.full(m, 1e-8), np.zeros(m)], np.inf),
                      constraints=[LinearConstraint(matrix, 0, np.inf)],
                      options={"maxiter": 3000, "ftol": 1e-11})
    return {"success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "cost": float(result.fun),
            "min_constraint_slack": float((matrix @ result.x).min()),
            "epsilon": result.x[:m].tolist(), "payment": result.x[m:].tolist()}


def audit_menu(menu):
    utilities = utility_matrix(menu)
    truthful = np.diag(utilities)
    regret = np.maximum(0, np.maximum(utilities.max(axis=1), 0) - truthful)
    return {"max_regret_including_exit": float(regret.max()),
            "minimum_truthful_utility": float(truthful.min()),
            "max_monotonicity_violation": float(np.maximum(np.diff(menu.epsilon), 0).max()),
            "utility_matrix": utilities.tolist(), "regret": regret.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("native_protocol.json"))
    parser.add_argument("--pdf", type=Path, default=Path(__file__).with_name(
        "Heterogeneous_Differential-Private_Federated_Learning_Trading_Privacy_for_Utility_Truthfully.pdf"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if sha256(args.pdf) != protocol["source_sha256"]:
        raise ValueError("Source PDF differs from registered source")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "native_results.json").exists():
        raise FileExistsError("Preserve previous runs; choose a fresh output directory")
    (args.out / "executed_protocol.json").write_bytes(args.protocol.read_bytes())
    start = time.perf_counter()
    cases = [(f"uniform_alpha_{alpha:g}", protocol["uniform_probabilities"], alpha)
             for alpha in protocol["alphas"]]
    cases.append(("figure4_ironing", protocol["figure4_probabilities"],
                  protocol["figure4_alpha_assumption"]))
    results, rows = {}, []
    for name, probabilities, alpha in cases:
        menu = solve_discrete_contracts(protocol["types"], probabilities, alpha,
                                       protocol["z1"], protocol["z2"])
        audit = audit_menu(menu)
        reference = full_constraint_reference(menu.types, menu.probabilities,
                                               alpha, menu.z1, menu.z2)
        cost = operator_cost(menu)
        gap = abs(cost - reference["cost"]) / max(1.0, abs(cost))
        results[name] = {
            "alpha": alpha, "probabilities": list(probabilities),
            "raw_epsilon": menu.raw_epsilon.tolist(), "epsilon": menu.epsilon.tolist(),
            "payment": menu.payment.tolist(),
            "blocks_one_based": [[i + 1 for i in block] for block in menu.blocks],
            "operator_cost_per_client": cost,
            "payment_per_client": float(menu.probabilities @ menu.payment),
            "epsilon_per_client": float(menu.probabilities @ menu.epsilon),
            "native_audit": audit, "full_p1_reference": reference,
            "relative_objective_gap": gap}
        for i in range(len(menu.types)):
            rows.append([name, alpha, menu.types[i], menu.probabilities[i],
                         menu.raw_epsilon[i], menu.epsilon[i], menu.payment[i],
                         audit["utility_matrix"][i][i], audit["regret"][i]])
    comparisons = []
    uniform = results["uniform_alpha_1"]
    for reading in protocol["figure5_readings"]:
        actual = uniform[reading["quantity"]][reading["type_index"]]
        difference = abs(actual - reading["plot_value"])
        comparisons.append({**reading, "reimplementation": actual,
                            "absolute_difference": difference,
                            "relative_difference_to_plot_reading": difference / abs(reading["plot_value"]),
                            "within_preregistered_plot_tolerance": difference <= reading["absolute_tolerance"]})
    ironing = results["figure4_ironing"]
    blocks_match = ironing["blocks_one_based"] == protocol["figure4_expected_blocks_one_based"]
    tol = protocol["verification"]
    algorithm_checks = all(
        case["native_audit"]["max_regret_including_exit"] <= tol["ic_ir_tolerance"]
        and case["native_audit"]["minimum_truthful_utility"] >= -tol["ic_ir_tolerance"]
        and case["native_audit"]["max_monotonicity_violation"] == 0
        and case["full_p1_reference"]["success"]
        and case["full_p1_reference"]["min_constraint_slack"] >= -tol["ic_ir_tolerance"]
        and case["relative_objective_gap"] <= tol["objective_relative_tolerance"]
        for case in results.values())
    bindings = [Path(__file__), Path(__file__).with_name("mechanism.py"), args.protocol,
                Path(__file__).with_name("MAPPING.md"), Path(__file__).with_name("UNSPECIFIED.md")]
    report = {
        "scope": protocol["scope"], "source_sha256": sha256(args.pdf),
        "source_bindings": {str(p): sha256(p) for p in bindings},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "cases": results, "plot_comparisons": comparisons,
        "figure4": {"reported_distinct_contracts": protocol["figure4_reported_distinct_contracts"],
                    "reimplementation_distinct_contracts": len(ironing["blocks_one_based"]),
                    "blocks_match": blocks_match, "alpha_is_assumption": True},
        "algorithm_checks_passed": algorithm_checks,
        "plot_checks_passed": all(x["within_preregistered_plot_tolerance"] for x in comparisons),
        "elapsed_seconds_including_reference_validation": time.perf_counter() - start,
        "randomness": "none; deterministic native optimization",
        "author_review": "NOT_REVIEWED", "fl_reproduction_completed": False,
        "common_contract_adapter_completed": False, "r3_4_closed": False}
    (args.out / "native_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.out / "contracts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "alpha", "type", "probability", "raw_epsilon", "epsilon",
                         "payment", "truthful_utility", "regret_including_exit"])
        writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ("figure4", "plot_comparisons",
                                           "algorithm_checks_passed", "plot_checks_passed")}, indent=2))
    if not algorithm_checks:
        raise SystemExit("Numerical validation failed; saved results must not be hidden")


if __name__ == "__main__":
    main()

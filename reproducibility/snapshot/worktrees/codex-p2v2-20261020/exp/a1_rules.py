"""A1/A2 shared decision rules; no training and no checkpoint selection on tests."""
from collections import defaultdict

import numpy as np

from p2_results import cell_key, paired_interval

EXTERNAL = ("PAC", "PAC-capped-legacy", "VCG", "VCG-capped-legacy", "CSRA", "MFG-Pricing",
            "Posted-price", "Lin", "DM-RegretNet", "RegretNet-r2")
TASK_EXTERNAL = EXTERNAL[:-1] + ("RegretNet-old-selected",)
LN_EXTERNAL = ("PAC", "CSRA", "MFG-Pricing", "Posted-price", "Lin")


def confirmation_cells(taskaware=False):
    cells = []
    for dataset in ("MNIST", "CIFAR10"):
        budgets = (30., 100.) if taskaware or dataset == "CIFAR10" else (10., 30., 100.)
        for alpha in (.5, .1):
            for mode, values in (("budget", budgets), ("privacy", (.5, .8))):
                cells.extend({"dataset": dataset, "alpha": alpha, "mode": mode, "value": value} for value in values)
    return cells


def indexed(rows):
    lookup = defaultdict(dict)
    for row in rows:
        key = (cell_key(row), row["arm"])
        if row["seed"] in lookup[key]:
            raise ValueError("duplicate result for cell/arm/seed")
        if not np.isfinite(row["accuracy"]):
            raise ValueError("nonfinite accuracy")
        lookup[key][row["seed"]] = row
    return lookup


def matched(row):
    if row["mode"] == "budget":
        return True
    if "matching_pass" in row:
        return bool(row["matching_pass"])
    return abs(row["epsilon_per_client_round"] / row["value"] - 1) <= .005 + 1e-6


def confirmation_rule(rows, candidate, cells, seeds=(231, 232, 233), external=EXTERNAL, exclusions=()):
    """Best arm is fixed by its mean over all seeds, then seed-paired differences."""
    lookup = indexed(rows)
    expected_seeds = set(seeds)
    excluded = {}
    for record in exclusions:
        if "seed" in record:
            omitted_seeds = [record["seed"]]
        else:
            # Task-aware omits the entire arm/cell if any paired seed is
            # unreachable. Keep the original calibration evidence intact.
            calibration = record.get("calibrations", {})
            if (record["mode"] != "privacy" or set(calibration) != {str(s) for s in seeds}
                    or not any(c.get("feasible") is False for c in calibration.values())):
                raise ValueError("whole-cell exclusion needs paired unreachability evidence")
            omitted_seeds = seeds
        for seed in omitted_seeds:
            excluded[(cell_key(record), record["arm"], seed)] = record
    per_cell, omissions, problems = [], [], []
    for cell in cells:
        key = cell_key(cell)
        own = lookup[(key, candidate)]
        valid_own = set(own) == expected_seeds and all(matched(r) for r in own.values())
        if not valid_own:
            problems.append({"cell": key, "arm": candidate, "reason": "candidate missing a seed or unmatched"})
        available = {}
        for arm in external:
            group = lookup[(key, arm)]
            if set(group) == expected_seeds and all(matched(r) for r in group.values()):
                available[arm] = group
                continue
            missing = expected_seeds - set(group)
            known = cell["mode"] == "privacy" and bool(missing) and all((key, arm, s) in excluded for s in missing)
            row = {"cell": key, "arm": arm, "missing_seeds": sorted(missing),
                   "reason": "unreachable; no complete three-seed mean" if known else "missing or unmatched evidence",
                   "details": {str(s): excluded.get((key, arm, s)) for s in sorted(missing)}}
            omissions.append(row)
            if not known or any(not matched(r) for r in group.values()):
                problems.append(row)
        if not available:
            problems.append({"cell": key, "reason": "no complete external comparator"})
        if not valid_own or not available:
            continue
        best = max(available, key=lambda arm: np.mean([available[arm][s]["accuracy"] for s in seeds]))
        diffs = [own[s]["accuracy"] - available[best][s]["accuracy"] for s in seeds]
        per_cell.append({**cell, "external_arm": best,
                         "external_mean_accuracy": float(np.mean([available[best][s]["accuracy"] for s in seeds])),
                         "seeds": list(seeds), "paired_differences_pp": [100 * d for d in diffs]})
    datasets = {}
    for dataset in ("MNIST", "CIFAR10"):
        group = [r for r in per_cell if r["dataset"] == dataset]
        needed = sum(c["dataset"] == dataset for c in cells)
        complete = len(group) == needed and not any(p.get("cell", (None,))[0] == dataset for p in problems)
        if not complete:
            datasets[dataset] = {"status": "incomplete", "pass": None, "cells_expected": needed,
                                 "cells_with_pairs": len(group), "seed_differences_pp": None, "interval_pp": None}
            continue
        d = np.mean([r["paired_differences_pp"] for r in group], axis=0)
        passed = bool(np.all(d > 0)) if dataset == "CIFAR10" else bool(d.mean() >= -.3 - 1e-12)
        datasets[dataset] = {"status": "evaluated", "pass": passed, "cells_expected": needed,
                             "cells_with_pairs": len(group), "seeds": list(seeds), "seed_differences_pp": d.tolist(),
                             "interval_pp": paired_interval(d)}
    complete = not problems and all(v["status"] == "evaluated" for v in datasets.values())
    return {"candidate": candidate, "status": "evaluated" if complete else "incomplete",
            "pass": all(v["pass"] for v in datasets.values()) if complete else None,
            "datasets": datasets, "per_cell": per_cell, "external_arms": list(external),
            "omissions": omissions, "problems": problems,
            "tie_rule": "first in the frozen external arm list",
            "interval_scope": "paired t interval across independent seeds; descriptive; no multiplicity correction"}


def stage2_candidate_gate(rows, candidate, cells, seeds=(14, 15)):
    lookup = indexed(rows)
    differences, reasons = [], []
    for cell in cells:
        key = cell_key(cell)
        def usable(arm):
            group = lookup[(key, arm)]
            return group if set(group) == set(seeds) and all(matched(r) for r in group.values()) else None
        own = usable(candidate)
        options = {a: usable(a) for a in ("CSRA", "PAC", "V8F")}
        options = {a: rows for a, rows in options.items() if rows}
        if not own or not options or (cell["dataset"] == "MNIST" and "V8F" not in options):
            reasons.append(f"missing or unmatched paired gate cell: {key}")
            continue
        best = max(options, key=lambda a: np.mean([r["accuracy"] for r in options[a].values()])) if cell["dataset"] == "CIFAR10" else "V8F"
        diff = [own[s]["accuracy"] - options[best][s]["accuracy"] for s in seeds]
        differences.append({"cell": key, "reference": best, "paired_seed_differences": diff, "mean": float(np.mean(diff))})
    gains = {ds: float(np.mean([r["mean"] for r in differences if r["cell"][0] == ds]))
             if any(r["cell"][0] == ds for r in differences) else None for ds in ("MNIST", "CIFAR10")}
    if gains["CIFAR10"] is None or gains["CIFAR10"] < .005 - 1e-12:
        reasons.append("CIFAR10 gain below 0.5 pp")
    if gains["MNIST"] is None or gains["MNIST"] < -.003 - 1e-12:
        reasons.append("MNIST difference below -0.3 pp")
    return {"candidate": candidate, "gate_pass": not reasons, "reasons": reasons, "gains": gains,
            "paired_cell_differences": differences, "test_results_used": False}


def adoption_priority(mf_n10=None, mf_n100=None, p2=None, taskaware=None):
    def both(a, b):
        return False if a is False or b is False else True if a is True and b is True else None
    routes = [("MF-Net", both(mf_n10, mf_n100)), ("mfg_p2", p2), ("task-aware", taskaware), ("V8F", True)]
    chosen = next(name for name, passed in routes if passed is True)
    rank = [name for name, _ in routes].index(chosen)
    pending = [name for name, passed in routes[:rank] if passed is None]
    return {"model": chosen, "status": "provisional" if pending else "final", "pending_higher_priority_routes": pending,
            "route_pass": dict(routes), "mf_separate_large_population_study": mf_n100 is True and mf_n10 is False,
            "criterion": "A1 priority, A2 task-aware mapping; no ranking by test accuracy"}

"""CPU-only allocation schedules and preregistered budget matching."""
from __future__ import annotations

from functools import lru_cache
import gc
from pathlib import Path
import sys

import numpy as np
import torch

from common import ROOT, WORK, OUT, data_for_cell, market_reports, digest
from p2_core import allocation, array_hash, exact_metrics, REFERENCES, load_any, AllocationInfeasible
from p2_state import CONFIG, AMENDMENT, read, immutable_json, update


def job(name, command, folder, done, gpu=False):
    return {"id": name, "command": command, "out": str(folder), "done": str(folder / done), "gpu": gpu}


@lru_cache(maxsize=96)
def market(dataset, alpha, seed, rounds=100, smoke_size=None):
    # Only public sizes and deterministic indices needed, never evaluate images.
    _, _, split, _ = data_for_cell(dataset, alpha, seed, stage="dev", smoke_size=smoke_size)
    reports = market_reports(rounds, split["client_sizes"], seed)
    gc.collect()
    return reports, split


def deploy_kappa(dataset, alpha, decision):
    rule = decision["selected_rule"][dataset]
    if rule == "Oracle-khat":
        return read(OUT / "kappa_hat.json")["by_cell"][f"{dataset}_a{alpha}"]
    return {"Oracle-k0": .01, "Oracle-k3": 3., "Oracle-k30": 30.}[rule]


def arm_kappa(arm, dataset, alpha, decision=None):
    if arm in ("Oracle-k0", "Oracle-k0-datasize"):
        return 0.
    if arm in ("Oracle-k3", "Oracle-k30"):
        return float(arm[len("Oracle-k"):])
    if arm == "Oracle-khat" or decision is None:
        return read(OUT / "kappa_hat.json")["by_cell"][f"{dataset}_a{alpha}"]
    return deploy_kappa(dataset, alpha, decision)


def calibration(reports, arm, checkpoint, target, kappa, reference=None, raw_sizes=None):
    if not np.isfinite(target) or target <= 0:
        raise ValueError("invalid privacy target")
    model = load_any(checkpoint) if checkpoint else None
    def amount(budget):
        e = allocation(arm, reports, budget, kappa, checkpoint, reference, target, model, raw_sizes)["executed"]
        return float(e.double().mean())
    points = [(float(b), amount(float(b))) for b in np.unique(np.r_[.5, np.linspace(1, 10, 10), np.linspace(10, 110, 26)])]
    sampled = points[:]
    for (lo, left), (hi, right) in zip(points[:-1], points[1:]):
        if (left - target) * (right - target) > 0:
            continue
        for _ in range(24):
            mid = (lo + hi) / 2
            value = amount(mid)
            sampled.append((mid, value))
            if abs(value / target - 1) <= .0025:
                break
            if (left - target) * (value - target) <= 0:
                hi = mid
            else:
                lo, left = mid, value
    best = min(sampled, key=lambda x: (abs(x[1] / target - 1), x[0]))
    return {"budget": best[0], "mean_100_rounds": best[1], "relative_error_100": best[1] / target - 1,
            "feasible": abs(best[1] / target - 1) <= .005, "sampled_budgets": sampled,
            "reason": "matched" if abs(best[1] / target - 1) <= .005 else "no matching budget found in [0.5,110]"}


def build_one(phase, cell, seed, arm, kappa, rounds, budget, checkpoint=None, reference=None,
              smoke_size=None, confirmation_gate=None):
    aggr = "datasize" if arm == "Oracle-k0-datasize" else "eps"
    stage = "test" if phase.startswith("stage3") else "smoke" if smoke_size else "dev"
    tag = f'{cell["dataset"]}_{cell["mode"]}{cell["value"]:g}_a{cell["alpha"]}_s{seed}/{arm}'
    reports, split = market(cell["dataset"], cell["alpha"], seed, rounds, smoke_size)
    target = cell["value"] if cell["mode"] == "privacy" else None
    detail = allocation(arm, reports, budget, kappa, checkpoint, reference, target, raw_sizes=split["client_sizes"])
    folder = OUT / phase / tag
    spec_path = OUT / "allocations" / phase / tag / "allocation.json"
    inputs = [CONFIG, AMENDMENT, OUT / "kappa_hat.json"]
    inputs += [p for p in (checkpoint, reference) if p is not None]
    if arm in REFERENCES:
        inputs.append(REFERENCES[arm][0])
    if confirmation_gate:
        inputs.append(confirmation_gate)
    spec = {**cell, "seed": seed, "arm": arm, "kappa": float(kappa), "budget": float(budget),
            "rounds": rounds, "aggregation": aggr, "stage": stage, "smoke_size": smoke_size,
            "comparison": cell, "split_sha256": split["sha256"], "reports_sha256": array_hash(reports),
            "allocation_sha256": array_hash(detail["executed"]), "payment_sha256": array_hash(detail["payment"]),
            "input_sha256": {str(p): digest(p) for p in inputs},
            "non_strategic_reference": arm.startswith("Oracle-") or arm.startswith("Aligned@"),
            "voluntary": detail["voluntary"],
            "reference_sha256": digest(reference) if reference else None,
            "confirmation_gate_sha256": digest(confirmation_gate) if confirmation_gate else None,
            **{name: detail[name].double().tolist() for name in ("proposed", "offered_payment", "executed", "payment")}}
    spec["precomputed_metrics"] = exact_metrics(detail["executed"], reports, kappa)
    if arm in ("mf_p2", "Uniform-DP@mf_p2"):
        spec["verified_raw_sizes"] = split["client_sizes"]
        spec["mf_input_rule"] = "verified raw sizes; value/cap draws and P2 reporting shares unchanged"
    if arm.startswith("Aligned@"):
        reference_e = allocation(arm.split("@", 1)[1], reports, budget, kappa)["executed"]
        diff = detail["executed"].sum(-1) - reference_e.double().sum(-1)
        spec["aligned_total_difference"] = diff.tolist()
    from a3_fingerprint import fingerprint
    fl_identity = fingerprint(spec, split, reports)
    spec["fl_fingerprint"] = fl_identity["sha256"]
    immutable_json(spec_path.with_name("fl_inputs.json"), fl_identity)
    immutable_json(spec_path, spec)
    cmd = [sys.executable, "-u", str(WORK / "exp/run_cell.py"), "--dataset", cell["dataset"],
           "--alpha", str(cell["alpha"]), "--seed", str(seed), "--arm", arm, "--budget", repr(float(budget)),
           "--stage", stage, "--rounds", str(rounds), "--aggr", aggr, "--out", str(folder),
           "--allocation-spec", str(spec_path), "--spec-sha256", digest(spec_path)]
    actual_checkpoint = checkpoint or (REFERENCES[arm][0] if arm in REFERENCES else None)
    if actual_checkpoint:
        cmd += ["--checkpoint", str(actual_checkpoint)]
    if reference:
        cmd += ["--reference", str(reference)]
    if smoke_size:
        cmd += ["--smoke-size", str(smoke_size), "--eval-every", "1"]
    item = job(tag, cmd, folder, "result.json", gpu=True)
    item.update({"phase": phase, "cell": cell, "seed": seed, "arm": arm, "kappa": float(kappa),
                 "budget": float(budget), "allocation_spec": str(spec_path), "allocation_spec_sha256": digest(spec_path)})
    item["fl_fingerprint"] = spec["fl_fingerprint"]
    return item


def prepare(phase, cells, seeds, arms, rounds, decision=None, reference=None, smoke_size=None,
            authorization=None):
    saved = OUT / f"jobs_{phase}.json"
    if saved.exists():
        jobs = read(saved)
        for j in jobs:
            if digest(j["allocation_spec"]) != j["allocation_spec_sha256"]:
                raise RuntimeError("saved job allocation hash changed")
        return jobs
    jobs, calibrations, excluded = [], [], []
    anchor_arm = "mf_p2" if any(a["arm"] == "Uniform-DP@mf_p2" for a in arms) else "mfg_p2"
    for cell in cells:
        for seed in seeds:
            reports, split = market(cell["dataset"], cell["alpha"], seed, 100, smoke_size)
            matched_reference_budget = None
            for entry in arms:
                arm = entry["arm"]
                kappa = arm_kappa(arm, cell["dataset"], cell["alpha"], decision)
                ckpt = entry.get("checkpoint")
                budget = float(cell["value"])
                if cell["mode"] == "privacy":
                    if arm in ("Uniform-DP", "Uniform-DP@mf_p2"):
                        if matched_reference_budget is None:
                            excluded.append({**cell, "seed": seed, "arm": arm,
                                             "reason": "selected MFG-P2 anchor has no matching budget"})
                            continue
                        try:
                            got = allocation(arm, reports, matched_reference_budget, kappa, reference=reference,
                                             raw_sizes=split["client_sizes"])
                        except AllocationInfeasible as error:
                            excluded.append({**cell, "seed": seed, "arm": arm, "reason": str(error)})
                            continue
                        amount = float(got["executed"].double().mean())
                        row = {"budget": matched_reference_budget, "mean_100_rounds": amount,
                               "relative_error_100": amount / cell["value"] - 1,
                               "feasible": abs(amount / cell["value"] - 1) <= .005,
                               "reason": "use the selected MFG-P2 budget to match its per-round totals"}
                    else:
                        row = calibration(reports, arm, ckpt, cell["value"], kappa, reference, split["client_sizes"])
                    calibrations.append({**cell, "seed": seed, "arm": arm, **row})
                    if not row["feasible"]:
                        excluded.append(calibrations[-1])
                        continue
                    budget = row["budget"]
                if arm == anchor_arm:
                    matched_reference_budget = budget
                try:
                    item = build_one(phase, cell, seed, arm, kappa, rounds, budget, ckpt, reference, smoke_size,
                                     authorization or (OUT / "stage2_selection.json" if phase.startswith("stage3") else None))
                except AllocationInfeasible as error:
                    excluded.append({**cell, "seed": seed, "arm": arm, "reason": str(error)})
                    continue
                jobs.append(item)
                update(f"preparing_{phase}", status="running", prepared_jobs=len(jobs),
                       excluded=len(excluded), current=item["id"])
    immutable_json(OUT / f"calibration_{phase}.json", {"rows": calibrations, "exclusions": excluded})
    immutable_json(saved, jobs)
    return jobs

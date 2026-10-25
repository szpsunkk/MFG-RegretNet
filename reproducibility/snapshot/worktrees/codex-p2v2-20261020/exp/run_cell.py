#!/usr/bin/env python3
"""One repaired RQ4 cell, with exact accounting, event evidence and resumable state."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common import (FROZEN, WORK, OUT, RepairedRunner, atomic_json, data_for_cell, digest,
                    evaluate, isolated_rng, market_reports, privacy_summary)
from mechanisms import OLD_MODELS, V8F, allocate
from tdsc_dpsgd_runner import DPSGDClientRequest, DPSGDConfig, DPSGDAccountingEvent, EXPANDED_RDP_ORDERS
from tdsc_rdp_cache import exact_rdp_cache
from p2_core import array_hash, exact_metrics
from p2_state import check_identity, read, CONFIG, AMENDMENT


def save_resume(path, runner, next_round, records, measurements, identity, elapsed):
    state = {"identity": identity, "next_round": next_round, "records": records,
             "measurements": measurements, "elapsed_seconds": elapsed,
             "model": {k: v.cpu() for k, v in runner.model.state_dict().items()},
             "ledgers": [ledger.export() for ledger in runner._external_ledgers],
             "opacus": [list(acc.history) for acc in runner._opacus_accountants]}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def restore_resume(path, runner, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise RuntimeError("resume identity differs: cannot mix protocol versions")
    runner.model.load_state_dict(state["model"])
    for ledger, accountant, saved, history in zip(runner._external_ledgers, runner._opacus_accountants,
                                                  state["ledgers"], state["opacus"]):
        ledger.state = np.asarray(saved["rdp_state"], dtype=float)
        ledger.state_version = saved["state_version"]
        ledger.events = [(e["event_id"], DPSGDAccountingEvent(e["sample_rate"], e["optimizer_steps"], e["noise_multiplier"]))
                         for e in saved["events"]]
        ledger._committed_ids = {name for name, _ in ledger.events}
        runner._committed_training_event_ids.update(ledger._committed_ids)
        accountant.history = history
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["MNIST", "CIFAR10"], required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--reference", type=Path, default=V8F)
    ap.add_argument("--stage", choices=["dev", "test", "smoke"], required=True)
    ap.add_argument("--aggr", choices=["eps", "datasize"], default="eps")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-cumulative-epsilon", type=float)
    ap.add_argument("--smoke-size", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allocation-spec", type=Path, required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--shared-from", type=Path)
    a = ap.parse_args()
    if a.rounds < 1 or a.eval_every < 1 or a.budget <= 0:
        ap.error("rounds, eval-every and budget must be positive")
    torch.set_num_threads(1)
    check_identity()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a.out.mkdir(parents=True, exist_ok=True)
    # Shared pilot/coordinator jobs may refer to the same cell. Serialize them
    # before checking completion, loading state or opening any output artifact.
    run_lock = open(a.out / ".run.lock", "a")
    fcntl.flock(run_lock, fcntl.LOCK_EX)
    final = a.out / "result.json"
    clients, validation, split, factory = data_for_cell(a.dataset, a.alpha, a.seed, a.batch_size, a.stage, a.smoke_size)
    sizes = [len(client) for client in clients]
    reports = market_reports(a.rounds, sizes, a.seed)
    if digest(a.allocation_spec) != a.spec_sha256:
        raise RuntimeError("allocation specification differs from job manifest hash")
    spec = read(a.allocation_spec)
    expected = {"dataset": a.dataset, "alpha": a.alpha, "seed": a.seed, "arm": a.arm,
                "budget": a.budget, "rounds": a.rounds, "aggregation": a.aggr,
                "stage": a.stage, "smoke_size": a.smoke_size}
    if any(spec[k] != value for k, value in expected.items()):
        raise RuntimeError("job differs from frozen allocation specification")
    if spec["reports_sha256"] != array_hash(reports.numpy()) or spec["split_sha256"] != split["sha256"]:
        raise RuntimeError("reports or client partition differs from precomputation")
    if a.arm in ("mf_p2", "Uniform-DP@mf_p2") and spec.get("verified_raw_sizes") != sizes:
        raise RuntimeError("MF-Net raw size inputs differ from actual datasets")
    if a.stage == "test" and not spec.get("confirmation_gate_sha256"):
        raise RuntimeError("official test access requires frozen Stage 2 decision")
    for path, expected_hash in spec["input_sha256"].items():
        if digest(path) != expected_hash:
            raise RuntimeError(f"stage input changed: {path}")
    epsilon, payments = np.asarray(spec["executed"], float), np.asarray(spec["payment"], float)
    if array_hash(epsilon) != spec["allocation_sha256"] or array_hash(payments) != spec["payment_sha256"]:
        raise RuntimeError("allocation/payment content hash differs")
    if epsilon.shape != (a.rounds, 10) or payments.shape != epsilon.shape:
        raise ValueError("invalid precomputed allocation shape")
    lr = .1 if a.dataset == "MNIST" else .05
    cfg = DPSGDConfig(noise_multiplier=1., max_grad_norm=1., learning_rate=lr, local_epochs=1,
                      batch_size=a.batch_size, delta=1e-5, rdp_orders=EXPANDED_RDP_ORDERS,
                      dataset_name=f"{a.dataset}-a{a.alpha}-repair-v1")
    with isolated_rng(a.seed):
        initial = factory()
    runner = RepairedRunner(initial, 10, cfg, seed=a.seed, device=a.device)
    source_paths = sorted(FROZEN.glob("*.py")) + [Path(__file__).resolve(), WORK / "exp/common.py", WORK / "exp/mechanisms.py"]
    source_paths += sorted((FROZEN / "baselines").rglob("*.py"))
    checkpoint = a.checkpoint or (OLD_MODELS[a.arm][1] if a.arm in OLD_MODELS else None)
    source_paths += [WORK / "exp/contracts.py"]
    meta = {"protocol": "p2-v2-A4-A5-20261020", "stage": a.stage, "dataset": a.dataset, "alpha": a.alpha,
            "seed": a.seed, "budget": a.budget, "arm": a.arm, "rounds": a.rounds,
            "aggregation": a.aggr, "config": asdict(cfg), "client_sizes": sizes,
            "split_sha256": split["sha256"], "source_sha256": {str(p.relative_to(WORK)): digest(p) for p in source_paths},
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
            "reference_sha256": digest(a.reference) if a.arm == "Uniform-DP" else None,
            "allocation_sha256": hashlib.sha256(epsilon.tobytes()).hexdigest(),
            "max_cumulative_epsilon_add_remove": a.max_cumulative_epsilon,
            "contract_rule": "all-or-nothing delivery; epsilon >= 0.01; exact DP calibration before payment",
            "public_validation": "classwise second half of reserved training records; first half for proxy training",
            "comparison": "fixed economic budget; Uniform-DP matches per-round total epsilon only",
            "evaluation_data": "official test set" if a.stage == "test" else "held-out original training records",
            "torch_version": torch.__version__, "numpy_version": np.__version__, "device": a.device,
            "smoke_size": a.smoke_size, "eval_every": a.eval_every}
    meta.update({"protocol_sha256": digest(CONFIG), "amendment_sha256": digest(AMENDMENT),
                 "kappa": spec["kappa"], "allocation_spec_sha256": digest(a.allocation_spec),
                 "source_sha256": read(OUT / "launch_manifest.json")["source_sha256"],
                 "input_sha256": spec["input_sha256"], "reference_sha256": spec.get("reference_sha256"),
                 "comparison": spec["comparison"], "non_strategic_reference": spec["non_strategic_reference"]})
    if "verified_raw_sizes" in spec:
        meta.update(verified_raw_sizes=spec["verified_raw_sizes"], mf_input_rule=spec["mf_input_rule"])
    if "fl_fingerprint" in spec:
        meta["fl_fingerprint"] = spec["fl_fingerprint"]
    meta["endpoint_sealed_until_stage_complete"] = True
    identity = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    if final.exists():
        saved = json.loads(final.read_text())
        if saved["identity"] != identity:
            raise RuntimeError("completed run identity differs: cannot reuse another configuration")
        for name in ("events", "privacy"):
            if saved[name + "_sha256"] != digest(a.out / (name + ".json")):
                raise RuntimeError("completed run evidence hash mismatch")
        print(f"COMPLETE {final}", flush=True)
        run_lock.close()
        return
    resume = a.out / "resume.pt"
    if a.shared_from:
        from a3_shared import materialize
        materialize(a.shared_from, a.out, meta, identity, spec, reports.numpy())
        run_lock.close()
        return
    records, measurements, start, elapsed_before = [], [], 0, 0.
    if resume.exists():
        state = restore_resume(resume, runner, identity)
        records, measurements = state["records"], state["measurements"]
        start, elapsed_before = state["next_round"], state["elapsed_seconds"]
    else:
        measurements.append({"round": 0, **evaluate(runner.model, validation, a.device)})
    atomic_json(a.out / "meta.json", meta)
    began = time.monotonic()
    for r in range(start, a.rounds):
        check_identity()
        requests, unavailable = [], []
        for i, eps in enumerate(epsilon[r]):
            if eps <= 0:
                continue
            try:
                runner.preview_participation(client_id=i, dataset_size=sizes[i], round_index=r,
                                             allocated_event_epsilon=float(eps))
            except ValueError as error:
                if "privacy budget is too low" not in str(error).lower():
                    raise
                unavailable.append({"client_id": i, "reason": str(error), "payment_settled": False})
                continue
            requests.append(DPSGDClientRequest(client_id=i, dataset=clients[i], allocated_event_epsilon=float(eps),
                            aggregation_weight=float(eps) if a.aggr == "eps" else float(sizes[i]),
                            payment=float(payments[r, i]), max_acceptable_event_epsilon=float(reports[r, i, 1]),
                            max_epsilon_add_remove=a.max_cumulative_epsilon))
        outcome = runner.run_round(r, requests)
        released = outcome.released_client_ids
        if unavailable or sorted(released) != np.flatnonzero(epsilon[r] > 0).tolist():
            atomic_json(a.out / "allocation_execution_mismatch.json", {
                "round": r + 1, "expected": np.flatnonzero(epsilon[r] > 0).tolist(),
                "released": released, "unavailable": unavailable, "outcome": asdict(outcome)})
            raise RuntimeError("precomputed settlement differs from actual calibrated release; stop for review")
        records.append({"round": r + 1, "reports": reports[r].tolist(), "proposed_epsilon": spec["proposed"][r],
                        "proposed_payments": spec["offered_payment"][r], "calibration_refusals": unavailable,
                        "executed_epsilon": epsilon[r].tolist(), "settled_payments": payments[r].tolist(),
                        "outcome": asdict(outcome),
                        "executed_epsilon_total": float(epsilon[r, released].sum()),
                        "true_settled_utility": float(sum(payments[r, i] - float(reports[r, i, 0]) * epsilon[r, i]
                                                          for i in released))})
        if (r + 1) % a.eval_every == 0 or r >= a.rounds - 5:
            metrics = evaluate(runner.model, validation, a.device)
            measurements.append({"round": r + 1, **metrics})
            elapsed = elapsed_before + time.monotonic() - began
            progress = {"status": "running", "round": r + 1, "total_rounds": a.rounds,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed / (r + 1) * (a.rounds - r - 1), "identity": identity}
            atomic_json(a.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if (r + 1) % 10 == 0 or r + 1 == a.rounds:
            save_resume(resume, runner, r + 1, records, measurements, identity,
                        elapsed_before + time.monotonic() - began)
    snapshot = runner.privacy_snapshot(include_prv_crosscheck=False)
    if not snapshot["all_accountant_parity_pass"]:
        raise RuntimeError("accountant parity failed")
    final_measurements = [m for m in measurements if m["round"] > a.rounds - min(5, a.rounds)]
    if len(final_measurements) != min(5, a.rounds):
        raise AssertionError("endpoint must contain exactly the last five measured rounds")
    # Keep the scheduled raw evaluations, but defer every endpoint average to
    # the complete-stage analyzer. A partial campaign has no derived accuracy.
    summary = {name: None for name in ("accuracy", "macro_f1", "loss")}
    summary.update({"seconds": elapsed_before + time.monotonic() - began,
                    "payment_per_round": float(np.mean([r["outcome"]["settled_payment_total"] for r in records])),
                    "epsilon_per_client_round": float(np.mean([r["executed_epsilon_total"] for r in records]) / 10),
                    "participants_per_round": float(np.mean([r["outcome"]["release_count"] for r in records])),
                    "zero_release_round_fraction": float(np.mean([r["outcome"]["release_count"] == 0 for r in records])),
                    "privacy": privacy_summary(snapshot), "accountant_parity_pass": True})
    summary.update(exact_metrics(epsilon, reports.numpy(), spec["kappa"]))
    atomic_json(a.out / "events.json", records)
    atomic_json(a.out / "privacy.json", snapshot)
    atomic_json(final, {"meta": meta, "identity": identity, "summary": summary, "measurements": measurements,
                        "events_sha256": digest(a.out / "events.json"), "privacy_sha256": digest(a.out / "privacy.json")})
    atomic_json(a.out / "progress.json", {"status": "complete", "round": a.rounds,
                                         "total_rounds": a.rounds, "elapsed_seconds": summary["seconds"]})
    print(json.dumps({"complete": str(final), "rounds": a.rounds}), flush=True)
    run_lock.close()


if __name__ == "__main__":
    with exact_rdp_cache(maxsize=262144):
        main()

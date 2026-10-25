"""Materialize an economic arm around a verified identical FL treatment."""
import copy
from pathlib import Path
import shutil
import time
import numpy as np
from common import OUT, atomic_json, digest
from p2_state import read
from p2_core import exact_metrics


def materialize(source, target, meta, identity, spec, reports):
    started = time.monotonic()
    source, target = Path(source), Path(target)
    result = read(source / "result.json")
    expected = spec.get("fl_fingerprint")
    if not expected or result["meta"].get("fl_fingerprint") != expected:
        raise RuntimeError("shared execution needs equal precomputed FL fingerprints")
    for name in ("events", "privacy"):
        if digest(source / (name + ".json")) != result[name + "_sha256"]:
            raise RuntimeError("shared source evidence changed")
    events = copy.deepcopy(read(source / "events.json"))
    e, p = np.asarray(spec["executed"]), np.asarray(spec["payment"])
    if len(events) != len(e):
        raise RuntimeError("shared source is incomplete")
    for r, event in enumerate(events):
        if event["executed_epsilon"] != spec["executed"][r]:
            raise RuntimeError("fingerprint/evidence allocation mismatch")
        event["proposed_epsilon"] = spec["proposed"][r]
        event["proposed_payments"] = spec["offered_payment"][r]
        event["settled_payments"] = spec["payment"][r]
        outcome = event["outcome"]
        released = outcome["released_client_ids"]
        if released != np.flatnonzero(e[r] > 0).tolist() or outcome["skipped_events"]:
            raise RuntimeError("unexpected shared execution acceptance")
        for row in outcome["participation_events"]:
            row["payment"] = float(p[r, row["client_id"]])
        for key in ("selected_payment_total", "settled_payment_total", "payment_total"):
            outcome[key] = sum(float(p[r, i]) for i in released)
        event["true_settled_utility"] = float(sum(p[r, i] - float(reports[r, i, 0]) * e[r, i] for i in released))
    summary = copy.deepcopy(result["summary"])
    summary["payment_per_round"] = float(np.mean([r["outcome"]["settled_payment_total"] for r in events]))
    summary.update(exact_metrics(e, reports, spec["kappa"]))
    # Accuracy, F1 and loss are copied verbatim, never re-averaged or inspected.
    summary["seconds"] = 0.
    receipt = {"shared_execution_with": str(source), "source_result_sha256": digest(source / "result.json"),
               "fl_fingerprint": expected, "canonical_training_seconds": result["summary"]["seconds"],
               "economic_record_rebuilt": True, "independent_training": False}
    atomic_json(target / "events.json", events)
    shutil.copy2(source / "privacy.json", target / "privacy.json")
    atomic_json(target / "meta.json", meta)
    atomic_json(target / "sharing.json", receipt)
    atomic_json(target / "result.json", {"meta": meta, "identity": identity, "summary": summary,
                "measurements": result["measurements"], **receipt,
                "events_sha256": digest(target / "events.json"), "privacy_sha256": digest(target / "privacy.json")})
    atomic_json(target / "progress.json", {"status": "complete", "round": len(events),
                "total_rounds": len(events), "elapsed_seconds": time.monotonic() - started,
                "shared_execution_with": str(source)})

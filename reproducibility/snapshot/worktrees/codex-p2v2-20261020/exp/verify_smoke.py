"""Independently recompute small actual FL runs and preserve their checks."""
import json
from pathlib import Path

import numpy as np
import torch

from common import ROOT, WORK, OUT, atomic_json, digest
from contracts import MIN_EVENT_EPSILON
from mechanisms import allocate


def read(path):
    return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(1)
    checked = []
    for dataset in ("MNIST", "CIFAR10"):
        folder = OUT / "smoke" / dataset
        result, events = read(folder / "result.json"), read(folder / "events.json")
        assert result["meta"]["stage"] == "smoke"
        assert len(events) == 5
        for field in ("events", "privacy"):
            assert result[field + "_sha256"] == digest(folder / (field + ".json"))
        total_epsilon, total_payment, releases = 0., 0., 0
        for row in events:
            e = np.asarray(row["proposed_epsilon"])
            p = np.asarray(row["proposed_payments"])
            true = np.asarray(row["reports"])
            released = row["outcome"]["released_client_ids"]
            assert np.all(e[released] >= MIN_EVENT_EPSILON)
            assert np.all(e[released] <= true[released, 1] + 1e-7)
            assert np.all(p[released] - true[released, 0] * e[released] >= -1e-9)
            assert p[released].sum() <= result["meta"]["budget"] + 2e-5
            assert abs(row["executed_epsilon_total"] - e[released].sum()) < 1e-9
            assert abs(row["outcome"]["settled_payment_total"] - p[released].sum()) < 1e-8
            assert not row["calibration_refusals"]
            for event in row["outcome"]["participation_events"]:
                assert event["accountant_parity_pass"]
                assert event["optimizer_steps"] == event["opacus_hook_optimizer_steps"]
                assert event["realized_isolated_event_epsilon"] <= event["allocated_event_epsilon"] + 1e-9
            releases += len(released)
            total_epsilon += e[released].sum()
            total_payment += p[released].sum()
        metrics = [r for r in result["measurements"] if r["round"] > 0]
        assert len(metrics) == 5
        for row in metrics:
            confusion = np.asarray(row["confusion"])
            assert confusion.sum() == row["examples"]
            assert abs(np.trace(confusion) / confusion.sum() - row["accuracy"]) < 1e-12
        for field in ("accuracy", "macro_f1", "loss"):
            assert abs(np.mean([r[field] for r in metrics]) - result["summary"][field]) < 1e-12
        assert abs(total_epsilon / 50 - result["summary"]["epsilon_per_client_round"]) < 1e-10
        assert abs(total_payment / 5 - result["summary"]["payment_per_round"]) < 1e-10
        checked.append({"dataset": dataset, "rounds": len(events), "releases": releases,
                        "result_sha256": digest(folder / "result.json"), "passed": True})
    original = read(OUT / "smoke/MNIST/result.json")
    resumed = read(OUT / "smoke/MNIST_resume/result.json")
    assert original["identity"] == resumed["identity"]
    assert original["measurements"] == resumed["measurements"]
    assert original["events_sha256"] == resumed["events_sha256"]
    assert original["privacy_sha256"] == resumed["privacy_sha256"]
    for key in original["summary"]:
        if key != "seconds":
            assert original["summary"][key] == resumed["summary"][key]
    protected = read(ROOT / "main_final2/revisions/codex_rq4_repair_20261016/paper_code_audit_20261017/evidence.json")["protected_file_hashes"]
    unchanged = {name: digest(ROOT / name) == value for name, value in protected.items()}
    assert all(unchanged.values()), "a protected historical artifact changed; inspect before continuing"
    frozen = sorted((WORK / "frozen").rglob("*.py"))
    assert all(digest(p) == digest(WORK.parent / "codex-rq4-repair-20261016/frozen" / p.relative_to(WORK / "frozen")) for p in frozen)
    # Exercise all analytical arms before any expensive confirmation queue.
    g = torch.Generator().manual_seed(701)
    reports = torch.rand(16, 10, 3, generator=g)
    reports[..., 1] = .1 + 4.9 * reports[..., 1]
    reports[..., 2] = 10 + 490 * reports[..., 2]
    arms = ["PAC", "VCG", "PAC-capped-legacy", "VCG-capped-legacy", "CSRA", "MFG-Pricing", "Lin",
            "Posted-price", "Uniform-DP", "Balanced-1", "Balanced-2"]
    for arm in arms:
        for budget in (10., 30., 100.):
            e, p = allocate(arm, reports, budget)
            assert torch.isfinite(e).all() and torch.isfinite(p).all()
            assert (p.sum(-1) <= budget + 2e-5).all()
            assert (e <= reports[..., 1] + 1e-7).all()
            if arm != "Uniform-DP":
                assert (p - reports[..., 0] * e >= -1e-7).all()
    value = {"actual_fl_smoke": checked, "completed_checkpoint_restore_exact": True,
             "protected_artifacts_unchanged": unchanged, "identical_frozen_python_files": len(frozen),
             "analytical_arms_checked": arms, "budgets": [10, 30, 100],
             "scope": "functional checks only; small smoke accuracy is not a model comparison"}
    atomic_json(OUT / "smoke/validation.json", value)
    print(json.dumps({"passed": True, "actual_fl_runs": len(checked), "release_events": sum(r["releases"] for r in checked),
                      "protected_files": len(unchanged), "output": str(OUT / "smoke/validation.json")}))


if __name__ == "__main__":
    main()

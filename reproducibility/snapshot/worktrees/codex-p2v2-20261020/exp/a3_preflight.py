"""Non-scientific A3 hardware probes, two-host regression and twin validation."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import importlib.metadata
import platform
import socket

import torch
from common import ROOT, WORK, OUT, atomic_json, digest
from p2_state import read, immutable_json, check_identity
from p2_jobs import build_one, arm_kappa
from a3_resources import resources, can_start, environment, gpu_pids, token, RESERVE_MIB

BASE = ROOT / "run/tdsc_revision/codex_p2_101_20261019"


def run_batch(jobs, label):
    folder = OUT / "operations" / label
    done_path = folder / "timing.json"
    if done_path.exists():
        return read(done_path)
    live, samples = [], []
    began = time.monotonic()
    start = resources()
    for item in jobs:
        pending = len([p for p, _ in live if p.poll() is None and p.pid not in gpu_pids()])
        if not can_start(resources(), pending):
            raise RuntimeError("insufficient memory for concurrent probe; no unsafe admission")
        path = Path(item["out"])
        path.mkdir(parents=True, exist_ok=True)
        log = (path / "job.log").open("a")
        proc = subprocess.Popen(item["command"], cwd=WORK, env=environment(), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        live.append((proc, log))
        time.sleep(.5)
    while any(p.poll() is None for p, _ in live):
        check_identity()
        sample = resources()
        samples.append(sample)
        atomic_json(folder / "status.json", {"status": "running", "active": sum(p.poll() is None for p, _ in live),
                    "total": len(live), "seconds": time.monotonic() - began, "resources": sample})
        time.sleep(2)
    for p, log in live:
        log.close()
        if p.returncode:
            raise RuntimeError(f"probe failed with {p.returncode}; see {folder}")
    elapsed = time.monotonic() - began
    final = resources()
    samples += [start, final]
    for j in jobs:
        if not Path(j["done"]).exists():
            raise RuntimeError("probe missing completed evidence")
    value = {"k": len(jobs), "seconds": elapsed,
             "aggregate_rounds_per_hour": sum(read(j["allocation_spec"])["rounds"] for j in jobs) * 3600 / elapsed,
             "min_free_mib": min(r["free_mib"] for r in samples), "max_used_mib": max(r["used_mib"] for r in samples),
             "swap_in_delta": final["pswpin"] - start["pswpin"], "swap_out_delta": final["pswpout"] - start["pswpout"],
             "samples": samples, "scientific_use": False, "threads_per_process": 1,
             "jobs": [{"out": j["out"], "result_sha256": digest(j["done"])} for j in jobs]}
    value["eligible"] = value["min_free_mib"] >= RESERVE_MIB and value["swap_in_delta"] == value["swap_out_delta"] == 0
    atomic_json(done_path, value)
    return value


def copy_job(item, out):
    new = copy.deepcopy(item)
    new["command"][0] = sys.executable
    new["command"][new["command"].index("--out") + 1] = str(out)
    new["out"], new["done"] = str(out), str(out / "result.json")
    return new


def probes(machine):
    target = OUT / "operations" / machine / "concurrency.json"
    if target.exists():
        return read(target)
    cell = {"dataset": "CIFAR10", "alpha": .5, "mode": "budget", "value": 30.}
    kappa = arm_kappa("PAC", "CIFAR10", .5)
    job = build_one("ops_probe_" + machine, cell, 14, "PAC", kappa, 3, 30.)
    rows = []
    for k in (4, 6, 8, 10):
        label = f"{machine}/probe_k{k}"
        copies = [copy_job(job, OUT / "operations" / label / f"copy{i}") for i in range(k)]
        row = run_batch(copies, label)
        rows.append(row)
        print(json.dumps({"machine": machine, **{n: row[n] for n in ("k", "aggregate_rounds_per_hour", "min_free_mib", "eligible")}}), flush=True)
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise RuntimeError("no safe measured concurrency")
    best = max(eligible, key=lambda x: (x["aggregate_rounds_per_hour"], -x["k"]))
    value = {"machine": machine, "selected_k": best["k"], "rate_rounds_per_hour": best["aggregate_rounds_per_hour"],
             "probe_dataset": "CIFAR10", "full_partition": True, "rounds": 3, "seed": 14,
             "rows": rows, "threads": 1, "thread_change_requested": False, "peak_reservation_mib": 1536}
    immutable_json(target, value)
    return value


def regression():
    target = OUT / "operations/local/regression.json"
    if target.exists():
        return read(target)
    original = read(BASE / "jobs_stage0.json")
    jobs, originals = [], []
    for ds in ("MNIST", "CIFAR10"):
        old = next(j for j in original if j["arm"] == "PAC" and j["cell"]["dataset"] == ds)
        spec = read(old["allocation_spec"])
        assert spec["rounds"] == 40 and Path(old["done"]).exists()
        new = copy_job(old, OUT / "operations/local/regression" / ds)
        new["command"][2] = str(WORK / "exp/run_cell.py")
        jobs.append(new)
        originals.append(old)
    run_batch(jobs, "local/regression_runs")
    rows = []
    for old, new in zip(originals, jobs):
        a, b = read(old["done"]), read(new["done"])
        comparisons = {"split": a["meta"]["split_sha256"] == b["meta"]["split_sha256"],
                       "allocation": a["meta"]["allocation_sha256"] == b["meta"]["allocation_sha256"],
                       "events": a["events_sha256"] == b["events_sha256"],
                       "privacy": a["privacy_sha256"] == b["privacy_sha256"]}
        # This diagnostic is explicitly allowed by A3.2, after full Stage 0.
        a_acc = {m["round"]: m["accuracy"] for m in a["measurements"]}
        differences = [{"round": m["round"], "difference_pp": 100 * (m["accuracy"] - a_acc[m["round"]])} for m in b["measurements"]]
        rows.append({"dataset": old["cell"]["dataset"], "original_job": old["id"], "checks": comparisons,
                     "original_result_sha256": digest(old["done"]), "local_result_sha256": digest(new["done"]),
                     "accuracy_differences_diagnostic_only": differences, "pass": all(comparisons.values())})
    value = {"local_eligible": all(r["pass"] for r in rows), "rows": rows,
             "event_hash_definition": "whole original events.json, including model/update digests; no relaxed projection"}
    immutable_json(target, value)
    return value


def twin_check(machine):
    target = OUT / "operations" / machine / "shared_validation.json"
    if target.exists():
        return read(target)
    jobs = []
    for arm in ("PAC", "VCG"):
        cell = {"dataset": "MNIST", "alpha": .5, "mode": "budget", "value": 100.}
        jobs.append(build_one("ops_twins_" + machine, cell, 14, arm, arm_kappa(arm, "MNIST", .5), 3, 100.))
    equal = jobs[0]["fl_fingerprint"] == jobs[1]["fl_fingerprint"]
    if equal:
        run_batch(jobs, machine + "/twin_runs")
        results = [read(j["done"]) for j in jobs]
        events = [read(Path(j["out"]) / "events.json") for j in jobs]
        trajectory = [[r["outcome"]["model_sha256_after"] for r in ev] for ev in events]
        exact = results[0]["measurements"] == results[1]["measurements"] and trajectory[0] == trajectory[1]
        from a3_shared import materialize
        import numpy as np
        rebuilt = OUT / "operations" / machine / "shared_economic_validation"
        rebuilt.mkdir(parents=True, exist_ok=True)
        materialize(jobs[0]["out"], rebuilt, results[1]["meta"], results[1]["identity"],
                    read(jobs[1]["allocation_spec"]), np.asarray([r["reports"] for r in events[1]]))
        economic_exact = (digest(rebuilt / "events.json") == results[1]["events_sha256"] and
                          digest(rebuilt / "privacy.json") == results[1]["privacy_sha256"])
    else:
        exact, economic_exact = False, False
    value = {"machine": machine, "same_fingerprint": equal, "bitwise_equal_trajectory": exact,
             "enabled": equal and exact and economic_exact, "economic_events_bitwise_equal": economic_exact,
             "arms": [j["id"] for j in jobs], "rounds": 3,
             "full_partition": True, "fl_fingerprint": jobs[0]["fl_fingerprint"],
             "scientific_use": False, "scope": "MNIST full-partition independent twin executions"}
    immutable_json(target, value)
    return value


def profile_run(machine):
    target = OUT / "operations" / machine / "profile" / "timing_breakdown.json"
    if target.exists():
        return
    cell = {"dataset": "CIFAR10", "alpha": .5, "mode": "budget", "value": 30.}
    item = build_one("ops_profile_" + machine, cell, 14, "PAC", arm_kappa("PAC", "CIFAR10", .5), 3, 30.)
    item = copy_job(item, target.parent)
    item["command"][2] = str(WORK / "ops/profile_cell.py")
    run_batch([item], machine + "/profile_timing")
    if not target.exists():
        raise RuntimeError("timing diagnostic did not write its breakdown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", choices=("101", "local"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_identity()
    env_path = OUT / "operations" / args.machine / "environment.json"
    env = {"hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
           "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "opacus", "numpy", "scipy")},
           "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "threads": 1,
           "source_sha256": read(OUT / "launch_manifest.json")["source_sha256"]}
    immutable_json(env_path, env)
    if not read(BASE / "stage0_decision.json")["gate_pass"]:
        raise RuntimeError("Stage 0 failed; no A3 continuation")
    if args.machine == "local":
        parent = read(ROOT / "run/tdsc_revision/codex_taskaware_20261017/status.json")
        jobs_path = ROOT / "run/tdsc_revision/codex_taskaware_20261017/jobs_confirmation.json"
        if jobs_path.exists():
            if len(read(jobs_path)) != 822 or not all(Path(j["done"]).exists() for j in read(jobs_path)):
                raise RuntimeError("all 822 task-aware confirmations must finish first")
        elif parent.get("status") != "complete":
            raise RuntimeError("task-aware completion not established")
        regression()
    probes(args.machine)
    twin_check(args.machine)
    profile_run(args.machine)
    atomic_json(OUT / "operations" / args.machine / "PREFLIGHT_DONE.json", {"complete": True, "time": time.time()})


if __name__ == "__main__":
    main()

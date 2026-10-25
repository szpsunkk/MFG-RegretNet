"""Dependency schedule for P2 and A1, with immutable whole-cell assignments."""
import fcntl
import json
from pathlib import Path
import sys
import time
import torch
from common import ROOT, WORK, OUT, PAPER, atomic_json, digest
from p2_state import CONFIG, read, seal, immutable_json, freeze_stage_inputs, update
from p2_core import FAMILIES, REFERENCES
from p2_jobs import job
from p2_results import collect, stage2_select
from p2_economics import screen
from a1_inputs import inherit, register_mf, BASE
from a1_rules import stage2_candidate_gate, confirmation_rule, confirmation_cells
from p2_campaign import audit_smoke


def cell_key(cell):
    return f'{cell["dataset"]}/{cell["alpha"]:g}/{cell["mode"]}/{cell["value"]:g}'


def assign(stage):
    target = OUT / "assignments" / (stage + ".json")
    if target.exists():
        return True
    remote = OUT / "operations/101/PREFLIGHT_DONE.json"
    local = OUT / "operations/local/PREFLIGHT_DONE.json"
    if not remote.exists() or (stage == "stage3" and not local.exists()):
        return False
    rates, evidence, environments = {}, {}, {}
    for machine in ("101", "local"):
        probe = OUT / "operations" / machine / "concurrency.json"
        if not probe.exists() or not (OUT / "operations" / machine / "PREFLIGHT_DONE.json").exists():
            continue
        if machine == "local" and not read(OUT / "operations/local/regression.json")["local_eligible"]:
            continue
        got = read(probe)
        factor = (got["selected_k"] - 2) / got["selected_k"] if machine == "local" else 1.
        rates[machine] = got["rate_rounds_per_hour"] * factor
        evidence[machine] = digest(probe)
        environments[machine] = digest(OUT / "operations" / machine / "environment.json")
    cells = read(CONFIG)["stage2_development"]["cells"] if stage == "stage2" else confirmation_cells()
    seeds = [14, 15] if stage == "stage2" else [231, 232, 233]
    load, mapping = {m: 0. for m in rates}, {}
    for cell in cells:
        # Equal full-cell workloads. Only measured throughput affects placement.
        machine = min(rates, key=lambda m: ((load[m] + 1) / rates[m], m))
        mapping[cell_key(cell)] = machine
        load[machine] += 1
    value = {"stage": stage, "cells": mapping,
             "blocks": [{"cell": c, "seed": seed, "machine": mapping[cell_key(c)]} for c in cells for seed in seeds],
             "throughput_only": True, "effective_rates": rates, "probe_sha256": evidence,
             "environment_sha256": environments, "local_ln_reserved_slots": 2,
             "late_arms_follow_same_assignment": True,
             "local_pending_at_stage2_assignment": not local.exists(), "created_unix": time.time()}
    immutable_json(target, value)
    immutable_json(target.with_suffix(".sha256.json"), {"sha256": digest(target)})
    return True


def task(name, script, args, folder, done, priority):
    item = job("cpu/" + name, [sys.executable, "-u", str(WORK / "exp" / script), *args], folder, done)
    item["priority"] = priority
    return item


def prepare_task(group, priority):
    return task("prepare_" + group, "a3_prepare.py", ["--group", group], OUT / "preparation" / group,
                "../../jobs_" + group + ".json", priority)


def load_group(group, priority):
    p = OUT / ("jobs_" + group + ".json")
    if not p.exists():
        return []
    jobs = read(p)
    for j in jobs:
        if digest(j["allocation_spec"]) != j["allocation_spec_sha256"]:
            raise RuntimeError("allocation changed after preparation")
        j["id"] = group + "/" + j["id"]
        j["priority"] = priority
        if group == "stage3_p2":
            j["priority"] = {"mfg_p2": 3, "Uniform-DP": 4, "mfg_nob_p2": 5, "regretnet_s_p2": 5}[j["arm"]]
    return jobs


def ready(group):
    p = OUT / ("jobs_" + group + ".json")
    return p.exists() and all(Path(j["done"]).exists() for j in read(p))


def analyze_stage2(mf=False):
    target = OUT / ("mf_stage2_selection.json" if mf else "stage2_selection.json")
    group = "stage2_mf" if mf else "stage2_p2"
    if target.exists() or not ready(group) or not ready("stage2_early"):
        return
    # Both complete job manifests are checked before any result is opened.
    rows = collect(group, read(OUT / ("jobs_" + group + ".json")), analysis_authorized=True)
    rows += collect("stage2_early", read(OUT / "jobs_stage2_early.json"), analysis_authorized=True)
    cells = read(CONFIG)["stage2_development"]["cells"]
    if mf:
        value = stage2_candidate_gate(rows, "mf_p2", cells)
        ckpt = read(OUT / "mf_registration.json")["checkpoint"]
        value["selected"] = {"mf_p2": {"arm": "mf_p2", "kind": "mf_p2", "checkpoint": ckpt,
                                       "checkpoint_sha256": digest(ckpt)}}
    else:
        value = stage2_select(rows, read(OUT / "stage1_screen.json")["candidates"], cells)
    value["decision_input_sha256"] = {str(OUT / ("jobs_" + g + ".json")): digest(OUT / ("jobs_" + g + ".json"))
                                        for g in (group, "stage2_early")}
    immutable_json(target, value)
    freeze_stage_inputs([target])


def analyze_stage3(mf=False):
    target = OUT / ("mf_stage3_decision.json" if mf else "p2_stage3_decision.json")
    group = "stage3_mf" if mf else "stage3_p2"
    if target.exists() or not ready(group) or not ready("stage3_early"):
        return
    rows = collect(group, read(OUT / ("jobs_" + group + ".json")), analysis_authorized=True)
    rows += collect("stage3_early", read(OUT / "jobs_stage3_early.json"), analysis_authorized=True)
    exclusions = read(OUT / ("calibration_" + group + ".json"))["exclusions"]
    exclusions += read(OUT / "calibration_stage3_early.json")["exclusions"]
    value = confirmation_rule(rows, "mf_p2" if mf else "mfg_p2", confirmation_cells(), exclusions=exclusions)
    value["sharing_disclosure"] = ("Equal FL-input fingerprints share one training trajectory and retain separate economic records; "
        "identical treatments therefore do not increase the per-cell maximum through repeated CIFAR GPU nondeterminism.")
    immutable_json(target, value)


def gate_pass(name):
    p = OUT / (name + ".json")
    return p.exists() and read(p)["gate_pass"]


def route_finished(prefix):
    for stage in ("stage1_screen", "stage2_selection"):
        p = OUT / (prefix + stage + ".json")
        if p.exists() and not read(p)["gate_pass"]:
            return True
    return (OUT / ("mf_stage3_decision.json" if prefix else "p2_stage3_decision.json")).exists()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    with (OUT / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        seal()
        for name in ("kappa_hat.json", "stage0_decision.json"):
            inherit(name)
        freeze_stage_inputs([OUT / "kappa_hat.json", OUT / "stage0_decision.json"])
        if not gate_pass("stage0_decision"):
            atomic_json(OUT / "ALL_DONE", {"reason": "Stage 0 gate failed; no A3 scientific continuation"})
            update("complete", status="complete", terminal_stage="stage0_gate_failed")
            return
        config = read(CONFIG)
        while True:
            queue = []
            for f in (OUT / "workers").glob("*_failure.json") if (OUT / "workers").exists() else []:
                raise RuntimeError(f"worker failure: {f}")
            registration = register_mf()
            queue.append(task("bank", "train_p2.py", ["--bank"], OUT / "bank", "../oracle_bank.json", 0))
            if (OUT / "oracle_bank.json").exists():
                for family in FAMILIES:
                    folder = OUT / "mechanisms" / family
                    queue.append(task("train_" + family, "train_p2.py", ["--kind", family, "--out", str(folder)], folder, "training.json", 0))
            # Independent comparator preparations are CPU tasks with lower priority.
            for group in ("stage2_early", "stage3_early"):
                queue.append(prepare_task(group, 20))
                queue += load_group(group, 20)
            queue.append(task("econ_oracle", "p2_economics.py", ["--oracle"], OUT / "economics", "oracle.json", 1))
            if (OUT / "economics/oracle.json").exists():
                entries = [{"arm": name, "checkpoint": str(REFERENCES[name][0])} for name in ("V8F", "RegretNet-r2", "No-population-r2")]
                entries += [{"arm": f, "checkpoint": str(OUT / "mechanisms" / f / "model.pt")} for f in FAMILIES
                            if (OUT / "mechanisms" / f / "training.json").exists()]
                if registration:
                    entries.append({"arm": "mf_p2", "checkpoint": registration["checkpoint"]})
                for e in entries:
                    arm = e["arm"]
                    queue.append(task("econ_" + arm, "p2_economics.py", ["--arm", arm, "--checkpoint", e["checkpoint"]],
                                      OUT / "economics" / arm, "../" + arm + ".json", 6 if arm == "mf_p2" else 1))
            refs_ready = all((OUT / "economics" / (a + ".json")).exists() for a in ("V8F", "RegretNet-r2", "No-population-r2"))
            if refs_ready and all((OUT / "economics" / (a + ".json")).exists() for a in FAMILIES) and not (OUT / "stage1_screen.json").exists():
                screen()
                freeze_stage_inputs([OUT / "stage1_screen.json"])
            if refs_ready and (OUT / "economics/mf_p2.json").exists() and not (OUT / "mf_stage1_screen.json").exists():
                screen(("mf_p2",), "mf_stage1_screen")
                freeze_stage_inputs([OUT / "mf_stage1_screen.json"])
            for suffix, prefix, priority in (("p2", "", 2), ("mf", "mf_", 6)):
                if gate_pass(prefix + "stage1_screen"):
                    smoke_group = "smoke_stage2_" + suffix
                    queue.append(prepare_task(smoke_group, priority))
                    queue += load_group(smoke_group, priority)
                    if ready(smoke_group):
                        proof = OUT / (smoke_group + "_validation.json")
                        if not proof.exists():
                            immutable_json(proof, audit_smoke(read(OUT / ("jobs_" + smoke_group + ".json"))))
                        queue.append(prepare_task("stage2_" + suffix, priority))
                        queue += load_group("stage2_" + suffix, priority)
                        analyze_stage2(mf=suffix == "mf")
                if gate_pass(prefix + "stage2_selection"):
                    queue.append(prepare_task("stage3_" + suffix, priority + 1))
                    queue += load_group("stage3_" + suffix, priority + 1)
                    analyze_stage3(mf=suffix == "mf")
            assign("stage2")
            assign("stage3")
            # No checkpoint means pending, not an invented failure or replacement.
            original_done, mf_done = route_finished(""), route_finished("mf_")
            no_usable_route = original_done and mf_done and not gate_pass("stage2_selection") and not gate_pass("mf_stage2_selection")
            if no_usable_route:
                unused = [j for j in queue if j["gpu"] and j["phase"].endswith("early") and Path(j["done"]).exists()]
                atomic_json(OUT / "run_not_used.json", [{"id": j["id"], "path": j["done"], "status": "run, not used"} for j in unused])
                queue = [j for j in queue if not j["gpu"] or not j["phase"].endswith("early")]
            atomic_json(OUT / "queue.json", {"jobs": queue, "updated_unix": time.time(), "priority_based_on_results": False})
            workers = [read(p) for p in (OUT / "workers").glob("*.json") if not p.name.endswith("_failure.json")] if (OUT / "workers").exists() else []
            complete = sum(Path(j["done"]).exists() for j in queue)
            active = [r for w in workers for r in w.get("active", [])]
            update("a3_dependency_schedule", status="running", complete=complete, total=len(queue), active=active,
                   mf_checkpoint_ready=registration is not None, original_route_finished=original_done,
                   mf_route_finished=mf_done, sealed_results=True,
                   assignment_sha256={s: digest(OUT / "assignments" / (s + ".json")) for s in ("stage2", "stage3")
                                      if (OUT / "assignments" / (s + ".json")).exists()})
            if original_done and mf_done and not active and all(Path(j["done"]).exists() for j in queue):
                atomic_json(OUT / "ALL_DONE", {"completed_unix": time.time(), "status": "complete"})
                update("complete", status="complete", complete=complete, total=len(queue))
                return
            time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update("failed", status="failed", error=repr(error))
        raise

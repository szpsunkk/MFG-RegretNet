#!/usr/bin/env python3
"""A1 continuation: preserve original P2, then screen and confirm the fixed MF-Net."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import sys
import time

import torch

from common import ROOT, WORK, OUT, PAPER, atomic_json, digest
from p2_state import seal, check_identity, read, immutable_json, freeze_stage_inputs, update, CONFIG
from p2_campaign import run_jobs, audit_smoke
from p2_jobs import job, prepare, market, calibration, deploy_kappa, build_one
from p2_core import REFERENCES, FAMILIES, AllocationInfeasible
from p2_economics import screen
from p2_results import collect, report
from a1_inputs import BASE, register_mf, inherit, reuse_jobs, verify_reuse_receipts
from a1_rules import stage2_candidate_gate, confirmation_rule, confirmation_cells


def parent_active():
    marker = str(ROOT / "worktrees/codex-p2-101-20261019/exp/")
    active = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "stat").read_text().split()[2] == "Z":
                continue
            if any(p.decode(errors="replace").startswith(marker) for p in (proc / "cmdline").read_bytes().split(b"\0")):
                active.append(int(proc.name))
        except (OSError, IndexError):
            continue
    return active


def final(reason, terminal):
    report(reason)
    atomic_json(OUT / "ALL_DONE", {"status": "complete", "terminal_stage": terminal,
                                  "reason": reason, "completed_unix": time.time()})
    update("complete", status="complete", terminal_stage=terminal, reason=reason)


def await_parent(poll):
    while True:
        check_identity()
        registration = register_mf()
        parent = read(BASE / "status.json")
        active = parent_active()
        if parent.get("status") == "failed":
            raise RuntimeError("original P2 has an operational failure; preserve it for diagnosis")
        if parent.get("status") == "running" and not active:
            raise RuntimeError("original P2 reports running but has no live controller or children")
        if parent.get("status") == "complete" and not active:
            if not (BASE / "ALL_DONE").exists():
                raise RuntimeError("original P2 complete status lacks terminal evidence")
            return registration
        update("waiting_original_p2", status="waiting", original_phase=parent.get("phase"),
               original_complete=parent.get("complete"), original_total=parent.get("total"),
               original_processes=active, mf_checkpoint_ready=registration is not None,
               reason="A1 is sealed; original P2 retains GPU priority and unchanged stages")
        time.sleep(poll)


def original_uniform(cells, selected, decision):
    """Preserve the original MFG-P2 anchor in addition to the MF-Net control."""
    phase = "stage3_original_uniform"
    saved = OUT / ("jobs_" + phase + ".json")
    if saved.exists():
        return read(saved), read(OUT / ("calibration_" + phase + ".json"))["exclusions"]
    checkpoint = Path(selected["mfg_p2"]["checkpoint"])
    jobs, exclusions = [], []
    for cell in cells:
        for seed in (231, 232, 233):
            reports, split = market(cell["dataset"], cell["alpha"], seed, 100)
            kappa = deploy_kappa(cell["dataset"], cell["alpha"], decision)
            budget = cell["value"]
            if cell["mode"] == "privacy":
                row = calibration(reports, "mfg_p2", checkpoint, cell["value"], kappa)
                if not row["feasible"]:
                    exclusions.append({**cell, "seed": seed, "arm": "Uniform-DP", "reason": "original MFG-P2 anchor unreachable"})
                    continue
                budget = row["budget"]
            try:
                jobs.append(build_one(phase, cell, seed, "Uniform-DP", kappa, 100, budget,
                                      reference=checkpoint, confirmation_gate=OUT / "stage2_selection.json"))
            except AllocationInfeasible as error:
                exclusions.append({**cell, "seed": seed, "arm": "Uniform-DP", "reason": str(error)})
    immutable_json(saved, jobs)
    immutable_json(OUT / ("calibration_" + phase + ".json"), {"exclusions": exclusions})
    return jobs, exclusions


def run_mf(registration, workers, poll=30):
    decision = read(OUT / "stage0_decision.json")
    for name in ("oracle", "V8F", "RegretNet-r2", "No-population-r2"):
        inherit("economics/" + name + ".json")
    freeze_stage_inputs([OUT / "economics" / (name + ".json") for name in
                         ("oracle", "V8F", "RegretNet-r2", "No-population-r2")])
    checkpoint = Path(registration["checkpoint"])
    econ = OUT / "economics/mf_p2"
    command = [sys.executable, "-u", str(WORK / "exp/p2_economics.py"), "--arm", "mf_p2", "--checkpoint", str(checkpoint)]
    run_jobs("mf_economics", [job("mf_p2", command, econ, "../mf_p2.json")], 1)
    screened = screen(("mf_p2",))
    freeze_stage_inputs([OUT / "stage1_screen.json"])
    if not screened["gate_pass"]:
        final("MF-Net未通过原P2同口径经济筛选；保留失败，原P2结论独立。", "mf_stage1_screen_failed")
        return
    entry = {"arm": "mf_p2", "kind": "mf_p2", "checkpoint": str(checkpoint)}
    smoke_cells = [{"dataset": ds, "alpha": .5, "mode": "budget", "value": 30.} for ds in ("MNIST", "CIFAR10")]
    smoke = prepare("smoke_stage2", smoke_cells, [14], [entry], 2, decision, smoke_size=64)
    run_jobs("smoke_stage2", smoke, workers)
    immutable_json(OUT / "smoke_stage2_validation.json", audit_smoke(smoke))
    cells = read(CONFIG)["stage2_development"]["cells"]
    jobs = prepare("stage2", cells, [14, 15], [entry] + [{"arm": a} for a in ("V8F", "CSRA", "PAC", "Oracle-P2")], 40, decision)
    merged = reuse_jobs("stage2", jobs)
    run_jobs("stage2_a1", merged, workers)
    verify_reuse_receipts("stage2")
    rows = collect("stage2", merged)
    gate = stage2_candidate_gate(rows, "mf_p2", cells)
    gate.update(selected={"mf_p2": {**entry, "checkpoint_sha256": digest(checkpoint)}},
                selection_rule="fixed ln-v1 Stage A checkpoint; no P2 seed selection")
    immutable_json(OUT / "stage2_selection.json", gate)
    freeze_stage_inputs([OUT / "stage2_selection.json"])
    report("MF-Net独立Stage 2完成，原mfg_p2门槛不被此结果改写。")
    if not gate["gate_pass"]:
        final("MF-Net未通过独立Stage 2，未启动它的官方测试确认。", "mf_stage2_gate_failed")
        return
    selection_path = BASE / "stage2_selection.json"
    while True:
        check_identity()
        selected = read(selection_path).get("selected", {}) if selection_path.exists() else {}
        missing = sorted(set(FAMILIES) - set(selected))
        if not missing:
            break
        # The amendment requires the full original arm list, not invented selections.
        update("waiting_original_selected_comparators", status="waiting", missing=missing,
               reason="MF-Net passed; complete original comparison list needs selected checkpoints")
        time.sleep(poll)
    freeze_stage_inputs([selection_path, *[Path(selected[k]["checkpoint"]) for k in FAMILIES]])
    cells = confirmation_cells()
    arms = [selected[k] for k in FAMILIES] + [{"arm": a} for a in (
        "V8F", "RegretNet-r2", "No-population-r2", "DM-RegretNet", "MFG-Pricing", "Posted-price",
        "PAC", "PAC-capped-legacy", "VCG", "VCG-capped-legacy", "CSRA", "Lin", "Oracle-P2")]
    arms += [entry, {"arm": "Uniform-DP@mf_p2"}]
    jobs = prepare("stage3", cells, [231, 232, 233], arms, 100, decision, checkpoint)
    extra, excluded_uniform = original_uniform(cells, selected, decision)
    jobs += extra
    merged = reuse_jobs("stage3", jobs)
    immutable_json(OUT / "a1_stage3_jobs.json", merged)
    immutable_json(OUT / "a1_stage3_control_exclusions.json", excluded_uniform)
    freeze_stage_inputs([OUT / "a1_stage3_jobs.json"])
    run_jobs("stage3_a1", merged, workers)
    verify_reuse_receipts("stage3")
    rows = collect("stage3", merged)
    analysis = confirmation_rule(rows, "mf_p2", cells, exclusions=read(OUT / "calibration_stage3.json")["exclusions"])
    immutable_json(OUT / "mf_stage3_decision.json", analysis)
    final("MF-Net独立确认完成；A1统一采用规则及A2映射由同一分析脚本计算。", "mf_stage3_complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--fl-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.fl_workers <= 4 or not 5 <= args.poll_seconds <= 60:
        parser.error("FL workers 1..4, poll seconds 5..60")
    if socket.gethostname() != "lab-server-101":
        raise RuntimeError("A1 computation is authorized on server101, not the local GPU")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        seal()
        if (OUT / "ALL_DONE").exists():
            return
        try:
            registration = await_parent(args.poll_seconds)
            for name in ("kappa_hat.json", "stage0_decision.json"):
                inherit(name)
            freeze_stage_inputs([OUT / "kappa_hat.json", OUT / "stage0_decision.json", BASE / "launch_manifest.json"])
            if not read(OUT / "stage0_decision.json")["gate_pass"]:
                final("原P2 Stage 0未通过；按A1保持该门槛，MF-Net不进入P2。", "stage0_gate_failed")
                return
            while registration is None:
                check_identity()
                update("waiting_mf_stage_a_checkpoint", status="waiting", original_p2_finished=True,
                       reason="requires the one ln-v1 Stage A selection; no substitute model")
                time.sleep(args.poll_seconds)
                registration = register_mf()
            run_mf(registration, args.fl_workers, args.poll_seconds)
        except Exception as error:
            update("failed", status="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()

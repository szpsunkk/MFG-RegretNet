#!/usr/bin/env python3
"""Execute the gated P2 protocol on the separately authorized server101."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from common import ROOT, WORK, OUT, PAPER, atomic_json, digest
from p2_core import FAMILIES, REFERENCES, STAGE0_ARMS, allocation, array_hash
from p2_state import (CONFIG, read, seal, check_identity, immutable_json, note,
                      update, wait_previous, freeze_stage_inputs)
from p2_jobs import job, prepare, market
from p2_results import collect, report, stage0_decision, stage2_select
from p2_economics import screen


def process_token(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        return fields[21] if fields[2] != "Z" else None
    except (OSError, IndexError):
        return None


def memory_available():
    if not torch.cuda.is_available():
        return True
    free, _ = torch.cuda.mem_get_info()
    return free >= 3 * 1024 ** 3


def run_jobs(phase, jobs, workers):
    check_identity()
    immutable_json(OUT / f"jobs_{phase}.json", jobs)
    pending, live, failures = [], {}, []
    done = 0
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    for item in jobs:
        folder = Path(item["out"])
        proc_file = folder / "process.json"
        if Path(item["done"]).exists():
            done += 1
        elif proc_file.exists() and process_token(read(proc_file)["pid"]) == read(proc_file)["start_token"]:
            prior = read(proc_file)
            if prior["command"] != item["command"]:
                raise RuntimeError("live job command differs on reattach")
            live[prior["pid"]] = (None, None, item, prior["start_token"])
        else:
            pending.append(item)
    while pending or live:
        check_identity()
        stopped = (OUT / "STOP_REQUESTED").exists()
        memory_wait = False
        while pending and len(live) < workers and not failures and not stopped:
            if pending[0]["gpu"] and not memory_available():
                memory_wait = True
                break
            item = pending.pop(0)
            folder = Path(item["out"])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "job.log").open("a", buffering=1)
            log.write("\nCOMMAND " + json.dumps(item["command"]) + "\n")
            process = subprocess.Popen(item["command"], cwd=WORK, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            token = process_token(process.pid)
            atomic_json(folder / "process.json", {"pid": process.pid, "start_token": token, "command": item["command"]})
            live[process.pid] = (process, log, item, token)
        for pid, (process, log, item, token) in list(live.items()):
            alive = process.poll() is None if process is not None else process_token(pid) == token
            if alive:
                continue
            if log:
                log.close()
            del live[pid]
            rc = process.returncode if process is not None else None
            if (rc is None or rc == 0) and Path(item["done"]).exists():
                done += 1
            else:
                failures.append({"id": item["id"], "returncode": rc, "out": item["out"]})
        active = []
        for pid, (_, _, item, _) in live.items():
            progress = Path(item["out"]) / "progress.json"
            active.append({"pid": pid, "id": item["id"], "progress": read(progress) if progress.exists() else None})
        update(phase, status="running", complete=done, total=len(jobs), pending=len(pending), active=active,
               failed=failures, memory_wait=memory_wait)
        if not live and (stopped or failures):
            atomic_json(OUT / f"failures_{phase}.json", failures)
            raise RuntimeError(f"{phase} stopped after draining jobs: {failures}")
        if pending or live:
            time.sleep(10)


def kappa_jobs(config):
    jobs = []
    for cell in config["stage_minus1_kappa_hat"]["cells"]:
        for seed in config["seeds"]["development"]:
            name = f'{cell["dataset"]}_a{cell["alpha"]}_s{seed}'
            folder = OUT / "stage_minus1" / name
            cmd = [sys.executable, "-u", str(WORK / "exp/estimate_kappa.py"), "--dataset", cell["dataset"],
                   "--alpha", str(cell["alpha"]), "--seed", str(seed), "--out", str(folder)]
            jobs.append(job(name, cmd, folder, "kappa.json", gpu=True))
    return jobs


def freeze_kappa(jobs):
    records = [read(j["done"]) for j in jobs]
    values = {}
    for ds in ("MNIST", "CIFAR10"):
        for alpha in (.5, .1):
            group = [r for r in records if r["dataset"] == ds and r["alpha"] == alpha]
            if len(group) != 2 or {r["seed"] for r in group} != {14, 15}:
                raise RuntimeError("kappa estimates require both paired seeds")
            values[f"{ds}_a{alpha}"] = float(np.exp(np.mean([np.log(r["measurements"]["warm"]["kappa"]) for r in group])))
    value = {"by_cell": values, "per_run": records, "measurement_stage": "after five FedAvg rounds",
             "initial_measurement_used": False, "job_hashes": {j["done"]: digest(j["done"]) for j in jobs},
             "protocol_sha256": digest(CONFIG)}
    immutable_json(OUT / "kappa_hat.json", value)
    return value


def audit_smoke(jobs):
    checked = []
    for j in jobs:
        spec = read(j["allocation_spec"])
        reports, _ = market(spec["dataset"], spec["alpha"], spec["seed"], spec["rounds"], spec["smoke_size"])
        ckpt = next((p for p in spec["input_sha256"] if p.endswith(".pt") and spec["arm"] in (*FAMILIES, "mf_p2")), None)
        detail = allocation(spec["arm"], reports, spec["budget"], spec["kappa"], checkpoint=ckpt,
                            raw_sizes=spec.get("verified_raw_sizes"))
        for key in ("proposed", "offered_payment", "executed", "payment"):
            if not np.array_equal(detail[key].double().numpy(), np.asarray(spec[key])):
                raise AssertionError(f"smoke reference reconstruction differs: {key}")
        events = read(Path(j["out"]) / "events.json")
        for r, e in enumerate(events):
            if not np.array_equal(e["executed_epsilon"], spec["executed"][r]):
                raise AssertionError("smoke execution allocation differs")
            if not np.array_equal(e["settled_payments"], spec["payment"][r]):
                raise AssertionError("smoke payment differs")
            for event in e["outcome"]["participation_events"]:
                if not event["accountant_parity_pass"] or event["optimizer_steps"] != event["opacus_hook_optimizer_steps"]:
                    raise AssertionError("smoke accountant mismatch")
        checked.append({"id": j["id"], "passed": True, "result_sha256": digest(j["done"])})
    return checked


def finish(reason, terminal):
    report(reason)
    note(reason)
    atomic_json(OUT / "ALL_DONE", {"status": "complete", "terminal_stage": terminal, "reason": reason,
                                  "completed_unix": time.time()})
    update("complete", status="complete", terminal_stage=terminal, reason=reason,
           report=str(PAPER / "RESULTS_zh.md"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-workers", type=int, default=2)
    ap.add_argument("--fl-workers", type=int, default=4)
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.cpu_workers <= 2 or not 1 <= args.fl_workers <= 4:
        ap.error("protocol allows 1–2 CPU workers and 1–4 FL workers")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--cpu-workers", str(args.cpu_workers),
               "--fl-workers", str(args.fl_workers)]
        with (OUT / "campaign.log").open("a", buffering=1) as log:
            proc = subprocess.Popen(cmd, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": proc.pid, "log": str(OUT / "campaign.log")}), flush=True)
        return
    lock = (OUT / ".campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    seal()
    if (OUT / "ALL_DONE").exists():
        print("COMPLETE", OUT / "ALL_DONE")
        return
    config = read(CONFIG)
    try:
        wait_previous()
        note("101服务器的主机、账号和独立运行授权已核验；开始 Stage −1。本机前序结果未被标记为完成。")
        jobs = kappa_jobs(config)
        run_jobs("stage_minus1", jobs, args.fl_workers)
        kappa = freeze_kappa(jobs)
        freeze_stage_inputs([OUT / "kappa_hat.json"])
        note("8 次 κ 估计完成并冻结，准备 Stage 0。")
        cells0 = [{**c, "mode": "budget", "value": 30.} for c in config["stage0_oracle_check"]["cells"]]
        arms0 = [{"arm": a} for a in STAGE0_ARMS]
        # Complete CPU schedule and kappa digest exist before any Stage 0 FL.
        stage0_jobs = prepare("stage0", cells0, [14, 15], arms0, 40)
        immutable_json(OUT / "stage0_manifest.json", {"jobs_sha256": digest(OUT / "jobs_stage0.json"),
                       "kappa_hat_sha256": digest(OUT / "kappa_hat.json"), "runs": len(stage0_jobs)})
        smoke = prepare("smoke_stage0", [c for c in cells0 if c["alpha"] == .5], [14], arms0, 2, smoke_size=64)
        run_jobs("smoke_stage0", smoke, args.fl_workers)
        immutable_json(OUT / "smoke_stage0_validation.json", audit_smoke(smoke))
        run_jobs("stage0", stage0_jobs, args.fl_workers)
        rows = collect("stage0", stage0_jobs)
        decision = stage0_decision(rows, kappa)
        immutable_json(OUT / "stage0_decision.json", decision)
        freeze_stage_inputs([OUT / "stage0_decision.json", OUT / "jobs_stage0.json"])
        report("Stage 0 完成，依据预定门槛作出决定。")
        if not decision["gate_pass"]:
            finish("Stage 0 未通过 CIFAR ≥1 pp 门槛，按协议终止 P2；未启动机制训练或测试集确认。", "stage0_gate_failed")
            return
        note("Stage 0 通过，冻结部署 κ 及训练范围，开始共享预热库和三家族训练。")
        bank_job = job("oracle-bank", [sys.executable, "-u", str(WORK / "exp/train_p2.py"), "--bank"], OUT / "bank", "../oracle_bank.json")
        run_jobs("oracle_bank", [bank_job], 1)
        freeze_stage_inputs([OUT / "oracle_bank.pt", OUT / "oracle_bank.json"])
        training = []
        for family in FAMILIES:
            folder = OUT / "mechanisms" / family
            cmd = [sys.executable, "-u", str(WORK / "exp/train_p2.py"), "--kind", family, "--out", str(folder)]
            training.append(job(family, cmd, folder, "training.json"))
        run_jobs("training", training, args.cpu_workers)
        freeze_stage_inputs([OUT / "mechanisms" / family / name for family in FAMILIES for name in ("model.pt", "training.json")])
        econ_oracle = job("economic-oracle", [sys.executable, "-u", str(WORK / "exp/p2_economics.py"), "--oracle"], OUT / "economics", "oracle.json")
        run_jobs("economic_oracle", [econ_oracle], 1)
        economies = []
        arms1 = [{"arm": family, "kind": family, "checkpoint": str(OUT / "mechanisms" / family / "model.pt")} for family in FAMILIES]
        for entry in arms1 + [{"arm": name, "checkpoint": str(REFERENCES[name][0])} for name in ("V8F", "RegretNet-r2", "No-population-r2")]:
            arm = entry["arm"]
            folder = OUT / "economics" / arm
            cmd = [sys.executable, "-u", str(WORK / "exp/p2_economics.py"), "--arm", arm, "--checkpoint", entry["checkpoint"]]
            economies.append(job(arm, cmd, folder, "../" + arm + ".json"))
        run_jobs("economics", economies, args.cpu_workers)
        screened = screen()
        freeze_stage_inputs([OUT / "stage1_screen.json", OUT / "economics/oracle.json"])
        if not screened["gate_pass"]:
            finish("Stage 1 有家族未通过预定经济筛选，终止 P2 并保留全部失败候选。", "stage1_gate_failed")
            return
        note("三个家族经济筛选通过；开始相同预算/购买隐私的 Stage 2 CNN 开发。")
        smoke2 = prepare("smoke_stage2", [c for c in cells0 if c["alpha"] == .5], [14], arms1, 2, decision, smoke_size=64)
        run_jobs("smoke_stage2", smoke2, args.fl_workers)
        immutable_json(OUT / "smoke_stage2_validation.json", audit_smoke(smoke2))
        arms2 = arms1 + [{"arm": a} for a in ("V8F", "CSRA", "PAC", "Oracle-P2")]
        cells2 = config["stage2_development"]["cells"]
        stage2_jobs = prepare("stage2", cells2, [14, 15], arms2, 40, decision)
        run_jobs("stage2", stage2_jobs, args.fl_workers)
        rows = collect("stage2", stage2_jobs)
        selected = stage2_select(rows, screened["candidates"], cells2)
        immutable_json(OUT / "stage2_selection.json", selected)
        freeze_stage_inputs([OUT / "stage2_selection.json"])
        report("Stage 2 开发完成，已冻结检查点和独立确认决定。")
        if not selected["gate_pass"]:
            finish("Stage 2 未通过预定配对精度/隐私匹配门槛；停止，未使用官方测试集作选择。", "stage2_gate_failed")
            return
        note("Stage 2 通过；以种子 231/232/233 开始独立确认。")
        cells3 = []
        for ds, budgets in (("MNIST", (10., 30., 100.)), ("CIFAR10", (30., 100.))):
            for alpha in (.5, .1):
                for b in budgets:
                    cells3.append({"dataset": ds, "alpha": alpha, "mode": "budget", "value": b})
                for target in (.5, .8):
                    cells3.append({"dataset": ds, "alpha": alpha, "mode": "privacy", "value": target})
        arms3 = list(selected["selected"].values()) + [{"arm": a} for a in (
            "V8F", "RegretNet-r2", "No-population-r2", "DM-RegretNet", "MFG-Pricing", "Posted-price", "Uniform-DP",
            "PAC", "PAC-capped-legacy", "VCG", "VCG-capped-legacy", "CSRA", "Lin", "Oracle-P2")]
        stage3_jobs = prepare("stage3", cells3, [231, 232, 233], arms3, 100, decision,
                              Path(selected["selected"]["mfg_p2"]["checkpoint"]))
        run_jobs("stage3", stage3_jobs, args.fl_workers)
        collect("stage3", stage3_jobs)
        finish("全部符合预定可达条件的 P2 独立确认已完成；不可达格子单独列出，所有结果保留。", "stage3_complete")
    except Exception as error:
        update("failed", status="failed", error=repr(error))
        note(f"运行异常停止：{error!r}。未标记 ALL_DONE；保留已完成结果和日志。")
        report(f"运行异常停止：{error!r}。需要核查，不能当作科研晋级失败。")
        raise


if __name__ == "__main__":
    main()

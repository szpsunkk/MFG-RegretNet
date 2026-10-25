#!/usr/bin/env python3
"""Durable public-task training, real CNN development, and gated confirmation.

Never changes the parent queue. Includes all planned failures/infeasible cells in
the output manifest. Candidate selection never reads an official test result.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from common import ROOT, WORK, OUT, PAPER, atomic_json, digest, data_for_cell, market_reports
from contracts import settlement
from mechanisms import allocate, load_model, V8F
from train_mechanism import economic_evaluation, raw_contract

PARENT = ROOT / "run/tdsc_revision/codex_rq4_repair_20261016/round2"
CONFIG = WORK / "protocol.json"


def read(path):
    return json.loads(Path(path).read_text())


def status(phase, **kwargs):
    atomic_json(OUT / "status.json", {"phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **kwargs})


def identity():
    files = sorted((WORK / "exp").glob("*.py")) + sorted((WORK / "frozen").rglob("*.py")) + [CONFIG]
    return {str(p.relative_to(WORK)): digest(p) for p in files}


def artifact_identity():
    paths = [OUT / "public_bank.pt", OUT / "public_bank.manifest.json", OUT / "noise_table.json",
             PARENT / "selection.json"]
    paths += [Path(r["checkpoint"]) for r in read(PARENT / "selection.json")["selection"].values()]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def check_identity():
    manifest = read(OUT / "launch_manifest.json")
    if identity() != manifest["source_sha256"]:
        raise RuntimeError("campaign source changed after launch; preserve existing results and create an explicit new version")
    if artifact_identity() != manifest["input_sha256"]:
        raise RuntimeError("public proxy, calibration table or reference checkpoints changed after launch")


def job(name, command, folder, done):
    return {"id": name, "command": command, "out": str(folder), "done": str(folder / done)}


def run_jobs(phase, jobs, workers):
    check_identity()
    saved_jobs = OUT / f"jobs_{phase}.json"
    if saved_jobs.exists() and read(saved_jobs) != jobs:
        raise RuntimeError("resumed job commands differ from the recorded phase")
    atomic_json(saved_jobs, jobs)
    pending = [j for j in jobs if not Path(j["done"]).exists()]
    complete = len(jobs) - len(pending)
    live, failed = {}, []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    while pending or live:
        check_identity()
        stop = (OUT / "STOP_REQUESTED").exists()
        while pending and len(live) < workers and not failed and not stop:
            item = pending.pop(0)
            folder = Path(item["out"])
            folder.mkdir(parents=True, exist_ok=True)
            log = open(folder / "job.log", "a", buffering=1)
            log.write("\nCOMMAND " + json.dumps(item["command"]) + "\n")
            proc = subprocess.Popen(item["command"], cwd=WORK, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            live[proc.pid] = (proc, log, item)
        for pid, (proc, log, item) in list(live.items()):
            if proc.poll() is None:
                continue
            log.close()
            del live[pid]
            if proc.returncode == 0 and Path(item["done"]).exists():
                complete += 1
            else:
                failed.append({"id": item["id"], "returncode": proc.returncode, "out": item["out"]})
        active = []
        for pid, (_, _, item) in live.items():
            progress = Path(item["out"]) / "progress.json"
            active.append({"pid": pid, "id": item["id"], "progress": read(progress) if progress.exists() else None})
        status(phase, status="running", total=len(jobs), complete=complete,
               pending=len(pending), active=active, failed=failed)
        if not live and (failed or stop):
            raise RuntimeError(f"{phase} drained after failure or stop: {failed}")
        if pending or live:
            time.sleep(10)


def wait_parent():
    while True:
        check_identity()
        old = read(PARENT / "isoprivacy_queue_20261017/status.json")
        if old.get("complete") == old.get("total") == 252 and not old.get("active"):
            return
        status("waiting_for_parent_queue", status="waiting", parent_complete=old.get("complete"),
               parent_total=old.get("total"), reason="Preserve the earlier 252-run experiment and its GPU capacity")
        if (OUT / "STOP_REQUESTED").exists():
            raise RuntimeError("STOP_REQUESTED while waiting for parent queue")
        time.sleep(15)


def training_jobs(config):
    jobs = []
    for kind in config["mechanism_families"]:
        for ceiling in config["candidate_ceilings"]:
            for weight in config["task_loss_weights"]:
                tag = f"{kind}_c{ceiling:g}_task{weight:g}"
                folder = OUT / "mechanisms" / tag
                cmd = [sys.executable, "-u", str(WORK / "exp/train_taskaware.py"), "--kind", kind,
                       "--ceiling", str(ceiling), "--task-weight", str(weight), "--seed", str(config["mechanism_seed"]),
                       "--task-scale", str(config["task_difference_scale"]),
                       "--procurement-floor-weight", str(config["procurement_floor_weight"]),
                       "--strong-final-steps", str(config["strong_final_steps"]),
                       "--dual-update-rate", str(config["dual_update_rate"]),
                       "--warm-steps", str(config["warm_steps"]), "--steps", str(config["training_steps"]),
                       "--device", "cpu", "--out", str(folder)]
                jobs.append(job(tag, cmd, folder, "evaluation.json"))
    return jobs


def references():
    old = read(PARENT / "selection.json")["selection"]
    answer = {}
    for kind in ("mfg", "mfg_nob", "regretnet_s", "dm_s"):
        path = Path(old[kind]["checkpoint"])
        name = {"mfg": "V8F", "mfg_nob": "No-population-old", "regretnet_s": "RegretNet-old-selected", "dm_s": "DM-RegretNet"}[kind]
        target = OUT / "reference_economics" / f"{kind}.json"
        if not target.exists():
            model = load_model(path, kind)
            econ = economic_evaluation(model, "cpu", amounts=(10., 30., 100.))
            atomic_json(target, {"checkpoint_sha256": digest(path), "budgets": econ})
        else:
            econ = read(target)["budgets"]
            if read(target)["checkpoint_sha256"] != digest(path):
                raise RuntimeError("reference checkpoint changed")
        answer[kind] = {"arm": name, "checkpoint": str(path), "economics": econ, "kind": kind,
                        "candidate": False, "eligible": True}
    return answer


def candidates(config, refs):
    result = list(refs.values())
    for item in training_jobs(config):
        saved = read(Path(item["out"]) / "training.json")["config"]
        evaluated = read(Path(item["out"]) / "evaluation.json")
        if evaluated["checkpoint_sha256"] != digest(Path(item["out"]) / "model.pt"):
            raise RuntimeError("candidate checkpoint differs from its economic evaluation")
        econ = evaluated["budgets"]
        kind = saved["kind"]
        reference = refs["mfg" if kind == "mfg_nob" else kind]["economics"]
        envelope = {b: max(.02, v["normalized_mean_found_regret"]) for b, v in reference.items()}
        viable = (econ["10"]["mean_executed_epsilon"] >= 1. and econ["100"]["mean_executed_epsilon"] >= 2.
                  and all(v["max_budget_excess"] <= 2e-5 for v in econ.values()))
        strict = viable and all(v["normalized_mean_found_regret"] <= .02 for v in econ.values())
        eligible = viable and all(v["normalized_mean_found_regret"] <= envelope[b] for b, v in econ.items())
        result.append({"arm": item["id"], "checkpoint": str(Path(item["out"]) / "model.pt"),
                       "kind": kind, "candidate": True, "eligible": eligible, "strict_002": strict,
                       "economics": econ, "envelope": envelope, "task_weight": saved["task_weight"],
                       "ceiling": saved["ceiling"]})
    atomic_json(OUT / "candidates.json", result)
    return result


def market(dataset, alpha, seed, rounds):
    _, _, split, _ = data_for_cell(dataset, alpha, seed, stage="dev")
    sizes = split["client_sizes"]
    gc.collect()
    return market_reports(rounds, sizes, seed), sizes


def calibrate(reports, arm, checkpoint, target, reference=V8F):
    if not np.isfinite(target) or target <= 0:
        raise ValueError("privacy matching requires a positive finite target")
    model = load_model(checkpoint) if checkpoint else None

    def amount(budget):
        if model is None:
            e, _ = allocate(arm, reports, budget, reference_path=reference)
        else:
            with torch.no_grad():
                e, p = raw_contract(model, reports, torch.full((len(reports), 1), float(budget)))
                e, _, _ = settlement(reports, e, p)
        return float(e.double().mean())

    points = [(float(b), amount(float(b))) for b in np.unique(np.r_[np.linspace(1, 10, 10), np.linspace(10, 110, 26)])]
    all_points = points[:]
    for (lo, left), (hi, right) in zip(points[:-1], points[1:]):
        if (left - target) * (right - target) > 0:
            continue
        for _ in range(24):
            mid = (lo + hi) / 2
            value = amount(mid)
            all_points.append((mid, value))
            if abs(value / target - 1) <= .0025:
                break
            if (left - target) * (value - target) <= 0:
                hi = mid
            else:
                lo, left = mid, value
    best = min(all_points, key=lambda x: (abs(x[1] / target - 1), x[0]))
    return {"budget": best[0], "purchased_epsilon_mean": best[1],
            "relative_error": best[1] / target - 1, "feasible": abs(best[1] / target - 1) <= .005,
            "searched_budget_range": [1., 110.], "observed_min": min(x[1] for x in all_points),
            "observed_max": max(x[1] for x in all_points), "sampled_budgets": len(all_points)}


def fl_jobs(config, phase, arms, reference=V8F):
    development = phase == "development"
    rounds = config["development_rounds"] if development else config["confirmation_rounds"]
    seeds = config["development_seeds"] if development else config["confirmation_seeds"]
    cells = config["development_cells"] if development else [
        {"dataset": ds, "alpha": al, "mode": mode, "value": val}
        for ds in config["confirmation_datasets"] for al in config["confirmation_alphas"]
        for mode, values in (("budget", config["confirmation_budgets"]), ("privacy", config["confirmation_purchased_epsilon"]))
        for val in values]
    jobs, calibrations, exclusions = [], [], []
    for cell in cells:
        data = {s: market(cell["dataset"], cell["alpha"], s, rounds)[0] for s in seeds}
        for arm in arms:
            calibrated = {}
            if cell["mode"] == "privacy":
                for seed in seeds:
                    row = calibrate(data[seed], arm["arm"], arm.get("checkpoint"), cell["value"], reference)
                    calibrated[seed] = row
                    calibrations.append({**cell, "arm": arm["arm"], "seed": seed, **row})
                if not all(r["feasible"] for r in calibrated.values()):
                    exclusions.append({**cell, "arm": arm["arm"], "reason": "target not reached for every paired seed within declared budget search", "calibrations": calibrated})
                    continue
            for seed in seeds:
                budget = cell["value"] if cell["mode"] == "budget" else calibrated[seed]["budget"]
                tag = f'{cell["dataset"]}_{cell["mode"]}{cell["value"]:g}_a{cell["alpha"]}_s{seed}/{arm["arm"]}'
                folder = OUT / phase / tag
                cmd = [sys.executable, "-u", str(WORK / "exp/run_cell.py"), "--dataset", cell["dataset"],
                       "--alpha", str(cell["alpha"]), "--budget", repr(budget), "--seed", str(seed),
                       "--arm", arm["arm"], "--stage", "dev" if development else "test", "--aggr", "eps",
                       "--rounds", str(rounds), "--reference", str(reference), "--out", str(folder)]
                if arm.get("checkpoint"):
                    cmd += ["--checkpoint", arm["checkpoint"]]
                item = job(tag, cmd, folder, "result.json")
                item.update({"cell": cell, "seed": seed, "arm": arm["arm"], "budget": budget})
                jobs.append(item)
        status(f"preparing_{phase}", status="running", prepared_jobs=len(jobs), cell=cell)
    atomic_json(OUT / f"calibration_{phase}.json", {"rows": calibrations, "exclusions": exclusions})
    return jobs


def collect(phase, jobs):
    records = []
    for job_ in jobs:
        result = read(job_["done"])
        meta = result["meta"]
        expected = {"dataset": job_["cell"]["dataset"], "alpha": job_["cell"]["alpha"],
                    "seed": job_["seed"], "arm": job_["arm"], "budget": job_["budget"]}
        if any(meta[key] != val for key, val in expected.items()):
            raise RuntimeError("completed run metadata differs from its scheduled cell")
        for name in ("events", "privacy"):
            if result[name + "_sha256"] != digest(Path(job_["out"]) / (name + ".json")):
                raise RuntimeError("completed run evidence hash mismatch")
        row = {**job_["cell"], "seed": job_["seed"], "arm": job_["arm"], "budget": job_["budget"],
               **result["summary"], "path": job_["done"]}
        if not row["accountant_parity_pass"]:
            raise RuntimeError("accountant parity failed")
        if row["mode"] == "privacy":
            row["actual_target_error"] = row["epsilon_per_client_round"] / row["value"] - 1
            row["matching_pass"] = abs(row["actual_target_error"]) <= .005 + 1e-6
        else:
            row["matching_pass"] = True
        records.append(row)
    atomic_json(OUT / f"summary_{phase}.json", records)
    return records


def select(config, rows, records):
    def cell_key(r):
        return (r["dataset"], r["alpha"], r["mode"], r["value"], r["seed"])
    expected = {cell_key({**c, "seed": s}) for c in config["development_cells"]
                for s in config["development_seeds"]}
    for row in rows:
        found = [r for r in records if r["arm"] == row["arm"]]
        row["development_count"] = len(found)
        row["complete"] = (len(found) == len(expected) and {cell_key(r) for r in found} == expected
                           and all(r["matching_pass"] and np.isfinite(r["accuracy"]) and np.isfinite(r["loss"])
                                   for r in found))
        row["dev_accuracy"] = float(np.mean([r["accuracy"] for r in found])) if found else None
        row["dev_loss"] = float(np.mean([r["loss"] for r in found])) if found else None
        row["by_regime"] = {mode: float(np.mean([r["accuracy"] for r in found if r["mode"] == mode]))
                            for mode in ("budget", "privacy")} if row["complete"] else {}
    selected = {}
    for kind in config["mechanism_families"]:
        choices = [r for r in rows if r["kind"] == kind and r["eligible"] and r["complete"]]
        if choices:
            selected[kind] = max(choices, key=lambda r: (r["dev_accuracy"], -r["dev_loss"], r["arm"]))
    mfg = selected.get("mfg")
    reference = next(r for r in rows if r["arm"] == "V8F")
    promote = bool(mfg and mfg["candidate"] and mfg.get("task_weight", 0) > 0 and reference["complete"]
                   and all(mfg["by_regime"][mode] > reference["by_regime"][mode] for mode in ("budget", "privacy")))
    value = {"selected": selected, "all_candidates": rows, "promotion_gate_pass": promote,
             "test_results_used": False, "frozen_unix": time.time(), "protocol_sha256": digest(CONFIG)}
    atomic_json(OUT / "selection.json", value)
    return value


def report(selection, phase, records):
    PAPER.mkdir(parents=True, exist_ok=True)
    lines = ["# 任务目标改进实验", "", f"阶段：{phase}。", "",
             "保留原神经分配/支付头；公开图像代理训练，独立开发验证，测试不参与选择。", "",
             f"开发阶段晋级条件通过：{selection['promotion_gate_pass']}。这不等于已证明在测试集上超过全部基准。", "",
             "| 模型 | 开发运行数 | 开发准确率 (%) | 严格 0.02 筛选 | 相对参考筛选 |", "|---|---:|---:|---|---|"]
    for row in selection["all_candidates"]:
        acc = f'{100 * row["dev_accuracy"]:.3f}' if row["dev_accuracy"] is not None else "未执行"
        lines.append(f'| {row["arm"]} | {row["development_count"]} | {acc} | {row.get("strict_002", "参考模型")} | {row["eligible"]} |')
    lines += ["", "完整逐运行结果、不可达隐私目标、失败候选与选择依据均保存在实验目录。",
              "任务代理采用公开图像线性分类器与末端噪声近似；其结果必须通过真实 CNN FL 验证。",
              "同购买 ε 总量不等于同累计 RDP；当前累计上限未启用。", ""]
    lines += ["Uniform-DP 是无付款的自愿参与统计对照，不具备采购 IR 保证，不能解读为零成本激励机制。", ""]
    if records:
        lines += ["| 数据集 | α | 比较条件 | 方法 | 准确率均值 (%) | Macro-F1 (%) | CE | 实付/轮 | 购买 ε/人/轮 | 秒/运行 | 匹配通过/运行数 |",
                  "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
        keys = sorted(set((r["dataset"], r["alpha"], r["mode"], r["value"], r["arm"]) for r in records))
        for key in keys:
            group = [r for r in records if (r["dataset"], r["alpha"], r["mode"], r["value"], r["arm"]) == key]
            means = {name: np.mean([r[name] for r in group]) for name in
                     ("accuracy", "macro_f1", "loss", "payment_per_round", "epsilon_per_client_round", "seconds")}
            lines.append(f'| {key[0]} | {key[1]} | {key[2]}={key[3]} | {key[4]} | {100 * means["accuracy"]:.3f} | '
                         f'{100 * means["macro_f1"]:.3f} | {means["loss"]:.4f} | {means["payment_per_round"]:.3f} | '
                         f'{means["epsilon_per_client_round"]:.4f} | {means["seconds"]:.1f} | '
                         f'{sum(r["matching_pass"] for r in group)}/{len(group)} |')
        lines += ["", "预算比较固定的是支出上限，实付额另列；购买 ε 比较按实际发布后的总量核验。",
                  "时间为并发运行中的墙钟时间，不能直接作为机制本身的加速比。", ""]
    (PAPER / "RESULTS_zh.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-workers", type=int, default=2)
    ap.add_argument("--fl-workers", type=int, default=6)
    ap.add_argument("--until", choices=["training", "development", "complete"], default="complete")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()
    if args.training_workers < 1 or args.fl_workers < 1:
        ap.error("worker counts must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--training-workers", str(args.training_workers),
               "--fl-workers", str(args.fl_workers), "--until", args.until]
        with open(OUT / "campaign.log", "a", buffering=1) as log:
            proc = subprocess.Popen(cmd, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": proc.pid, "log": str(OUT / "campaign.log")}), flush=True)
        return
    lock = open(OUT / ".campaign.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    config = read(CONFIG)
    manifest = OUT / "launch_manifest.json"
    if not manifest.exists():
        atomic_json(manifest, {"created_unix": time.time(), "source_sha256": identity(),
                              "protocol_sha256": digest(CONFIG), "input_sha256": artifact_identity()})
    check_identity()
    try:
        run_jobs("training", training_jobs(config), args.training_workers)
        if args.until == "training":
            status("training_complete", status="complete")
            return
        refs = references()
        rows = candidates(config, refs)
        controls = [{"arm": a} for a in ("CSRA", "MFG-Pricing", "Posted-price", "Uniform-DP", "Balanced-1", "Balanced-2")]
        dev_arms = [r for r in rows if r["eligible"]] + controls
        dev_jobs = fl_jobs(config, "development", dev_arms)
        wait_parent()
        run_jobs("development", dev_jobs, args.fl_workers)
        dev_records = collect("development", dev_jobs)
        selection = select(config, rows, dev_records)
        report(selection, "development", dev_records)
        if args.until == "development" or not selection["promotion_gate_pass"]:
            status("development_complete", status="complete", promotion_gate_pass=selection["promotion_gate_pass"],
                   report=str(PAPER / "RESULTS_zh.md"), reason="Complete declared development stage; confirmation gate not met" if not selection["promotion_gate_pass"] else "Requested development stage complete")
            return
        arms = list(selection["selected"].values()) + list(refs.values()) + [
            {"arm": a} for a in ("PAC", "VCG", "PAC-capped-legacy", "VCG-capped-legacy", "CSRA", "MFG-Pricing", "Lin", "Posted-price", "Uniform-DP", "Balanced-1", "Balanced-2")]
        arms = list({a["arm"]: a for a in arms}.values())
        test_jobs = fl_jobs(config, "confirmation", arms, Path(selection["selected"]["mfg"]["checkpoint"]))
        run_jobs("confirmation", test_jobs, args.fl_workers)
        test_records = collect("confirmation", test_jobs)
        report(selection, "confirmation", test_records)
        status("complete", status="complete", runs=len(test_records), report=str(PAPER / "RESULTS_zh.md"))
    except Exception as error:
        status("failed", status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()

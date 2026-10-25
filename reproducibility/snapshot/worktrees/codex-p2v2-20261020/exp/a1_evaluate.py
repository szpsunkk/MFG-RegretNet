#!/usr/bin/env python3
"""One A1/A2 adoption analysis for completed P2, MF-Net and task-aware routes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import ROOT, OUT, PAPER, atomic_json, digest
from p2_state import read
from a1_rules import confirmation_rule, confirmation_cells, adoption_priority, EXTERNAL, TASK_EXTERNAL, LN_EXTERNAL


def pending(reason):
    return {"status": "pending", "pass": None, "reason": reason}


def resolve_path(path):
    path = Path(path)
    if not path.exists() and str(path).startswith("/data/home/skk/paper/FL-market/"):
        path = ROOT / path.relative_to("/data/home/skk/paper/FL-market")
    return path


def load_rows(folder, jobs, require_n=None, candidate=None, checkpoint_hash=None):
    rows, evidence, splits = [], {}, {}
    for job in jobs:
        # Mirrored manifests retain server101 absolute paths.
        path = resolve_path(job["done"])
        if not path.exists():
            raise FileNotFoundError(path)
        result = read(path)
        meta, summary = result["meta"], result["summary"]
        expected = {**{k: job["cell"][k] for k in ("dataset", "alpha")},
                    **{k: job[k] for k in ("seed", "arm", "budget")}}
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("confirmation metadata differs from its job")
        if meta.get("rounds") != 100 or meta.get("stage") != "test" or not summary["accountant_parity_pass"]:
            raise ValueError("adoption needs complete 100-round test runs with valid accounting")
        if result["identity"] != hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest():
            raise ValueError("confirmation identity differs from its metadata")
        if job["arm"] == candidate and meta.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("confirmation used another checkpoint than the frozen selection")
        if "allocation_spec" in job:
            if (digest(resolve_path(job["allocation_spec"])) != job["allocation_spec_sha256"]
                    or meta.get("allocation_spec_sha256") != job["allocation_spec_sha256"]
                    or meta.get("kappa") != job["kappa"]):
                raise ValueError("confirmation allocation specification or kappa differs")
        split_key = (job["cell"]["dataset"], job["cell"]["alpha"], job["seed"])
        split = (meta.get("split_sha256"), meta.get("client_sizes"))
        if not all(split) or splits.setdefault(split_key, split) != split:
            raise ValueError("confirmation arms do not share the same client partition")
        final = [m for m in result["measurements"] if m["round"] >= 96]
        if len(final) != 5 or {m["round"] for m in final} != set(range(96, 101)):
            raise ValueError("confirmation endpoint lacks exactly rounds 96..100")
        for metric in ("accuracy", "macro_f1", "loss"):
            if not np.isclose(np.mean([m[metric] for m in final]), summary[metric], atol=1e-12, rtol=0):
                raise ValueError("confirmation endpoint differs from its five round measurements")
        if require_n is not None and meta.get("n_clients", meta.get("n_agents", meta.get("N"))) != require_n:
            raise ValueError("LN adoption evidence does not attest N=100")
        for name in ("events", "privacy"):
            if digest(path.parent / (name + ".json")) != result[name + "_sha256"]:
                raise ValueError("confirmation evidence hash mismatch")
        evidence[str(path)] = digest(path)
        rows.append({**job["cell"], "seed": job["seed"], "arm": job["arm"], **summary})
    return rows, evidence


def evaluate_route(folder, route):
    folder = Path(folder)
    status_file = folder / "status.json"
    if not status_file.exists():
        return pending("campaign not available")
    status = read(status_file)
    if status.get("status") != "complete":
        return pending("campaign " + status.get("status", "not complete"))
    if route == "taskaware":
        if status.get("phase") == "development_complete" and status.get("promotion_gate_pass") is False:
            return {"status": "scientific_stop", "pass": False, "reason": "task-aware development gate failed"}
        selection = read(folder / "selection.json")
        if selection["selected"]["mfg"]["arm"] != "mfg_c2_task1":
            raise ValueError("task-aware selected model differs from the A2 mapping")
        candidate, seeds, external, cells = "mfg_c2_task1", (211, 212, 213), TASK_EXTERNAL, confirmation_cells(True)
        phase, required_n = "confirmation", None
        checkpoint_hash = digest(resolve_path(selection["selected"]["mfg"]["checkpoint"]))
    elif route in ("p2", "mf"):
        terminal = status.get("terminal_stage", "")
        if "gate_failed" in terminal or "screen_failed" in terminal:
            return {"status": "scientific_stop", "pass": False, "reason": terminal}
        if terminal not in ("stage3_complete", "mf_stage3_complete"):
            return pending("no full confirmation terminal state")
        candidate = "mfg_p2" if route == "p2" else "mf_p2"
        seeds, external, cells, phase, required_n = (231, 232, 233), EXTERNAL, confirmation_cells(), "stage3", None
        selected = read(folder / "stage2_selection.json")["selected"][candidate]
        checkpoint_hash = selected["checkpoint_sha256"]
        if digest(resolve_path(selected["checkpoint"])) != checkpoint_hash:
            raise ValueError("selected P2 checkpoint has changed")
    else:
        terminal = status.get("terminal_stage", "")
        if "gate_failed" in terminal:
            return {"status": "scientific_stop", "pass": False, "reason": terminal}
        candidate, seeds, external = "MF-Net", (251, 252, 253), LN_EXTERNAL
        cells = [{"dataset": ds, "alpha": a, "mode": "budget", "value": 300.}
                 for ds in ("MNIST", "CIFAR10") for a in (.5, .1)]
        phase, required_n = "confirmation", 100
        from a1_inputs import verify_handoff
        if not (folder / "p2_handoff/p2_handoff.json").exists():
            return pending("LN confirmation needs the common Stage A checkpoint handoff")
        handoff = verify_handoff(folder / "p2_handoff")
        checkpoint_hash = handoff["checkpoint_sha256"]
        registration = OUT / "mf_registration.json"
        if registration.exists() and read(registration)["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("N=10 and N=100 must use the same selected MF-Net checkpoint")
    manifest = folder / ("a1_stage3_jobs.json" if route == "mf" else "jobs_" + phase + ".json")
    if not manifest.exists():
        return pending("confirmation job manifest missing")
    try:
        rows, evidence = load_rows(folder, read(manifest), required_n, candidate, checkpoint_hash)
    except FileNotFoundError as error:
        return pending("return still incomplete: " + str(error))
    calibration = folder / ("calibration_" + phase + ".json")
    value = read(calibration) if calibration.exists() else {}
    exclusions = value.get("exclusions", value.get("infeasible", []))
    result = confirmation_rule(rows, candidate, cells, seeds, external, exclusions)
    result.update(source_campaign=str(folder), jobs_sha256=digest(manifest), result_sha256=evidence,
                  checkpoint_sha256=checkpoint_hash,
                  adoption_rule="A2 (added during confirmation)" if route == "taskaware" else "A1/ln-v1")
    return result


def evaluate_all(write=False):
    folders = {"p2": ROOT / "run/tdsc_revision/codex_p2_101_20261019",
               "mf": OUT, "taskaware": ROOT / "run/tdsc_revision/codex_taskaware_20261017",
               "ln": ROOT / "run/tdsc_revision/codex_ln_20261019"}
    routes = {route: evaluate_route(folder, route) for route, folder in folders.items()}
    selection = adoption_priority(routes["mf"]["pass"], routes["ln"]["pass"], routes["p2"]["pass"], routes["taskaware"]["pass"])
    attempted = [name for name, folder in folders.items() if any((folder / phase).exists() for phase in
                 ("stage_minus1", "stage0", "stage2", "mechanisms", "confirmation", "stage_a", "stage_A", "economics/mf_p2"))]
    # MF-Net at N=10 and N=100 is one model route, not two independent choices.
    route_names = sorted({"MF-Net" if x in ("mf", "ln") else x for x in attempted})
    value = {"selection": selection, "routes": routes, "attempted_routes": route_names,
             "number_of_routes_attempted": len(route_names),
             "timing_disclosure": "A2 was added while task-aware confirmation ran; Codex had seen accuracy fields in progress outputs. See A1_EXECUTION_20261019.md."}
    if write:
        atomic_json(PAPER / "ADOPTION_STATUS.json", value)
        lines = ["# A1/A2统一采用判定", "", "当前采用状态：" + json.dumps(selection, ensure_ascii=False), "",
                 f"已尝试模型路线数：{len(route_names)}；{', '.join(route_names)}。未就绪路线不算科学失败。", "",
                 "| 路线 | 数据集 | d_s（百分点） | 均值及95% t区间（百分点） | 通过 |",
                 "|---|---|---|---|---|"]
        for name, record in routes.items():
            if "datasets" not in record:
                lines.append(f"| {name} | 全部 | {record['status']} | {record.get('reason', '')} | {record['pass']} |")
            for dataset, result in record.get("datasets", {}).items():
                lines.append(f"| {name} | {dataset} | {result['seed_differences_pp']} | {result['interval_pp']} | {result['pass']} |")
        lines += ["", value["timing_disclosure"], "", "完整逐单元最佳基线、不可达排除和输入哈希见ADOPTION_STATUS.json。",
                  "所有路线按固定优先级判定，不按测试精度高低选择路线。本文档不修改论文。", ""]
        (PAPER / "ADOPTION_zh.md").write_text("\n".join(lines))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = evaluate_all(args.write)
    print(json.dumps({"selection": value["selection"], "routes": {k: {f: v.get(f) for f in ("status", "pass", "reason")} for k, v in value["routes"].items()}}, ensure_ascii=False))

"""Complete-cell stage gates and seed-paired reporting, with no hidden exclusions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from common import OUT, PAPER, digest, atomic_json
from p2_core import STAGE0_ARMS, FAMILIES
from p2_jobs import deploy_kappa
from p2_state import read, immutable_json


def cell_key(row, seed=False):
    key = (row["dataset"], row["alpha"], row["mode"], row["value"])
    return key + (row["seed"],) if seed else key


def collect(phase, jobs, analysis_authorized=False):
    if phase.startswith(("stage2", "stage3")) and not analysis_authorized:
        raise RuntimeError("A3 analysis requires all decision-relevant manifests complete")
    if not all(Path(j["done"]).exists() for j in jobs):
        raise RuntimeError("cannot read results of an incomplete stage")
    rows = []
    for j in jobs:
        result = read(j["done"])
        for field in ("events", "privacy"):
            if digest(Path(j["out"]) / (field + ".json")) != result[field + "_sha256"]:
                raise RuntimeError("run evidence hash mismatch")
        expected = {**{k: j["cell"][k] for k in ("dataset", "alpha")},
                    **{k: j[k] for k in ("seed", "arm", "budget", "kappa")}}
        if any(result["meta"][k] != v for k, v in expected.items()):
            raise RuntimeError("result differs from frozen job")
        if result["meta"]["allocation_spec_sha256"] != j["allocation_spec_sha256"]:
            raise RuntimeError("result allocation spec mismatch")
        summary = dict(result["summary"])
        if result["meta"].get("endpoint_sealed_until_stage_complete"):
            if not analysis_authorized:
                raise RuntimeError("endpoint remains sealed")
            rounds = result["meta"]["rounds"]
            measurements = [m for m in result["measurements"] if m["round"] > rounds - min(5, rounds)]
            if len(measurements) != min(5, rounds):
                raise RuntimeError("incomplete endpoint measurements")
            for name in ("accuracy", "macro_f1", "loss"):
                summary[name] = float(np.mean([m[name] for m in measurements]))
        row = {"phase": phase, **j["cell"], "seed": j["seed"], "arm": j["arm"], "budget": j["budget"],
               "kappa": j["kappa"], **summary, "path": j["done"]}
        row["actual_target_error"] = row["epsilon_per_client_round"] / row["value"] - 1 if row["mode"] == "privacy" else None
        row["matching_pass"] = row["mode"] != "privacy" or abs(row["actual_target_error"]) <= .005 + 1e-6
        if not row["accountant_parity_pass"]:
            raise RuntimeError("accountant parity failure")
        rows.append(row)
    immutable_json(OUT / f"summary_{phase}.json", rows)
    return rows


def stage0_decision(rows, kappa_hat):
    expected = {(ds, alpha, s, a) for ds in ("MNIST", "CIFAR10") for alpha in (.5, .1)
                for s in (14, 15) for a in STAGE0_ARMS}
    observed = [(r["dataset"], r["alpha"], r["seed"], r["arm"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Stage 0 must have all 72 distinct paired results")
    means = {ds: {arm: float(np.mean([r["accuracy"] for r in rows if r["dataset"] == ds and r["arm"] == arm]))
                  for arm in STAGE0_ARMS} for ds in ("MNIST", "CIFAR10")}
    rules = ("Oracle-k0", "Oracle-khat", "Oracle-k3", "Oracle-k30")
    selected = {ds: max(rules, key=lambda arm: means[ds][arm]) for ds in means}
    delta = max(means["CIFAR10"][arm] for arm in rules[1:]) - max(means["CIFAR10"][a] for a in ("CSRA", "PAC", "Oracle-k0"))
    bounds = [.003, 300.]
    for ds, rule in selected.items():
        kvals = [kappa_hat["by_cell"][f"{ds}_a{a}"] for a in (.5, .1)] if rule == "Oracle-khat" else [{"Oracle-k0": .01, "Oracle-k3": 3., "Oracle-k30": 30.}[rule]]
        for k in kvals:
            if not .03 <= k <= 30:
                bounds = [min(bounds[0], k / 10), max(bounds[1], 10 * k)]
    contrasts = {ds: {f"{a} minus {b}": means[ds][a] - means[ds][b] for a, b in (
        ("Aligned@PAC", "PAC"), ("Aligned@CSRA", "CSRA"), ("Oracle-k0-datasize", "Oracle-k0"))} for ds in means}
    return {"gate_pass": bool(delta >= .01 - 1e-12), "cifar_gain_pp": 100 * delta,
            "required_gain_pp": 1., "selected_rule": selected, "means": means,
            "diagnostic_differences": contrasts, "training_kappa_range": bounds,
            "per_run": rows, "test_results_used": False,
            "reference_scope": "best aligned-family grid candidate; no global optimality or CNN upper-bound guarantee"}


def stage2_select(rows, candidates, cells, seeds=(14, 15)):
    expected = {cell_key(c) for c in cells}
    selected, completeness = {}, {}
    for family in FAMILIES:
        entries = [c for c in candidates if c["kind"] == family and c["eligible"]]
        choices = []
        for entry in entries:
            found = [r for r in rows if r["arm"] == entry["arm"]]
            unique = {(cell_key(r), r["seed"]) for r in found}
            duplicate = len(unique) != len(found)
            entered = {cell_key(r) for r in found}
            complete = bool(entered) and not duplicate and all(
                (key, s) in unique for key in entered for s in seeds)
            valid = complete and all(np.isfinite(r["accuracy"]) and np.isfinite(r["loss"]) for r in found)
            completeness[entry["arm"]] = {"complete_entered_cells": complete,
                                         "entered_cells": len(entered), "runs": len(found)}
            if valid:
                choices.append({**entry, "mean_accuracy": float(np.mean([r["accuracy"] for r in found])),
                                "mean_loss": float(np.mean([r["loss"] for r in found]))})
        if choices:
            selected[family] = max(choices, key=lambda x: (x["mean_accuracy"], -x["mean_loss"]))
    reasons, differences = [], []
    if len(selected) != len(FAMILIES):
        reasons.append("some P2 family has no complete eligible candidate")
    mfg = selected.get("mfg_p2", {}).get("arm")
    def usable(arm, key):
        group = [r for r in rows if r["arm"] == arm and cell_key(r) == key]
        return group if len(group) == len(seeds) and {r["seed"] for r in group} == set(seeds) and all(r["matching_pass"] for r in group) else []
    for key in sorted(expected):
        ours = usable(mfg, key)
        v8f = usable("V8F", key)
        available = {arm: usable(arm, key) for arm in ("CSRA", "PAC", "V8F")}
        available = {a: g for a, g in available.items() if g}
        if not ours or (key[0] == "MNIST" and not v8f) or not available:
            reasons.append(f"missing or unmatched paired gate cell: {key}")
            continue
        other = max(available, key=lambda a: np.mean([r["accuracy"] for r in available[a]])) if key[0] == "CIFAR10" else "V8F"
        lookup = {r["seed"]: r["accuracy"] for r in available[other]}
        diffs = [r["accuracy"] - lookup[r["seed"]] for r in ours]
        differences.append({"cell": key, "reference": other, "paired_seed_differences": diffs, "mean": float(np.mean(diffs))})
    gains = {ds: float(np.mean([r["mean"] for r in differences if r["cell"][0] == ds]))
             if any(r["cell"][0] == ds for r in differences) else None for ds in ("MNIST", "CIFAR10")}
    if gains["CIFAR10"] is None or gains["CIFAR10"] < .005 - 1e-12:
        reasons.append("CIFAR10 gain below 0.5 pp")
    if gains["MNIST"] is None or gains["MNIST"] < -.003 - 1e-12:
        reasons.append("MNIST difference below -0.3 pp")
    return {"selected": selected, "gate_pass": not reasons, "reasons": reasons, "completeness": completeness,
            "paired_cell_differences": differences, "gains": gains, "test_results_used": False}


def paired_interval(values):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("a paired interval needs at least two finite independent seeds")
    mean = float(values.mean())
    half = float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"n_seeds": len(values), "mean_difference": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in record.items()})


def report(reason):
    records = []
    for phase in ("stage0", "stage2", "stage3"):
        path = OUT / f"summary_{phase}.json"
        if path.exists():
            records += read(path)
    write_csv(PAPER / "per_run.csv", records)
    groups, paired = [], []
    for phase, key, arm in sorted({(r["phase"], cell_key(r), r["arm"]) for r in records}):
        group = [r for r in records if r["phase"] == phase and cell_key(r) == key and r["arm"] == arm]
        row = {"phase": phase, "dataset": key[0], "alpha": key[1], "mode": key[2], "value": key[3],
               "arm": arm, "n_seeds": len(group), "seeds": sorted(r["seed"] for r in group),
               "matching_passes": sum(r["matching_pass"] for r in group),
               "kappa_values": sorted(set(r["kappa"] for r in group))}
        for metric in ("accuracy", "macro_f1", "loss", "payment_per_round", "epsilon_per_client_round",
                       "tv", "participants_per_round", "j_kappa", "seconds"):
            vals = [r[metric] for r in group]
            row[metric] = float(np.mean(vals)) if all(v is not None for v in vals) else None
            if metric == "accuracy":
                row["accuracy_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        groups.append(row)
        targets = ([{"PAC": "Aligned@PAC", "CSRA": "Aligned@CSRA", "Oracle-k0": "Oracle-k0-datasize"}.get(arm)]
                   if phase == "stage0" else [a for a in ("mfg_p2", "mf_p2") if a != arm])
        for target_arm in filter(None, targets):
            ours = {r["seed"]: r for r in records if r["phase"] == phase and cell_key(r) == key and r["arm"] == target_arm}
            expected_seeds = {231, 232, 233} if phase == "stage3" else {14, 15}
            complete = len(group) == len(expected_seeds) and set(ours) == {r["seed"] for r in group} == expected_seeds
            if complete:
                for metric in ("accuracy", "macro_f1", "loss", "payment_per_round", "epsilon_per_client_round"):
                    paired.append({"phase": phase, "dataset": key[0], "alpha": key[1], "mode": key[2], "value": key[3],
                        "method": target_arm, "reference": arm,
                        "metric": metric, **paired_interval([ours[r["seed"]][metric] - r[metric] for r in group]),
                        "all_privacy_matches": all(r["matching_pass"] and ours[r["seed"]]["matching_pass"] for r in group),
                        "interval_scope": f"descriptive paired t; n={len(expected_seeds)} seeds; no multiplicity correction"})
    write_csv(PAPER / "summary.csv", groups)
    write_csv(PAPER / "paired.csv", paired)
    lines = ["# P2 A1补充实验结果", "", reason, "", "具体协议见A1、A2和A1_EXECUTION_20261019.md；原P2完整结果保留在独立目录codex_p2_101_20261019。",
             "本表仅展示本协调器生成或验证复用的运行。所有路线的采用判定见ADOPTION_zh.md；验证日志见结果目录validation。", "",
             "| 阶段 | 数据集 | α | 条件 | 方法 | κ | n | 精度 % ± SD | F1 % | CE | 实付/轮 | ε/人/轮 | TV | 参与数 | Jκ | 匹配 |",
             "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in groups:
        j = f'{r["j_kappa"]:.4g}' if r["j_kappa"] is not None else "∞"
        sd = f'{100*r["accuracy_sd"]:.3f}' if r["accuracy_sd"] is not None else "未估计"
        lines.append(f'| {r["phase"]} | {r["dataset"]} | {r["alpha"]} | {r["mode"]}={r["value"]} | {r["arm"]} | '
                     f'{r["kappa_values"]} | {r["n_seeds"]} | {100*r["accuracy"]:.3f} ± {sd} | {100*r["macro_f1"]:.3f} | {r["loss"]:.4f} | '
                     f'{r["payment_per_round"]:.3f} | {r["epsilon_per_client_round"]:.4f} | {r["tv"]:.4f} | '
                     f'{r["participants_per_round"]:.2f} | {j} | {r["matching_passes"]}/{r["n_seeds"]} |')
    for phase in ("stage0", "stage2", "stage3"):
        p = OUT / f"calibration_{phase}.json"
        if p.exists() and read(p)["exclusions"]:
            lines += ["", f"## {phase} 未进入运行的不可达格子", "",
                      "| 数据集 | α | 目标 | 种子 | 方法 | 原因 |", "|---|---:|---:|---:|---|---|"]
            for r in read(p)["exclusions"]:
                lines.append(f'| {r["dataset"]} | {r["alpha"]} | {r["value"]} | {r["seed"]} | {r["arm"]} | {r["reason"]} |')
    for name in ("stage0_decision", "stage1_screen", "stage2_selection"):
        p = OUT / (name + ".json")
        if p.exists():
            value = read(p)
            lines += ["", f"{name}：晋级={value['gate_pass']}。完整逐项决定：{p}。"]
            if name == "stage1_screen":
                for c in value["candidates"]:
                    lines.append(f"- {c['arm']}：经济筛选={c['eligible']}；严格 0.02={c['strict_002']}；原因={c['reasons']}")
    if (OUT / "kappa_hat.json").exists():
        lines += ["", "κ估计：" + json.dumps(read(OUT / "kappa_hat.json")["by_cell"], ensure_ascii=False)]
    lines += ["", "统计单位为独立种子，开发 n=2、确认 n=3。末五轮先在运行内平均。paired.csv 为 Stage 0 的三个规定诊断对照，以及 Stage 2/3 中有完整数据的 MFG-P2、MF-Net 减各参考的种子配对均值差及 t 95% 描述性区间；未多重校正，不据此宣称全面显著优越。",
              "40 轮开发和 100 轮预算标定的差异逐项保留，实际偏差超 0.5% 的运行不用于同隐私晋级。",
              "Oracle 是对齐族网格参考，无策略保证，也不是全局最优或 CNN 精度上界。Uniform-DP 为自愿参与统计对照。",
              "κ估计为非隐私开发分析；Jκ 为训练代理。正式运行沿用真实 DP-SGD 和 RDP 账本，累计上限未启用。同购买量不等于同累计隐私。",
              "原P2在101保留优先级，A1新增MF计算等待它释放资源。本机task-aware保持独立运行。源码或输入哈希变化会阻止继续启动。",
              "并发墙钟不能解释为机制本身加速比。失败作业见 status.json、jobs_*.json 和相应 job.log。", ""]
    (PAPER / "RESULTS_zh.md").write_text("\n".join(lines))

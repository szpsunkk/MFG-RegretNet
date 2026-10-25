"""RQ4 from the complete P2-v2 Stage 3: independent recomputation of the A1.3 decision and the paper's tables.

Runs only when every Stage 3 run is complete (candidate, ablations, Uniform-DP and all comparators) and Codex's
decision file exists, so it never reads a partial stage (A3.5). Nothing here chooses settings, seeds or arms:
every one of the 18 settings and 3 seeds is reported.

Steps
  1. Jobs: <base>/jobs_stage3_v2.json (P2-v2 arms) + codex_p2_a3_20261019/jobs_stage3_early.json (comparators).
  2. Per run: check the result against its frozen job (dataset, alpha, seed, arm, budget, kappa, allocation spec),
     A_final = mean test accuracy of rounds 96-100 recomputed from the stored confusion matrices, and the
     privacy match of the purchased-privacy settings (|eps per client and round / target - 1| <= 0.5%).
  3. A1.3 decision, implemented here from the amendments (A1.3, A4.7): per setting, the best external IR-feasible
     arm by its three-seed mean among arms with complete matched evidence (ties: first in the frozen list), the
     seed-paired margins; per dataset the mean margin over its settings for each seed; pass if CIFAR-10 is
     positive in every seed and the MNIST mean is >= -0.3 points. Compared with Codex's decision file.
  4. Outputs: rq4_recomputed.json, rq4_main_table.tex (per dataset and group of settings, per seed),
     rq4_supp_table.tex (all 18 settings x all arms), rq4_macros.tex (numbers used in the text).
Usage: python rq4_fill.py --base RUN/continuation_A7/QV-sel --old RUN_A3 --decision FILE --out DIR [--remap OLD=NEW]
       (test_rq4_fill.py runs it on synthetic runs only.)
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXTERNAL = ("PAC", "PAC-capped-legacy", "VCG", "VCG-capped-legacy", "CSRA", "MFG-Pricing",
            "Posted-price", "Lin", "DM-RegretNet", "RegretNet-r2")
CANDIDATE = "mfg_p2v2"
SEEDS = (231, 232, 233)
DISPLAY = {
    "mfg_p2v2": "MFG-RegretNet (Ours)", "Uniform-DP": "Uniform-DP", "Oracle-P2": "Non-strategic reference",
    "RegretNet-r2": "RegretNet", "DM-RegretNet": "DM-RegretNet", "MFG-Pricing": "MFG-Pricing",
    "Posted-price": "Posted price", "PAC": "PAC", "PAC-capped-legacy": "PAC (capped)", "VCG": "VCG",
    "VCG-capped-legacy": "VCG (capped)", "CSRA": "CSRA", "Lin": "Lin et al.",
}
# Tables show MFG-RegretNet and the ten external arms of the decision (PAPER_EXPERIMENTS_PLAN_20261023.md);
# Uniform-DP and the non-strategic reference only with --extra-arms. The decision recomputation reads every run.
ORDER = ["mfg_p2v2", *EXTERNAL]


def cells():
    """a1_rules.confirmation_cells(): 18 settings."""
    out = []
    for dataset in ("MNIST", "CIFAR10"):
        budgets = (30., 100.) if dataset == "CIFAR10" else (10., 30., 100.)
        for alpha in (.5, .1):
            for mode, values in (("budget", budgets), ("privacy", (.5, .8))):
                out.extend({"dataset": dataset, "alpha": alpha, "mode": mode, "value": v} for v in values)
    return out


def key(c):
    return (c["dataset"], float(c["alpha"]), c["mode"], float(c["value"]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with open(path) as fh:
        return json.load(fh)


def run_row(job):
    res = read(job["done"])
    meta = res["meta"]
    expect = {"dataset": job["cell"]["dataset"], "alpha": job["cell"]["alpha"], "seed": job["seed"],
              "arm": job["arm"], "budget": job["budget"], "kappa": job["kappa"]}
    bad = {k: (meta.get(k), v) for k, v in expect.items() if meta.get(k) != v}
    if bad or meta.get("allocation_spec_sha256") != job["allocation_spec_sha256"]:
        raise RuntimeError(f"{job['id']}: result differs from its frozen job: {bad}")
    if meta.get("stage") != "test" or meta.get("rounds") != 100:
        raise RuntimeError(f"{job['id']}: not a 100-round official-test run")
    end = [m for m in res["measurements"] if 96 <= m["round"] <= 100]
    if sorted(m["round"] for m in end) != [96, 97, 98, 99, 100]:
        raise RuntimeError(f"{job['id']}: endpoint rounds incomplete")
    accs = []
    for m in end:
        cm = np.asarray(m["confusion"], dtype=float)
        acc = float(np.trace(cm) / cm.sum())
        if abs(acc - m["accuracy"]) > 1e-12:
            raise RuntimeError(f"{job['id']}: accuracy differs from its confusion matrix")
        accs.append(acc)
    summ = res["summary"]
    cell = job["cell"]
    if cell["mode"] == "privacy":
        err = summ["epsilon_per_client_round"] / cell["value"] - 1
        matched = abs(err) <= .005 + 1e-6
    else:
        err, matched = None, True
    return {"dataset": cell["dataset"], "alpha": float(cell["alpha"]), "mode": cell["mode"],
            "value": float(cell["value"]), "seed": job["seed"], "arm": job["arm"], "accuracy": float(np.mean(accs)),
            "epsilon_per_client_round": summ.get("epsilon_per_client_round"), "target_error": err,
            "matched": matched, "result_sha256": sha(job["done"])}


def decide(rows, exclusions):
    lookup = defaultdict(dict)
    for r in rows:
        k = (key(r), r["arm"])
        if r["seed"] in lookup[k]:
            raise RuntimeError(f"duplicate run {k} seed {r['seed']}")
        lookup[k][r["seed"]] = r
    excluded = {(key(e), e["arm"], e["seed"]) for e in exclusions if "seed" in e}
    per_cell, problems, omissions = [], [], []
    for c in cells():
        k = key(c)
        own = lookup[(k, CANDIDATE)]
        ok_own = set(own) == set(SEEDS) and all(r["matched"] for r in own.values())
        if not ok_own:
            problems.append({"cell": k, "arm": CANDIDATE, "reason": "candidate missing or unmatched"})
        avail = {}
        for arm in EXTERNAL:
            g = lookup[(k, arm)]
            if set(g) == set(SEEDS) and all(r["matched"] for r in g.values()):
                avail[arm] = g
                continue
            missing = set(SEEDS) - set(g)
            known = c["mode"] == "privacy" and missing and all((k, arm, s) in excluded for s in missing)
            omissions.append({"cell": k, "arm": arm, "missing_seeds": sorted(missing),
                              "unmatched_seeds": sorted(s for s, r in g.items() if not r["matched"]),
                              "reason": "listed as unreachable" if known else "missing or unmatched"})
            if not known or any(not r["matched"] for r in g.values()):
                problems.append(omissions[-1])
        if not ok_own or not avail:
            if not avail:
                problems.append({"cell": k, "reason": "no complete external comparator"})
            continue
        means = {a: float(np.mean([avail[a][s]["accuracy"] for s in SEEDS])) for a in avail}
        best = max(avail, key=lambda a: (means[a], -EXTERNAL.index(a)))
        diffs = [100 * (own[s]["accuracy"] - avail[best][s]["accuracy"]) for s in SEEDS]
        per_cell.append({"cell": k, "external_arm": best, "external_mean": means[best],
                         "candidate_mean": float(np.mean([own[s]["accuracy"] for s in SEEDS])),
                         "margins_pp": diffs})
    datasets = {}
    for ds in ("MNIST", "CIFAR10"):
        group = [r for r in per_cell if r["cell"][0] == ds]
        needed = sum(c["dataset"] == ds for c in cells())
        if len(group) != needed or any(p.get("cell", (None,))[0] == ds for p in problems):
            datasets[ds] = {"status": "incomplete", "pass": None}
            continue
        d = np.mean([r["margins_pp"] for r in group], axis=0)
        passed = bool(np.all(d > 0)) if ds == "CIFAR10" else bool(d.mean() >= -.3 - 1e-12)
        datasets[ds] = {"status": "evaluated", "pass": passed, "seed_margins_pp": d.tolist(),
                        "mean_margin_pp": float(d.mean())}
    complete = not problems and all(v["status"] == "evaluated" for v in datasets.values())
    return {"status": "evaluated" if complete else "incomplete",
            "pass": all(v["pass"] for v in datasets.values()) if complete else None,
            "datasets": datasets, "per_cell": per_cell, "omissions": omissions, "problems": problems}


def compare(mine, codex):
    """Differences between this recomputation and Codex's decision file (empty list = identical)."""
    diffs = []
    if mine["pass"] != codex.get("pass"):
        diffs.append(("pass", mine["pass"], codex.get("pass")))
    theirs = {(c["dataset"], float(c["alpha"]), c["mode"], float(c["value"])): c for c in codex.get("per_cell", [])}
    for row in mine["per_cell"]:
        t = theirs.get(row["cell"])
        if t is None:
            diffs.append(("missing cell in Codex", row["cell"]))
            continue
        if t["external_arm"] != row["external_arm"]:
            diffs.append(("external arm", row["cell"], row["external_arm"], t["external_arm"]))
        if np.max(np.abs(np.array(t["paired_differences_pp"]) - row["margins_pp"])) > 1e-9:
            diffs.append(("margins", row["cell"], row["margins_pp"], t["paired_differences_pp"]))
    for ds, v in mine["datasets"].items():
        t = codex.get("datasets", {}).get(ds, {})
        if v.get("pass") != t.get("pass"):
            diffs.append(("dataset pass", ds, v.get("pass"), t.get("pass")))
        if v.get("seed_margins_pp") is not None and t.get("seed_differences_pp") is not None and \
                np.max(np.abs(np.array(v["seed_margins_pp"]) - t["seed_differences_pp"])) > 1e-9:
            diffs.append(("dataset margins", ds, v["seed_margins_pp"], t["seed_differences_pp"]))
    return diffs


def pp(x, signed=False):
    return (f"{x:+.2f}" if signed else f"{x:.2f}").replace("-", "$-$")


def group_name(c):
    return {"budget": "budget", "privacy": "purchased privacy"}[c[2]]


def main_table(decision):
    """Per dataset and group of settings: mean accuracy of ours and of the best external arm, per-seed margins."""
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"& & \multicolumn{2}{c}{$\mathcal{A}_{\mathrm{final}}$ (\%)} & \multicolumn{4}{c}{Margin (points)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
             r"Dataset & Settings & Ours & Best external & 231 & 232 & 233 & Mean \\", r"\midrule"]
    for ds, label in (("MNIST", "MNIST"), ("CIFAR10", "CIFAR-10")):
        rows = [r for r in decision["per_cell"] if r["cell"][0] == ds]
        for mode in ("budget", "privacy"):
            g = [r for r in rows if r["cell"][2] == mode]
            if not g:
                continue
            m = np.mean([r["margins_pp"] for r in g], axis=0)
            lines.append(f"{label} & {group_name(g[0]['cell'])} ({len(g)}) & "
                         f"{pp(100 * np.mean([r['candidate_mean'] for r in g]))} & "
                         f"{pp(100 * np.mean([r['external_mean'] for r in g]))} & "
                         + " & ".join(pp(x, True) for x in m) + f" & {pp(m.mean(), True)} \\\\")
        d = decision["datasets"][ds]
        if d.get("seed_margins_pp"):
            rule = "all seeds $>0$" if ds == "CIFAR10" else r"mean $\ge -0.3$"
            verdict = "met" if d["pass"] else "not met"
            lines.append(f"{label} & all ({len(rows)}) & & & " + " & ".join(pp(x, True) for x in d["seed_margins_pp"])
                         + f" & {pp(d['mean_margin_pp'], True)} \\\\")
            lines.append(rf"\multicolumn{{8}}{{l}}{{\footnotesize Rule for {label}: {rule}; {verdict}.}} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def supp_table(rows, decision):
    """All 18 settings x all arms: mean (sd) over the three seeds; best external arm of each setting in bold."""
    by = defaultdict(dict)
    for r in rows:
        by[(key(r), r["arm"])][r["seed"]] = r
    best = {r["cell"]: r["external_arm"] for r in decision["per_cell"]}
    cols = cells()
    out = []
    for ds in ("MNIST", "CIFAR10"):
        cs = [c for c in cols if c["dataset"] == ds]
        head = " & ".join(rf"\rotatebox{{90}}{{$\alpha{{=}}{c['alpha']:g}$, "
                          + (rf"$B{{=}}{c['value']:g}$" if c["mode"] == "budget" else rf"$\bar\epsilon{{=}}{c['value']:g}$")
                          + "}" for c in cs)
        out += [rf"\begin{{tabular}}{{l{'c' * len(cs)}}}", r"\toprule", f"{'CIFAR-10' if ds == 'CIFAR10' else ds} & {head} \\\\",
                r"\midrule"]
        for arm in ORDER:
            cells_txt = []
            for c in cs:
                g = by.get((key(c), arm), {})
                if set(g) != set(SEEDS):
                    cells_txt.append("--")
                    continue
                acc = np.array([100 * g[s]["accuracy"] for s in SEEDS])
                txt = f"{acc.mean():.1f}"
                if not all(g[s]["matched"] for s in SEEDS):
                    txt += r"$^{\dagger}$"
                if best.get(key(c)) == arm:
                    txt = rf"\textbf{{{txt}}}"
                cells_txt.append(txt)
            if all(t == "--" for t in cells_txt):
                continue
            out.append(f"{DISPLAY.get(arm, arm)} & " + " & ".join(cells_txt) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def macros(decision):
    d = decision["datasets"]
    verdict = {True: "met", False: "not met", None: "incomplete"}[decision["pass"]]
    lines = [f"% generated by rq4_fill.py; decision {verdict}"]
    lines.append(rf"\newcommand{{\RQfourDecision}}{{{verdict}}}")
    for ds, tag in (("MNIST", "Mnist"), ("CIFAR10", "Cifar")):
        if d[ds].get("seed_margins_pp"):
            for s, x in zip(SEEDS, d[ds]["seed_margins_pp"]):
                lines.append(rf"\newcommand{{\RQfour{tag}Seed{'ABC'[SEEDS.index(s)]}}}{{{pp(x, True)}}}")
            lines.append(rf"\newcommand{{\RQfour{tag}Mean}}{{{pp(d[ds]['mean_margin_pp'], True)}}}")
            lines.append(rf"\newcommand{{\RQfour{tag}Pass}}{{{'met' if d[ds]['pass'] else 'not met'}}}")
    wins = sum(np.mean(r["margins_pp"]) > 0 for r in decision["per_cell"])
    lines.append(rf"\newcommand{{\RQfourSettingsAhead}}{{{wins}}}")
    lines.append(rf"\newcommand{{\RQfourSettings}}{{{len(decision['per_cell'])}}}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="run/.../continuation_A7/QV-sel")
    ap.add_argument("--old", required=True, help="run/tdsc_revision/codex_p2_a3_20261019")
    ap.add_argument("--decision", required=True, help="Codex's stage3 adoption file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-arms", default="", help="comma list from Uniform-DP,Oracle-P2 (author's decision)")
    ap.add_argument("--remap", default="", help="OLD=NEW prefix of the result paths (runs recorded on another host)")
    a = ap.parse_args()
    extra = [x for x in a.extra_arms.split(",") if x]
    assert set(extra) <= {"Uniform-DP", "Oracle-P2"}, extra
    ORDER[1:1] = extra
    base, old, out = Path(a.base), Path(a.old), Path(a.out)
    jobs = read(base / "jobs_stage3_v2.json") + read(old / "jobs_stage3_early.json")
    if a.remap:
        src, dst = a.remap.split("=", 1)
        for j in jobs:
            if j["done"].startswith(src):
                j["done"] = dst + j["done"][len(src):]
    missing = [j["id"] for j in jobs if not Path(j["done"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(jobs)} runs incomplete; nothing read (A3.5)")
    if not Path(a.decision).exists():
        raise SystemExit("Codex's decision file does not exist yet; nothing read")
    exclusions = read(old / "calibration_stage3_early.json")["exclusions"] + \
        read(base / "calibration_stage3_v2.json")["exclusions"]
    rows = [run_row(j) for j in jobs]
    mine = decide(rows, exclusions)
    codex = read(a.decision)
    diffs = compare(mine, codex)
    out.mkdir(parents=True, exist_ok=True)
    report = {"runs": len(rows), "decision": mine, "differences_to_codex": [list(map(str, d)) for d in diffs],
              "codex_decision_sha256": sha(a.decision), "codex_selected_model": codex.get("selected_model"),
              "rows": rows}
    (out / "rq4_recomputed.json").write_text(json.dumps(report, indent=1))
    (out / "rq4_main_table.tex").write_text(main_table(mine))
    (out / "rq4_supp_table.tex").write_text(supp_table(rows, mine))
    (out / "rq4_macros.tex").write_text(macros(mine))
    print({"runs": len(rows), "status": mine["status"], "pass": mine["pass"], "differences_to_codex": len(diffs)})


if __name__ == "__main__":
    main()

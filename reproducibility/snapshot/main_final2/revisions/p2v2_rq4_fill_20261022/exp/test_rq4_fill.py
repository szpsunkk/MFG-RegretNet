"""rq4_fill.py on synthetic Stage 3 runs only (no real result is read).

1. Builds fake jobs and result files with the structure of Codex's runs (meta, summary, measurements with
   confusion matrices), including unreachable purchased-privacy comparators listed in the calibration files.
2. Computes the decision with Codex's own confirmation_rule (its source extracted from a1_rules.py and
   p2_results.py, executed without importing Codex's modules) and writes it as the "Codex decision".
3. Runs rq4_fill.py and requires: zero differences to that decision, the refusal on an incomplete stage,
   and LaTeX tables that compile.
Scenarios: a passing and a failing synthetic outcome, plus a tie between two external arms.
"""
import ast
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

import rq4_fill as F

WORK = Path(__file__).resolve().parents[4] / "worktrees/codex-p2v2-20261020/exp"
ARMS_V2 = ["mfg_p2v2", "nob_p2v2", "profile_p2v2", "Uniform-DP"]
ARMS_OLD = ["V8F", "RegretNet-r2", "No-population-r2", "DM-RegretNet", "MFG-Pricing", "Posted-price", "PAC",
            "PAC-capped-legacy", "VCG", "VCG-capped-legacy", "CSRA", "Lin", "Oracle-P2"]
PDFLATEX = "/usr/local/texlive/2022/bin/x86_64-linux/pdflatex"


def codex_rule():
    ns = {"np": np, "defaultdict": defaultdict, "student_t": student_t}
    for path, names in ((WORK / "p2_results.py", {"cell_key", "paired_interval"}),
                        (WORK / "a1_rules.py", {"indexed", "matched", "confirmation_rule"})):
        tree = ast.parse(path.read_text())
        body = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "EXTERNAL" for t in n.targets)]
        body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def confusion(acc, n=10000):
    m = np.zeros((10, 10), dtype=int)
    m[0, 0] = int(round(acc * n))
    m[0, 1] = n - m[0, 0]
    return m


def build(root, rng, shift, tie=False, unreachable=(("CSRA", 0.5), ("PAC", 0.8)), unmatched_lin=False):
    base, old = root / "base", root / "old"
    base.mkdir(parents=True), old.mkdir(parents=True)
    jobs_v2, jobs_old, excl_v2, excl_old = [], [], [], []
    for c in F.cells():
        for seed in F.SEEDS:
            level = 0.97 if c["dataset"] == "MNIST" else 0.55
            for arm in ARMS_V2 + ARMS_OLD:
                if c["mode"] == "privacy" and (arm, c["value"]) in unreachable:
                    (excl_old if arm in ARMS_OLD else excl_v2).append({**c, "seed": seed, "arm": arm, "feasible": False})
                    continue
                acc = level + rng.normal(0, 0.01) + (shift if arm == "mfg_p2v2" else 0.)
                if tie and arm in ("PAC", "VCG"):
                    acc = level + 0.02
                acc = min(max(acc, 0.05), 0.999)
                jid = f"{c['dataset']}_{c['mode']}{c['value']:g}_a{c['alpha']:g}_s{seed}/{arm}"
                out = root / "runs" / jid
                out.mkdir(parents=True)
                ms = [{"round": r, "accuracy": float(np.trace(confusion(acc)) / 10000), "macro_f1": 0., "loss": 0.,
                       "confusion": confusion(acc).tolist()} for r in range(91, 101)]
                eps = c["value"] * (1 + (0.02 if (arm == "Lin" and unmatched_lin) else 0.002)) if c["mode"] == "privacy" else 1.0
                res = {"meta": {"dataset": c["dataset"], "alpha": c["alpha"], "seed": seed, "arm": arm,
                                "budget": c["value"] if c["mode"] == "budget" else 110., "kappa": 3.,
                                "allocation_spec_sha256": "x" * 64, "rounds": 100, "stage": "test"},
                       "summary": {"epsilon_per_client_round": eps, "accountant_parity_pass": True},
                       "measurements": ms}
                (out / "result.json").write_text(json.dumps(res))
                job = {"id": jid, "cell": dict(c), "seed": seed, "arm": arm,
                       "budget": res["meta"]["budget"], "kappa": 3., "allocation_spec_sha256": "x" * 64,
                       "done": str(out / "result.json"), "out": str(out)}
                (jobs_v2 if arm in ARMS_V2 else jobs_old).append(job)
    (base / "jobs_stage3_v2.json").write_text(json.dumps(jobs_v2))
    (old / "jobs_stage3_early.json").write_text(json.dumps(jobs_old))
    (base / "calibration_stage3_v2.json").write_text(json.dumps({"rows": [], "exclusions": excl_v2}))
    (old / "calibration_stage3_early.json").write_text(json.dumps({"rows": [], "exclusions": excl_old}))
    rows = []
    for j in jobs_v2 + jobs_old:
        r = F.run_row(j)
        rows.append({**r, "matching_pass": r["matched"]})
    ns = codex_rule()
    decision = ns["confirmation_rule"](rows, "mfg_p2v2", F.cells(), exclusions=excl_old + excl_v2)
    (root / "decision.json").write_text(json.dumps(decision))
    return base, old


def compile_tables(out):
    doc = out / "check.tex"
    doc.write_text("\\documentclass{article}\\usepackage{booktabs,graphicx}\\begin{document}\n"
                   "\\input{rq4_macros.tex}\\input{rq4_main_table.tex}\\input{rq4_supp_table.tex}\n"
                   "\\RQfourDecision{} \\RQfourCifarMean{} \\RQfourMnistSeedA{}\\end{document}\n")
    r = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", doc.name], cwd=out,
                       capture_output=True, text=True)
    return r.returncode == 0


def main():
    ok = True
    for name, shift, tie, unmatched in (("passing", 0.02, False, False), ("failing", -0.02, False, False),
                                        ("tie", 0.0, True, False), ("unmatched comparator", 0.02, False, True)):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, old = build(root, np.random.default_rng(7), shift, tie, unmatched_lin=unmatched)
            out = root / "out"
            cmd = [sys.executable, str(Path(__file__).parent / "rq4_fill.py"), "--base", str(base), "--old", str(old),
                   "--decision", str(root / "decision.json"), "--out", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            rep = json.loads((out / "rq4_recomputed.json").read_text()) if r.returncode == 0 else None
            codex = json.loads((root / "decision.json").read_text())
            good = r.returncode == 0 and rep["differences_to_codex"] == [] and rep["decision"]["pass"] == codex["pass"]
            good = good and (compile_tables(out) if codex["pass"] is not None else rep["decision"]["status"] == "incomplete")
            print(name, "Codex pass:", codex["pass"], "| rq4_fill:", rep and rep["decision"]["pass"],
                  "| differences:", rep and len(rep["differences_to_codex"]), "| tables compile:", good)
            ok &= good
            # refusal on an incomplete stage
            Path(json.loads((base / "jobs_stage3_v2.json").read_text())[0]["done"]).unlink()
            r = subprocess.run(cmd[:-1] + [str(root / "out2")], capture_output=True, text=True)
            refused = r.returncode != 0 and "incomplete" in (r.stdout + r.stderr) and not (root / "out2").exists()
            print(name, "refuses incomplete stage:", refused)
            ok &= refused
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

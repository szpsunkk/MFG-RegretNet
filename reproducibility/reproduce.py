"""Verify frozen artifacts, recompute all RQ4 endpoints, and regenerate paper assets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from runtime import HERE, FIGURES, RQ4, activate, records, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "tables", "figures"))
    parser.add_argument("--out", type=Path, default=HERE / "outputs/paper")
    args = parser.parse_args()
    print("Verified release files:", verify_release(), flush=True)
    if args.command == "verify":
        return
    activate()
    import rq4_fill as F
    jobs = records(HERE / "outputs/records", privacy=True)
    exclusions = json.loads((HERE / "evidence/rq4_exclusions.json").read_text())
    rows = [F.run_row(j) for j in jobs]
    decision = F.decide(rows, exclusions)
    archived = json.loads((HERE / "evidence/rq4_recomputed.json").read_text())
    # JSON round-trip normalizes tuple keys/values in the unchanged analyzer.
    if json.loads(json.dumps(decision)) != archived["decision"] or rows != archived["rows"]:
        raise AssertionError("recomputed results differ from the published evidence")
    codex = json.loads((HERE / "evidence/stage3_adoption.json").read_text())
    if F.compare(decision, codex):
        raise AssertionError("independent decision differs from the frozen adoption decision")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rq4_recomputed.json").write_text(json.dumps(archived, indent=2) + "\n")
    (args.out / "rq4_main_table.tex").write_text(F.main_table(decision))
    (args.out / "rq4_supp_table.tex").write_text(F.supp_table(rows, decision))
    (args.out / "rq4_macros.tex").write_text(F.macros(decision))
    for label, name, subset in (("candidate", "jobs_stage3_v2.json", jobs[:216]),
                                ("comparators", "jobs_stage3_early.json", jobs[216:])):
        directory = HERE / "outputs/jobs" / label
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(subset))
    subprocess.run([sys.executable, str(RQ4 / "rq4_privacy.py"),
                    "--base", str(HERE / "outputs/jobs/candidate"),
                    "--old", str(HERE / "outputs/jobs/comparators"),
                    "--decision", str(args.out / "rq4_recomputed.json"), "--out", str(args.out)], check=True)
    if json.loads((args.out / "rq4_privacy.json").read_text()) != json.loads((HERE / "evidence/rq4_privacy.json").read_text()):
        raise AssertionError("privacy recomputation differs from the paper")
    print(json.dumps({"runs": len(rows), "settings": len(decision["per_cell"]),
                      "decision": decision["pass"], "datasets": decision["datasets"]}), flush=True)
    if args.command == "figures":
        import rq4_curves as C
        import rq4_curves_main as S
        for ds, tag in (("MNIST", "mnist"), ("CIFAR10", "cifar10")):
            C.figure(jobs, ds, S.full_settings(ds), args.out / f"fig_rq4_curves_{tag}_all.pdf", False)
            S.figure(jobs, ds, archived["decision"]["per_cell"],
                     args.out / f"fig_rq4_curves_{tag}_main.pdf", 10, False)
            S.figure_all(jobs, ds, archived["decision"]["per_cell"],
                         args.out / f"fig_rq4_curves_{tag}_all_smoothed.pdf", 10, False)
        for name, evidence in (("make_paper_assets.py", "paper_data.json"), ("make_rq2_figure.py", "rq2_timing.json"),
                               ("make_csra_assets.py", "csra_declared.json")):
            flag = "--data" if name == "make_paper_assets.py" else "--input"
            extra = ["--extra", str(HERE / "evidence/rq3_decomposition_extra.json")] if name == "make_paper_assets.py" else []
            subprocess.run([sys.executable, str(FIGURES / name), flag, str(HERE / "evidence" / evidence),
                            "--out", str(args.out), *extra], check=True)
    print("Output:", args.out.resolve())


if __name__ == "__main__":
    main()

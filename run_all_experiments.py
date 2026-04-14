#!/usr/bin/env python3
"""
按 实验思路.md 运行完整实验套件：Phase 4 (RQ1–RQ3) → Phase 5 (表格与图) → 可选 RQ4/RQ5/RQ6/消融。
可指定 config.yaml 或命令行参数。
"""
from __future__ import division, print_function

import argparse
import os
import subprocess
import sys

def run(cmd, cwd=None):
    print("[Run]", cmd)
    r = subprocess.call(cmd, shell=True, cwd=cwd or os.getcwd())
    if r != 0:
        print("Exit code", r)
    return r

def main():
    parser = argparse.ArgumentParser(description="Run full experiment suite (实验思路)")
    parser.add_argument("--config", type=str, default="config.yaml", help="config YAML (optional)")
    parser.add_argument("--out-dir", type=str, default="run/privacy_paper")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--num-profiles", type=int, default=1000)
    parser.add_argument("--n-list", type=str, default="10,50,100")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="", help="MFG-RegretNet checkpoint")
    parser.add_argument("--mfg-regretnet-ckpt-by-n", type=str, default="", help="e.g. 10:ckpt10.pt,50:ckpt50.pt,100:ckpt100.pt")
    parser.add_argument("--regretnet-ckpt", type=str, default="", help="RegretNet checkpoint")
    parser.add_argument("--skip-rq1", action="store_true")
    parser.add_argument("--skip-rq2", action="store_true")
    parser.add_argument("--skip-rq3", action="store_true")
    parser.add_argument("--run-rq4", action="store_true", help="run RQ4 FL convergence (sample JSON if no FL deps)")
    parser.add_argument("--run-rq5", action="store_true")
    parser.add_argument("--run-rq6", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--log-scale", action="store_true", help="RQ2 figure log-log")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Phase 4
    cmd_p4 = [
        "python", "run_phase4_eval.py",
        "--n-agents", str(args.n_agents),
        "--budget", str(args.budget),
        "--num-profiles", str(args.num_profiles),
        "--seeds", args.seeds,
        "--n-list", args.n_list,
        "--out-dir", out_dir,
    ]
    if args.skip_rq1: cmd_p4.append("--skip-rq1")
    if args.skip_rq2: cmd_p4.append("--skip-rq2")
    if args.skip_rq3: cmd_p4.append("--skip-rq3")
    if args.mfg_regretnet_ckpt: cmd_p4.extend(["--mfg-regretnet-ckpt", args.mfg_regretnet_ckpt])
    if args.mfg_regretnet_ckpt_by_n: cmd_p4.extend(["--mfg-regretnet-ckpt-by-n", args.mfg_regretnet_ckpt_by_n])
    if args.regretnet_ckpt: cmd_p4.extend(["--regretnet-ckpt", args.regretnet_ckpt])

    if run(" ".join(cmd_p4)) != 0:
        print("Phase 4 failed.")
        return 1

    # Phase 5
    summary_path = os.path.join(out_dir, "phase4_summary.json")
    if not os.path.isfile(summary_path):
        print("phase4_summary.json not found.")
        return 1
    cmd_p5 = ["python", "run_phase5_tables_figures.py", "--input", summary_path, "--out-dir", out_dir]
    if args.no_figures: cmd_p5.append("--no-figures")
    if args.log_scale: cmd_p5.append("--log-scale")
    if run(" ".join(cmd_p5)) != 0:
        print("Phase 5 failed.")
        return 1

    # Optional RQ4
    if args.run_rq4:
        rq4_out = os.path.join(out_dir, "rq4_accuracy.json")
        run("python exp_rq/rq4_fl_convergence.py --out " + rq4_out + " --sample")
        run("python run_phase5_tables_figures.py --input " + summary_path + " --out-dir " + out_dir + " --accuracy-json " + rq4_out)

    # Optional RQ5
    if args.run_rq5:
        run("python exp_rq/rq5_privacy_utility.py --out-dir " + os.path.join(out_dir, "rq5"))

    # Optional RQ6
    if args.run_rq6:
        ckpt = args.mfg_regretnet_ckpt or ""
        run("python exp_rq/rq6_robustness.py --out-dir " + os.path.join(out_dir, "rq6") + (" --mfg-regretnet-ckpt " + ckpt if ckpt else ""))

    # Optional ablation
    if args.run_ablation:
        cmd_ab = ["python", "exp_rq/ablation_study.py", "--out-dir", os.path.join(out_dir, "ablation")]
        if args.regretnet_ckpt: cmd_ab.extend(["--regretnet-ckpt", args.regretnet_ckpt])
        if args.mfg_regretnet_ckpt: cmd_ab.extend(["--mfg-regretnet-ckpt", args.mfg_regretnet_ckpt])
        run(" ".join(cmd_ab))

    print("Done. Tables:", os.path.join(out_dir, "tables"))
    print("Figures:", os.path.join(out_dir, "figures"))
    return 0

if __name__ == "__main__":
    sys.exit(main())

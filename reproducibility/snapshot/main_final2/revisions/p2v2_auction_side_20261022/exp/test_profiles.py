"""common_p2v2's profile generators reproduce the paper's original generators bit for bit.

Needs the original code (main_final2/code/final_20e327e and the step-11 script), so it runs on the machine that
has them; the result is recorded with the outputs.
"""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import torch

import common_p2v2 as C

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "main_final2/code/final_20e327e"))
from datasets_fl_benchmark import generate_privacy_paper_bids  # noqa: E402


def original_hetero_gen(scenario):
    src = (ROOT / "main_final2/revisions/step11_scale_hetero_20261014/exp/hetero_eval.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "gen")
    env = {"np": np, "torch": torch, "SCEN": scenario, "SCALE": math.exp(0.5 * 2.3263478740),
           "ev": type("ev", (), {"N": 10})}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "hetero_eval.gen", "exec"), env)
    return env["gen"]


def main():
    ok = True
    for seed, count in ((1001, 50), (1005, 50), (50000, 30), (42, 30)):
        a = generate_privacy_paper_bids(10, 1, count, seed=seed)
        b = C.generate_privacy_paper_bids(10, count, seed)
        same = torch.equal(a, b)
        ok &= same
        print("generate_privacy_paper_bids seed", seed, "identical" if same else "DIFFERENT")
    full = C.test_profiles(per_seed=40)
    ref = torch.cat([generate_privacy_paper_bids(10, 1, 40, seed=s) for s in C.TEST_SEEDS]).double()
    ok &= torch.equal(full, ref)
    print("test_profiles", "identical" if torch.equal(full, ref) else "DIFFERENT")
    for scenario in ("bimodal", "realistic"):
        gen = original_hetero_gen(scenario)
        ref = torch.cat([gen(25, 7000 + s) for s in C.TEST_SEEDS]).double()
        mine = C.hetero_profiles(scenario, per_seed=25)
        same = torch.equal(ref, mine)
        ok &= same
        print("hetero", scenario, "identical" if same else "DIFFERENT")
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

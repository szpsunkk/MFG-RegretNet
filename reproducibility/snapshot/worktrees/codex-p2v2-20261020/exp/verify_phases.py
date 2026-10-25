"""Required repository phase checks, isolated on CPU and training-only evaluation.

This is a functional check of the inherited scripts, not a P2 experiment. Their
historical Laplace FL results are never included in the DP-SGD comparison.
"""
import argparse
import copy
import os
import runpy
import sys

import torch
from torch.utils.data import Subset

from common import ROOT, OUT
import datasets_fl_benchmark as datasets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("run functional phase checks with CUDA_VISIBLE_DEVICES='' while predecessor is active")
    torch.set_num_threads(1)
    scratch = OUT / "validation/phase_smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    sys.path.append(str(ROOT))  # client/experiments used only by the legacy smoke scripts
    for name, dataset_folder in (("load_mnist", "mnist"), ("load_cifar10", "cifar10")):
        original = getattr(datasets, name)
        def training_only_loader(*args, original=original, dataset_folder=dataset_folder, **kwargs):
            train, official = original(root=str(ROOT / "data" / dataset_folder), download=False)
            clean = copy.copy(train)
            clean.transform = official.transform
            # No official test prediction, including in the inherited smoke path.
            return train, Subset(clean, list(range(200)))
        setattr(datasets, name, training_only_loader)
    filename = {1: "run_phase1_full_check.py", 2: "run_phase2_verify.py", 3: "run_phase3_verify.py"}[args.phase]
    print("FUNCTIONAL ONLY: CPU; isolated output; training-only 200-record evaluation; not P2 accuracy evidence", flush=True)
    runpy.run_path(str(ROOT / filename), run_name="__main__")


if __name__ == "__main__":
    main()

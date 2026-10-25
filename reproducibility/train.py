"""Portable replay of the final CPU/float64 QV-sel training schedule."""
import argparse
import json
from pathlib import Path
import time

import torch

from runtime import HERE, WORK, SNAPSHOT, activate, definitions, sha, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("mfg_p2v2", "nob_p2v2", "profile_p2v2"), default="mfg_p2v2")
    parser.add_argument("--steps", type=int, default=12000, help="values below 12000 are marked smoke runs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 12000:
        parser.error("steps must be in [1,12000]")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("choose an empty --out directory")
    verify_release()
    activate()
    import p2v2_mech as M
    from p2_reference import sample_log_uniform
    sample = definitions(WORK / "exp/train_mechanism.py", ["sample_profiles"], {"torch": torch})["sample_profiles"]
    table_path = SNAPSHOT / "run/tdsc_revision/codex_p2v2_20261020/A7_tau/table.json"
    table = json.loads(table_path.read_text())
    torch.set_num_threads(1)
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    net = M.SelectorModel(args.family).double()
    for mode in ("budget", "fixed_total"):
        getattr(net, "tau_" + mode).copy_(torch.tensor(table[mode], dtype=torch.float64))
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    began = time.monotonic()
    for step in range(args.steps):
        lr = .001 if step < 10000 else .00003
        for group in optimizer.param_groups:
            group["lr"] = lr
        reports, budget = sample(64, generator, "cpu")
        kappa = sample_log_uniform(64, .003, 300., generator, "cpu")
        optimizer.zero_grad(set_to_none=True)
        loss = net.objective(reports.double(), budget.double(), kappa.double()).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "loss": float(loss.detach()), "lr": lr}
            history.append(row)
            print(json.dumps(row), flush=True)
    target = M.load(args.family)
    max_difference = max(float((v - target.state_dict()[k]).abs().max()) for k, v in net.state_dict().items())
    torch.save({"family": args.family, "gamma": 1., "seed": 42, "selected_proxy": "QV-sel",
                "tau_sha256": sha(table_path), "steps": args.steps, "state_dict": net.state_dict(),
                "partial_smoke": args.steps != 12000}, args.out / "model.pt")
    (args.out / "training.json").write_text(json.dumps({"history": history, "steps": args.steps,
        "partial_smoke": args.steps != 12000, "max_abs_difference_to_frozen_checkpoint": max_difference,
        "seconds": time.monotonic() - began, "torch": torch.__version__}, indent=2) + "\n")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()

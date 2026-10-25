"""Prepare one immutable tranche of jobs as its dependency becomes available."""
import argparse
from pathlib import Path
import torch
from common import OUT
from p2_state import read, CONFIG, check_identity
from p2_jobs import prepare
from p2_core import FAMILIES
from a1_rules import confirmation_cells

EXTERNAL = ("V8F", "RegretNet-r2", "No-population-r2", "DM-RegretNet", "MFG-Pricing",
            "Posted-price", "PAC", "PAC-capped-legacy", "VCG", "VCG-capped-legacy", "CSRA", "Lin", "Oracle-P2")


def build(group):
    check_identity()
    decision = read(OUT / "stage0_decision.json")
    assert decision["gate_pass"]
    stage3 = group.startswith("stage3")
    smoke = group.startswith("smoke_")
    cells = confirmation_cells() if stage3 else read(CONFIG)["stage2_development"]["cells"]
    seeds, rounds = ([231, 232, 233], 100) if stage3 else ([14, 15], 40)
    if smoke:
        cells = [{"dataset": ds, "alpha": .5, "mode": "budget", "value": 30.} for ds in ("MNIST", "CIFAR10")]
        seeds, rounds = [14], 2
    reference = None
    gate = OUT / "stage0_decision.json"
    if group.endswith("early"):
        arms = [{"arm": a} for a in (EXTERNAL if stage3 else ("V8F", "CSRA", "PAC", "Oracle-P2"))]
    elif group.endswith("p2"):
        gate = OUT / ("stage2_selection.json" if stage3 else "stage1_screen.json")
        assert read(gate)["gate_pass"]
        if stage3:
            selected = read(gate)["selected"]
            arms = [selected[k] for k in FAMILIES] + [{"arm": "Uniform-DP"}]
            reference = Path(selected["mfg_p2"]["checkpoint"])
        else:
            arms = [{"arm": k, "kind": k, "checkpoint": str(OUT / "mechanisms" / k / "model.pt")} for k in FAMILIES]
    elif group.endswith("mf"):
        gate = OUT / ("mf_stage2_selection.json" if stage3 else "mf_stage1_screen.json")
        assert read(gate)["gate_pass"]
        reference = Path(read(OUT / "mf_registration.json")["checkpoint"])
        arms = [{"arm": "mf_p2", "kind": "mf_p2", "checkpoint": str(reference)}]
        if stage3:
            arms.append({"arm": "Uniform-DP@mf_p2"})
        else:
            reference = None
    else:
        raise ValueError(group)
    return prepare(group, cells, seeds, arms, rounds, decision, reference,
                   smoke_size=64 if smoke else None, authorization=gate)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    build(a.group)

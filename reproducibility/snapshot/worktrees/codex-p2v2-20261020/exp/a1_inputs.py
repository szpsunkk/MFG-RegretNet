"""Pinned Stage A handoff and verifiable reuse of the original P2 artifacts."""
from pathlib import Path
import shutil

import torch

from common import ROOT, OUT, WORK, digest
from p2_state import read, immutable_json, freeze_stage_inputs
from a1_mf import load_mf

BASE = ROOT / "run/tdsc_revision/codex_p2_101_20261019"
INBOX = ROOT / "run/tdsc_revision/codex_p2_a1_input_20261019"
LN_PROTOCOL_HASH = "839dca288bdcea44ea5e5549b757515e2d4a48aa270ae703f4035e689043c535"
MF_REFERENCE_HASH = "ca79cd5e16715f4bad9d0fa72c7fe6bb837dab8c647f4f194f1efb112975ec2b"


def verify_handoff(folder):
    folder = Path(folder)
    handoff = read(folder / "p2_handoff.json")
    required = {"schema": "p2-a1-mf-handoff-v1", "protocol": "ln-v1", "family": "MF-Net",
                "protocol_sha256": LN_PROTOCOL_HASH, "reference_sha256": MF_REFERENCE_HASH,
                "selection_used_official_test": False, "selected_before_fl": True}
    if any(handoff.get(k) != v for k, v in required.items()):
        raise ValueError("handoff does not attest the frozen ln-v1 Stage A selection")
    if handoff.get("training_seed") not in (61, 62, 63):
        raise ValueError("MF-Net seed is outside ln-v1")
    # Fixed filenames prevent paths in an external manifest from escaping its folder.
    for filename, key in (("selected_mf.pt", "checkpoint_sha256"), ("stage_a_selection.json", "selection_sha256")):
        if digest(folder / filename) != handoff.get(key):
            raise ValueError("Stage A handoff hash mismatch: " + filename)
    selection = read(folder / "stage_a_selection.json")
    selected = selection.get("selected", {}).get("MF-Net", {})
    if selected.get("checkpoint_sha256") != handoff["checkpoint_sha256"] or selected.get("seed") != handoff["training_seed"]:
        raise ValueError("handoff does not match the selected MF-Net in the Stage A evidence")
    if selection.get("selection_used_official_test") is not False or selection.get("dev_profile_seed") != 71:
        raise ValueError("selection must use ln-v1 development profiles only")
    saved = torch.load(folder / "selected_mf.pt", map_location="cpu", weights_only=False)
    load_mf(saved)
    return handoff


def register_mf():
    target = OUT / "mf_registration.json"
    if not (INBOX / "p2_handoff.json").exists():
        return read(target) if target.exists() else None
    handoff = verify_handoff(INBOX)
    destination = OUT / "mf_input"
    proposed = {"handoff": handoff, "checkpoint": str(destination / "selected_mf.pt"),
                "checkpoint_sha256": handoff["checkpoint_sha256"],
                "selection_sha256": handoff["selection_sha256"], "within_p2_seed_selection": False}
    if target.exists():
        if read(target) != proposed:
            raise RuntimeError("MF-Net is already frozen; another seed/checkpoint cannot replace it")
        return read(target)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("selected_mf.pt", "stage_a_selection.json", "p2_handoff.json"):
        source, dest = INBOX / name, destination / name
        if dest.exists() and digest(source) != digest(dest):
            raise RuntimeError("partial MF-Net registration differs")
        if not dest.exists():
            shutil.copy2(source, dest)
    immutable_json(target, proposed)
    freeze_stage_inputs([target, *[destination / name for name in ("selected_mf.pt", "stage_a_selection.json", "p2_handoff.json")]])
    return proposed


def inherit(name):
    source, target = BASE / name, OUT / name
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) != digest(source):
        raise RuntimeError("inherited frozen artifact differs: " + name)
    if not target.exists():
        shutil.copy2(source, target)
    return target


def verified_result(job):
    result = read(job["done"])
    expected = {**{k: job["cell"][k] for k in ("dataset", "alpha")},
                **{k: job[k] for k in ("seed", "arm", "budget", "kappa")}}
    if any(result["meta"].get(k) != v for k, v in expected.items()):
        raise RuntimeError("reused result differs from its job")
    if result["meta"]["allocation_spec_sha256"] != job["allocation_spec_sha256"]:
        raise RuntimeError("reused result allocation hash differs")
    if not result["summary"]["accountant_parity_pass"]:
        raise RuntimeError("reused result has accountant mismatch")
    for name in ("events", "privacy"):
        if digest(Path(job["out"]) / (name + ".json")) != result[name + "_sha256"]:
            raise RuntimeError("reused result evidence changed")
    return result


def reusable(old_job, new_job):
    if not Path(old_job["done"]).exists():
        return False
    old, new = read(old_job["allocation_spec"]), read(new_job["allocation_spec"])
    for job, spec in ((old_job, old), (new_job, new)):
        if digest(job["allocation_spec"]) != job["allocation_spec_sha256"]:
            raise RuntimeError("allocation specification changed before reuse")
        for path, expected in spec["input_sha256"].items():
            if digest(path) != expected:
                raise RuntimeError("an input to a reused run changed")
    fields = ("dataset", "alpha", "seed", "arm", "budget", "kappa", "rounds", "aggregation", "stage",
              "smoke_size", "split_sha256", "reports_sha256", "allocation_sha256", "payment_sha256",
              "executed", "payment", "proposed", "offered_payment")
    if any(old[k] != new[k] for k in fields):
        return False
    verified_result(old_job)
    return True


def reuse_jobs(phase, new_jobs):
    manifest = BASE / ("jobs_" + phase + ".json")
    previous = {j["id"]: j for j in read(manifest)} if manifest.exists() else {}
    merged, receipt = [], []
    for new in new_jobs:
        old = previous.get(new["id"])
        selected = old if old is not None and reusable(old, new) else new
        merged.append(selected)
        if selected is old:
            receipt.append({"id": old["id"], "original_result": old["done"],
                            "original_result_sha256": digest(old["done"]),
                            "original_spec_sha256": digest(old["allocation_spec"]),
                            "a1_spec_sha256": digest(new["allocation_spec"])})
    immutable_json(OUT / ("reuse_" + phase + ".json"), receipt)
    return merged


def verify_reuse_receipts(phase):
    for row in read(OUT / ("reuse_" + phase + ".json")):
        if digest(row["original_result"]) != row["original_result_sha256"]:
            raise RuntimeError("reused result changed after it was accepted")

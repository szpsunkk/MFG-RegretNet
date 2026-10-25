"""A3 source seals, portable paths and progress without scientific endpoints."""
import json
import os
from pathlib import Path
import time

from common import ROOT, WORK, OUT, PAPER, digest, atomic_json
from p2_core import REFERENCES

CONFIG = WORK / "protocol.json"
DESIGN = ROOT / "main_final2/revisions/p2_mechanism_design_20261018"
AMENDMENT = ROOT / "main_final2/revisions/codex_p2_101_20261019/AMENDMENT_20261018_before_runs.md"
A3 = DESIGN / "AMENDMENT_A3_20261019.md"
PARENT = ROOT / "run/tdsc_revision/codex_taskaware_20261017"
EXPECTED = {
    CONFIG: "ef54f45251d3c7b5b567745774af5040050bd32182c358cfe3ae81bfe62b2d8e",
    WORK / "exp/p2_reference.py": "d4dfdbc7c253e3de452df4f3d900dad067f071d71e92416994be5195723fabcb",
    WORK / "exp/mf_reference.py": "ca79cd5e16715f4bad9d0fa72c7fe6bb837dab8c647f4f194f1efb112975ec2b",
    A3: "8165bdd4924265f5fe0dfb7cb80c8604714d3e928dd1da1579c6f0f9ed5cb4cf",
    DESIGN / "AMENDMENT_A4_20261020.md": "1d3588d43c5a42ad77b153a6dd50dac54a86e4d78bd1e8bb0debdbad80b427fd",
    DESIGN / "AMENDMENT_A5_20261020.md": "4b55952441e27e7293ddc0b6570e06f993bdf3569800c3499d5adec1ded07225",
    WORK / "exp/qv_reference.py": "24398d1fc7f5fe074186c9951508738ff73a03f8593f5dc7aa3771052380ecb0",
    WORK / "exp/p2v2_reference.py": "722f72832650bb1a23dd6464f486bc6fbb2c7965f74c4865d06749c4df5ccc12",
    DESIGN / "AMENDMENT_A6_20261020.md": "57e0492e63bf429aab486de8375aeda7e17f10d222d50a44f3d7cf5b09443600",
    WORK / "exp/qv_sel_reference.py": "f7a3d65b75fa5cef2932919133e80243a20f88d091508929882fc156e991a782",
}


def relocate(value):
    """Resolve captured machine paths, without rewriting bytes or their hashes."""
    if isinstance(value, str):
        for prefix in ("/data/home/skk/paper/FL-market/", "/home/skk/paper/FL-market/"):
            if value.startswith(prefix):
                return str(ROOT / value[len(prefix):])
    elif isinstance(value, list):
        return [relocate(v) for v in value]
    elif isinstance(value, dict):
        return {relocate(k): relocate(v) for k, v in value.items()}
    return value


def read(path):
    return relocate(json.loads(Path(path).read_text()))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != relocate(value):
            raise RuntimeError(f"frozen artifact differs: {path}")
    else:
        atomic_json(path, value)


def source_identity():
    files = sorted((WORK / "exp").glob("*.py")) + sorted((WORK / "frozen").rglob("*.py"))
    files += sorted((WORK / "ops").glob("*.py"))
    files += [CONFIG, AMENDMENT, A3, DESIGN / "AMENDMENT_A1_20261019.md", DESIGN / "AMENDMENT_A2_20261019.md"]
    files += [DESIGN / "AMENDMENT_A4_20261020.md", DESIGN / "AMENDMENT_A5_20261020.md"]
    files += [DESIGN / "AMENDMENT_A6_20261020.md"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def input_identity():
    paths = sorted(set(v[0] for v in REFERENCES.values()))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def seal():
    check_reference_hashes()
    path = OUT / "launch_manifest.json"
    if not path.exists():
        atomic_json(path, {"created_unix": time.time(), "source_sha256": source_identity(),
                           "input_sha256": input_identity(), "protocol_sha256": digest(CONFIG),
                           "amendment_sha256": digest(AMENDMENT), "a3_sha256": digest(A3)})
    check_identity()


def check_reference_hashes():
    for p, expected in EXPECTED.items():
        if digest(p) != expected:
            raise RuntimeError(f"reference changed: {p}")


def terminal_previous(old, parent):
    """Compatibility check for the inherited tests, not an A3 scheduling gate."""
    if old.get("failed") or parent.get("status") == "failed" or parent.get("failed"):
        return False, "predecessor failure"
    old_done = old.get("complete") == old.get("total") == 252 and not old.get("active")
    parent_done = parent.get("status") == "complete" and (parent.get("phase") == "complete" or
        (parent.get("phase") == "development_complete" and parent.get("promotion_gate_pass") is False))
    return old_done and parent_done, "legacy terminal check"


def check_identity():
    manifest = read(OUT / "launch_manifest.json")
    if manifest["source_sha256"] != source_identity() or manifest["input_sha256"] != input_identity():
        raise RuntimeError("A3 source/input changed after sealing")
    registry = OUT / "frozen_stage_inputs.json"
    if registry.exists():
        for p, expected in read(registry).items():
            if digest(p) != expected:
                raise RuntimeError(f"frozen input changed: {p}")


def freeze_stage_inputs(paths):
    registry = OUT / "frozen_stage_inputs.json"
    value = read(registry) if registry.exists() else {}
    for path in paths:
        key, hashed = str(path), digest(path)
        if key in value and value[key] != hashed:
            raise RuntimeError(f"cannot replace frozen input: {key}")
        value[key] = hashed
    atomic_json(registry, value)


PROGRESS_KEYS = {"status", "round", "total_rounds", "elapsed_seconds", "estimated_remaining_seconds",
                 "identity", "phase", "step", "profiles", "total", "seconds", "strong_search"}


def safe_progress(value):
    return {k: v for k, v in (value or {}).items() if k in PROGRESS_KEYS}


def update(phase, **fields):
    if "active" in fields:
        fields["active"] = [{**r, "progress": safe_progress(r.get("progress"))} for r in fields["active"]]
    atomic_json(OUT / "status.json", {"phase": phase, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def note(message):
    PAPER.mkdir(parents=True, exist_ok=True)
    with (PAPER / "PLAN_AND_PROGRESS_zh.md").open("a") as f:
        f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M:%S %z')}: {message}\n")


def wait_previous():
    raise RuntimeError("Use the A3 dependency scheduler")

"""Portable entrypoints around the byte-preserved experiment sources."""
import hashlib
import ast
import json
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "snapshot"
WORK = SNAPSHOT / "worktrees/codex-p2v2-20261020"
AUCTION = SNAPSHOT / "main_final2/revisions/p2v2_auction_side_20261022/exp"
FIGURES = SNAPSHOT / "main_final2/revisions/p2v2_paper_figs_20261023/exp"
RQ4 = SNAPSHOT / "main_final2/revisions/p2v2_rq4_fill_20261022/exp"
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate():
    os.environ["FL_ROOT"] = str(SNAPSHOT)
    for folder in (WORK / "exp", AUCTION, RQ4, FIGURES):
        sys.path.insert(0, str(folder))


def records(destination, privacy=False):
    """Extract the two original JSON files per run and verify every byte."""
    destination = Path(destination).resolve()
    jobs = json.loads((HERE / "evidence/rq4_index.json").read_text())
    with zipfile.ZipFile(HERE / "evidence/rq4_records.zip") as archive:
        expected = {f'{j["key"]}/{name}.json' for j in jobs for name in ("result", "allocation")}
        if set(archive.namelist()) != expected:
            raise ValueError("unexpected archive members")
        for job in jobs:
            for name in ("result", "allocation"):
                relative = f'{job["key"]}/{name}.json'
                path = destination / relative
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(relative))
                if sha(path) != job[name + "_sha256"]:
                    raise ValueError("record hash mismatch: " + relative)
            job["done"] = str(destination / job["key"] / "result.json")
            job["allocation_spec"] = str(destination / job["key"] / "allocation.json")
    if privacy:
        with zipfile.ZipFile(HERE / "evidence/rq4_privacy_ledgers.zip") as archive:
            expected = {j["key"] + "/privacy.json" for j in jobs}
            if set(archive.namelist()) != expected:
                raise ValueError("unexpected privacy archive members")
            for job in jobs:
                rel = job["key"] + "/privacy.json"
                path = destination / rel
                if not path.exists():
                    path.write_bytes(archive.read(rel))
                if sha(path) != job["privacy_sha256"]:
                    raise ValueError("privacy ledger hash mismatch: " + rel)
    return jobs


def verify_release():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for relative, expected in manifest.items():
        if sha(HERE.parent / relative) != expected:
            raise ValueError("release hash mismatch: " + relative)
    return len(manifest)


def definitions(path, names, namespace):
    """Load selected unmodified definitions without historical scheduler imports.

    Used for the training sampler and the original Selector parity check. The
    release manifest verifies these source files before they are executed.
    """
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("missing frozen definition")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace

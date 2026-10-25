"""Allocation-only preregistration for the author's 32 development runs.

Never opens FL result files or calls an accuracy analyser. The original
run_cell.py remains byte-for-byte unchanged. All four arms are prepared before
one immutable registration is written and before any FL process is launched.
"""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

from common import ROOT, WORK, OUT, PAPER, atomic_json, digest, data_for_cell, market_reports
import p2_core as P
from p2_state import CONFIG, AMENDMENT, source_identity, input_identity, check_reference_hashes

ORIGINAL = ROOT / "worktrees/codex-p2v2-20261020"
PRIOR = ROOT / "run/tdsc_revision/codex_p2v2_20261020"
TASK = ROOT / "CODEX_TASK_20261023_aggregation_2x2.md"
TASK_SHA256 = "2a1d784247134d3f81b45560115c5d4d60413fc2c49b6a00361924665468ed96"
CHECKPOINT = PRIOR / "continuation_A7/QV-sel/training/g1/mfg_p2v2/model.pt"
DECISION = PRIOR / "stage3_adoption_A7.json"
REFERENCE = ROOT / "main_final2/revisions/p2v2_auction_side_20261022/exp"
ANALYSIS = ROOT / "main_final2/revisions/agg2x2_design_20261023/exp"
EXPECTED = {
    ANALYSIS / "agg2x2_analysis.py": "dd3d2cd4454b95340bd4152a37a4c96277924d1e7ab9d73a52ee3738812cf116",
    ANALYSIS / "test_agg2x2_analysis.py": "933b6a61e012a35b2aa3f53ceb36571aaff14f35029efa50959b1f239b6d39f5",
    REFERENCE / "csra_declared.py": "ef4844ade9d65734594a00add0253747d13b1c5d6dbd3685b70e746f5cad7bda",
    REFERENCE / "common_p2v2.py": "a4984138a4e1116af9e90740004cc71f1ec12757cfa1502a54b5d663694aae81",
    CHECKPOINT: "baddeed7201f2d4dfe82bce9aeb65a0fed86d29759618cab4a7130f57a2b382b",
    PRIOR / "A7_tau/table.json": "777b80c70115950ba1cb968bc35e9e3e66cf10c93ab6295175eb8b891d7dd3c7",
    DECISION: "670eff4c9ca61dcb58f15cbf7f6dabfbba470ed1e38df77790869cdf61d6bc37",
    ORIGINAL / "a7flow/model.py": "1defd361aac95809ea713ae1be503befeebf0b106744c8fd900a0a6be94248de",
}
ARMS = ("mfg_p2v2", "Uniform-DP", "CSRA", "CSRA-declared")
BLOCKS = {"mfg_p2v2": "A", "Uniform-DP": "A", "CSRA": "B", "CSRA-declared": "C"}
PROHIBITED = set(range(10, 17)) | {18, 19, 20} | set(range(61, 64)) | set(range(71, 74))
PROHIBITED |= set(range(101, 104)) | set(range(211, 214)) | set(range(231, 234)) | set(range(301, 311))


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def immutable(path, value):
    path = Path(path)
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise RuntimeError("immutable artifact differs: " + str(path))
    else:
        atomic_json(path, normalized)


def check_hashes():
    expected = dict(EXPECTED, **{})
    expected[TASK] = TASK_SHA256
    rows = {str(p.relative_to(ROOT)): {"expected": h, "actual": digest(p)} for p, h in expected.items()}
    if any(v["expected"] != v["actual"] for v in rows.values()):
        raise RuntimeError("task/reference SHA256 mismatch; no work permitted")
    return rows


def source_parity():
    allowed = {"exp/common.py", "exp/mechanisms.py"}
    changes = []
    for p in ORIGINAL.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(ORIGINAL)
        target = WORK / rel
        if not target.is_file() or digest(target) != digest(p):
            changes.append(str(rel))
    if set(changes) != allowed:
        raise RuntimeError("unexpected changes to inherited workspace: " + repr(changes))
    return {"changed_inherited_files": changes, "run_cell_unchanged": True,
            "run_cell_sha256": digest(WORK / "exp/run_cell.py"),
            "settlement_unchanged": digest(WORK / "exp/contracts.py") == digest(ORIGINAL / "exp/contracts.py"),
            "selector_unchanged": digest(WORK / "a7flow/model.py") == EXPECTED[ORIGINAL / "a7flow/model.py"]}


def seed_audit():
    """Inspect protocol declarations and path/queue identities, never outcomes."""
    names = subprocess.check_output([
        "rg", "--files", "--hidden", "--no-ignore", "-g", "!**/.git/**",
        "-g", "!data/**", "-g", "!**/node_modules/**", "-g", "!**/__pycache__/**",
    ], cwd=ROOT, text=True).splitlines()
    own = (str(WORK.relative_to(ROOT)) + "/", str(OUT.relative_to(ROOT)) + "/",
           str(PAPER.relative_to(ROOT)) + "/")
    names = [f for f in names if not f.startswith(own)]
    used = set(PROHIBITED)
    evidence = {}
    documents = {}
    contexts = []
    identities = {}
    pattern = re.compile(r"(?:^|[/_\-])(?:s|seed)[_\-=]?(\d+)(?=[/_.\-]|$)", re.I)

    def remember(seed, path):
        if isinstance(seed, int) and not isinstance(seed, bool):
            used.add(seed)
            evidence.setdefault(str(seed), [])
            if len(evidence[str(seed)]) < 8:
                evidence[str(seed)].append(path)

    def json_seeds(obj, path, under_seed=False):
        if isinstance(obj, dict):
            for k, v in obj.items():
                json_seeds(v, path + "/" + str(k), under_seed or "seed" in k.lower())
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                json_seeds(v, path + "/" + str(i), under_seed)
        elif under_seed:
            remember(obj, path)

    for name in names:
        for match in pattern.finditer(name):
            remember(int(match.group(1)), name)
        p = ROOT / name
        low = p.name.lower()
        document = (p.suffix.lower() in (".md", ".json", ".yaml", ".yml") and
                    ("protocol" in low or "amendment" in low or low.startswith("codex_task")))
        queue = (p.suffix == ".json" and (low.startswith("queue_") or low.startswith("jobs_") or
                                         low in ("protocol.json", "registration.json")))
        if not document and not queue:
            continue
        if p == TASK:
            continue
        if p.stat().st_size > 25 * 1024**2:
            raise RuntimeError("seed identity file too large for safe inspection: " + name)
        text = p.read_text()
        if document:
            documents[name] = digest(p)
        if p.suffix == ".json":
            try:
                obj = json.loads(text)
            except ValueError:
                continue
            json_seeds(obj, name)
            if queue:
                identities[name] = digest(p)
        else:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                # Keep the exact contexts for human audit. Nearby occurrence
                # counts/budgets are not automatically treated as seed IDs.
                if re.search(r"seed|种子", line, re.I):
                    snippet = " ".join(lines[max(0, i - 1):i + 3])
                    if re.search(r"(?<![\d.])(24|25|26|27)(?![\d.])", snippet):
                        contexts.append({"path": name, "line": i + 1, "text": snippet})
                    for match in re.finditer(r"(?:seeds?|种子)\s*[:：=]?\s*[*`\[({ ]*(\d+(?:\s*[,/、]\s*\d+)*)", line, re.I):
                        for v in re.findall(r"\d+", match.group(1)):
                            remember(int(v), name + ":" + str(i + 1))
    return {"time": now(), "path_count": len(names), "documents_sha256": documents,
            "queue_identity_sha256": identities, "used_seeds": sorted(used),
            "seed_evidence": evidence, "candidate_document_contexts": contexts,
            "scope": "all indexed and ignored repository paths; protocol/task/amendment declarations; queue seed fields; no result payloads"}


def configure_selector():
    for name in ("table.json", "table_sha256.json"):
        src, dst = PRIOR / "A7_tau" / name, OUT / "A7_tau" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and digest(dst) != digest(src):
            raise RuntimeError("copied tau differs")
        if not dst.exists():
            shutil.copy2(src, dst)
    sys.path.insert(0, str(WORK / "next"))
    sys.path.insert(0, str(WORK / "a7flow"))
    from model import load_model
    from stage3_v2 import detail_for
    return load_model(CHECKPOINT), detail_for


@contextlib.contextmanager
def selector_reference(model, detail_for):
    """Use the frozen A7 model when p2_core requests its reference contract.

    Only the new job builder adapts the model-loading interface. The call to
    p2_core.allocation('Uniform-DP', ...) and its complete matching rule are
    unchanged, including AllocationInfeasible and the minimum-event rule.
    """
    old_load, old_raw = P.load_any, P.raw
    def load(path, kind=None):
        if Path(path).resolve() == CHECKPOINT.resolve():
            return model
        return old_load(path, kind)
    def raw(net, reports, budgets, kappas):
        if net is model:
            assert torch.all(budgets == 30) and torch.all(kappas == 30)
            detail = detail_for(net, reports, 30., 30., None)
            return detail["proposed"], detail["offered_payment"]
        return old_raw(net, reports, budgets, kappas)
    P.load_any, P.raw = load, raw
    try:
        yield
    finally:
        P.load_any, P.raw = old_load, old_raw


def declared_check(reports, detail):
    sys.path.insert(0, str(REFERENCE))
    import csra_declared
    from mechanisms import allocate, csra_qms_batch
    original_e, original_p = csra_qms_batch(reports, torch.full((40, 1), 30.))
    e, p = allocate("CSRA-declared", reports, 30., return_offer=True)
    reference_e, reference_p = csra_declared.csra(reports.double(), torch.full((40,), 30., dtype=torch.float64), "declared")
    winner_equal = torch.equal(e > 0, reference_e > 0) and torch.equal(e > 0, original_e > 0)
    payment_error = float((p.double() - reference_p).abs().max())
    exposure_equal = torch.equal(e, torch.where(original_e > 0, reports[..., 1].to(e.dtype), torch.zeros_like(e)))
    row = {"winner_set_equal": winner_equal, "payment_max_abs_difference": payment_error,
           "original_payments_bitwise_equal": torch.equal(p, original_p),
           "release_equals_declared_capacity": exposure_equal, "markets": len(reports)}
    if not (winner_equal and payment_error <= 1e-5 and exposure_equal and row["original_payments_bitwise_equal"]):
        raise RuntimeError("Block C reference implementation check failed: " + repr(row))
    from contracts import settlement
    executed, paid, _ = settlement(reports.to(e.dtype), e, p)
    if not torch.equal(executed, detail["executed"]) or not torch.equal(paid, detail["payment"]):
        raise RuntimeError("Block C settlement differs")
    row["settlement_bitwise_equal"] = True
    return row


def build_job(block, alpha, seed, arm, aggregation, reports, split, detail, registration):
    tag = f"CIFAR10_budget30_a{alpha}_s{seed}/{arm}/{aggregation}"
    folder = OUT / "runs" / block / tag
    spec_path = OUT / "allocations" / block / tag / "allocation.json"
    inputs = [TASK, OUT / "registration.json", OUT / "registration_receipt.json", CONFIG, AMENDMENT,
              CHECKPOINT, OUT / "A7_tau/table.json", DECISION]
    if arm == "CSRA-declared":
        inputs += [REFERENCE / "csra_declared.py", REFERENCE / "common_p2v2.py"]
    spec = {"dataset": "CIFAR10", "mode": "budget", "value": 30., "alpha": alpha,
            "seed": seed, "arm": arm, "kappa": 30., "budget": 30., "rounds": 40,
            "aggregation": aggregation, "stage": "dev", "smoke_size": None,
            "comparison": {"diagnostic": "aggregation_2x2", "block": block, "budget": 30.},
            "split_sha256": split["sha256"], "reports_sha256": P.array_hash(reports),
            "allocation_sha256": P.array_hash(detail["executed"]),
            "payment_sha256": P.array_hash(detail["payment"]),
            "input_sha256": {str(p): digest(p) for p in inputs},
            "non_strategic_reference": arm == "Uniform-DP", "voluntary": detail["voluntary"],
            "reference_sha256": digest(CHECKPOINT) if arm == "Uniform-DP" else None,
            "confirmation_gate_sha256": None, "selected_proxy": "QV-sel",
            "post_hoc_block": block == "C",
            **{key: detail[key].double().tolist() for key in ("proposed", "offered_payment", "executed", "payment")}}
    immutable(spec_path, spec)
    command = [sys.executable, "-u", str(WORK / "exp/run_cell.py"), "--dataset", "CIFAR10",
               "--alpha", str(alpha), "--seed", str(seed), "--arm", arm, "--budget", "30.0",
               "--stage", "dev", "--rounds", "40", "--aggr", aggregation,
               "--out", str(folder), "--allocation-spec", str(spec_path), "--spec-sha256", digest(spec_path)]
    if arm == "mfg_p2v2":
        command += ["--checkpoint", str(CHECKPOINT)]
    if arm == "Uniform-DP":
        command += ["--reference", str(CHECKPOINT)]
    return {"id": block + "/" + tag, "block": block, "alpha": alpha, "seed": seed,
            "arm": arm, "aggregation": aggregation, "done": str(folder / "result.json"),
            "out": str(folder), "command": command, "allocation_spec": str(spec_path),
            "allocation_spec_sha256": digest(spec_path), "allocation_sha256": spec["allocation_sha256"],
            "payment_sha256": spec["payment_sha256"]}


def prepare():
    torch.set_num_threads(1)
    hashes = check_hashes()
    check_reference_hashes()
    parity = source_parity()
    PAPER.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "PREPARED.json").exists():
        raise RuntimeError("already prepared: inspect sealed registration instead of rebuilding")
    immutable(OUT / "validation/task_hashes.json", {"time": now(), "files": hashes})
    immutable(OUT / "validation/source_parity.json", parity)
    analysis_proof = json.loads((OUT / "validation/analysis_tests.json").read_text())
    assert analysis_proof["exit_code"] == 0 and analysis_proof["cases_passed"] == 14
    assert analysis_proof["test_sha256"] == EXPECTED[ANALYSIS / "test_agg2x2_analysis.py"]
    decision = json.loads(DECISION.read_text())
    external = {str(r["alpha"]): r["external_arm"] for r in decision["per_cell"]
                if r["dataset"] == "CIFAR10" and r["mode"] == "budget" and r["value"] == 30}
    del decision
    if external != {"0.1": "CSRA", "0.5": "CSRA"}:
        raise RuntimeError("Block B arm disagrees with the task: " + repr(external))
    audit = seed_audit()
    immutable(OUT / "validation/seed_audit.json", audit)
    used = set(audit["used_seeds"])
    model, detail_for = configure_selector()
    selected, skipped, prepared, c_checks = {}, [], {}, []
    for alpha in (.1, .5):
        selected[str(alpha)] = []
        seed = 24
        while len(selected[str(alpha)]) < 2:
            if seed in used:
                skipped.append({"alpha": alpha, "seed": seed, "reason": "previous seed use",
                                "evidence": audit["seed_evidence"].get(str(seed), ["explicit task exclusion"])})
                seed += 1
                continue
            clients, _, split, _ = data_for_cell("CIFAR10", alpha, seed, 64, "dev")
            reports = market_reports(40, [len(c) for c in clients], seed)
            try:
                detail = {"mfg_p2v2": detail_for(model, reports, 30., 30., None)}
                with selector_reference(model, detail_for):
                    ref = P.allocation("selected-reference", reports, 30., 30., checkpoint=CHECKPOINT)
                    if not torch.equal(ref["executed"], detail["mfg_p2v2"]["executed"]):
                        raise RuntimeError("Uniform reference differs from frozen A7 mechanism")
                    detail["Uniform-DP"] = P.allocation("Uniform-DP", reports, 30., 30., reference=CHECKPOINT)
                for arm in ("CSRA", "CSRA-declared"):
                    detail[arm] = P.allocation(arm, reports, 30., 30.)
            except P.AllocationInfeasible as error:
                skipped.append({"alpha": alpha, "seed": seed, "reason": str(error), "allocation_only": True})
                seed += 1
                continue
            c_check = {"alpha": alpha, "seed": seed, **declared_check(reports, detail["CSRA-declared"])}
            c_checks.append(c_check)
            prepared[(alpha, seed)] = (reports, split, detail)
            selected[str(alpha)].append(seed)
            print("ALLOCATION_CHECKED", alpha, seed, flush=True)
            seed += 1
    registration = {"seeds": selected, "block_b_arm": external, "block_c_arm": "CSRA-declared"}
    immutable(OUT / "validation/seed_selection.json", {"time": now(), "selected": selected,
              "skipped": skipped, "all_arm_allocations_prechecked": True,
              "accuracy_accessed": False, "test_predictions": False})
    immutable(OUT / "validation/block_c.json", {"checks": c_checks, "all_pass": True})
    immutable(OUT / "registration.json", registration)
    receipt = {"time": now(), "sha256": digest(OUT / "registration.json"),
               "task_sha256": TASK_SHA256, "all_blocks_registered_before_any_FL": True}
    immutable(OUT / "registration_receipt.json", receipt)
    jobs = []
    for block in ("A", "B", "C"):
        for (alpha, seed), (reports, split, details) in prepared.items():
            for arm in ARMS:
                if BLOCKS[arm] != block:
                    continue
                for agg in ("eps", "datasize"):
                    jobs.append(build_job(block, alpha, seed, arm, agg, reports, split, details[arm], registration))
    for j in jobs:
        pair = next(k for k in jobs if all(k[x] == j[x] for x in ("alpha", "seed", "arm")) and k["aggregation"] != j["aggregation"])
        assert all(j[k] == pair[k] for k in ("allocation_sha256", "payment_sha256"))
    immutable(OUT / "queue_agg2x2.json", jobs)
    immutable(OUT / "validation/aggregation_pairs.json", {"pairs": 16, "all_allocation_and_payment_hashes_equal": True})
    atomic_json(OUT / "jobs_agg2x2.json", [{k: (None if k == "done" else v) for k, v in j.items()
                if k in ("block", "alpha", "seed", "arm", "aggregation", "done", "allocation_spec_sha256")} for j in jobs])
    extra_sources = sorted((WORK / "a7flow").glob("*.py")) + sorted((WORK / "next").glob("*.py"))
    extra_sources += sorted((WORK / "a7").glob("*.py"))
    source_all = source_identity()
    source_all.update({str(p.relative_to(ROOT)): digest(p) for p in extra_sources})
    inputs = dict(input_identity())
    inputs.update({str(p.relative_to(ROOT)): digest(p) for p in EXPECTED})
    immutable(OUT / "launch_manifest.json", {"created": now(), "created_unix": datetime.now().timestamp(),
              "source_sha256": source_identity(), "input_sha256": input_identity(),
              "all_source_sha256": source_all, "study_input_sha256": inputs,
              "protocol_sha256": digest(CONFIG), "amendment_sha256": digest(AMENDMENT),
              "task_sha256": TASK_SHA256, "registration_sha256": receipt["sha256"],
              "queue_sha256": digest(OUT / "queue_agg2x2.json"), "blocks": {"A": 16, "B": 8, "C": 8},
              "endpoint_access": "Claude only after complete blocks; Codex reads progress and hashes only"})
    from p2_state import check_identity
    check_identity()
    immutable(OUT / "PREPARED.json", {"time": now(), "total": 32, "registration_sha256": receipt["sha256"],
              "launch_manifest_sha256": digest(OUT / "launch_manifest.json"), "source_parity": parity})
    immutable(OUT / "run_not_used.json", [])
    for name in ("registration.json", "registration_receipt.json", "launch_manifest.json", "queue_agg2x2.json"):
        (OUT / name).chmod(0o444)
    atomic_json(PAPER / "registration.json", registration)
    atomic_json(PAPER / "registration_receipt.json", receipt)
    print(json.dumps({"prepared": len(jobs), "registration": registration, "receipt": receipt,
                      "skipped": skipped, "block_c_all_pass": True}), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()

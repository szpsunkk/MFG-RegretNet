"""Replay a frozen RQ4 allocation with the original split, model and DP-SGD runner.

Only campaign scheduling and machine-local paths are replaced. Inputs are
verified against the original byte-preserved specification before training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from runtime import HERE, activate, records, sha, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--job", help="run_0000 etc., from --list")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="contains mnist/ and cifar10/")
    parser.add_argument("--out", type=Path, default=HERE / "outputs/fl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check-only", action="store_true", help="validate full dataset split/reports, without training")
    parser.add_argument("--max-rounds", type=int, default=100, help="<100 is a marked partial smoke run")
    args = parser.parse_args()
    if args.list:
        for j in json.loads((HERE / "evidence/rq4_index.json").read_text()):
            print(j["key"], j["id"], json.dumps(j["cell"], sort_keys=True))
        return
    if not args.job or not 1 <= args.max_rounds <= 100:
        parser.error("--job is required and --max-rounds must be in [1,100]")
    verify_release()
    jobs = records(HERE / "outputs/records")
    job = next((j for j in jobs if j["key"] == args.job), None)
    if job is None:
        parser.error("unknown job key")
    spec = json.loads(Path(job["allocation_spec"]).read_text())
    original = json.loads(Path(job["done"]).read_text())
    activate()
    import common as C
    from tdsc_dpsgd_runner import DPSGDConfig, DPSGDClientRequest, EXPANDED_RDP_ORDERS
    from tdsc_rdp_cache import exact_rdp_cache
    # Portability only: keep every data transformation in the frozen common.py.
    load_mnist, load_cifar = C.load_mnist, C.load_cifar10
    C.load_mnist = lambda **kw: load_mnist(root=str(args.data_root / "mnist"), download=False)
    C.load_cifar10 = lambda **kw: load_cifar(root=str(args.data_root / "cifar10"), download=False)
    out = args.out.resolve() / args.job
    C.OUT = out / "split_manifest"
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset, alpha, seed = spec["dataset"], spec["alpha"], spec["seed"]
    clients, validation, split, factory = C.data_for_cell(dataset, alpha, seed, 64, "test", None)
    sizes = [len(c) for c in clients]
    reports = C.market_reports(100, sizes, seed)
    def array_hash(x):
        return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()
    if split["sha256"] != spec["split_sha256"] or array_hash(reports.numpy()) != spec["reports_sha256"]:
        raise ValueError("dataset split or market reports differ from the frozen protocol")
    epsilon, payment = np.asarray(spec["executed"]), np.asarray(spec["payment"])
    if (array_hash(epsilon) != spec["allocation_sha256"] or array_hash(payment) != spec["payment_sha256"]
            or epsilon.shape != (100, 10) or payment.shape != (100, 10)):
        raise ValueError("frozen allocation/payment content mismatch")
    if not spec.get("confirmation_gate_sha256"):
        raise ValueError("not a frozen confirmation job")
    print(json.dumps({"job": args.job, "split_verified": True, "reports_verified": True,
                      "allocation_verified": True, "client_sizes": sizes}), flush=True)
    if args.check_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "STARTED.json").exists():
        raise FileExistsError("choose a fresh --out directory; prior runs are preserved")
    config = DPSGDConfig(noise_multiplier=1., max_grad_norm=1., learning_rate=.1 if dataset == "MNIST" else .05,
                         local_epochs=1, batch_size=64, delta=1e-5, rdp_orders=EXPANDED_RDP_ORDERS,
                         dataset_name=f"{dataset}-a{alpha}-repair-v1")
    if json.loads(json.dumps(asdict(config))) != original["meta"]["config"]:
        raise ValueError("DP-SGD configuration differs from original run")
    with C.isolated_rng(seed):
        initial = factory()
    runner = C.RepairedRunner(initial, 10, config, seed=seed, device=args.device)
    meta = {"release": "2026-10-25", "job": args.job, "source_result_sha256": job["result_sha256"],
            "allocation_spec_sha256": job["allocation_spec_sha256"], "config": asdict(config),
            "rounds": args.max_rounds, "partial_smoke": args.max_rounds != 100, "device": args.device,
            "torch": torch.__version__, "numpy": np.__version__, "split_sha256": split["sha256"]}
    C.atomic_json(out / "STARTED.json", meta)
    measurements = [{"round": 0, **C.evaluate(runner.model, validation, args.device)}]
    events = []
    began = time.monotonic()
    with exact_rdp_cache(maxsize=262144):
        for r in range(args.max_rounds):
            requests = []
            for i, e in enumerate(epsilon[r]):
                if e <= 0:
                    continue
                runner.preview_participation(client_id=i, dataset_size=sizes[i], round_index=r,
                                             allocated_event_epsilon=float(e))
                requests.append(DPSGDClientRequest(client_id=i, dataset=clients[i], allocated_event_epsilon=float(e),
                    aggregation_weight=float(e) if spec["aggregation"] == "eps" else float(sizes[i]),
                    payment=float(payment[r, i]), max_acceptable_event_epsilon=float(reports[r, i, 1]),
                    max_epsilon_add_remove=original["meta"].get("max_cumulative_epsilon_add_remove")))
            outcome = runner.run_round(r, requests)
            if sorted(outcome.released_client_ids) != np.flatnonzero(epsilon[r] > 0).tolist():
                raise RuntimeError("actual DP release differs from the frozen allocation")
            events.append({"round": r + 1, "outcome": asdict(outcome)})
            if (r + 1) % 5 == 0 or r >= 95 or r + 1 == args.max_rounds:
                measurements.append({"round": r + 1, **C.evaluate(runner.model, validation, args.device)})
            C.atomic_json(out / "progress.json", {"round": r + 1, "total_rounds": args.max_rounds,
                                                  "seconds": time.monotonic() - began})
            print(f"{args.job}: round {r + 1}/{args.max_rounds}", flush=True)
    privacy = runner.privacy_snapshot(include_prv_crosscheck=False)
    if not privacy["all_accountant_parity_pass"]:
        raise AssertionError("independent accountant parity failed")
    C.atomic_json(out / "events.json", events)
    C.atomic_json(out / "privacy.json", privacy)
    final = [m["accuracy"] for m in measurements if 96 <= m["round"] <= 100]
    C.atomic_json(out / "result.json", {"meta": meta, "measurements": measurements,
        "final_accuracy": float(np.mean(final)) if len(final) == 5 else None,
        "privacy": C.privacy_summary(privacy), "events_sha256": sha(out / "events.json"),
        "privacy_sha256": sha(out / "privacy.json"), "seconds": time.monotonic() - began})
    print("Saved:", out)


if __name__ == "__main__":
    main()

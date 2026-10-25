"""Pre-execution FL identity. Economic payments deliberately do not enter it."""
from dataclasses import asdict
import hashlib
import json
from functools import lru_cache

import numpy as np
import torch
from common import WORK, FROZEN, RepairedRunner, isolated_rng, Net, CIFAR10Net, digest
from tdsc_dpsgd_runner import DPSGDConfig, EXPANDED_RDP_ORDERS, tensor_state_sha256, _event_seed
from tdsc_rdp_cache import exact_rdp_cache


def hash_payload(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=128)
def identity_runner(dataset, alpha, seed):
    with isolated_rng(seed):
        model = (Net if dataset == "MNIST" else CIFAR10Net)()
    cfg = DPSGDConfig(noise_multiplier=1., max_grad_norm=1.,
                      learning_rate=.1 if dataset == "MNIST" else .05, local_epochs=1,
                      batch_size=64, delta=1e-5, rdp_orders=EXPANDED_RDP_ORDERS,
                      dataset_name=f"{dataset}-a{alpha}-repair-v1")
    return RepairedRunner(model, 10, cfg, seed=seed, device="cpu")


def fingerprint(spec, split, reports):
    runner = identity_runner(spec["dataset"], spec["alpha"], spec["seed"])
    executed = np.asarray(spec["executed"], dtype=float)
    rows = []
    # No cumulative cap is enabled in p2-v1. Thus the exact release decision
    # depends only on the frozen per-event contract. Refuse unsupported caps.
    if spec.get("max_cumulative_epsilon") is not None:
        raise ValueError("cumulative-capped runs require a stateful fingerprint")
    with exact_rdp_cache(maxsize=262144):
        for r, amounts in enumerate(executed):
            row = []
            for i, eps in enumerate(amounts):
                base = {"client": i, "accepted": bool(eps > 0), "executed_epsilon": float(eps),
                        "aggregation_weight": 0., "noise_multiplier": None, "sample_rate": None,
                        "optimizer_steps": 0, "rng_seeds": None}
                if eps > 0:
                    event, calibration, _ = runner._cached_allocated_event_profile(
                        dataset_size=split["client_sizes"][i], allocated_event_epsilon=float(eps), event_delta=1e-5)
                    if calibration["realized_isolated_event_epsilon"] > float(reports[r, i, 1]) + runner.config.event_epsilon_tolerance:
                        raise ValueError("allocation would be refused by the actual executor")
                    base.update(asdict(event))
                    base["aggregation_weight"] = float(eps) if spec["aggregation"] == "eps" else float(split["client_sizes"][i])
                    base["rng_seeds"] = {stream: _event_seed(dataset_name=runner.config.dataset_name,
                        base_seed=spec["seed"], target_epsilon=runner.config.target_epsilon,
                        round_index=r, client_id=i, stream=stream) for stream in
                        ("poisson-sampler", "dp-gaussian-noise", "data-augmentation")}
                row.append(base)
            rows.append(row)
    sources = sorted(FROZEN.rglob("*.py")) + [WORK / "exp/common.py", WORK / "exp/run_cell.py"]
    payload = {"schema": "p2-a3-fl-input-v1", "split_sha256": spec["split_sha256"],
               "seed": spec["seed"], "initial_model_sha256": tensor_state_sha256(runner.model.state_dict()),
               "config": asdict(runner.config), "executor_sha256": {str(p.relative_to(WORK)): digest(p) for p in sources},
               "evaluation": {"stage": spec["stage"], "smoke_size": spec["smoke_size"],
                              "eval_every": 1 if spec["smoke_size"] else 5, "last_five_each_round": True},
               "rounds": rows}
    return {"sha256": hash_payload(payload), "payload": payload}

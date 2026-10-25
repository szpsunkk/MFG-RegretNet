#!/usr/bin/env python3
"""Offline, nonprivate development-only update dispersion through the real runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from common import FROZEN, RepairedRunner, atomic_json, data_for_cell, digest, isolated_rng
from p2_state import check_identity, CONFIG, AMENDMENT
from p2_reference import dp_noise_energy, kappa_from_updates
from tdsc_dpsgd_runner import DPSGDClientRequest, DPSGDConfig, EXPANDED_RDP_ORDERS
from tdsc_rdp_cache import exact_rdp_cache


class CaptureRunner(RepairedRunner):
    def _train_one(self, **kwargs):
        delta, row = super()._train_one(**kwargs)
        names = [name for name, p in self.model.named_parameters() if p.is_floating_point()]
        self.captured.append(torch.cat([delta[name].flatten().detach().cpu() for name in names]).double().numpy())
        return delta, row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("MNIST", "CIFAR10"), required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, choices=(14, 15), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    check_identity()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "kappa.json").exists():
        return
    clients, _, split, factory = data_for_cell(args.dataset, args.alpha, args.seed, stage="dev")
    cfg = DPSGDConfig(noise_multiplier=1e-6, max_grad_norm=1., learning_rate=.1 if args.dataset == "MNIST" else .05,
                      local_epochs=1, batch_size=64, delta=1e-5, rdp_orders=EXPANDED_RDP_ORDERS,
                      dataset_name=f"{args.dataset}-a{args.alpha}-repair-v1")
    with isolated_rng(args.seed):
        model = factory()
    runner = CaptureRunner(model, 10, cfg, seed=args.seed, device=args.device)
    n_params = sum(p.numel() for p in runner.model.parameters() if p.is_floating_point())
    energy, tiny, events = [], [], []
    for i, client in enumerate(clients):
        preview = runner.preview_participation(client_id=i, dataset_size=len(client), round_index=0,
                                              allocated_event_epsilon=1.)
        kwargs = dict(sample_rate=preview["sample_rate"], optimizer_steps=preview["optimizer_steps"],
                      dataset_size=len(client), lr=cfg.learning_rate, clip=1., n_params=n_params)
        energy.append(dp_noise_energy(preview["noise_multiplier"], **kwargs))
        tiny.append(dp_noise_energy(1e-6, **kwargs))
        events.append({"client": i, "n": len(client), "sigma": preview["noise_multiplier"],
                       "q": preview["sample_rate"], "T": preview["optimizer_steps"],
                       "dp_noise_energy": energy[-1], "warmup_noise_energy": tiny[-1]})
    if not all(np.isfinite(e) and e > 0 for e in energy) or max(np.array(tiny) / energy) >= 1e-6:
        raise AssertionError("invalid noise energy or tiny-noise approximation")
    began = time.monotonic()
    measurements, histories = {}, []
    for r in range(6):
        check_identity()
        runner.captured = []
        requests = [DPSGDClientRequest(client_id=i, dataset=client, noise_multiplier=1e-6,
                                      aggregation_weight=float(len(client)), payment=0.)
                    for i, client in enumerate(clients)]
        result = runner.run_round(r, requests)
        if result.release_count != 10:
            raise AssertionError("nonprivate measurement did not collect every client")
        if r in (0, 5):
            u = np.asarray(runner.captured)
            measured = kappa_from_updates(u, energy)
            if not all(np.isfinite(x) and x > 0 for x in measured.values()):
                raise AssertionError("nonpositive or nonfinite kappa measurement")
            measured["update_sha256"] = hashlib.sha256(u.tobytes()).hexdigest()
            measured["update_norms"] = np.linalg.norm(u, axis=1).tolist()
            measurements["initial" if r == 0 else "warm"] = measured
        histories.append(asdict(result))
        atomic_json(args.out / "progress.json", {"round": r + 1, "total_rounds": 6,
                                                 "seconds": time.monotonic() - began})
    runner.captured = []
    atomic_json(args.out / "nonprivate_events.json", histories)
    atomic_json(args.out / "kappa.json", {"dataset": args.dataset, "alpha": args.alpha, "seed": args.seed,
                "n_params": n_params, "clients": events, "measurements": measurements,
                "split_sha256": split["sha256"], "config": asdict(cfg),
                "events_sha256": digest(args.out / "nonprivate_events.json"),
                "runner_sha256": digest(FROZEN / "tdsc_dpsgd_runner.py"),
                "protocol_sha256": digest(CONFIG), "amendment_sha256": digest(AMENDMENT),
                "source_scope": "offline nonprivate development diagnostic; no image accuracy evaluated",
                "seconds": time.monotonic() - began})


if __name__ == "__main__":
    with exact_rdp_cache(maxsize=262144):
        main()

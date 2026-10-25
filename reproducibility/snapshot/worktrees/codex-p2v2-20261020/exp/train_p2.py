#!/usr/bin/env python3
"""Fair three-family P2 training; no image accuracy used for training choices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import OUT, atomic_json, digest
from contracts import settlement, utility
from p2_core import FAMILIES, P2Head, raw, training_term, search, deviation_utility
from p2_reference import p2_oracle, sample_log_uniform
from p2_state import check_identity, read, immutable_json, CONFIG, AMENDMENT
from train_mechanism import sample_profiles


def make_bank(count=4096):
    check_identity()
    out = OUT / "oracle_bank.pt"
    if out.exists():
        if digest(out) != read(OUT / "oracle_bank.json")["sha256"]:
            raise RuntimeError("oracle warm bank changed")
        return
    bounds = read(OUT / "stage0_decision.json")["training_kappa_range"]
    generator = torch.Generator().manual_seed(42 + 1000)
    reports, budgets = sample_profiles(count, generator, "cpu")
    kappas = sample_log_uniform(count, *bounds, generator)
    erows, prows = [], []
    began = time.monotonic()
    for idx, (rows, budget, kappa) in enumerate(zip(reports.double().numpy(), budgets[:, 0], kappas[:, 0])):
        value, cap, size = rows.T
        eps, _, _ = p2_oracle(cap, value, size / size.sum(), float(budget), float(kappa))
        price = np.interp(float(budget), [5, 10, 15, 20, 25, 110], [.44, .63, .77, .89, 1., 1.])
        pay = np.maximum(value, price) * eps
        pay *= min(1., float(budget) / max(float(pay.sum()), 1e-12))
        erows.append(eps)
        prows.append(pay)
        if (idx + 1) % 100 == 0:
            check_identity()
            atomic_json(OUT / "bank/progress.json", {"profiles": idx + 1, "total": count,
                                                    "seconds": time.monotonic() - began})
    value = {"reports": reports, "budgets": budgets, "kappas": kappas,
             "epsilon": torch.tensor(np.asarray(erows), dtype=torch.float32),
             "payment": torch.tensor(np.asarray(prows), dtype=torch.float32)}
    tmp = out.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(out)
    immutable_json(OUT / "oracle_bank.json", {"sha256": digest(out), "profiles": count, "seed": 42,
                   "derived_stream": 1042, "kappa_range": bounds,
                   "stage0_sha256": digest(OUT / "stage0_decision.json"),
                   "scope": "supervised warm targets; budget scaling can reduce payment below cost"})


def train(kind, folder, warm_steps=6000, steps=6000):
    torch.set_num_threads(1)
    check_identity()
    if (folder / "training.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    bounds = read(OUT / "stage0_decision.json")["training_kappa_range"]
    bank = torch.load(OUT / "oracle_bank.pt", map_location="cpu", weights_only=False)
    if digest(OUT / "oracle_bank.pt") != read(OUT / "oracle_bank.json")["sha256"]:
        raise RuntimeError("warm bank hash mismatch")
    torch.manual_seed(42)
    g = torch.Generator().manual_seed(42 + 2000)
    model = P2Head(kind, bounds)
    if kind.startswith("mfg"):
        with torch.no_grad():
            [m for m in model.network.payment_head.modules() if isinstance(m, torch.nn.Linear)][-1].bias.fill_(-4.)
    config = {"kind": kind, "seed": 42, "kappa_range": bounds, "warm_steps": warm_steps, "steps": steps,
              "batch_size": 64, "strong_final_steps": 2000, "normal_learning_rate": 1e-4,
              "final_learning_rate": 3e-5, "dual_update_rate": 5., "oracle_bank_sha256": digest(OUT / "oracle_bank.pt"),
              "protocol_sha256": digest(CONFIG), "amendment_sha256": digest(AMENDMENT),
              "source_sha256": read(OUT / "launch_manifest.json")["source_sha256"],
              "search_domain": "v [0,1]; cap [0,true_cap]; size [10,true_size]"}
    start, log = time.monotonic(), []
    def save(name):
        torch.save({"kind": kind, "config": config, "state_dict": model.state_dict()}, folder / name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(warm_steps):
        idx = torch.randint(len(bank["reports"]), (64,), generator=g)
        reports, budget, kappa = (bank[name][idx] for name in ("reports", "budgets", "kappas"))
        e, p = raw(model, reports, budget, kappa)
        loss = ((e - bank["epsilon"][idx]) / reports[..., 1]).square().mean() + ((p - bank["payment"][idx]) / 2).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 300 == 0:
            check_identity()
            atomic_json(folder / "progress.json", {"phase": "warm", "step": step + 1, "loss": float(loss),
                                                  "seconds": time.monotonic() - start})
    save("warm.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dual = torch.full((1, 10), 2.)
    for step in range(steps):
        reports, budget = sample_profiles(64, g, "cpu")
        kappa = sample_log_uniform(64, *bounds, g)
        strong = step >= max(0, steps - 2000)
        for group in optimizer.param_groups:
            group["lr"] = 3e-5 if strong else 1e-4
        deviations = search(model, reports, budget, kappa, strong=strong)
        e, p = raw(model, reports, budget, kappa)
        actual, _, _ = settlement(reports, e, p)
        truthful = utility(reports, e, p)
        regret = (deviation_utility(model, reports, budget, kappa, deviations) - truthful).clamp_min(0) / (budget / 10)
        ir = (reports[..., 0] * e - p).clamp_min(0) / (budget / 10)
        objective = training_term(actual, reports, kappa).mean()
        loss = objective + (dual * (regret - .02)).mean() + 5 * ir.mean()
        loss += 5 * (regret - .02).clamp_min(0).square().mean() + 10 * ir.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("nonfinite P2 training loss/gradient")
        optimizer.step()
        if (step + 1) % 25 == 0:
            with torch.no_grad():
                dual.add_(5 * (regret - .02).mean(0)).clamp_(0, 100)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            check_identity()
            row = {"phase": "constrained", "step": step + 1, "p2_objective": float(objective),
                   "loss": float(loss), "regret": float(regret.mean()), "ir": float(ir.mean()),
                   "mean_executed_epsilon": float(actual.mean()),
                   "empty_market_fraction": float((actual.sum(-1) == 0).float().mean()),
                   "strong_search": strong, "seconds": time.monotonic() - start}
            log.append(row)
            atomic_json(folder / "progress.json", row)
            print(json.dumps(row), flush=True)
    save("model.pt")
    atomic_json(folder / "training.json", {"config": config, "log": log,
                                          "checkpoint_sha256": digest(folder / "model.pt")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", action="store_true")
    ap.add_argument("--kind", choices=FAMILIES)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.bank:
        make_bank()
    elif args.kind and args.out:
        train(args.kind, args.out)
    else:
        ap.error("use --bank, or --kind and --out")

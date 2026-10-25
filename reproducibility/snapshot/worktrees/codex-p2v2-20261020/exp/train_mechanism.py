#!/usr/bin/env python3
"""Matched-budget neural allocation/payment training with executed-utility attacks.

Two predefined objectives: quantity, or quantity with exposure-share balance.
Both are public procurement proxies, not image cross-entropy. Image validation
is performed separately and never uses the official image test set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import FROZEN, WORK, atomic_json, digest, isolated_rng
from mechanisms import make_model
from contracts import utility as executed_utility, settlement


def sample_profiles(count, generator, device):
    reports = torch.rand(count, 10, 3, generator=generator, device=device)
    reports[..., 1] = .1 + 4.9 * reports[..., 1]
    # Sizes are public covariates. Include heterogeneous populations, using no image labels.
    reports[..., 2] = 10 + 490 * reports[..., 2].square()
    budget = 1 + 109 * torch.rand(count, 1, generator=generator, device=device)
    tight = torch.rand(count, 1, generator=generator, device=device) < .5
    budget = torch.where(tight, 1 + 24 * torch.rand(count, 1, generator=generator, device=device), budget)
    return reports, budget


def raw_contract(model, reports, budget):
    allocation, payment = model((reports, budget))
    return allocation.squeeze(-1) * reports[..., 1], payment


def settled_utility(exposure, payment, truthful):
    return executed_utility(truthful, exposure, payment)


def own_deviation_utility(model, truthful, budget, misreports):
    """Every client deviates unilaterally; all opponents and verified sizes stay fixed."""
    batch, agents, features = truthful.shape
    profiles = truthful[:, None].expand(batch, agents, agents, features).clone()
    arange = torch.arange(agents, device=truthful.device)
    profiles[:, arange, arange] = misreports
    eps, pay = raw_contract(model, profiles.reshape(-1, agents, features), budget.repeat_interleave(agents, 0))
    eps = eps.view(batch, agents, agents)[:, arange, arange]
    pay = pay.view(batch, agents, agents)[:, arange, arange]
    return settled_utility(eps, pay, truthful)


def search(model, truthful, budget, strong=False):
    batch, agents, _ = truthful.shape
    vg = torch.linspace(0, 1, 5 if strong else 3, device=truthful.device)
    eg = torch.tensor([0., 1e-6, .01, .05, .1, .2, .4, .6, .8, 1.], device=truthful.device) if strong else torch.tensor([0., 1e-4, .1, .5, 1.], device=truthful.device)
    grid = torch.cartesian_prod(vg, eg)
    with torch.no_grad():
        best = truthful.clone()
        best_u = own_deviation_utility(model, truthful, budget, best)
        for part in grid.split(10):
            count = len(part)
            candidates = truthful[:, None].expand(batch, count, agents, 3).clone()
            candidates[..., 0] = part[None, :, None, 0]
            # Full [0,5] report domain, including overstatement; true capacity is checked in utility.
            candidates[..., 1] = 5 * part[None, :, None, 1]
            utilities = own_deviation_utility(model, truthful.repeat_interleave(count, 0),
                                              budget.repeat_interleave(count, 0), candidates.reshape(-1, agents, 3))
            values, index = utilities.view(batch, count, agents).max(1)
            chosen = candidates[torch.arange(batch, device=truthful.device)[:, None], index,
                                torch.arange(agents, device=truthful.device)[None]]
            best = torch.where((values > best_u)[..., None], chosen, best)
            best_u = torch.maximum(values, best_u)
    candidate = best.detach().clone().requires_grad_(True)
    for _ in range(25 if strong else 8):
        utility = own_deviation_utility(model, truthful, budget, candidate)
        gradient, = torch.autograd.grad(utility.sum(), candidate)
        with torch.no_grad():
            improved = utility > best_u
            best = torch.where(improved[..., None], candidate, best)
            best_u = torch.maximum(utility, best_u)
            candidate[..., 0].add_(.03 * gradient[..., 0]).clamp_(0, 1)
            candidate[..., 1].add_(.03 * gradient[..., 1]).clamp_(0, 5)
            candidate[..., 2].copy_(truthful[..., 2])
    with torch.no_grad():
        utility = own_deviation_utility(model, truthful, budget, candidate)
        best = torch.where((utility > best_u)[..., None], candidate, best)
    return best.detach()


def procurement_objective(exposure, truthful, balance):
    total = exposure.sum(1, keepdim=True)
    shares = exposure / total.clamp_min(1e-8)
    target = truthful[..., 2] / truthful[..., 2].sum(1, keepdim=True)
    quantity = total[:, 0] / truthful[..., 1].sum(1).clamp_min(1e-8)
    mismatch = (shares - target).square().sum(1)
    # Zero-release procurement has zero gain and its finite share-mismatch cost.
    return -quantity.mean() + float(balance) * mismatch.mean()


def economic_evaluation(model, device, amounts=(10., 25., 50.)):
    model.eval()
    old_deter = model.deter_train
    model.deter_train = False
    generator = torch.Generator(device=device).manual_seed(9102026)
    rows = {}
    for amount in amounts:
        values = []
        for _ in range(8):
            truthful, _ = sample_profiles(64, generator, device)
            budget = torch.full((64, 1), amount, device=device)
            deviation = search(model, truthful, budget, strong=True)
            with torch.no_grad():
                eps, pay = raw_contract(model, truthful, budget)
                raw_u = pay - truthful[..., 0] * eps
                utility = settled_utility(eps, pay, truthful)
                dev_u = own_deviation_utility(model, truthful, budget, deviation)
                gain = (dev_u - utility).clamp_min(0)
                actual, _, _ = settlement(truthful, eps, pay)
                values.append([float(gain.mean()), float(gain.max()), float(actual.sum(1).mean()),
                               float((raw_u < -1e-9).float().mean()), float((pay.sum(1) - amount).clamp_min(0).max())])
        vals = np.asarray(values)
        rows[str(int(amount))] = {"mean_found_regret": float(vals[:, 0].mean()),
                                  "max_found_regret": float(vals[:, 1].max()),
                                  "normalized_mean_found_regret": float(vals[:, 0].mean() / (amount / 10)),
                                  "mean_executed_epsilon": float(vals[:, 2].mean()),
                                  "decline_fraction": float(vals[:, 3].mean()),
                                  "max_budget_excess": float(vals[:, 4].max()), "profiles": 512}
    model.deter_train = old_deter
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["mfg", "mfg_nob", "regretnet_s", "dm_s"], required=True)
    ap.add_argument("--balance", type=float, choices=[0., .5], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warm-steps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "evaluation.json").exists():
        print("COMPLETE", a.out, flush=True)
        return
    torch.manual_seed(a.seed)
    generator = torch.Generator(device=a.device).manual_seed(a.seed + 1000)
    model = make_model(a.kind).to(a.device)
    if a.kind in ("mfg", "mfg_nob"):
        with torch.no_grad():
            [m for m in model.payment_head.modules() if isinstance(m, torch.nn.Linear)][-1].bias.fill_(-4.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    began = time.monotonic()
    for step in range(a.warm_steps):
        truthful, budget = sample_profiles(64, generator, a.device)
        price = torch.tensor(np.interp(budget[:, 0].cpu().numpy(), [5, 10, 15, 20, 25, 110],
                                      [.44, .63, .77, .89, 1., 1.]), device=a.device, dtype=truthful.dtype)[:, None]
        accepted = torch.sigmoid((price - truthful[..., 0]) / .02)
        desired = accepted * truthful[..., 1]
        factor = (budget / (price * desired.sum(1, keepdim=True)).clamp_min(1e-8)).clamp(max=1)
        desired = desired * factor
        eps, pay = raw_contract(model, truthful, budget)
        loss = ((eps - desired) / 5).square().mean() + ((pay - price * desired) / 5).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dual_regret = torch.full((1, 10), 5., device=a.device)
    dual_ir = torch.full((1, 10), 5., device=a.device)
    log = []
    for step in range(a.steps):
        truthful, budget = sample_profiles(64, generator, a.device)
        deviation = search(model, truthful, budget)
        eps, pay = raw_contract(model, truthful, budget)
        utility = settled_utility(eps, pay, truthful)
        gain = (own_deviation_utility(model, truthful, budget, deviation) - utility).clamp_min(0)
        raw_u = pay - truthful[..., 0] * eps
        unit = budget / 10
        regret = gain / unit
        ir = (-raw_u).clamp_min(0) / unit
        # Smooth participation only in the training surrogate; evaluation uses exact gates.
        participation = torch.sigmoid(raw_u / (.02 * unit))
        objective = procurement_objective(eps * participation, truthful, a.balance)
        loss = objective + (dual_regret * regret).mean() + (dual_ir * ir).mean()
        loss += 10 * (regret.square().mean() + ir.square().mean())
        if a.kind == "dm_s":
            allocation = eps / truthful[..., 1]
            loss += .02 * (allocation * (1 - allocation)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("nonfinite training gradient")
        optimizer.step()
        if (step + 1) % 25 == 0:
            with torch.no_grad():
                dual_regret.add_(2 * regret.mean(0)).clamp_(max=500)
                dual_ir.add_(2 * ir.mean(0)).clamp_(max=500)
        if (step + 1) % 100 == 0 or step + 1 == a.steps:
            row = {"step": step + 1, "objective": float(objective), "regret": float(gain.mean()),
                   "normalized_regret": float(regret.mean()), "ir": float(ir.mean()),
                   "purchased_epsilon": float(eps.sum(1).mean()), "seconds": time.monotonic() - began}
            log.append(row)
            print(json.dumps(row), flush=True)
            atomic_json(a.out / "progress.json", row)
    config = {"kind": a.kind, "balance": a.balance, "seed": a.seed, "steps": a.steps,
              "warm_steps": a.warm_steps, "budget_range": [5, 110], "tight_budget_mixture_probability": .5,
              "warm_target": "posted-price-public-budget-table", "batch_size": 64,
              "objective": "normalized total procurement plus exposure/data-share mismatch",
              "report_domain": {"v": [0, 1], "epsilon": [0, 5], "size": "verified public, fixed under deviation"},
              "utility": "actual accepted payment minus true cost; rejected contract settles zero",
              "critic": False, "source_sha256": {str(p.relative_to(WORK)): digest(p) for p in
                                                   [Path(__file__).resolve(), WORK / "exp/mechanisms.py", FROZEN / "models_scaled.py"]}}
    torch.save({"kind": a.kind, "seed": a.seed, "config": config,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, a.out / "model.pt")
    atomic_json(a.out / "training.json", {"config": config, "log": log})
    evaluation = economic_evaluation(model, a.device)
    atomic_json(a.out / "evaluation.json", {"checkpoint_sha256": digest(a.out / "model.pt"), "budgets": evaluation,
                "scope": "strong deviation search lower bounds; not certified upper bounds"})
    print(json.dumps({"evaluation": evaluation, "seconds": time.monotonic() - began}), flush=True)


if __name__ == "__main__":
    main()

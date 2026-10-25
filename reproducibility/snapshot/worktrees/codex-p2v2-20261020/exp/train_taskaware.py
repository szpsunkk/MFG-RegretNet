#!/usr/bin/env python3
"""Train original neural heads against an explicit public image task proxy."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import OUT, WORK, atomic_json, digest
from contracts import balanced_target, settlement, MIN_EVENT_EPSILON
from mechanisms import make_model
from task_proxy import TaskProxy
from train_mechanism import sample_profiles, raw_contract, settled_utility, own_deviation_utility, search, economic_evaluation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["mfg", "mfg_nob", "regretnet_s"], required=True)
    ap.add_argument("--ceiling", type=float, default=1.)
    ap.add_argument("--task-weight", type=float, default=1.)
    ap.add_argument("--task-scale", type=float, default=100.)
    ap.add_argument("--procurement-floor-weight", type=float, default=10.)
    ap.add_argument("--warm-steps", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--strong-final-steps", type=int, default=2000)
    ap.add_argument("--dual-update-rate", type=float, default=5.)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-strong-evaluation", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "evaluation.json").exists():
        print("COMPLETE", args.out, flush=True)
        return
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 20261017)
    proxy = TaskProxy(OUT / "public_bank.pt", OUT / "noise_table.json", args.device)
    model = make_model(args.kind).to(args.device)
    if args.kind.startswith("mfg"):
        with torch.no_grad():
            [x for x in model.payment_head.modules() if isinstance(x, torch.nn.Linear)][-1].bias.fill_(-4.)
    config = {**vars(args), "out": str(args.out), "proxy_sha256": digest(OUT / "public_bank.pt"),
              "noise_table_sha256": digest(OUT / "noise_table.json"), "minimum_event_epsilon": MIN_EVENT_EPSILON,
              "report_domain": {"v": [0, 1], "cap": [0, 5], "size": "verified public"},
              "execution": "true-cap/IR/minimum-release gates; no cumulative cap; exact calibration in final evaluation",
              "source_sha256": {str(p.relative_to(WORK)): digest(p) for p in sorted((WORK / "exp").glob("*.py"))}}

    def save(name):
        torch.save({"kind": args.kind, "seed": args.seed, "config": config,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, args.out / name)

    def batch(count):
        data = proxy.sample(count, generator)
        reports, budget = sample_profiles(count, generator, args.device)
        reports[..., 2] = (500 * data["sizes"] / data["sizes"].max(-1, keepdim=True).values).clamp_min(10)
        return data, reports, budget

    start = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(args.warm_steps):
        _, reports, budget = batch(32)
        target_e, target_p = balanced_target(reports, budget, args.ceiling)
        e, p = raw_contract(model, reports, budget)
        loss = ((e - target_e) / reports[..., 1]).square().mean() + ((p - target_p) / 2).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 300 == 0 or step + 1 == args.warm_steps:
            row = {"phase": "warm", "step": step + 1, "loss": float(loss), "elapsed_seconds": time.monotonic() - start}
            atomic_json(args.out / "progress.json", row)
            print(json.dumps(row), flush=True)
    save("warm.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dual = torch.full((1, 10), 2., device=args.device)
    log = []
    for step in range(args.steps):
        data, reports, budget = batch(16)
        strong = step >= max(0, args.steps - args.strong_final_steps)
        for group in optimizer.param_groups:
            group["lr"] = 3e-5 if strong else 1e-4
        deviations = search(model, reports, budget, strong=strong)
        e, p = raw_contract(model, reports, budget)
        actual, _, _ = settlement(reports, e, p)
        truthful_u = settled_utility(e, p, reports)
        regret = (own_deviation_utility(model, reports, budget, deviations) - truthful_u).clamp_min(0) / (budget / 10)
        ir = (reports[..., 0] * e - p).clamp_min(0) / (budget / 10)
        noise = torch.randn(data["base"].shape, device=args.device, generator=generator)
        target_e, target_p = balanced_target(reports, budget, args.ceiling)
        target_e, _, _ = settlement(reports, target_e, target_p)
        target_loss = proxy.loss(target_e, data, noise).detach()
        task_loss = proxy.loss(actual, data, noise)
        # The task proxy alone is nearly invariant to a common exposure scaling
        # once noise is small. Require useful procurement rather than rewarding
        # regret reduction obtained only by buying almost nothing. The public
        # reference is the same for task-trained and no-task control candidates.
        reference = ((e - target_e) / reports[..., 1]).square().mean() + ((p - target_p) / 2).square().mean()
        task_improvement = (task_loss - target_loss) / target_loss.clamp_min(.1)
        procurement_ratio = actual.sum(-1) / target_e.sum(-1).clamp_min(.01)
        shortfall = (1 - procurement_ratio).clamp_min(0).square().mean()
        loss = (.1 * reference + args.task_weight * args.task_scale * task_improvement
                + args.procurement_floor_weight * shortfall)
        loss += (dual * (regret - .02)).mean() + 5 * ir.mean()
        loss += 5 * (regret - .02).clamp_min(0).square().mean() + 10 * ir.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("non-finite task-training gradient")
        optimizer.step()
        if (step + 1) % 25 == 0:
            with torch.no_grad():
                dual.add_(args.dual_update_rate * (regret - .02).mean(0)).clamp_(0, 100)
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {"phase": "task", "step": step + 1, "loss": float(loss), "public_proxy_ce": float(task_loss),
                   "teacher_proxy_ce": float(target_loss), "normalized_search_regret": float(regret.mean()),
                   "ir": float(ir.mean()), "mean_executed_epsilon": float(actual.mean()),
                   "mean_procurement_fraction_of_reference": float(procurement_ratio.mean()),
                   "procurement_shortfall_penalty": float(shortfall),
                   "reference_fit_loss": float(reference), "scaled_task_difference": float(args.task_scale * task_improvement),
                   "strong_search": strong, "mean_regret_multiplier": float(dual.mean()),
                   "elapsed_seconds": time.monotonic() - start}
            log.append(row)
            atomic_json(args.out / "progress.json", row)
            print(json.dumps(row), flush=True)
    save("model.pt")
    atomic_json(args.out / "training.json", {"config": config, "log": log})
    if not args.skip_strong_evaluation:
        economic = economic_evaluation(model, args.device, amounts=(10., 30., 100.))
        atomic_json(args.out / "evaluation.json", {"checkpoint_sha256": digest(args.out / "model.pt"),
                    "budgets": economic, "scope": "auction execution lower bounds; calibrated-domain assumptions validated separately"})
    print(json.dumps({"complete": str(args.out / "model.pt"), "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()

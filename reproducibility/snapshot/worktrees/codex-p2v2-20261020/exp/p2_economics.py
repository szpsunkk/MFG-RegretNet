"""Holdout executed-utility attacks, economic screening and all reference failures."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import OUT, digest, atomic_json
from contracts import settlement, utility
from p2_core import FAMILIES, REFERENCES, load_any, raw, search, deviation_utility, exact_metrics
from p2_reference import p2_oracle, j_kappa
from p2_state import read, check_identity, immutable_json
from p2_jobs import deploy_kappa
from train_mechanism import sample_profiles


def holdout():
    return sample_profiles(512, torch.Generator().manual_seed(42 + 3000), "cpu")[0]


def kappas():
    decision = read(OUT / "stage0_decision.json")
    return sorted({deploy_kappa(ds, alpha, decision) for ds in ("MNIST", "CIFAR10") for alpha in (.5, .1)})


def oracle_economics():
    path = OUT / "economics/oracle.json"
    if path.exists():
        return read(path)
    result = {}
    truth = holdout().double().numpy()
    for budget in (10., 30., 100.):
        for kappa in kappas():
            js = []
            for i, row in enumerate(truth):
                v, cap, size = row.T
                _, _, info = p2_oracle(cap, v, size / size.sum(), budget, kappa)
                js.append(info["j"])
                if (i + 1) % 64 == 0:
                    check_identity()
                    atomic_json(OUT / "economics/oracle_progress.json", {"budget": budget, "kappa": kappa, "profile": i + 1})
            if not np.isfinite(js).all():
                raise ValueError("oracle benchmark empty/infinite on economic holdout")
            result[f"{budget:g}/{kappa!r}"] = js
    immutable_json(path, result)
    return result


def own_epsilon(model, truth, budget, kappa, changed):
    b, n, features = truth.shape
    profiles = truth[:, None].expand(b, n, n, features).clone()
    ix = torch.arange(n)
    profiles[:, ix, ix] = changed
    e, p = raw(model, profiles.reshape(-1, n, features), budget.repeat_interleave(n, 0), kappa.repeat_interleave(n, 0))
    return settlement(truth, e.view(b, n, n)[:, ix, ix], p.view(b, n, n)[:, ix, ix])[0]


def evaluate_economics(arm, path):
    target = OUT / "economics" / (arm + ".json")
    if target.exists():
        if read(target)["checkpoint_sha256"] != digest(path):
            raise RuntimeError("evaluated checkpoint changed")
        return
    model = load_any(path, REFERENCES[arm][1] if arm in REFERENCES else None)
    truth = holdout()
    oracle = read(OUT / "economics/oracle.json")
    result = {}
    for amount in (10., 30., 100.):
        for kval in kappas():
            gains, no_size, eps_all, pay_all, raw_bad, excess, monotone = [], [], [], [], [], [], []
            for batch in truth.split(64):
                check_identity()
                b, k = torch.full((len(batch), 1), amount), torch.full((len(batch), 1), kval)
                deviations = search(model, batch, b, k, strong=True, sizes=True)
                limited = search(model, batch, b, k, strong=True, sizes=False)
                with torch.no_grad():
                    e, p = raw(model, batch, b, k)
                    actual, paid, _ = settlement(batch, e, p)
                    u = utility(batch, e, p)
                    gains.extend((deviation_utility(model, batch, b, k, deviations) - u).clamp_min(0).numpy())
                    no_size.extend((deviation_utility(model, batch, b, k, limited) - u).clamp_min(0).numpy())
                    eps_all.extend(actual.numpy())
                    pay_all.extend(paid.numpy())
                    raw_bad.extend((p - batch[..., 0] * e < -1e-9).numpy())
                    excess.extend((p.sum(-1) - amount).clamp_min(0).numpy())
                    rates = []
                    for coordinate in range(3):
                        changed = batch.clone()
                        if coordinate == 0:
                            changed[..., 0] = (changed[..., 0] + .1).clamp_max(1)
                        elif coordinate == 1:
                            changed[..., 1] *= .9
                        else:
                            changed[..., 2] = (changed[..., 2] * .9).clamp_min(10)
                        rates.append((own_epsilon(model, batch, b, k, changed) > actual + 1e-3).numpy())
                    monotone.extend(np.stack(rates, axis=-1))
            e, p = np.asarray(eps_all), np.asarray(pay_all)
            share = truth[..., 2].numpy() / truth[..., 2].sum(-1, keepdim=True).numpy()
            js = np.array([j_kappa(x, s, kval) for x, s in zip(e, share)])
            ratio = js / np.asarray(oracle[f"{amount:g}/{kval!r}"])
            metrics = exact_metrics(e, truth.numpy(), kval)
            result[f"{amount:g}/{kval!r}"] = {**metrics, "profiles": 512, "budget": amount, "kappa": kval,
                "mean_found_regret": float(np.mean(gains)), "max_found_regret": float(np.max(gains)),
                "normalized_mean_found_regret": float(np.mean(gains) / (amount / 10)),
                "normalized_no_size_regret": float(np.mean(no_size) / (amount / 10)),
                "raw_ir_violation_fraction": float(np.mean(raw_bad)),
                "settled_ir_violation_fraction": float(np.mean(p - truth[..., 0].numpy() * e < -1e-7)),
                "max_budget_excess": float(np.max(excess)),
                "monotonicity_violation_rates": dict(zip(("value", "capacity", "size"), np.mean(monotone, axis=(0, 1)).tolist())),
                "j_over_oracle_mean": float(ratio.mean()) if np.isfinite(ratio).all() else None,
                "j_over_oracle_infinite": bool(np.isinf(ratio).any())}
            atomic_json(OUT / "economics" / f"{arm}_progress.json", {"budget": amount, "kappa": kval, "metrics": result[f"{amount:g}/{kval!r}"]})
    immutable_json(target, {"arm": arm, "checkpoint": str(path), "checkpoint_sha256": digest(path),
                   "metrics": result, "scope": "found deviation lower bounds, cap underreports and size underreports; not certificates"})


def screen(families=FAMILIES, output_name="stage1_screen"):
    refs = {arm: read(OUT / "economics" / (arm + ".json")) for arm in ("V8F", "RegretNet-r2", "No-population-r2")}
    candidates = []
    for family in families:
        record = read(OUT / "economics" / (family + ".json"))
        reasons, strict = [], True
        for key, row in record["metrics"].items():
            baseline = [refs[name]["metrics"][key] for name in ("RegretNet-r2", "No-population-r2")]
            envelope = max(b["normalized_mean_found_regret"] for b in baseline)
            if row["normalized_mean_found_regret"] > envelope + 1e-10:
                reasons.append(f"{key}: regret above paired reference envelope")
            strict &= row["normalized_mean_found_regret"] <= .02
            if row["budget"] in (30., 100.):
                old = refs["V8F"]["metrics"][key]
                if row["j_infinite"] or (not old["j_infinite"] and row["j_kappa"] >= old["j_kappa"]):
                    reasons.append(f"{key}: executed J not below V8F")
            for coordinate, value in row["monotonicity_violation_rates"].items():
                if value > max(b["monotonicity_violation_rates"][coordinate] for b in baseline) + 1e-12:
                    reasons.append(f"{key}: {coordinate} monotonicity worse than envelope")
            if row["max_budget_excess"] > 2e-5 or row["settled_ir_violation_fraction"] > 0:
                reasons.append(f"{key}: BF or executed IR violation")
        candidates.append({"arm": family, "kind": family, "checkpoint": record["checkpoint"],
                           "checkpoint_sha256": record["checkpoint_sha256"], "eligible": not reasons,
                           "strict_002": bool(strict), "reasons": reasons, "economics": record["metrics"]})
    value = {"candidates": candidates, "gate_pass": all(c["eligible"] for c in candidates),
             "references": refs, "test_accuracy_used": False}
    immutable_json(OUT / (output_name + ".json"), value)
    return value


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--checkpoint", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    check_identity()
    if args.oracle:
        oracle_economics()
    elif args.arm and args.checkpoint:
        evaluate_economics(args.arm, args.checkpoint)
    else:
        ap.error("use --oracle, or --arm and --checkpoint")

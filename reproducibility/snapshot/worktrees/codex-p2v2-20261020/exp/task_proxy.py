#!/usr/bin/env python3
"""Public image task proxy and its explicitly approximate terminal-noise model.

Uses only the proxy half of reserved training images. The linear image model is
an offline training surrogate; final deployment always uses the original CNN and
exact Opacus execution. The empty-market loss is the unchanged-model loss.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common import ROOT, OUT, atomic_json, digest, dirichlet_partition, isolated_rng, opacus_safe_size
from datasets_fl_benchmark import load_mnist, load_cifar10
from tdsc_dpsgd_runner import calibrate_noise_multiplier_for_event_epsilon, EXPANDED_RDP_ORDERS
from tdsc_rdp_cache import exact_rdp_cache
from contracts import MIN_EVENT_EPSILON

EPS_GRID = torch.tensor([.01, .03, .1, .3, .8, 1.6, 3., 5.])
STEP_GRID = torch.tensor([1., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024.])


def public_features(dataset):
    loader = load_mnist if dataset == "MNIST" else load_cifar10
    train, clean_test = loader(root=str(ROOT / ("data/mnist" if dataset == "MNIST" else "data/cifar10")), download=False)
    train = copy.copy(train)
    train.transform = clean_test.transform
    labels = np.asarray(train.targets)
    rng = np.random.default_rng(20261016)
    proxy, validation, private = [], [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        heldout = sorted(indices[:len(indices) // 10].tolist())
        proxy.extend(heldout[:len(heldout) // 2])
        validation.extend(heldout[len(heldout) // 2:])
        private.extend(indices[len(indices) // 10:].tolist())
    assert not set(proxy) & (set(validation) | set(private))
    xs, ys = [], []
    for x, y in DataLoader(Subset(train, sorted(proxy)), batch_size=256, shuffle=False):
        pooled = F.adaptive_avg_pool2d(x, (7, 7) if dataset == "MNIST" else (4, 4)).flatten(1)
        xs.append(F.pad(pooled, (0, 49 - pooled.shape[1])))
        ys.append(y.long())
    x, y = torch.cat(xs), torch.cat(ys)
    mean, std = x.mean(0), x.std(0).clamp_min(.2)
    x = (x - mean) / std
    x = torch.cat([x, torch.ones(len(x), 1)], 1)
    return x, y, {"proxy_indices": sorted(proxy), "development_indices": sorted(validation),
                  "private_count": len(private), "features": "pooled pixels, standardized on proxy only; final coordinate is bias"}


def clipped_gradient(x, y, weights):
    err = torch.softmax(x @ weights, -1) - F.one_hot(y, 10)
    norm = x.norm(dim=-1) * err.norm(dim=-1)
    clip = (1 / norm.clamp_min(1e-12)).clamp(max=1)
    return x.t() @ (err * clip[:, None]) / len(x)


def build_bank(out, profiles_per_alpha=3):
    torch.set_num_threads(1)
    banks, manifests = [], {}
    generator = torch.Generator().manual_seed(20261017)
    for dataset in ("MNIST", "CIFAR10"):
        x, y, manifest = public_features(dataset)
        manifests[dataset] = manifest
        weights = torch.zeros(50, 10)
        states = [weights.clone()]
        lr = .1 if dataset == "MNIST" else .05
        for step in range(128):
            ids = torch.randint(len(x), (64,), generator=generator)
            weights -= .2 * clipped_gradient(x[ids], y[ids], weights)
            if step + 1 in (32, 128):
                states.append(weights.clone())
        for alpha in (.1, .5):
            for rep in range(profiles_per_alpha):
                split, _ = dirichlet_partition(y.numpy(), 10, alpha, min_size=4,
                                               seed=20261017 + rep + int(100 * alpha))
                sizes = [max(64, opacus_safe_size(18 * len(ids), 64)) for ids in split]
                physical_steps = torch.tensor([n // 64 for n in sizes], dtype=torch.float32)
                for state_index, base in enumerate(states):
                    deltas = []
                    for ids, steps in zip(split, physical_steps):
                        ids = torch.tensor(ids)
                        local = base.clone()
                        for _ in range(int(steps)):
                            sample = ids[torch.randint(len(ids), (64,), generator=generator)]
                            local -= lr * clipped_gradient(x[sample], y[sample], local)
                        deltas.append(local - base)
                    validation = torch.randperm(len(x), generator=generator)[:256]
                    banks.append({"dataset": dataset, "alpha": alpha, "replicate": rep,
                                  "state_index": state_index, "base": base, "deltas": torch.stack(deltas),
                                  "x": x[validation], "y": y[validation], "steps": physical_steps,
                                  "sizes": torch.tensor(sizes, dtype=torch.float32), "learning_rate": lr})
                print(json.dumps({"bank_dataset": dataset, "alpha": alpha, "replicate": rep,
                                  "banks": len(banks)}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"banks": banks, "public_manifest": manifests,
                "scope": "public linear-model local clipped SGD; additive terminal DP noise approximation"}, out)
    atomic_json(out.with_suffix(".manifest.json"), {"public_data": manifests, "banks": len(banks), "sha256": digest(out)})


def noise_table(path):
    if path.exists():
        data = json.loads(path.read_text())
        assert data["epsilon_grid"] == EPS_GRID.tolist() and data["step_grid"] == STEP_GRID.tolist()
        return torch.tensor(data["sigma"])
    values = []
    with exact_rdp_cache(maxsize=262144):
        for steps in STEP_GRID:
            row = []
            for eps in EPS_GRID:
                result = calibrate_noise_multiplier_for_event_epsilon(
                    target_event_epsilon=float(eps), sample_rate=1 / int(steps), optimizer_steps=int(steps),
                    event_delta=1e-5, orders=EXPANDED_RDP_ORDERS)
                row.append(result["noise_multiplier"])
            values.append(row)
            print(json.dumps({"calibrated_noise_table_steps": int(steps)}), flush=True)
    atomic_json(path, {"epsilon_grid": EPS_GRID.tolist(), "step_grid": STEP_GRID.tolist(), "sigma": values,
                       "interpolation": "log-linear in steps and epsilon, proxy only; deployment uses exact inversion"})
    return torch.tensor(values)


class TaskProxy:
    def __init__(self, bank_path, table_path, device="cpu"):
        saved = torch.load(bank_path, map_location=device, weights_only=False)
        self.banks = saved["banks"]
        self.device = device
        self.log_sigma = noise_table(table_path).to(device).log()
        self.eg = EPS_GRID.to(device).log()
        self.sg = STEP_GRID.to(device).log()

    def sigma(self, exposure, steps):
        le = exposure.clamp(float(EPS_GRID[0]), float(EPS_GRID[-1])).log().contiguous()
        ls = steps.clamp(float(STEP_GRID[0]), float(STEP_GRID[-1])).log().contiguous()
        ei = torch.searchsorted(self.eg, le).clamp(1, len(self.eg) - 1)
        si = torch.searchsorted(self.sg, ls).clamp(1, len(self.sg) - 1)
        et = (le - self.eg[ei - 1]) / (self.eg[ei] - self.eg[ei - 1])
        st = (ls - self.sg[si - 1]) / (self.sg[si] - self.sg[si - 1])
        a = self.log_sigma[si - 1, ei - 1] * (1 - et) + self.log_sigma[si - 1, ei] * et
        b = self.log_sigma[si, ei - 1] * (1 - et) + self.log_sigma[si, ei] * et
        return (a * (1 - st) + b * st).exp()

    def sample(self, batch, generator):
        indices = torch.randint(len(self.banks), (batch,), generator=generator, device=self.device).tolist()
        rows = [self.banks[i] for i in indices]
        data = {key: torch.stack([r[key] for r in rows]) for key in ("base", "deltas", "x", "y", "steps", "sizes")}
        data["learning_rate"] = torch.tensor([r["learning_rate"] for r in rows], device=self.device)
        # Permute clients together with all their public proxy features.
        perm = torch.rand(batch, 10, generator=generator, device=self.device).argsort(-1)
        ar = torch.arange(batch, device=self.device)[:, None]
        for key in ("deltas", "steps", "sizes"):
            data[key] = data[key][ar, perm]
        return data

    def loss(self, exposure, data, noise):
        total = exposure.sum(-1, keepdim=True)
        shares = exposure / total.clamp_min(1e-12)
        delta = torch.einsum("bn,bnfc->bfc", shares, data["deltas"])
        delta = torch.where((total > 0)[:, :, None], delta, 0.)
        sigma = self.sigma(exposure, data["steps"])
        local_std = data["learning_rate"][:, None] * data["steps"].sqrt() * sigma / 64
        variance = ((shares * local_std) ** 2).sum(-1)
        # sqrt(0) has an infinite derivative: keep the empty-release branch
        # exactly unchanged while using a differentiable positive branch.
        std = torch.where(total[:, 0] > 0, variance.clamp_min(1e-20).sqrt(), 0.)
        model = data["base"] + delta + std[:, None, None] * noise
        logits = torch.einsum("bvf,bfc->bvc", data["x"], model)
        return F.cross_entropy(logits.reshape(-1, 10), data["y"].reshape(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "public_bank.pt")
    parser.add_argument("--profiles-per-alpha", type=int, default=3)
    args = parser.parse_args()
    with isolated_rng(20261017):
        build_bank(args.out, args.profiles_per_alpha)
    noise_table(args.out.with_name("noise_table.json"))


if __name__ == "__main__":
    main()

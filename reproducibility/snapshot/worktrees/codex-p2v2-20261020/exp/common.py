"""Frozen dependencies and reproducible experimental primitives for repaired RQ4."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[1]
FROZEN = WORK / "frozen"
sys.path.insert(0, str(FROZEN))
from datasets_fl_benchmark import load_cifar10, load_mnist
from FL import CIFAR10Net, Net
from tdsc_dpsgd_runner import TDSCDPSGDRunner, _event_seed

OUT = ROOT / "run/tdsc_revision/codex_p2v2_20261020"
PAPER = ROOT / "main_final2/revisions/codex_p2v2_20261020"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def isolated_rng(seed=None, device="cpu"):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        try:
            if seed is not None:
                # Do not seed unrelated CUDA devices as torch.manual_seed does.
                torch.random.default_generator.manual_seed(int(seed))
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(int(seed))
                random.seed(int(seed))
                np.random.seed(int(seed) % (2**32))
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


class RepairedRunner(TDSCDPSGDRunner):
    def _train_one(self, *, round_index, request, event, preview, event_contract):
        seed = _event_seed(dataset_name=self.config.dataset_name, base_seed=self.seed,
                           target_epsilon=self.config.target_epsilon, round_index=round_index,
                           client_id=request.client_id, stream="data-augmentation")
        with isolated_rng(seed, self.device):
            delta, row = super()._train_one(round_index=round_index, request=request,
                                           event=event, preview=preview, event_contract=event_contract)
        row["augmentation_seed"] = seed
        return delta, row


def dirichlet_partition(labels, n_clients, alpha, min_size=10, seed=0, max_attempts=10000):
    """Exchangeable per-class Dirichlet-multinomial split, conditional on min size."""
    labels = np.asarray(labels)
    if alpha <= 0 or n_clients <= 0 or min_size < 0:
        raise ValueError("invalid partition parameters")
    if len(labels) < n_clients * min_size:
        raise ValueError("not enough records to satisfy min_size")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for attempt in range(max_attempts):
        splits = [[] for _ in range(n_clients)]
        for label in classes:
            ids = rng.permutation(np.flatnonzero(labels == label))
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            counts = rng.multinomial(len(ids), proportions)
            for client, part in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
                splits[client].extend(part.tolist())
        if min(map(len, splits)) >= min_size:
            return [rng.permutation(ids).tolist() for ids in splits], attempt + 1
    raise RuntimeError("min_size rejection sampling exhausted; do not silently alter alpha")


def opacus_safe_size(size, batch):
    if size <= batch:
        return size
    steps = size // batch
    while steps > 1 and int(1.0 / (1.0 / steps)) != steps:
        steps -= 1
    return steps * batch


def data_for_cell(dataset, alpha, seed, batch=64, stage="test", smoke_size=None):
    loader = load_mnist if dataset == "MNIST" else load_cifar10
    train, test = loader(root=str(ROOT / ("data/mnist" if dataset == "MNIST" else "data/cifar10")), download=False)
    labels = np.asarray(train.targets)
    rng = np.random.default_rng(20261016)
    heldout, pool = [], []
    for label in range(10):
        idx = rng.permutation(np.flatnonzero(labels == label))
        count = len(idx) // 10
        heldout.extend(idx[:count].tolist())
        pool.extend(idx[count:].tolist())
    pool = np.asarray(sorted(pool))
    local, attempts = dirichlet_partition(labels[pool], 10, alpha, 10, seed)
    complete = [pool[ids].tolist() for ids in local]
    used = []
    for ids in complete:
        limit = len(ids) if smoke_size is None else min(len(ids), smoke_size)
        used.append(ids[:opacus_safe_size(limit, batch)])
    proxy_indices, dev_indices = [], []
    for label in range(10):
        ids = sorted(i for i in heldout if labels[i] == label)
        proxy_indices.extend(ids[:len(ids) // 2])
        dev_indices.extend(ids[len(ids) // 2:])
    manifest = {"dataset": dataset, "alpha": alpha, "seed": seed, "batch_size": batch,
                "split": "classwise-Dirichlet-multinomial-conditioned-on-min-size-v1",
                "split_attempts": attempts, "min_size": 10, "dev_split_seed": 20261016,
                "development_indices": sorted(dev_indices), "public_proxy_indices": sorted(proxy_indices),
                "full_client_indices": complete,
                "used_client_indices": used, "smoke_size": smoke_size,
                "client_sizes": list(map(len, used)),
                "class_counts": [np.bincount(labels[ids], minlength=10).tolist() for ids in used],
                "unused_train_indices": sorted(set(pool.tolist()) - set(sum(used, [])))}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["sha256"] = fingerprint
    manifest_path = OUT / "splits" / f"{dataset}_a{alpha}_s{seed}_b{batch}_n{smoke_size}.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("saved split does not match regenerated split")
    else:
        atomic_json(manifest_path, manifest)
    evaluation = test
    if stage != "test":
        clean_train = copy.copy(train)
        clean_train.transform = test.transform
        # Reserve half of each public class for mechanism/proxy training. The
        # other half selects candidates using actual CNN FL, never proxy data.
        evaluation = Subset(clean_train, sorted(dev_indices))
    if smoke_size is not None:
        evaluation = Subset(evaluation, range(min(200, len(evaluation))))
    clients = [Subset(train, ids) for ids in used]
    return clients, evaluation, manifest, Net if dataset == "MNIST" else CIFAR10Net


def market_reports(rounds, sizes, seed):
    """Local RandomState preserves the old public market draws without global RNG effects."""
    reported_sizes = np.maximum(10.0, 500.0 * np.asarray(sizes) / max(sizes))
    result = []
    for r in range(rounds):
        rng = np.random.RandomState(seed * 100000 + r)
        rows = []
        for d in reported_sizes:
            value, cap = rng.uniform(0, 1), rng.uniform(.1, 5)
            rng.randint(10, 501)  # historical public draw; replaced by verified dataset size
            rows.append([value, cap, d])
        result.append(rows)
    return torch.tensor(result, dtype=torch.float32)


def evaluate(model, dataset, device="cpu", batch=256):
    mode = model.training
    device = torch.device(device)
    confusion = torch.zeros(10, 10, dtype=torch.int64, device=device)
    loss_sum = 0.0
    with isolated_rng(872341, device), torch.no_grad():
        generator = torch.Generator().manual_seed(872341)
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, generator=generator, num_workers=0)
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            confusion += torch.bincount(10 * y + logits.argmax(1), minlength=100).reshape(10, 10)
    model.train(mode)
    conf = confusion.cpu().numpy()
    if conf.sum() == 0:
        raise ValueError("empty evaluation set")
    tp = np.diag(conf)
    den = conf.sum(0) + conf.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(10, float), where=den > 0)
    return {"accuracy": float(tp.sum() / conf.sum()), "macro_f1": float(f1.mean()),
            "loss": float(loss_sum / conf.sum()), "examples": int(conf.sum()),
            "confusion": conf.tolist()}


def privacy_summary(snapshot):
    all_eps = np.array([c["epsilon_add_remove"] for c in snapshot["clients"]])
    counts = np.array([c["participations"] for c in snapshot["clients"]])
    def metrics(values):
        return {"mean": float(values.mean()), "p50": float(np.quantile(values, .5, method="higher")),
                "p95": float(np.quantile(values, .95, method="higher")), "max": float(values.max())} if len(values) else None
    return {"all_clients": metrics(all_eps), "active_clients": metrics(all_eps[counts > 0]),
            "zero_release_fraction": float((counts == 0).mean()), "participations": counts.tolist()}

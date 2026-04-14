#!/usr/bin/env python3
"""
Phase 1 verification: MNIST/CIFAR-10 non-IID (Dirichlet) + privacy paper bids.
Run from project root: python run_phase1_verify.py
Requires: torch, numpy; for MNIST/CIFAR tests also torchvision.
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Project imports
from datasets_fl_benchmark import (
    load_mnist,
    load_cifar10,
    dirichlet_split,
    get_client_subsets,
    generate_privacy_paper_bids,
    calc_cost_privacy_paper,
)
from FL import Net, CIFAR10Net, Arguments, ldp_fed_sgd, test
from client import Clients, extr_noniid_dirt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_mnist_n_agents(n_agents=10, alpha=0.5, seed=42):
    """Verify MNIST + Dirichlet split + 1 FL round with N clients."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[Phase 1] MNIST, n_agents={n_agents}, Dirichlet alpha={alpha}")
    train_set, test_set = load_mnist()
    n_classes = 10
    idxs_client = extr_noniid_dirt(train_set, n_agents, n_classes, alpha=alpha)
    subsets = get_client_subsets(train_set, idxs_client)
    print(f"  Client sample counts: {[len(s) for s in subsets]}")
    # One round FL: plosses = ones (all participate), equal weights
    fl_args = Arguments()
    fl_args.device = DEVICE
    fl_args.rounds = 1
    fl_args.lr = 0.01
    fl_args.L = 1.0
    fl_args.sensi = 2.0 * fl_args.L
    model = Net().to(DEVICE)
    plosses = torch.ones(n_agents, device=DEVICE) * 0.5
    weights = torch.ones(n_agents, device=DEVICE) / n_agents
    local_sets = []
    for sub in subsets:
        n = len(sub)
        if n == 0:
            raise ValueError("Empty client subset: Dirichlet split gave 0 samples to a client.")
        batch_size = max(1, min(64, n))
        dl = DataLoader(sub, batch_size=batch_size, shuffle=True)
        x, y = next(iter(dl))
        local_sets.append((x, y))
    model = ldp_fed_sgd(model, fl_args, plosses, weights, [local_sets], 0)
    acc = test(model, test_set, fl_args, 0)
    print(f"  After 1 round FL, test accuracy: {acc:.4f}")
    return acc


def verify_cifar10_n_agents(n_agents=10, alpha=0.5, seed=42):
    """Verify CIFAR-10 + Dirichlet + 1 FL round."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[Phase 1] CIFAR-10, n_agents={n_agents}, Dirichlet alpha={alpha}")
    train_set, test_set = load_cifar10()
    n_classes = 10
    idxs_client = extr_noniid_dirt(train_set, n_agents, n_classes, alpha=alpha)
    subsets = get_client_subsets(train_set, idxs_client)
    print(f"  Client sample counts: {[len(s) for s in subsets]}")
    fl_args = Arguments()
    fl_args.device = DEVICE
    fl_args.rounds = 1
    fl_args.lr = 0.01
    model = CIFAR10Net().to(DEVICE)
    plosses = torch.ones(n_agents, device=DEVICE) * 0.5
    weights = torch.ones(n_agents, device=DEVICE) / n_agents
    local_sets = []
    for sub in subsets:
        n = len(sub)
        if n == 0:
            raise ValueError("Empty client subset in CIFAR-10.")
        batch_size = max(1, min(64, n))
        dl = DataLoader(sub, batch_size=batch_size, shuffle=True)
        x, y = next(iter(dl))
        local_sets.append((x, y))
    model = ldp_fed_sgd(model, fl_args, plosses, weights, [local_sets], 0)
    acc = test(model, test_set, fl_args, 0)
    print(f"  After 1 round FL, test accuracy: {acc:.4f}")
    return acc


def verify_privacy_paper_bids():
    """Verify privacy paper bid generation and cost."""
    print("[Phase 1] Privacy paper bids: v~U[0,1], eps~U[0.1,5], c=v*eps")
    bids = generate_privacy_paper_bids(n_agents=5, n_items=1, num_profiles=4, seed=42)
    assert bids.shape == (4, 5, 3), f"Expected (4,5,3), got {bids.shape}"
    cost = calc_cost_privacy_paper(bids)
    assert cost.shape == (4, 5)
    for b in range(4):
        for i in range(5):
            v, eps = bids[b, i, 0].item(), bids[b, i, 1].item()
            expected = v * eps
            got = cost[b, i].item()
            assert abs(got - expected) < 1e-5, f"cost mismatch {got} vs {expected}"
    print("  Bid shape OK, cost = v*eps OK.")


def verify_client_generate_mnist():
    """Verify Clients.generate_clients for MNIST (saves to data/mnist/)."""
    print("[Phase 1] Clients.generate_clients(MNIST, non-IID, alpha=0.5)")
    clients = Clients()
    clients.dirs = "data/mnist/niid/"
    clients.filename = "phase1_test_profiles.json"
    clients.min_pbudget = 0.1
    clients.max_pbudget = 5.0
    os.makedirs(clients.dirs, exist_ok=True)
    d = clients.generate_clients("MNIST", n_profiles=1, n_clients_per_profile=10, iid=False, alpha=0.5)
    assert len(d) == 10
    print(f"  Generated {len(d)} clients.")
    assert all("privacy_budget" in d[i] and "data_size" in d[i] for i in range(10))
    print("  Sample (eps, size):", [(d[i]["privacy_budget"], d[i]["data_size"]) for i in range(3)])


if __name__ == "__main__":
    verify_privacy_paper_bids()
    verify_client_generate_mnist()
    verify_mnist_n_agents(n_agents=10, alpha=0.5)
    verify_cifar10_n_agents(n_agents=10, alpha=0.5)
    print("[Phase 1] All checks passed.")

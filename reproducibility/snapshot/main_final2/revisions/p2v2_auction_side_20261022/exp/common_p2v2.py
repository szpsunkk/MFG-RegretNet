"""Profiles and small helpers shared by the P2-v2 auction-side drivers.

The profile generators repeat the paper's generators draw for draw, so the new rows use the same profiles as
the existing rows of each table (checked by test_profiles.py against the original code):
  * test_profiles: datasets_fl_benchmark.generate_privacy_paper_bids, seeds 1001-1005 (eval_rq1.py, eval_rq3.py);
  * hetero_profiles: step11 hetero_eval.gen (bimodal / realistic), seeds 7001-7005;
  * generate_privacy_paper_bids also feeds the adaptive attacker (seeds 50000 + 101 k), the coordinated attacks
    and the accountant (seeds 42-44).
"""
import json
import math
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch

N_AGENTS = 10
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005]
HETERO_SCALE = math.exp(0.5 * 2.3263478740)


def generate_privacy_paper_bids(n_agents, num_profiles, seed):
    """datasets_fl_benchmark.generate_privacy_paper_bids with n_items = 1 (float32, as the original)."""
    np.random.seed(seed)
    reports = np.zeros((num_profiles, n_agents, 3))
    for b in range(num_profiles):
        for i in range(n_agents):
            v = np.random.uniform(0.0, 1.0)
            eps = np.random.uniform(0.1, 5.0)
            size = np.random.randint(10, 500 + 1)
            reports[b, i, 0] = v
            reports[b, i, 1] = eps
            reports[b, i, 2] = size
    return torch.tensor(reports, dtype=torch.float32)


def test_profiles(per_seed=2000):
    """The paper's 10,000 RQ1/RQ3 profiles, promoted to float64 before any arithmetic."""
    return torch.cat([generate_privacy_paper_bids(N_AGENTS, per_seed, s) for s in TEST_SEEDS]).double()


def hetero_profiles(scenario, per_seed=1000):
    """step11 hetero_eval.gen for seeds 7001-7005 (float32 as the original), promoted to float64."""
    out = []
    for s in TEST_SEEDS:
        rng = np.random.default_rng(7000 + s)
        P, N = per_seed, N_AGENTS
        if scenario == "bimodal":
            v = rng.uniform(0.0, 1.0, (P, N))
            sens = rng.random((P, N)) < 0.5
            eps = np.where(sens, rng.uniform(0.1, 0.5, (P, N)), rng.uniform(2.0, 5.0, (P, N)))
        elif scenario == "realistic":
            v = np.minimum(rng.lognormal(0.0, 0.5, (P, N)) / HETERO_SCALE, 1.0)
            eps = rng.uniform(0.1, 5.0, (P, N))
        else:
            raise ValueError(scenario)
        d = rng.integers(10, 501, (P, N)).astype(float)
        out.append(torch.tensor(np.stack([v, eps, d], -1), dtype=torch.float32))
    return torch.cat(out).double()


def host_info():
    return {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S %z")}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]

"""A4 initialization/reference and monotonicity checks; no optimization."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from v2_model import P2V2, FAMILIES, WORK, cap_projection
from common import OUT, atomic_json, digest
import p2v2_reference as REF
import p2_reference as R


def profiles(rng, count):
    return np.stack([rng.random((count, 10)), .1 + 4.9 * rng.random((count, 10)),
                     10 + 490 * rng.random((count, 10)) ** 2], -1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--family', choices=FAMILIES, required=True)
    p.add_argument('--gamma', type=float, choices=(0., .5, 1.), required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    root = OUT / 'validation/v2' / f'g{a.gamma:g}' / a.family
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = P2V2(a.family, a.gamma).double().eval()
    rng = np.random.default_rng(202610203)
    raw = profiles(rng, 256)
    reports = torch.tensor(raw)
    budgets = torch.tensor(np.resize([10., 30., 100.], 256)[:, None])
    kappas = torch.tensor(np.resize([3., 30.], 256)[:, None])
    began = time.time()
    with torch.no_grad():
        got_e, got_p = [], []
        for lo in range(0, 256, 16):
            e, pay = model.mechanism(reports[lo:lo+16], budgets[lo:lo+16], kappas[lo:lo+16])
            got_e.extend(e.numpy()); got_p.extend(pay.numpy())
    errors_e, errors_p = [], []
    for i, row in enumerate(raw):
        s_net = model.numpy_total(float(budgets[i, 0]), float(kappas[i, 0]))
        original_sizes = row[:, 2].copy()
        # A5 changes projection shares, never the network's raw report features.
        reference_reports = row.copy(); reference_reports[:, 2] = original_sizes ** a.gamma
        def original_network(changed):
            network_reports = changed.copy()
            network_reports[:, 2] = original_sizes
            return s_net(network_reports)
        e, pay = REF.mechanism(reference_reports, float(budgets[i, 0]), original_network)
        errors_e.append(float(np.max(np.abs(e - got_e[i]))))
        errors_p.append(float(np.max(np.abs(pay - got_p[i]))))
        assert errors_e[-1] <= 1e-6 and errors_p[-1] <= 1e-6, (i, errors_e[-1], errors_p[-1])
        if (i + 1) % 16 == 0:
            atomic_json(root / 'progress.json', dict(phase='reference_parity', profiles=i+1, total=256, seconds=time.time()-began))
            print('REFERENCE', i+1, flush=True)
    monotonicity = []
    test_profiles = torch.tensor(profiles(np.random.default_rng(202610204), 150))
    with torch.no_grad():
        for init_seed in (1, 2, 3):
            torch.manual_seed(init_seed)
            net = P2V2(a.family, a.gamma).double().eval()
            worst = 0.
            for amount in (10., 30., 100.):
                for kval in (3., 30.):
                    b = torch.full((150, 1), amount, dtype=torch.float64)
                    k = torch.full((150, 1), kval, dtype=torch.float64)
                    for fixed in (None, 8.):
                        base = net.allocate(test_profiles, b, k, fixed)
                        for i in range(10):
                            for coordinate in range(3):
                                ch = test_profiles.clone()
                                if coordinate == 0:
                                    ch[:, i, 0] = (ch[:, i, 0] + .1).clamp_max(1.)
                                elif coordinate == 1:
                                    ch[:, i, 1] *= .9
                                else:
                                    ch[:, i, 2] = (ch[:, i, 2] * .9).clamp_min(10.)
                                increase = float((net.allocate(ch, b, k, fixed)[:, i] - base[:, i]).max())
                                worst = max(worst, increase)
            assert worst <= 1e-9, (init_seed, worst)
            monotonicity.append(dict(initialization_seed=init_seed, profiles=150, max_increase=worst,
                                     budgets=[10,30,100], kappas=[3,30], fixed_totals=[None,8]))
            print('MONOTONICITY', init_seed, worst, flush=True)
    result = dict(pass_checks=True, family=a.family, gamma=a.gamma, profiles=256,
        max_allocation_error=max(errors_e), max_payment_error=max(errors_p), monotonicity=monotonicity,
        dtype='float64', seconds=time.time()-began, model_source_sha256=digest(WORK / 'next/v2_model.py'),
        check_source_sha256=digest(Path(__file__)), reference_sha256=digest(WORK / 'exp/p2v2_reference.py'))
    atomic_json(root / 'checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()

"""A4.3 checks for the A7.4 threshold rule, before any selector training."""
import time
import numpy as np
import torch
from flow_state import TABLE, WORK, candidate, check, read, digest, atomic_json
from model import Selector, reference_mechanism
from check_v2 import profiles


def validate(family):
    check(); torch.set_num_threads(1)
    assert read(TABLE.parent / 'table_sha256.json')['sha256'] == digest(TABLE)
    folder = candidate('QV-sel') / 'validation/v2/g1' / family
    torch.manual_seed(42)
    model = Selector(family).double().eval()
    raw = profiles(np.random.default_rng(202610203), 256)
    reports = torch.tensor(raw)
    budgets = torch.tensor(np.resize([10., 30., 100.], 256)[:, None])
    kappas = torch.tensor(np.resize([3., 30.], 256)[:, None])
    started = time.time()
    errors_e, errors_p = [], []
    with torch.no_grad():
        for lo in range(0, 256, 16):
            e, p = model.mechanism(reports[lo:lo+16], budgets[lo:lo+16], kappas[lo:lo+16])
            for offset, row in enumerate(raw[lo:lo+16]):
                b, k = float(budgets[lo+offset, 0]), float(kappas[lo+offset, 0])
                expected_e, expected_p = reference_mechanism(row, b, k, model.numpy_total(b, k), model.table)
                errors_e.append(float(np.max(np.abs(expected_e-e[offset].numpy()))))
                errors_p.append(float(np.max(np.abs(expected_p-p[offset].numpy()))))
                if errors_e[-1] > 1e-6 or errors_p[-1] > 1e-6:
                    raise RuntimeError(f'A4.3 selector reference mismatch: {family}, {lo+offset}')
            atomic_json(folder / 'progress.json', {'phase': 'reference_parity', 'profiles': lo+16,
                'total': 256, 'seconds': time.time()-started})
    monotonicity = []
    test = torch.tensor(profiles(np.random.default_rng(202610204), 150))
    with torch.no_grad():
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            net = Selector(family).double().eval()
            worst = 0.
            for amount in (10., 30., 100.):
                for kval in (3., 30.):
                    b = torch.full((150, 1), amount, dtype=torch.float64)
                    k = torch.full((150, 1), kval, dtype=torch.float64)
                    for fixed in (None, 8.):
                        original = net.allocate(test, b, k, fixed)
                        for i in range(10):
                            for coordinate in range(3):
                                changed = test.clone()
                                if coordinate == 0:
                                    changed[:, i, 0] = (changed[:, i, 0] + .1).clamp_max(1.)
                                elif coordinate == 1:
                                    changed[:, i, 1] *= .9
                                else:
                                    changed[:, i, 2] = (changed[:, i, 2] * .9).clamp_min(10.)
                                worst = max(worst, float((net.allocate(changed, b, k, fixed)[:, i]-original[:, i]).max()))
            if worst > 1e-9:
                raise RuntimeError(f'A4.3 selector monotonicity failed: family={family}, seed={seed}, increase={worst}')
            monotonicity.append({'initialization_seed': seed, 'profiles': 150, 'max_increase': worst})
    atomic_json(folder / 'checks.json', {'pass_checks': True, 'family': family, 'selected_proxy': 'QV-sel',
        'profiles': 256, 'max_allocation_error': max(errors_e), 'max_payment_error': max(errors_p),
        'monotonicity': monotonicity, 'tau_sha256': digest(TABLE), 'model_source_sha256': digest(WORK / 'a7flow/model.py'),
        'seconds': time.time()-started})

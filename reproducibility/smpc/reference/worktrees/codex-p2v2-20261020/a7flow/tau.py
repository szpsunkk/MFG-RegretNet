"""A7.4 CPU threshold table, with exact grid objectives and no image outcomes."""
import time
import numpy as np
import torch
from flow_state import OUT, TABLE, check, digest, canonical_immutable, atomic_json
from train_mechanism import sample_profiles
from p2_core import array_hash
from v2_model import cap_projection

KAPPAS = np.array([.003, .01, .03, .1, .3, 1, 3, 10, 30, 100, 300])
AMOUNTS = np.array([1., 2, 5, 10, 20, 30, 50, 100, 110])
THRESHOLDS = np.arange(51, dtype=float) * .005


def interpolate(table, kappa, amount, mode):
    """Bilinear in log coordinates, constant at the nearest boundary."""
    x = np.clip(np.log(kappa), np.log(KAPPAS[0]), np.log(KAPPAS[-1]))
    y = np.clip(np.log(amount), np.log(AMOUNTS[0]), np.log(AMOUNTS[-1]))
    values = np.asarray(table[mode], dtype=float)
    along_amount = np.array([np.interp(y, np.log(AMOUNTS), row) for row in values])
    return float(np.interp(x, np.log(KAPPAS), along_amount))


def selected(shares, threshold):
    mask = shares >= np.asarray(threshold).reshape(-1, 1)
    empty = ~mask.any(-1)
    rows = np.flatnonzero(empty)
    mask[rows, shares[rows].argmax(-1)] = True
    return mask


def grid_j(reports, count, amount, kappas=KAPPAS, grid=400, fixed=False):
    """Evaluate all totals together, equivalent to the registered scalar scan.

    For each profile a threshold selects a prefix in stable size order. Reusing
    the exact prefix's grid objective across thresholds changes no candidate.
    """
    reports = torch.as_tensor(reports, dtype=torch.float64)
    caps, sizes = reports[..., 1], reports[..., 2]
    pi = sizes / sizes.sum(-1, keepdim=True)
    order = np.argsort(-sizes.numpy(), axis=-1, kind='stable')[:, :count]
    mask = torch.zeros_like(caps, dtype=torch.bool)
    mask.scatter_(1, torch.from_numpy(order.copy()), True)
    limited = torch.where(mask, caps, 0.)
    target = torch.where(mask, pi, 0.)
    target /= target.sum(-1, keepdim=True)
    high = torch.minimum(torch.full((len(caps),), float(amount)), limited.sum(-1))
    fractions = torch.ones(1, dtype=torch.float64) if fixed else torch.linspace(1. / grid, 1., grid, dtype=torch.float64)
    totals = high[:, None] * fractions[None, :]
    steps = len(fractions)
    e = cap_projection(totals.reshape(-1, 1), target.repeat_interleave(steps, 0), limited.repeat_interleave(steps, 0))
    e = torch.where(e >= .01, e, 0.).reshape(len(caps), steps, 10)
    sums = e.sum(-1)
    bias = ((e / sums.clamp_min(1e-300)[..., None] - pi[:, None]) ** 2).sum(-1)
    noise = (e > 0).sum(-1) / sums.clamp_min(1e-300) ** 2
    scores = torch.as_tensor(kappas)[None, :, None] * bias[:, None] + noise[:, None]
    scores = torch.where(sums[:, None] > 0, scores, torch.full_like(scores, float('inf')))
    return scores.min(-1).values.numpy()


def build():
    torch.set_num_threads(1)
    check()
    if TABLE.exists():
        return
    generator = torch.Generator(device='cpu').manual_seed(4242)
    profiles, _ = sample_profiles(4096, generator, 'cpu')
    values = profiles.double().numpy()
    shares = values[..., 2] / values[..., 2].sum(-1, keepdims=True)
    counts = np.stack([selected(shares, np.full(len(values), t)).sum(-1) for t in THRESHOLDS], -1)
    base = TABLE.parent
    total_tasks, done = 2 * 9 * 10, 0
    began = time.time()
    table, objective = {}, {}
    for mode in ('budget', 'fixed_total'):
        # [profile, kappa, amount, number of participants]
        bank = np.empty((4096, len(KAPPAS), len(AMOUNTS), 10))
        for b, amount in enumerate(AMOUNTS):
            for count in range(1, 11):
                check()
                cache = base / 'grids' / f'{mode}_{amount:g}_k{count}.npy'
                if cache.exists():
                    result = np.load(cache, allow_pickle=False)
                    assert result.shape == (4096, len(KAPPAS))
                else:
                    result = np.concatenate([grid_j(values[i:i+64], count, amount, fixed=mode == 'fixed_total')
                                             for i in range(0, 4096, 64)], 0)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with cache.with_suffix('.tmp').open('wb') as stream:
                        np.save(stream, result, allow_pickle=False)
                    cache.with_suffix('.tmp').replace(cache)
                bank[:, :, b, count-1] = result
                done += 1
                atomic_json(base / 'progress.json', {'phase': 'tau_table', 'step': done,
                    'total': total_tasks, 'seconds': time.time()-began})
        means = np.stack([bank[np.arange(4096), :, :, counts[:, t]-1].mean(0)
                          for t in range(len(THRESHOLDS))], -1)
        best = means.argmin(-1)  # ascending tau: exact ties select the smaller.
        table[mode] = THRESHOLDS[best].tolist()
        objective[mode] = np.take_along_axis(means, best[..., None], -1)[..., 0].tolist()
    canonical_immutable(TABLE, {'kappas': KAPPAS.tolist(), 'amounts': AMOUNTS.tolist(),
        'thresholds': THRESHOLDS.tolist(), **table, 'objective': objective, 'profile_seed': 4242,
        'profiles': 4096, 'profile_sha256': array_hash(profiles), 'grid': 400,
        'source_sha256': digest(__file__), 'interpolation': 'bilinear log kappa and log amount, edge-clamped'})
    canonical_immutable(base / 'table_sha256.json', {'sha256': digest(TABLE)})


if __name__ == '__main__':
    build()

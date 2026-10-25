"""A5 allocation-only preparation. No image prediction or endpoint access."""
from pathlib import Path
import sys
import numpy as np
import torch
from common import ROOT, WORK, OUT, PAPER, atomic_json, digest
from p2_state import read, immutable_json, check_identity, freeze_stage_inputs, DESIGN, CONFIG, AMENDMENT
from p2_core import allocation, array_hash, exact_metrics, REFERENCES
from p2_jobs import market, calibration
from a3_fingerprint import fingerprint
import qv_reference as Q
import qv_sel_reference as QS

ARMS = ('QV-g1', 'QV-sel', 'QV-g0.5', 'QV-g0', 'CSRA', 'PAC', 'PAC-capped-legacy',
        'VCG-capped-legacy', 'Lin', 'RegretNet-r2', 'V8F')
EXTERNAL = ARMS[4:-1]
SEEDS = (16, 20)
GAMMAS = {'QV-g1': 1., 'QV-sel': 'sel', 'QV-g0.5': .5, 'QV-g0': 0.}
CELLS = [dict(dataset=ds, alpha=a, mode=mode, value=value)
         for ds in ('MNIST', 'CIFAR10') for a in (.5, .1)
         for mode, value in (('budget', 30.), ('privacy', .8))]


def qv_detail(reports, budget, kappa, gamma, fixed_total=None):
    rows = [(QS.qv_sel(row, budget, kappa, fixed_total) if gamma == 'sel'
             else Q.qv_proxy(row, budget, kappa, gamma, fixed_total)) for row in reports.double().numpy()]
    e, p = (torch.tensor(np.stack([r[j] for r in rows]), dtype=torch.float64) for j in (0, 1))
    return dict(proposed=e, offered_payment=p, executed=e, payment=p, voluntary=False)


def fixed_total_calibration(reports, gamma, target, kappa):
    def amount(total):
        return float(qv_detail(reports, 110., kappa, gamma, total)['executed'].mean())
    points = [(float(x), amount(x)) for x in np.unique(np.r_[.5, np.linspace(1, 10, 10), np.linspace(10, 50, 26)])]
    sampled = points[:]
    for (lo, left), (hi, right) in zip(points[:-1], points[1:]):
        if (left - target) * (right - target) > 0:
            continue
        for _ in range(24):
            mid = (lo + hi) / 2
            value = amount(mid)
            sampled.append((mid, value))
            if abs(value / target - 1) <= .0025:
                break
            if (left - target) * (value - target) <= 0:
                hi = mid
            else:
                lo, left = mid, value
    total, value = min(sampled, key=lambda x: (abs(x[1] / target - 1), x[0]))
    ok = abs(value / target - 1) <= .005
    return dict(budget=110., fixed_total=total, mean_100_rounds=value, relative_error_100=value / target - 1,
                feasible=ok, sampled_totals=sampled, reason='matched' if ok else 'no total in [0.5,50] matches')


def build_contract_job(phase, cell, seed, arm, kappa, rounds, budget, detail, split, reports,
                       extra=None, inputs=(), checkpoint=None, gate=None, smoke_size=None):
    tag = f"{cell['dataset']}_{cell['mode']}{cell['value']:g}_a{cell['alpha']}_s{seed}/{arm}"
    folder = OUT / phase / tag
    spec_path = OUT / 'allocations' / phase / tag / 'allocation.json'
    sources = [CONFIG, AMENDMENT, DESIGN / 'AMENDMENT_A4_20261020.md', DESIGN / 'AMENDMENT_A5_20261020.md',
               DESIGN / 'AMENDMENT_A6_20261020.md',
               PAPER / 'a2_verification.json', *inputs]
    if arm in REFERENCES:
        checkpoint = REFERENCES[arm][0]
    if checkpoint:
        sources.append(checkpoint)
    if gate:
        sources.append(gate)
    spec = {**cell, 'seed': seed, 'arm': arm, 'kappa': kappa, 'budget': budget, 'rounds': rounds,
        'aggregation': 'eps', 'stage': 'test' if phase.startswith('stage3') else 'smoke' if smoke_size else 'dev',
        'smoke_size': smoke_size, 'comparison': cell, 'split_sha256': split['sha256'],
        'reports_sha256': array_hash(reports), 'allocation_sha256': array_hash(detail['executed']),
        'payment_sha256': array_hash(detail['payment']), 'input_sha256': {str(p): digest(p) for p in sources},
        'non_strategic_reference': arm.startswith('QV-'), 'voluntary': detail['voluntary'],
        'reference_sha256': None, 'confirmation_gate_sha256': digest(gate) if gate else None,
        **{key: detail[key].double().tolist() for key in ('proposed', 'offered_payment', 'executed', 'payment')},
        **(extra or {})}
    spec['precomputed_metrics'] = exact_metrics(detail['executed'], reports, kappa)
    identity = fingerprint(spec, split, reports)
    spec['fl_fingerprint'] = identity['sha256']
    immutable_json(spec_path.with_name('fl_inputs.json'), identity)
    immutable_json(spec_path, spec)
    cmd = [sys.executable, '-u', str(WORK / 'exp/run_cell.py'), '--dataset', cell['dataset'],
        '--alpha', str(cell['alpha']), '--seed', str(seed), '--arm', arm, '--budget', repr(budget),
        '--stage', spec['stage'], '--rounds', str(rounds), '--aggr', 'eps', '--out', str(folder),
        '--allocation-spec', str(spec_path), '--spec-sha256', digest(spec_path)]
    if checkpoint:
        cmd.extend(['--checkpoint', str(checkpoint)])
    if smoke_size:
        cmd.extend(['--smoke-size', str(smoke_size), '--eval-every', '1'])
    return dict(id=phase + '/' + tag, phase=phase, command=cmd, out=str(folder), done=str(folder / 'result.json'),
        gpu=True, cell=cell, seed=seed, arm=arm, budget=budget, kappa=kappa, priority=0,
        allocation_spec=str(spec_path), allocation_spec_sha256=digest(spec_path), fl_fingerprint=identity['sha256'])


def prepare():
    torch.set_num_threads(1)
    check_identity()
    if (OUT / 'QV_PREPARED.json').exists():
        return
    jobs, rows, exclusions = [], [], []
    for cell in CELLS:
        for seed in SEEDS:
            reports, split = market(cell['dataset'], cell['alpha'], seed)
            kappa = 3. if cell['dataset'] == 'MNIST' else 30.
            for arm in ARMS:
                check_identity()
                key = f"{cell['dataset']}_{cell['mode']}{cell['value']:g}_a{cell['alpha']}_s{seed}/{arm}"
                allocation_cache = OUT / 'preparation' / key / 'detail.json'
                if allocation_cache.exists():
                    cache = read(allocation_cache)
                    row = cache['calibration']
                    detail = {k: torch.tensor(cache[k], dtype=torch.float64) for k in ('proposed', 'offered_payment', 'executed', 'payment')} if row['feasible'] else None
                    if detail:
                        detail['voluntary'] = False
                else:
                    budget, total = cell['value'], None
                    row = dict(budget=budget, feasible=True, reason='fixed budget')
                    if cell['mode'] == 'privacy':
                        row = fixed_total_calibration(reports, GAMMAS[arm], cell['value'], kappa) if arm in GAMMAS else calibration(reports, arm, None, cell['value'], kappa, raw_sizes=split['client_sizes'])
                        budget, total = row['budget'], row.get('fixed_total')
                    detail = None
                    if row['feasible']:
                        detail = qv_detail(reports[:40], budget, kappa, GAMMAS[arm], total) if arm in GAMMAS else allocation(arm, reports[:40], budget, kappa, raw_sizes=split['client_sizes'])
                        if cell['mode'] == 'privacy':
                            row['mean_40_rounds'] = float(detail['executed'].double().mean())
                            row['relative_error_40'] = row['mean_40_rounds'] / cell['value'] - 1
                            row['matched_40'] = abs(row['relative_error_40']) <= .005
                    immutable_json(allocation_cache, {'calibration': row, **({k: detail[k].double().tolist() for k in ('proposed', 'offered_payment', 'executed', 'payment')} if detail else {})})
                rows.append({**cell, 'seed': seed, 'arm': arm, **row})
                if not row['feasible']:
                    exclusions.append(rows[-1])
                    continue
                item = build_contract_job('qv', cell, seed, arm, kappa, 40, row['budget'], detail, split, reports[:40],
                    extra={'gamma': GAMMAS.get(arm), 'fixed_total': row.get('fixed_total'), 'calibration': row}, inputs=[allocation_cache])
                jobs.append(item)
                # Publication is atomic and append-only. Each spec is immutable.
                atomic_json(OUT / 'qv_queue.json', {'jobs': jobs})
                atomic_json(OUT / 'preparation_status.json', {'phase': 'qv', 'prepared': len(jobs), 'excluded': len(exclusions), 'current': key})
                print('PREPARED', key, flush=True)
    immutable_json(OUT / 'jobs_qv.json', jobs)
    immutable_json(OUT / 'calibration_qv.json', {'rows': rows, 'exclusions': exclusions})
    freeze_stage_inputs([OUT / 'jobs_qv.json', OUT / 'calibration_qv.json'])
    immutable_json(OUT / 'QV_PREPARED.json', {'jobs': len(jobs), 'exclusions': len(exclusions), 'jobs_sha256': digest(OUT / 'jobs_qv.json')})


if __name__ == '__main__':
    prepare()

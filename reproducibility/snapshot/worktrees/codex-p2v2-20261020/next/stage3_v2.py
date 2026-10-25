"""Candidate confirmation jobs and complete-stage A1.3 adoption, no partial reads."""
import argparse
from pathlib import Path
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from v2_model import WORK, FAMILIES, load_model
from common import ROOT, OUT, PAPER, digest, atomic_json
from p2_state import check_identity, read, immutable_json, freeze_stage_inputs
from p2_core import array_hash, exact_metrics
from p2_jobs import market
from qv_jobs import fixed_total_calibration, build_contract_job
from mechanisms import fill_uniform
from a1_rules import confirmation_cells, confirmation_rule


def detail_for(model, reports, budget, kappa, total):
    e_rows, p_rows = [], []
    with torch.no_grad():
        for chunk in reports.split(16):
            b = torch.full((len(chunk), 1), budget, dtype=torch.float64)
            k = torch.full((len(chunk), 1), kappa, dtype=torch.float64)
            e, p = model.mechanism(chunk.double(), b, k, total)
            e_rows.append(e); p_rows.append(p)
    e, p = torch.cat(e_rows), torch.cat(p_rows)
    assert bool((e <= reports[..., 1].double() + 1e-9).all())
    assert bool((p.sum(-1) <= budget + 1e-9).all())
    assert bool((p >= e * reports[..., 0].double() - 1e-9).all())
    return dict(proposed=e, offered_payment=p, executed=e, payment=p, voluntary=False)


def normalized_immutable(path, value):
    # JSON encodes tuple-valued DPSGDConfig.rdp_orders as arrays. Normalize the
    # transient value too; content hashes remain identical to the first write.
    import json
    immutable_json(path, json.loads(json.dumps(value)))


def prepare(gamma):
    check_identity(); torch.set_num_threads(1)
    saved = OUT / 'jobs_stage3_v2.json'
    if saved.exists():
        return
    gate = OUT / 'attempts' / f'g{gamma:g}' / 'stage1_screen.json'
    assert read(gate)['gate_pass']
    checkpoints = {family: OUT / 'training' / f'g{gamma:g}' / family / 'model.pt' for family in FAMILIES}
    models = {family: load_model(path) for family, path in checkpoints.items()}
    freeze_stage_inputs([gate, *checkpoints.values()])
    import qv_jobs
    qv_jobs.immutable_json = normalized_immutable
    jobs, calibrations, exclusions = [], [], []
    for cell in confirmation_cells():
        for seed in (231, 232, 233):
            reports, split = market(cell['dataset'], cell['alpha'], seed)
            kappa = 3. if cell['dataset'] == 'MNIST' else 30.
            total, budget = None, cell['value']
            row = dict(budget=budget, feasible=True, reason='fixed budget')
            if cell['mode'] == 'privacy':
                row = fixed_total_calibration(reports, gamma, cell['value'], kappa)
                total, budget = row['fixed_total'], 110.
            anchor = None
            for arm in (*FAMILIES, 'Uniform-DP'):
                if not row['feasible']:
                    exclusions.append({**cell, 'seed':seed, 'arm':arm, **row}); continue
                if arm == 'Uniform-DP':
                    assert anchor is not None
                    e = torch.tensor(np.stack([fill_uniform(r[:, 1].double().numpy(), float(x.sum())) for r,x in zip(reports,anchor['executed'])]), dtype=torch.float64)
                    e = torch.where(e >= .01, e, 0.)
                    if bool(((e.sum(-1)-anchor['executed'].sum(-1)).abs() > 1e-6 + 1e-7*e.sum(-1)).any()):
                        exclusions.append({**cell, 'seed':seed, 'arm':arm, 'reason':'minimum-event contract prevents exact per-round total matching'}); continue
                    detail = dict(proposed=e, offered_payment=torch.zeros_like(e), executed=e, payment=torch.zeros_like(e), voluntary=True)
                    checkpoint = checkpoints['mfg_p2v2']
                else:
                    detail = detail_for(models[arm], reports, budget, kappa, total)
                    checkpoint = checkpoints[arm]
                if arm == 'mfg_p2v2':
                    anchor = detail
                calibrations.append({**cell, 'seed':seed, 'arm':arm, **row})
                item = build_contract_job('stage3_v2', cell, seed, arm, kappa, 100, budget,
                    detail, split, reports, extra=dict(gamma=gamma, fixed_total=total, calibration=row),
                    inputs=[OUT / 'qv_decision.json'], checkpoint=checkpoint, gate=gate)
                jobs.append(item)
                atomic_json(OUT / 'continuation_queue.json', {'jobs': jobs})
                atomic_json(OUT / 'stage3_preparation_status.json', dict(prepared=len(jobs), excluded=len(exclusions), current=item['id']))
                print('PREPARED', item['id'], flush=True)
    immutable_json(OUT / 'calibration_stage3_v2.json', dict(rows=calibrations, exclusions=exclusions))
    immutable_json(saved, jobs)
    freeze_stage_inputs([saved, OUT / 'calibration_stage3_v2.json'])
    immutable_json(OUT / 'STAGE3_PREPARED.json', dict(jobs=len(jobs), exclusions=len(exclusions), gamma=gamma))


def analyze():
    if not (OUT / 'STAGE3_PREPARED.json').exists():
        return None
    old = ROOT / 'run/tdsc_revision/codex_p2_a3_20261019'
    jobs = read(OUT / 'jobs_stage3_v2.json') + read(old / 'jobs_stage3_early.json')
    if not all(Path(j['done']).exists() for j in jobs):
        return None
    if (OUT / 'stage3_adoption.json').exists():
        return read(OUT / 'stage3_adoption.json')
    # This is the first permitted opening of any sealed old Stage 3 outcomes.
    from p2_results import collect, write_csv
    for job in jobs:
        result = read(job['done'])
        assert result['meta']['stage'] == 'test' and result['meta']['rounds'] == 100
        for m in result['measurements']:
            if 96 <= m['round'] <= 100:
                matrix = np.asarray(m['confusion'])
                assert abs(np.trace(matrix) / matrix.sum() - m['accuracy']) <= 1e-12
    rows = collect('stage3_p2v2_complete', jobs, analysis_authorized=True)
    exclusions = read(old / 'calibration_stage3_early.json')['exclusions'] + read(OUT / 'calibration_stage3_v2.json')['exclusions']
    decision = confirmation_rule(rows, 'mfg_p2v2', confirmation_cells(), exclusions=exclusions)
    deadline = datetime(2026, 10, 31, 23, 59, tzinfo=timezone(timedelta(hours=9))).timestamp()
    timely = time.time() <= deadline and not (OUT / 'DEADLINE_FALLBACK.json').exists()
    decision.update(dict(result_sha256={j['done']:digest(j['done']) for j in jobs},
        qv_decision_sha256=digest(OUT / 'qv_decision.json'),
        within_deadline=timely, selected_model='mfg_p2v2' if decision['pass'] and timely else 'V8F'))
    immutable_json(OUT / 'stage3_adoption.json', decision)
    immutable_json(PAPER / 'stage3_adoption.json', decision)
    write_csv(PAPER / 'stage3_per_run.csv', rows)
    atomic_json(OUT / 'ALL_DONE.json', {'adoption_sha256':digest(OUT / 'stage3_adoption.json')})
    return decision


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gamma', type=float); parser.add_argument('--analyze', action='store_true')
    args=parser.parse_args()
    if args.analyze:
        print(analyze())
    else:
        prepare(args.gamma)

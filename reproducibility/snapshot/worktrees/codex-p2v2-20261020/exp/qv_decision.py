"""Open QV endpoints only after all registered runnable jobs complete."""
import csv
import statistics
import time
from pathlib import Path
from common import OUT, PAPER, atomic_json, digest
from p2_state import read, immutable_json, check_identity
from qv_jobs import CELLS, SEEDS, GAMMAS, EXTERNAL


def cell_key(cell):
    return f"{cell['dataset']}_{cell['mode']}{cell['value']:g}_a{cell['alpha']}"


def decide():
    check_identity()
    if not (OUT / 'QV_PREPARED.json').exists():
        return None
    jobs = read(OUT / 'jobs_qv.json')
    if not all(Path(j['done']).exists() for j in jobs):
        return None
    path = OUT / 'qv_decision.json'
    if path.exists():
        return read(path)
    records, match, rows, hashes = {}, {}, [], {}
    for job in jobs:
        obj = read(job['done'])
        spec = read(job['allocation_spec'])
        assert obj['meta']['stage'] == 'dev' and obj['meta']['rounds'] == 40
        assert obj['meta']['allocation_spec_sha256'] == job['allocation_spec_sha256']
        assert all(obj[n + '_sha256'] == digest(Path(job['out']) / (n + '.json')) for n in ('events', 'privacy'))
        endpoint = []
        for row in obj['measurements']:
            if 36 <= row['round'] <= 40:
                matrix = row['confusion']
                acc = sum(matrix[i][i] for i in range(10)) / sum(map(sum, matrix))
                assert abs(acc - row['accuracy']) < 1e-12
                endpoint.append(100 * acc)
        assert len(endpoint) == 5
        key = cell_key(job['cell']), job['seed'], job['arm']
        records[key] = statistics.mean(endpoint)
        calibration = spec['calibration']
        match[key] = calibration.get('matched_40', True)
        rows.append(dict(cell=key[0], seed=key[1], arm=key[2], accuracy_pp=records[key],
            matched_40=match[key], epsilon_100=calibration.get('mean_100_rounds'),
            epsilon_40=calibration.get('mean_40_rounds'), seconds=obj['summary']['seconds'],
            shared_execution=bool(obj.get('shared_execution_with')), result_sha256=digest(job['done'])))
        hashes[job['done']] = digest(job['done'])
    cells = {}
    for cell in CELLS:
        key = cell_key(cell)
        usable = [a for a in EXTERNAL if all((key, s, a) in records and match[key, s, a] for s in SEEDS)]
        means = {a: statistics.mean(records[key, s, a] for s in SEEDS)
                 for a in (*GAMMAS, *EXTERNAL, 'V8F') if all((key, s, a) in records for s in SEEDS)}
        best = max(usable, key=means.get) if usable else None
        cells[key] = dict(best_external=best, all_arm_means=means, omitted_external=[a for a in EXTERNAL if a not in usable], proxies={})
        for arm, gamma in GAMMAS.items():
            complete = best is not None and all((key, s, arm) in records and match[key, s, arm] for s in SEEDS)
            cells[key]['proxies'][arm] = dict(complete_matched_pair=complete,
                delta_by_seed={str(s): records[key, s, arm] - records[key, s, best] for s in SEEDS} if complete else None,
                delta_to_v8f_by_seed={str(s): records[key, s, arm] - records[key, s, 'V8F'] for s in SEEDS}
                if all((key, s, a) in records for s in SEEDS for a in (arm, 'V8F')) else None)
    proxies = {}
    for arm, gamma in GAMMAS.items():
        datasets = {}
        for dataset in ('MNIST', 'CIFAR10'):
            selected = [v['proxies'][arm] for k, v in cells.items() if k.startswith(dataset + '_')]
            complete = len(selected) == 4 and all(v['complete_matched_pair'] for v in selected)
            ds = {str(s): statistics.mean(v['delta_by_seed'][str(s)] for v in selected) for s in SEEDS} if complete else None
            datasets[dataset] = dict(complete=complete, d_s=ds, D=statistics.mean(ds.values()) if complete else None)
        ok = all(d['complete'] for d in datasets.values()) and datasets['CIFAR10']['D'] > 0 and datasets['MNIST']['D'] >= -.3
        proxies[arm] = dict(gamma=gamma, datasets=datasets, eligible=ok,
            score=sum(d['D'] for d in datasets.values()) if all(d['complete'] for d in datasets.values()) else None)
    tie_order = list(GAMMAS)
    ranked = sorted([a for a in proxies if proxies[a]['eligible']], key=lambda a: (-proxies[a]['score'], tie_order.index(a)))
    decision = dict(created_unix=time.time(), runs=len(jobs), cells=cells, proxies=proxies,
        eligible_ranked=ranked, selected_gamma=proxies[ranked[0]]['gamma'] if ranked else None,
        proceed=bool(ranked), fallback=None if ranked else 'V8F', endpoint='mean rounds36-40; development split',
        actual_window_matching='required on all paired seeds, per inherited preregistered 20261018 amendment',
        source_result_sha256=hashes)
    immutable_json(path, decision)
    immutable_json(PAPER / 'qv_decision.json', decision)
    immutable_json(PAPER / 'qv_decision_sha256.json', {'sha256': digest(path)})
    with (PAPER / 'qv_per_run.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# P2v2 快速验证结果', '', 'QV 已完整结束，A5/A6 四个代理全部报告；单位为准确率百分点。', '',
             '| 版本 | MNIST D | CIFAR10 D | 合格 |', '|---|---:|---:|---|']
    for arm, v in proxies.items():
        lines.append(f"| {arm} | {v['datasets']['MNIST']['D']} | {v['datasets']['CIFAR10']['D']} | {v['eligible']} |")
    lines += ['', '合格版本顺序：' + (', '.join(ranked) or '无；停止 P2v2，保留 V8F'), '',
              '未匹配/缺少两种子的单元不能删除后晋级。完整逐单元差值见 qv_decision.json。Stage 3 比较记录继续封存。']
    (PAPER / 'RESULTS_zh.md').write_text('\n'.join(lines) + '\n')
    return decision


if __name__ == '__main__':
    result = decide()
    print('pending' if result is None else {'proceed': result['proceed'], 'eligible_ranked': result['eligible_ranked']})

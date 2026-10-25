"""Conditional A5 continuation; unresolved A6 thresholds never get invented."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from v2_model import WORK, FAMILIES
from common import ROOT, OUT, PAPER, atomic_json, digest
from p2_state import check_identity, read, immutable_json, freeze_stage_inputs
from a3_resources import environment


def phase_job(kind, family, gamma):
    if kind == 'training':
        folder = OUT / 'training' / f'g{gamma:g}' / family
        done = folder / 'training.json'
        command = [sys.executable, '-u', str(WORK / 'next/train_v2.py'), '--family', family, '--gamma', str(gamma)]
    else:
        folder = OUT / 'attempts' / f'g{gamma:g}' / 'workers' / family
        done = OUT / 'attempts' / f'g{gamma:g}' / 'economics' / (family + '.json')
        command = [sys.executable, '-u', str(WORK / 'next/screen_v2.py'), '--family', family, '--gamma', str(gamma)]
    return dict(id=f'{kind}/g{gamma:g}/{family}', phase=kind, command=command, out=str(folder), done=str(done),
                gpu=False, priority=1, family=family, gamma=gamma)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--detach', action='store_true'); args = parser.parse_args()
    if args.detach:
        sys.path.insert(0, str(WORK / 'ops'))
        from bootstrap import child
        print(json.dumps(child('continuation', Path(__file__).resolve()))); return
    with (OUT / '.continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [WORK / 'next' / n for n in ('v2_model.py', 'check_v2.py', 'validate_all.py', 'train_v2.py', 'screen_v2.py', 'stage3_v2.py', 'continue_v2.py')]
        immutable_json(OUT / 'next_source_manifest.json', {str(p):digest(p) for p in files})
        freeze_stage_inputs([*files, OUT / 'next_source_manifest.json', OUT / 'validation/reference_tests.json'])
        while True:
            check_identity()
            now = time.time()
            deadline = datetime(2026,10,31,23,59,tzinfo=timezone(timedelta(hours=9))).timestamp()
            if now > deadline and not (OUT / 'stage3_adoption.json').exists():
                immutable_json(OUT / 'DEADLINE_FALLBACK.json', dict(selected_model='V8F', deadline_jst='2026-10-31 23:59',
                    reason='no Stage 3 decision by deadline; running confirmation not interrupted'))
            if (OUT / 'FAILURE.json').exists() or (OUT / 'validation/V2_IMPLEMENTATION_FAILURE.json').exists():
                atomic_json(OUT / 'continuation_status.json', dict(status='implementation_or_execution_failure', time=now))
                return
            path = OUT / 'qv_decision.json'
            if not path.exists():
                atomic_json(OUT / 'continuation_status.json', dict(status='waiting_for_complete_qv', time=now))
                time.sleep(15); continue
            decision = read(path)
            if not decision['proceed']:
                atomic_json(OUT / 'continuation_status.json', dict(status='qv_not_eligible', time=now, selected_model='V8F'))
                return
            selected = None
            for arm in decision['eligible_ranked']:
                gamma = decision['proxies'][arm]['gamma']
                if gamma == 'sel':
                    atomic_json(OUT / 'continuation_status.json', dict(status='awaiting_A6_kappa_threshold_table_rule',
                        time=now, selected_proxy=arm, reason='A6 does not specify kappa grid/interpolation for continuous log-uniform training; no alternative rule assumed'))
                    selected = 'pending'; break
                check_files = [OUT / 'validation/v2' / f'g{gamma:g}' / f / 'checks.json' for f in FAMILIES]
                if not all(p.exists() for p in check_files):
                    atomic_json(OUT / 'continuation_status.json', dict(status='waiting_for_A4_checks', gamma=gamma, time=now))
                    selected = 'pending'; break
                assert all(read(p)['pass_checks'] for p in check_files)
                training = [phase_job('training', f, gamma) for f in FAMILIES]
                if not all(Path(j['done']).exists() for j in training):
                    atomic_json(OUT / 'continuation_queue.json', {'jobs':training})
                    atomic_json(OUT / 'continuation_status.json', dict(status='training', gamma=gamma, time=now))
                    selected = 'pending'; break
                economics = [phase_job('economics', f, gamma) for f in FAMILIES]
                if not all(Path(j['done']).exists() for j in economics):
                    freeze_stage_inputs([OUT / 'training' / f'g{gamma:g}' / f / 'model.pt' for f in FAMILIES])
                    atomic_json(OUT / 'continuation_queue.json', {'jobs':economics})
                    atomic_json(OUT / 'continuation_status.json', dict(status='stage1_economics', gamma=gamma, time=now))
                    selected = 'pending'; break
                screen = OUT / 'attempts' / f'g{gamma:g}' / 'stage1_screen.json'
                if not screen.exists():
                    subprocess.run([sys.executable, str(WORK / 'next/screen_v2.py'), '--gamma', str(gamma), '--finalize'],
                                   cwd=ROOT, env=environment(), check=True)
                if read(screen)['gate_pass']:
                    selected = gamma; break
            if selected == 'pending':
                time.sleep(15); continue
            if selected is None:
                immutable_json(OUT / 'STOP_ROUTE.json', dict(reason='all eligible QV variants failed unchanged Stage 1 screen',
                    selected_model='V8F', time=now, attempts=decision['eligible_ranked']))
                return
            from stage3_v2 import prepare, analyze
            prepare(selected)
            result = analyze()
            if result is not None:
                atomic_json(OUT / 'continuation_status.json', dict(status='complete', time=now, adopted=result['selected_model']))
                return
            atomic_json(OUT / 'continuation_status.json', dict(status='stage3_sealed', gamma=selected, time=now))
            time.sleep(20)


if __name__ == '__main__':
    main()

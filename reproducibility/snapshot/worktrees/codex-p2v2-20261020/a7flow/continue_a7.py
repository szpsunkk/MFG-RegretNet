"""Proceed automatically only after the A7 decision and all required checks."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from flow_state import (HERE, ROOT, OUT, PAPER, OPS, TABLE, GAMMA, FAMILIES, candidate, initialize,
                        seal, check, read, digest, atomic_json, canonical_immutable, progress_note)
from a3_resources import environment


def phase_job(phase, arm, family=None):
    base = candidate(arm)
    gamma = GAMMA[arm]
    if phase == 'tau':
        done, folder = TABLE, TABLE.parent
    elif phase == 'check':
        folder = base / 'validation/v2/g1' / family
        done = folder / 'checks.json'
    elif phase == 'training':
        folder = base / 'training' / f'g{gamma:g}' / family
        done = folder / 'training.json'
    else:
        folder = base / 'attempts' / f'g{gamma:g}' / 'workers' / family
        done = base / 'attempts' / f'g{gamma:g}' / 'economics' / (family + '.json')
    command = [sys.executable, '-u', str(HERE / 'run_phase.py'), '--phase', phase, '--proxy', arm]
    if family:
        command += ['--family', family]
    return {'id': f'A7/{phase}/{arm}/{family or "public"}', 'phase': phase, 'command': command,
        'out': str(folder), 'done': str(done), 'gpu': False, 'priority': 1, 'family': family, 'proxy': arm}


def publish(jobs, status, **fields):
    atomic_json(OUT / 'continuation_A7_queue.json', {'jobs': jobs})
    atomic_json(OUT / 'continuation_A7_status.json', {'status': status, 'time': time.time(), **fields})
    progress_note(status)


def finish_unused():
    """Wait for admitted comparisons to finish; never interrupt or open results."""
    while True:
        path = OUT / 'status_A7.json'
        saved = read(path) if path.exists() else {}
        progress_note('A7 stopped; finishing already admitted sealed comparisons')
        if not saved.get('active') and (OUT / 'run_not_used_A7.json').exists():
            canonical_immutable(PAPER / 'run_not_used_A7.json', read(OUT / 'run_not_used_A7.json'))
            progress_note('A7 stopped; V8F retained; sealed comparator drain complete')
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--detach', action='store_true'); args = parser.parse_args()
    if args.detach:
        from bootstrap import child
        print(json.dumps(child('continuation', Path(__file__).resolve()))); return
    with (OUT / '.continuation_A7.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal()
        with (OPS / 'continuation_tests.log').open('w') as log:
            tested = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_flow.py'],
                                    cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        if tested.returncode:
            raise RuntimeError('continuation regression checks failed')
        while True:
            check()
            if (OUT / 'A7_FAILURE.json').exists():
                publish([], 'execution_failure'); return
            deadline = datetime(2026, 10, 31, 23, 59, tzinfo=timezone(timedelta(hours=9))).timestamp()
            if time.time() > deadline and not (OUT / 'stage3_adoption_A7.json').exists():
                canonical_immutable(OUT / 'DEADLINE_FALLBACK_A7.json', {'selected_model': 'V8F',
                    'reason': 'no Stage 3 decision by deadline; running confirmation is not interrupted'})
            path = OUT / 'qv_decision_A7.json'
            if not path.exists():
                publish([], 'waiting_for_all_A7_runs'); time.sleep(15); continue
            decision = read(path)
            if not decision['proceed']:
                publish([], 'A7_not_eligible', selected_model='V8F')
                finish_unused(); return
            selected = None
            pending = False
            for arm in decision['eligible_ranked']:
                base = initialize(arm)
                gamma = GAMMA[arm]
                if arm == 'QV-sel' and not TABLE.exists():
                    publish([phase_job('tau', arm)], 'calibrating_A7_threshold_table', proxy=arm)
                    pending = True; break
                if arm == 'QV-sel':
                    checks = [phase_job('check', arm, f) for f in FAMILIES]
                    if not all(Path(j['done']).exists() for j in checks):
                        publish(checks, 'A4_3_selector_checks', proxy=arm); pending = True; break
                    if not all(read(j['done'])['pass_checks'] for j in checks):
                        raise RuntimeError('A4.3 selector check failed; training forbidden')
                training = [phase_job('training', arm, f) for f in FAMILIES]
                if not all(Path(j['done']).exists() for j in training):
                    publish(training, 'training', proxy=arm); pending = True; break
                economics = [phase_job('economics', arm, f) for f in FAMILIES]
                if not all(Path(j['done']).exists() for j in economics):
                    publish(economics, 'stage1_economics', proxy=arm); pending = True; break
                gate = base / 'attempts' / f'g{gamma:g}' / 'stage1_screen.json'
                if not gate.exists():
                    subprocess.run([sys.executable, str(HERE / 'run_phase.py'), '--phase', 'finalize', '--proxy', arm],
                                   cwd=ROOT, env=environment(), check=True)
                if read(gate)['gate_pass']:
                    selected = arm; break
            if pending:
                time.sleep(15); continue
            if selected is None:
                canonical_immutable(OUT / 'STOP_ROUTE_A7.json', {'reason': 'all A7-eligible variants failed unchanged Stage 1',
                    'selected_model': 'V8F', 'attempts': decision['eligible_ranked']})
                publish([], 'stage1_not_eligible', selected_model='V8F')
                finish_unused(); return
            base = candidate(selected)
            if not (base / 'STAGE3_PREPARED.json').exists():
                subprocess.run([sys.executable, str(HERE / 'run_phase.py'), '--phase', 'prepare_stage3', '--proxy', selected],
                               cwd=ROOT, env=environment(), check=True)
            publish(read(base / 'jobs_stage3_v2.json'), 'stage3_sealed', proxy=selected)
            subprocess.run([sys.executable, str(HERE / 'run_phase.py'), '--phase', 'analyze_stage3', '--proxy', selected],
                           cwd=ROOT, env=environment(), check=True)
            if (OUT / 'ALL_DONE_A7.json').exists():
                return
            time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        from state import safe_error
        safe_error('continuation', error)
        raise

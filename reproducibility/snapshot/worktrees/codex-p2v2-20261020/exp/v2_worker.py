"""101-only admission: QV first, sealed historical comparators second."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, WORK, OUT, atomic_json, digest
from p2_state import read, check_identity, safe_progress
from a3_resources import resources, can_start, gpu_pids, token, environment
from qv_decision import decide

OLD = ROOT / 'run/tdsc_revision/codex_p2_a3_20261019'


def queue():
    qv = read(OUT / 'qv_queue.json')['jobs'] if (OUT / 'qv_queue.json').exists() else []
    decision = read(OUT / 'qv_decision.json') if (OUT / 'qv_decision.json').exists() else None
    early = read(OLD / 'jobs_stage3_early.json')
    for j in early:
        j['id'] = 'old-stage3/' + j['id']
        j['priority'] = 10
        j['historical'] = True
    later = read(OUT / 'continuation_queue.json')['jobs'] if (OUT / 'continuation_queue.json').exists() else []
    stop = (decision is not None and not decision['proceed']) or (OUT / 'STOP_ROUTE.json').exists()
    return sorted(qv + later + early, key=lambda j: (j.get('priority', 0), j['id'])), stop


def same_block(a, b):
    return (a.get('fl_fingerprint') and a.get('fl_fingerprint') == b.get('fl_fingerprint')
            and a.get('cell') == b.get('cell') and a.get('seed') == b.get('seed')
            and a['phase'].split('_')[0] == b['phase'].split('_')[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    live, failures = {}, []
    initial = resources()
    started = time.time()
    while True:
        check_identity()
        jobs, stop = queue()
        for job in jobs:
            path = Path(job['out']) / 'process.json'
            if path.exists() and not Path(job['done']).exists():
                previous = read(path)
                pid = previous['pid']
                if pid not in live and token(pid) == previous['start_token']:
                    live[pid] = (None, None, job, previous['start_token'])
        for pid, (proc, log, job, tok) in list(live.items()):
            alive = proc.poll() is None if proc else token(pid) == tok
            if alive:
                continue
            if log:
                log.close()
            del live[pid]
            if not Path(job['done']).exists() or (proc is not None and proc.returncode):
                failures.append(dict(id=job['id'], out=job['out'], returncode=proc.returncode if proc else None))
                atomic_json(OUT / 'FAILURE.json', failures)
        snapshot = resources()
        observed = gpu_pids()
        missing = sum(j[2]['gpu'] and pid not in observed for pid, j in live.items())
        gpu_count = sum(j[2]['gpu'] for j in live.values())
        foreign = len(observed - set(live))
        early_count = sum(bool(j[2].get('historical')) for j in live.values())
        qv_pending = not (OUT / 'qv_decision.json').exists()
        cpu_count = len(live) - gpu_count
        no_swap = all(snapshot[k] == initial[k] for k in ('pswpin', 'pswpout'))
        for job in jobs:
            if failures or (OUT / 'PAUSE_ADMISSION').exists():
                break
            if stop and job.get('historical'):
                continue
            if Path(job['done']).exists() or any(j[2]['id'] == job['id'] for j in live.values()):
                continue
            if job.get('historical') and qv_pending and early_count >= 2:
                continue
            if job['gpu']:
                if gpu_count + foreign >= 10 or not no_swap or not can_start(snapshot, missing):
                    continue
                if any(same_block(job, j[2]) for j in live.values()):
                    continue
            elif cpu_count >= 2:
                continue
            spec = job.get('allocation_spec')
            if spec and digest(spec) != job['allocation_spec_sha256']:
                raise RuntimeError('allocation changed: ' + spec)
            command = list(job['command']); command[0] = sys.executable
            shared = next((j['out'] for j in jobs if j['id'] != job['id'] and Path(j['done']).exists() and same_block(job, j)), None)
            if shared and job['gpu']:
                command.extend(['--shared-from', shared])
            folder = Path(job['out']); folder.mkdir(parents=True, exist_ok=True)
            log = (folder / 'job.log').open('a')
            proc = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            tok = token(proc.pid)
            atomic_json(folder / 'process.json', dict(pid=proc.pid, start_token=tok, command=command, machine='101', queue_job=job['id']))
            live[proc.pid] = proc, log, job, tok
            if job['gpu']:
                gpu_count += 1; missing += 1
            else:
                cpu_count += 1
            if job.get('historical'):
                early_count += 1
        active = []
        for pid, (_, _, job, _) in live.items():
            p = Path(job['out']) / 'progress.json'
            active.append(dict(pid=pid, id=job['id'], historical=bool(job.get('historical')),
                               progress=safe_progress(read(p)) if p.exists() else {}))
        counts = {phase: dict(total=sum(j['phase'] == phase for j in jobs),
                  complete=sum(j['phase'] == phase and Path(j['done']).exists() for j in jobs)) for phase in sorted({j['phase'] for j in jobs})}
        atomic_json(OUT / 'status.json', dict(pid=os.getpid(), updated_unix=time.time(), started_unix=started,
            status='failed' if failures else 'draining' if stop else 'running', counts=counts, active=active,
            resources=snapshot, failures=failures, max_gpu_jobs=10, early_limit_during_qv=2, no_swapping=no_swap))
        if not failures:
            decide()
        if not live and (failures or stop or (OUT / 'ALL_DONE.json').exists()):
            if stop:
                unused = [{'id': j['id'], 'out': j['out'], 'completed': Path(j['done']).exists()}
                          for j in jobs if j.get('historical')]
                atomic_json(OUT / 'run_not_used.json', unused)
            return
        time.sleep(3)


if __name__ == '__main__':
    main()

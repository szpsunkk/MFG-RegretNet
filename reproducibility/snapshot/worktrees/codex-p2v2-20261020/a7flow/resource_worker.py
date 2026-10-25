"""Resource-only replacement controller; adopt all live scientific workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from flow_state import HERE, OUT, OPS, check, read, digest, atomic_json, canonical_immutable
from a3_resources import token, can_start as original_can_start, resources as original_resources
import worker

PEAK_MIB = 4096


def reserve(snapshot, missing=0):
    return original_can_start(snapshot, missing, peak_mib=PEAK_MIB)


def telemetry():
    return {**original_resources(), 'new_worker_reserve_mib': PEAK_MIB, 'free_guard_mib': 3072}


def takeover():
    check()
    old = read(OPS / 'worker_process.json')
    pid = old['pid']
    if token(pid) != old['start_token']:
        raise RuntimeError('original A7 controller is no longer the recorded process')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if command != old['command'] or not command[-1].endswith('/a7/worker.py'):
        raise RuntimeError('refuse to signal an unrecognized controller')
    snapshot = read(OUT / 'status_A7.json')
    active = {str(row['pid']): token(row['pid']) for row in snapshot['active']}
    canonical_immutable(OPS / 'resource_takeover_before.json', {'old_controller': old,
        'workers': active, 'observed_free_mib': snapshot['resources']['free_mib'],
        'new_worker_reserve_mib': PEAK_MIB, 'source_sha256': digest(__file__), 'time': time.time()})
    # Signal only the coordinator PID, never its process group or child workers.
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if token(pid) != old['start_token']:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('old controller has not exited; leave workers untouched')
    after = {p: token(int(p)) for p in active}
    canonical_immutable(OPS / 'resource_takeover_after.json', {'workers': after,
        'all_workers_preserved': after == active, 'time': time.time()})
    if after != active:
        # A normally completed worker is allowed, but an incomplete exit is not.
        jobs, _ = worker.queue()
        for p in active:
            if active[p] == after[p]:
                continue
            matched = [j for j in jobs if (Path(j['out']) / 'process.json').exists()
                       and read(Path(j['out']) / 'process.json')['pid'] == int(p)]
            if len(matched) != 1 or not Path(matched[0]['done']).exists():
                raise RuntimeError('a worker exited without completion during handoff')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--detach', action='store_true'); args = parser.parse_args()
    if args.detach:
        from bootstrap import child
        print(json.dumps(child('resource_worker', Path(__file__).resolve()))); return
    takeover()
    worker.can_start = reserve
    worker.resources = telemetry
    worker.main()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        from state import safe_error
        safe_error('resource_worker', error)
        raise

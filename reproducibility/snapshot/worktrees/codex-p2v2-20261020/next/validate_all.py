"""Run the nine preregistered torch checks on three CPU processes."""
import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from v2_model import WORK, FAMILIES
from common import ROOT, OUT, atomic_json, digest
from p2_state import read, immutable_json, check_identity
from a3_resources import environment


def run(family):
    for gamma in (1., .5, 0.):
        folder = OUT / 'validation/v2' / f'g{gamma:g}' / family
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'checks.json').exists():
            assert read(folder / 'checks.json')['model_source_sha256'] == digest(WORK / 'next/v2_model.py')
            continue
        with (folder / 'check.log').open('a') as log:
            r = subprocess.run([sys.executable, '-u', str(WORK / 'next/check_v2.py'), '--family', family,
                '--gamma', str(gamma)], cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            atomic_json(OUT / 'validation/V2_IMPLEMENTATION_FAILURE.json', dict(family=family, gamma=gamma,
                returncode=r.returncode, time=time.time(), training_allowed=False))
            raise RuntimeError('implementation check failed: ' + family)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--detach', action='store_true'); args=p.parse_args()
    if args.detach:
        sys.path.insert(0, str(WORK / 'ops'))
        from bootstrap import child
        print(json.dumps(child('v2_checks', Path(__file__).resolve()))); return
    check_identity()
    with (OUT / '.v2_validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [WORK / 'next' / name for name in ('v2_model.py', 'check_v2.py', 'validate_all.py')]
        hashes = {str(p): digest(p) for p in files}
        immutable_json(OUT / 'validation/next_validation_manifest.json', hashes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(run, FAMILIES))
        assert all(digest(p) == v for p, v in hashes.items())
        atomic_json(OUT / 'validation/ALL_TORCH_CHECKS_PASSED.json', dict(time=time.time(), checks=9, source_sha256=hashes))


if __name__ == '__main__':
    main()

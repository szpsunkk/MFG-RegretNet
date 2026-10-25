"""Complete missing remote legacy checks; preserve completed smoke evidence."""
import json
from pathlib import Path
import subprocess
import sys
import time
from v2_model import WORK
from common import ROOT, OUT, atomic_json, digest
from p2_state import check_identity, read, immutable_json
from a3_resources import environment


def main():
    check_identity()
    folder = OUT / 'validation'
    checks = []
    for name, count in (('qv_reference', 4), ('qv_sel_reference', 3)):
        log = folder / (name + '.log')
        assert sum(line.startswith('PASS ') for line in log.read_text().splitlines()) == count
        checks.append(dict(name=name, passed=count, log_sha256=digest(log)))
    for dataset in ('MNIST', 'CIFAR10'):
        base = OUT / 'smoke_qv' / f'{dataset}_budget30_a0.5_s16/QV-g1'
        result = read(base / 'result.json')
        assert result['meta']['rounds'] == 3 and result['meta']['stage'] == 'smoke'
        assert result['summary']['accountant_parity_pass']
        for name in ('events', 'privacy'):
            assert result[name + '_sha256'] == digest(base / (name + '.json'))
        checks.append(dict(name=dataset + '_smoke', result_sha256=digest(base / 'result.json')))
    for phase in (1, 2, 3):
        command = [sys.executable, '-u', str(WORK / 'exp/verify_phases.py'), '--phase', str(phase)]
        with (folder / f'phase{phase}.log').open('w') as log:
            r = subprocess.run(command, cwd=ROOT, env=dict(environment(), CUDA_VISIBLE_DEVICES=''),
                               stdout=log, stderr=subprocess.STDOUT)
        assert r.returncode == 0, ('phase', phase, r.returncode)
        checks.append(dict(name='phase' + str(phase), exit_code=0, log_sha256=digest(folder / f'phase{phase}.log')))
    immutable_json(folder / 'PREFLIGHT_DONE.json', dict(checks=checks, machine='101', completed_unix=time.time(),
        resume_script_sha256=digest(Path(__file__)), scientific_sources_changed=False,
        note='Original setup lacked legacy root check scripts. Completed smoke evidence is reused byte-for-byte.'))
    sys.path.insert(0, str(WORK / 'ops'))
    from bootstrap import child
    processes = {name: child(name, WORK / 'exp' / script) for name, script in
                 (('prepare_qv', 'qv_jobs.py'), ('worker', 'v2_worker.py'))}
    atomic_json(OUT / 'operations/LAUNCHED.json', dict(processes=processes, time=time.time()))
    print(json.dumps(processes), flush=True)


if __name__ == '__main__':
    main()

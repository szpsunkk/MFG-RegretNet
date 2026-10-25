"""Invoke unchanged training/screen/confirmation code under A7 artifact paths."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from flow_state import (ROOT, WORK, OUT, PAPER, TABLE, GAMMA, FAMILIES, candidate, initialize,
                        check, read, digest, atomic_json, canonical_immutable)


def train_selector(family):
    from model import Selector
    from train_mechanism import sample_profiles
    from p2_reference import sample_log_uniform
    base = initialize('QV-sel')
    for f in FAMILIES:
        proof = read(base / 'validation/v2/g1' / f / 'checks.json')
        assert proof['pass_checks'] and proof['tau_sha256'] == digest(TABLE)
    assert read(base / 'validation/reference_tests.json')['all_pass']
    torch.manual_seed(42)
    generator = torch.Generator(device='cpu').manual_seed(42)
    net = Selector(family).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    folder = base / 'training/g1' / family
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'training.json').exists():
        return
    start, history = 0, []
    resume = folder / 'training_resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        assert state['tau_sha256'] == digest(TABLE)
        assert state['continuation_sha256'] == digest(OUT / 'A7_continuation_manifest.json')
        net.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator']); start, history = state['step'], state['history']
    began = time.time()
    for step in range(start, 12000):
        if step % 100 == 0:
            check()
        lr = .001 if step < 10000 else .00003
        for group in optimizer.param_groups:
            group['lr'] = lr
        reports, budget = sample_profiles(64, generator, 'cpu')
        kappa = sample_log_uniform(64, .003, 300., generator, 'cpu')
        optimizer.zero_grad(set_to_none=True)
        loss = net.objective(reports.double(), budget.double(), kappa.double()).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite selector training objective')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        if (step+1) % 100 == 0:
            row = {'step': step+1, 'total': 12000, 'loss': float(loss.detach()), 'lr': lr, 'seconds': time.time()-began}
            history.append(row); atomic_json(folder / 'progress.json', row)
        if (step+1) % 1000 == 0:
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
                'step': step+1, 'history': history, 'tau_sha256': digest(TABLE),
                'continuation_sha256': digest(OUT / 'A7_continuation_manifest.json')}
            torch.save(state, folder / 'training_resume.tmp'); (folder / 'training_resume.tmp').replace(resume)
    saved = {'family': family, 'gamma': 1., 'selected_proxy': 'QV-sel', 'seed': 42, 'steps': 12000,
        'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'tau_sha256': digest(TABLE),
        'model_source_sha256': digest(WORK / 'a7flow/model.py'), 'trainer_sha256': digest(__file__),
        'qv_decision_sha256': digest(OUT / 'qv_decision_A7.json'),
        'config': {'batch': 64, 'lr_initial': .001, 'lr_last2000': .00003,
                   'kappa_range': [.003, 300.], 'clip_norm': 1., 'dtype': 'float64'}}
    torch.save(saved, folder / 'model.tmp'); (folder / 'model.tmp').replace(folder / 'model.pt')
    atomic_json(folder / 'training.json', {'family': family, 'selected_proxy': 'QV-sel', 'history': history,
        'checkpoint_sha256': digest(folder / 'model.pt'), 'seconds': time.time()-began})


def selector_fixed_total_calibration(model, reports, target, kappa):
    from stage3_v2 import detail_for
    def amount(total):
        return float(detail_for(model, reports, 110., kappa, total)['executed'].mean())
    points = [(float(x), amount(x)) for x in np.unique(np.r_[.5, np.linspace(1, 10, 10), np.linspace(10, 50, 26)])]
    sampled = points[:]
    for (lo, left), (hi, right) in zip(points[:-1], points[1:]):
        if (left-target) * (right-target) > 0:
            continue
        for _ in range(24):
            mid = (lo+hi)/2; value = amount(mid); sampled.append((mid, value))
            if abs(value/target-1) <= .0025:
                break
            if (left-target) * (value-target) <= 0:
                hi = mid
            else:
                lo, left = mid, value
    total, mean = min(sampled, key=lambda p: (abs(p[1]/target-1), p[0]))
    feasible = abs(mean/target-1) <= .005
    return {'budget': 110., 'fixed_total': total, 'mean_100_rounds': mean, 'relative_error_100': mean/target-1,
        'feasible': feasible, 'sampled_totals': sampled, 'reason': 'matched' if feasible else 'no total in [0.5,50] matches',
        'selector_tau_sha256': digest(TABLE)}


def configure_stage3(arm):
    import stage3_v2 as stage
    import qv_jobs
    from model import load_model
    base = initialize(arm)
    stage.OUT, stage.PAPER = base, PAPER / 'continuation_A7' / arm
    stage.check_identity, stage.load_model = check, load_model
    original_build = qv_jobs.build_contract_job
    def build(phase, *args, **kwargs):
        inputs = list(kwargs.get('inputs', []))
        inputs += [OUT / 'A7_continuation_manifest.json']
        if arm == 'QV-sel':
            inputs += [TABLE]
        kwargs['inputs'] = inputs
        kwargs['extra'] = dict(kwargs.get('extra', {}), selected_proxy=arm, amendment='A7')
        return original_build('stage3_A7', *args, **kwargs)
    stage.build_contract_job = build
    if arm == 'QV-sel':
        anchor = load_model(base / 'training/g1/mfg_p2v2/model.pt')
        stage.fixed_total_calibration = lambda reports, gamma, target, kappa: selector_fixed_total_calibration(anchor, reports, target, kappa)
    return stage


def run(phase, arm, family=None):
    torch.set_num_threads(1)
    check()
    base = initialize(arm)
    gamma = GAMMA[arm]
    if phase == 'tau':
        from tau import build
        build()
    elif phase == 'check':
        from check_sel import validate
        validate(family)
    elif phase == 'training':
        if arm == 'QV-sel':
            train_selector(family)
        else:
            import train_v2
            train_v2.OUT, train_v2.check_identity = base, check
            sys.argv = [str(Path(train_v2.__file__)), '--family', family, '--gamma', str(gamma)]
            train_v2.main()
    elif phase in ('economics', 'finalize'):
        import screen_v2
        from model import load_model
        screen_v2.OUT, screen_v2.check_identity, screen_v2.load_model = base, check, load_model
        sys.argv = [str(Path(screen_v2.__file__)), '--gamma', str(gamma)]
        sys.argv += ['--finalize'] if phase == 'finalize' else ['--family', family]
        screen_v2.main()
    elif phase == 'prepare_stage3':
        configure_stage3(arm).prepare(gamma)
    elif phase == 'analyze_stage3':
        result = configure_stage3(arm).analyze()
        if result is not None:
            canonical_immutable(OUT / 'stage3_adoption_A7.json', result)
            canonical_immutable(PAPER / 'stage3_adoption_A7.json', result)
            atomic_json(OUT / 'ALL_DONE_A7.json', {'selected_model': result['selected_model'],
                'decision_sha256': digest(OUT / 'stage3_adoption_A7.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('tau', 'check', 'training', 'economics', 'finalize', 'prepare_stage3', 'analyze_stage3'), required=True)
    parser.add_argument('--proxy', choices=GAMMA, required=True)
    parser.add_argument('--family', choices=FAMILIES)
    args = parser.parse_args()
    run(args.phase, args.proxy, args.family)

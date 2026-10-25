"""Single seed-42 final checkpoint, fixed A4 schedule, selected A5 target."""
import argparse
from pathlib import Path
import time
import torch
from v2_model import P2V2, FAMILIES, WORK
from common import OUT, atomic_json, digest
from p2_state import check_identity, read
from p2_reference import sample_log_uniform
from train_mechanism import sample_profiles


def main():
    p = argparse.ArgumentParser(); p.add_argument('--family', choices=FAMILIES, required=True)
    p.add_argument('--gamma', type=float, choices=(0., .5, 1.), required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.set_num_threads(1)
    check_identity()
    for family in FAMILIES:
        check = read(OUT / 'validation/v2' / f'g{a.gamma:g}' / family / 'checks.json')
        assert check['pass_checks'] and check['model_source_sha256'] == digest(WORK / 'next/v2_model.py')
    assert read(OUT / 'validation/reference_tests.json')['all_pass']
    decision = read(OUT / 'qv_decision.json')
    assert any(decision['proxies'][arm]['gamma'] == a.gamma for arm in decision['eligible_ranked'])
    torch.manual_seed(42)
    gen = torch.Generator(device=a.device).manual_seed(42)
    model = P2V2(a.family, a.gamma).double().to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    folder = OUT / 'training' / f'g{a.gamma:g}' / a.family
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'model.pt'
    if final.exists():
        return
    resume_path = folder / 'training_resume.pt'
    start, history = 0, []
    if resume_path.exists():
        state = torch.load(resume_path, map_location=a.device, weights_only=False)
        assert state['source_sha256'] == digest(WORK / 'next/v2_model.py')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        gen.set_state(state['generator']); start, history = state['step'], state['history']
    began = time.time()
    for step in range(start, 12000):
        check_identity() if step % 100 == 0 else None
        lr = 1e-3 if step < 10000 else 3e-5
        for group in optimizer.param_groups:
            group['lr'] = lr
        reports, budget = sample_profiles(64, gen, a.device)
        kappa = sample_log_uniform(64, .003, 300., gen, a.device)
        reports, budget, kappa = reports.double(), budget.double(), kappa.double()
        optimizer.zero_grad(set_to_none=True)
        loss = model.objective(reports, budget, kappa).mean()
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step+1) % 100 == 0:
            row = dict(step=step+1, total=12000, loss=float(loss.detach()), lr=lr, seconds=time.time()-began)
            history.append(row); atomic_json(folder / 'progress.json', row)
            print(row, flush=True)
        if (step+1) % 1000 == 0:
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), generator=gen.get_state(),
                         step=step+1, history=history, source_sha256=digest(WORK / 'next/v2_model.py'))
            torch.save(state, folder / 'training_resume.tmp')
            (folder / 'training_resume.tmp').replace(resume_path)
    saved = dict(family=a.family, gamma=a.gamma, seed=42, steps=12000,
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        config=dict(batch=64, lr_initial=.001, lr_last2000=.00003, kappa_range=[.003,300.], clip_norm=1., dtype='float64'),
        model_source_sha256=digest(WORK / 'next/v2_model.py'), trainer_sha256=digest(Path(__file__)),
        qv_decision_sha256=digest(OUT / 'qv_decision.json'))
    torch.save(saved, folder / 'model.tmp'); (folder / 'model.tmp').replace(final)
    atomic_json(folder / 'training.json', dict(family=a.family, gamma=a.gamma, history=history,
        checkpoint_sha256=digest(final), seconds=time.time()-began))


if __name__ == '__main__':
    main()

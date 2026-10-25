"""Use the unchanged P2 economic screen with the A4 mechanism adapter."""
import argparse
from pathlib import Path
import torch
from v2_model import WORK, FAMILIES, load_model
from common import ROOT, OUT, digest
from p2_state import read, immutable_json, check_identity, freeze_stage_inputs
from contracts import utility
import p2_core as core
import p2_economics as economics


def raw(model, reports, budget, kappa):
    device = next(model.parameters()).device
    return model.mechanism(reports.to(device=device, dtype=torch.float64),
                           budget.to(device=device, dtype=torch.float64), kappa.to(device=device, dtype=torch.float64))


def deviation(model, truthful, budget, kappa, misreports):
    batch, n, features = truthful.shape
    profiles = truthful[:, None].expand(batch, n, n, features).clone()
    ix = torch.arange(n, device=truthful.device)
    profiles[:, ix, ix] = misreports
    rows = profiles.reshape(-1, n, features).double()
    b, k = budget.repeat_interleave(n, 0).double(), kappa.repeat_interleave(n, 0).double()
    clients = ix.repeat(batch)
    pieces = []
    # Preserve the exact 64-interval formula; only bound intermediate memory.
    for start in range(0, len(rows), 32):
        stop = start + 32
        e, p = model.own_payment(rows[start:stop], b[start:stop], k[start:stop], clients[start:stop])
        true = truthful.reshape(-1, features)[start:stop].double()
        pieces.append(utility(true, e, p))
    return torch.cat(pieces).reshape(batch, n)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gamma', type=float, required=True)
    p.add_argument('--family', choices=FAMILIES); p.add_argument('--finalize', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    check_identity()
    attempt = OUT / 'attempts' / f'g{args.gamma:g}'
    attempt.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'run/tdsc_revision/codex_p2_a3_20261019/economics'
    for name in ('oracle', 'V8F', 'RegretNet-r2', 'No-population-r2'):
        immutable_json(attempt / 'economics' / (name + '.json'), read(old / (name + '.json')))
    freeze_stage_inputs([old / (n + '.json') for n in ('oracle', 'V8F', 'RegretNet-r2', 'No-population-r2')])
    economics.OUT = attempt
    economics.kappas = lambda: [3., 30.]
    # The same 512 float32-drawn profiles are promoted before arithmetic only.
    original_holdout = economics.holdout
    economics.holdout = lambda: original_holdout().double()
    economics.raw = raw
    core.deviation_utility = deviation
    economics.deviation_utility = deviation
    economics.load_any = lambda path, kind=None: load_model(path)
    if args.finalize:
        value = economics.screen(families=FAMILIES)
        print({'gate_pass': value['gate_pass'], 'gamma': args.gamma}, flush=True)
    else:
        assert args.family
        checkpoint = OUT / 'training' / f'g{args.gamma:g}' / args.family / 'model.pt'
        economics.evaluate_economics(args.family, checkpoint)


if __name__ == '__main__':
    main()

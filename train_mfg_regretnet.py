#!/usr/bin/env python3
"""
Train MFG-RegretNet for privacy paper (Phase 3).
Paper: b_MFG = (1/N)*sum_i b_i, budget projection (eq.49), AL training (Algorithm 2).
Hyperparams: T=200 outer epochs, T_in=25 (PGA steps R=25), L=64 batch, rho/λ updates.
"""
import os
import argparse
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

parser = argparse.ArgumentParser()
parser.add_argument('--random-seed', type=int, default=42)
parser.add_argument('--num-examples', type=int, default=102400)
parser.add_argument('--test-num-examples', type=int, default=10000)
parser.add_argument('--n-agents', type=int, default=10)
parser.add_argument('--n-items', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=200, help='T in paper')
parser.add_argument('--batch-size', type=int, default=64, help='L in paper')
parser.add_argument('--test-batch-size', type=int, default=500)
parser.add_argument('--model-lr', type=float, default=1e-3)
parser.add_argument('--max-budget-rate', type=float, default=5.0)
parser.add_argument('--min-budget-rate', type=float, default=0.1)
parser.add_argument('--activation', default='tanh')

parser.add_argument('--misreport-lr', type=float, default=0.01, help='PGA step size eta in paper')
parser.add_argument('--misreport-iter', type=int, default=25, help='R in paper')
parser.add_argument('--test-misreport-iter', type=int, default=25)

parser.add_argument('--rho-regret', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-regret', type=int, default=1)
parser.add_argument('--rho-incr-amount-regret', type=float, default=1.5)
parser.add_argument('--rho-ir', type=float, default=10.0)
parser.add_argument('--rho-incr-epoch-ir', type=int, default=1)
parser.add_argument('--rho-incr-amount-ir', type=float, default=1.5)
parser.add_argument('--rho-bc', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-bc', type=int, default=1)
parser.add_argument('--rho-incr-amount-bc', type=float, default=1.0)
parser.add_argument('--rho-deter', type=float, default=1.0)
parser.add_argument('--rho-incr-epoch-deter', type=int, default=1)
parser.add_argument('--rho-incr-amount-deter', type=float, default=0.0)

parser.add_argument('--regret-lagr-mult', type=float, default=1.0)
parser.add_argument('--ir-lagr-mult', type=float, default=1.0)
parser.add_argument('--bc-lagr-mult', type=float, default=1.0)
parser.add_argument('--deter-lagr-mult', type=float, default=1.0)
parser.add_argument('--lagr-update-iter-regret', type=int, default=25)
parser.add_argument('--lagr-update-iter-ir', type=int, default=25)
parser.add_argument('--lagr-update-iter-bc', type=int, default=10)
parser.add_argument('--lagr-update-iter-deter', type=int, default=10)

parser.add_argument('--resume', default='')
parser.add_argument('--L', type=float, default=1.0)
parser.add_argument('--aggr-method', type=str, default='OptAggr')

parser.add_argument('--hidden-layer-size', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=3)
parser.add_argument('--separate', action='store_true', default=True)
parser.add_argument('--normalized-loss', type=int, default=2)
parser.add_argument('--name', default='mfg_regretnet_privacy')
parser.add_argument('--gpu', type=str, default='0')
# RQ3：在仍优化 regret/IR 的前提下，温和提高总支付与参与者效用（见 exp_rq/RQ3_PROCESS.md）
parser.add_argument(
    '--lambda-revenue-util', type=float, default=0.0,
    help='鼓励 sum(p)/B（仅 MFG-RegretNet）；建议 0.03~0.12 试 RQ3 η_rev；0 保持原行为',
)
parser.add_argument(
    '--lambda-participant-welfare', type=float, default=0.0,
    help='鼓励 sum_i u_i（ truthful 效用和，与 W 一致）；建议 1e-4~1e-2；0 关闭',
)

args, _unknown = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.nn.parallel import DataParallel
import numpy as np
from datasets import Dataloader
from datasets_fl_benchmark import generate_privacy_paper_bids
from regretnet import MFGRegretNet, train_loop, test_loop

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_privacy_paper_data(n_agents, n_items, num_profiles, seed=None):
    """Build (num_profiles, n_agents, n_items+4) for train_loop: reports (v,eps,size) + val_type (2)."""
    reports = generate_privacy_paper_bids(
        n_agents=n_agents, n_items=n_items, num_profiles=num_profiles,
        v_min=0.0, v_max=1.0, eps_min=0.1, eps_max=5.0, seed=seed
    )
    # reports: (num_profiles, n_agents, n_items+2) = (P, N, 3)
    reports = reports.numpy()
    # Append val_type: (0, 1) for additive / scale so cost = v*eps is used
    val_type = np.zeros((reports.shape[0], reports.shape[1], 2))
    val_type[:, :, 0] = 0
    val_type[:, :, 1] = 1.0
    data = np.concatenate([reports, val_type], axis=2)  # (P, N, 5)
    return torch.tensor(data, dtype=torch.float32)


def main():
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    train_data = build_privacy_paper_data(
        args.n_agents, args.n_items, args.num_examples, seed=args.random_seed
    )
    train_data = train_data.to(DEVICE)
    train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = build_privacy_paper_data(
        args.n_agents, args.n_items, args.test_num_examples, seed=args.random_seed + 1
    )
    test_data = test_data.to(DEVICE)
    test_loader = Dataloader(test_data[:args.test_num_examples], batch_size=args.test_batch_size, shuffle=False)

    model = MFGRegretNet(
        args.n_agents, args.n_items,
        hidden_layer_size=args.hidden_layer_size,
        n_hidden_layers=args.n_hidden_layers,
        activation=args.activation,
        separate=args.separate,
    )
    model.glorot_init()
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', {})
        if sd:
            # Saved with DataParallel so keys may be 'module.xxx'; strip prefix for bare model
            first_key = next(iter(sd.keys()), '')
            if first_key.startswith('module.'):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd)
    model = DataParallel(model.to(DEVICE))

    os.makedirs('result', exist_ok=True)
    writer = SummaryWriter(log_dir=f"run/{args.name}-{args.n_agents}-{args.n_items}", comment=str(args))

    # train_loop expects args with n_agents, n_items, etc.
    train_loop(model, train_loader, test_loader, args, device=DEVICE, writer=writer)
    writer.close()

    test_result = test_loop(model, test_loader, args, device=DEVICE)
    import json
    print(json.dumps(test_result, indent=2))


if __name__ == '__main__':
    main()

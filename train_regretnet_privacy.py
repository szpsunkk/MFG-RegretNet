#!/usr/bin/env python3
"""
在隐私论文合成 bid 上训练标准 RegretNet（与 MFG 同分布），供 RQ1 对比。
产出：result/regretnet_privacy_<epoch>_checkpoint.pt
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--num-examples", type=int, default=32768)
parser.add_argument("--test-num-examples", type=int, default=4096)
parser.add_argument("--n-agents", type=int, default=10)
parser.add_argument("--n-items", type=int, default=1)
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--test-batch-size", type=int, default=500)
parser.add_argument("--model-lr", type=float, default=1e-3)
parser.add_argument("--misreport-lr", type=float, default=0.01)
parser.add_argument("--misreport-iter", type=int, default=25)
parser.add_argument("--test-misreport-iter", type=int, default=25)
parser.add_argument("--max-budget-rate", type=float, default=5.0)
parser.add_argument("--min-budget-rate", type=float, default=0.1)
parser.add_argument("--activation", default="tanh")
parser.add_argument("--hidden-layer-size", type=int, default=100)
parser.add_argument("--n-hidden-layers", type=int, default=3)
parser.add_argument("--separate", action="store_true", default=True)
parser.add_argument("--normalized-loss", type=int, default=2)
parser.add_argument("--p-activation", default="softmax")
parser.add_argument("--a-activation", default="softmax")
parser.add_argument("--smoothing", type=float, default=0.125)
parser.add_argument("--normalized-input", type=int, default=-1)
parser.add_argument("--name", default="regretnet_privacy")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--resume", default="")
parser.add_argument("--L", type=float, default=1.0)
parser.add_argument("--aggr-method", type=str, default="OptAggr")
parser.add_argument("--privacy-cost", action="store_true", default=False,
                    help="Use privacy paper cost c(v,eps)=v*eps for IR/regret in training")
parser.add_argument("--fixed-budget", type=float, default=50.0,
                    help="Fixed budget for training (0=use dynamic budget from valuation*size)")
# RegretNet 拉格朗日（与 train.py 一致）
parser.add_argument("--rho-regret", type=float, default=1.0)
parser.add_argument("--rho-incr-epoch-regret", type=int, default=1)
parser.add_argument("--rho-incr-amount-regret", type=float, default=1.0)
parser.add_argument("--rho-ir", type=float, default=1.0)
parser.add_argument("--rho-incr-epoch-ir", type=int, default=1)
parser.add_argument("--rho-incr-amount-ir", type=float, default=1.0)
parser.add_argument("--rho-bc", type=float, default=1.0)
parser.add_argument("--rho-incr-epoch-bc", type=int, default=1)
parser.add_argument("--rho-incr-amount-bc", type=float, default=1.0)
parser.add_argument("--rho-deter", type=float, default=1.0)
parser.add_argument("--rho-incr-epoch-deter", type=int, default=1)
parser.add_argument("--rho-incr-amount-deter", type=float, default=0.0)
parser.add_argument("--regret-lagr-mult", type=float, default=1.0)
parser.add_argument("--ir-lagr-mult", type=float, default=1.0)
parser.add_argument("--bc-lagr-mult", type=float, default=1.0)
parser.add_argument("--deter-lagr-mult", type=float, default=1.0)
parser.add_argument("--lagr-update-iter-regret", type=int, default=10)
parser.add_argument("--lagr-update-iter-ir", type=int, default=10)
parser.add_argument("--lagr-update-iter-bc", type=int, default=10)
parser.add_argument("--lagr-update-iter-deter", type=int, default=10)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
from torch.nn.parallel import DataParallel
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

from datasets import Dataloader
from regretnet import RegretNet, train_loop, test_loop
from train_mfg_regretnet import build_privacy_paper_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    train_data = build_privacy_paper_data(
        args.n_agents, args.n_items, args.num_examples, seed=args.random_seed
    ).to(DEVICE)
    train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = build_privacy_paper_data(
        args.n_agents, args.n_items, args.test_num_examples, seed=args.random_seed + 1
    ).to(DEVICE)
    test_loader = Dataloader(test_data[: args.test_num_examples], batch_size=args.test_batch_size, shuffle=False)

    model = RegretNet(
        args.n_agents,
        args.n_items,
        activation=args.activation,
        hidden_layer_size=args.hidden_layer_size,
        n_hidden_layers=args.n_hidden_layers,
        p_activation=args.p_activation,
        a_activation=args.a_activation,
        separate=args.separate,
        smoothing=args.smoothing,
        normalized_input=args.normalized_input,
    )
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", {})
        if sd:
            k0 = next(iter(sd.keys()), "")
            if k0.startswith("module."):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd)
    model = DataParallel(model.to(DEVICE))

    os.makedirs("result", exist_ok=True)
    writer = SummaryWriter(
        log_dir="run/{}-{}-{}".format(args.name, args.n_agents, args.n_items),
        comment=str(args),
    )
    train_loop(model, train_loader, test_loader, args, device=DEVICE, writer=writer)
    writer.close()
    import json

    print(json.dumps(test_loop(model, test_loader, args, device=DEVICE), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RQ4: FL 收敛性（实验思路 3.4）
运行多轮 FL，每轮记录全局模型精度，输出收敛曲线数据 (rounds + methods) 供 Phase 5 画图。
依赖：FL.py, client, experiments.acc_eval_mechs，以及数据集（如 NSL-KDD/Bank）。
若无法导入完整 FL 流程，可生成示例 JSON 用于测试图表。
"""
from __future__ import division, print_function

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="RQ4: FL convergence — accuracy vs round")
    parser.add_argument("--out", type=str, default="run/privacy_paper/rq4_accuracy.json", help="output JSON path")
    parser.add_argument("--rounds", type=int, default=200, help="FL rounds")
    parser.add_argument("--rnd-step", type=int, default=10, help="record accuracy every this many rounds")
    parser.add_argument("--sample", action="store_true", help="write sample JSON only (no FL run)")
    parser.add_argument("--dataset", type=str, default="NSL-KDD", help="dataset for FL")
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--regretnet-ckpt", type=str, default="")
    parser.add_argument("--mfg-regretnet-ckpt", type=str, default="")
    args = parser.parse_args()

    if args.sample:
        # 示例数据，用于测试 Phase 5 的 RQ4 图
        rounds = [1] + [(r + 1) * args.rnd_step for r in range(args.rounds // args.rnd_step)]
        methods = {
            "PAC": [0.3 + 0.5 * (1 - 1/x) for x in rounds],
            "VCG": [0.32 + 0.48 * (1 - 1/x) for x in rounds],
            "MFG-RegretNet": [0.35 + 0.52 * (1 - 1/x) for x in rounds],
        }
        data = {"rounds": rounds, "methods": methods}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print("Wrote sample RQ4 JSON:", args.out)
        return 0

    # 尝试运行完整 FL 评估
    try:
        import numpy as np
        from experiments import acc_eval_mechs, load_auc_model, DEVICE
        from experiments import Exp_Args
        from client import Clients
        from FL import Arguments
        from datasets import load_nslkdd, load_bank
    except ImportError as e:
        print("RQ4 full run requires FL/client/datasets. Error:", e)
        print("Use --sample to generate sample JSON, or run acc_eval_mechs_parallel from experiments.py and save rounds+methods to", args.out)
        return 1

    exp_args = Exp_Args()
    exp_args.n_rounds = args.rounds
    exp_args.rnd_step = args.rnd_step
    exp_args.n_agents = args.n_agents
    exp_args.dataset = args.dataset
    exp_args.vary_budget = False
    exp_args.n_processes = 1
    exp_args.n_runs = 1

    trade_mech_ls = [
        ["PAC", "ConvlAggr", "", 1],
        ["VCG", "ConvlAggr", "", 1],
    ]
    if args.regretnet_ckpt and os.path.isfile(args.regretnet_ckpt):
        trade_mech_ls.append(["RegretNet", "ConvlAggr", args.regretnet_ckpt, 1])
    if args.mfg_regretnet_ckpt and os.path.isfile(args.mfg_regretnet_ckpt):
        trade_mech_ls.append(["MFG-RegretNet", "ConvlAggr", args.mfg_regretnet_ckpt, 1])

    if exp_args.dataset == "NSL-KDD":
        train_data, test_data = load_nslkdd()
        # fl_args 等需与 experiments 中一致
    elif exp_args.dataset == "Bank" or exp_args.dataset == "Banking":
        train_data, test_data = load_bank()
    else:
        print("Unsupported dataset. Use --sample or set --dataset NSL-KDD|Bank")
        return 1

    clients = Clients()
    clients.dirs = "data/nslkdd/iid/" if "NSL" in exp_args.dataset else "data/bank/"
    clients.filename = "test_profiles_2mp.json"
    if not os.path.isfile(os.path.join(clients.dirs, clients.filename)):
        print("Client data not found. Use --sample to generate sample JSON.")
        return 1
    clients.load_json()

    from FL import Arguments, Logistic, ldp_fed_sgd, test
    fl_args = Arguments()
    fl_args.rounds = exp_args.n_rounds
    fl_args.device = DEVICE
    fl_args.rnd_step = exp_args.rnd_step
    # 根据数据集设置 input_size, output_size, shape
    if "NSL" in exp_args.dataset or "KDD" in exp_args.dataset:
        fl_args.input_size = 122
        fl_args.output_size = 5
        fl_args.shape = (-1, 122)
    else:
        fl_args.input_size = 48
        fl_args.output_size = 2
        fl_args.shape = (-1, 48)

    acc_mech_ls = acc_eval_mechs(trade_mech_ls, train_data, test_data, clients, fl_args, exp_args, run=0)
    rounds = [1] + [(r + 1) * exp_args.rnd_step for r in range(exp_args.n_rounds // exp_args.rnd_step)]
    method_names = []
    for t in trade_mech_ls:
        name = t[0]
        if t[1] != "ConvlAggr":
            name += "+" + t[1]
        method_names.append(name)
    methods = {name: list(acc) for name, acc in zip(method_names, acc_mech_ls)}
    data = {"rounds": rounds[: len(acc_mech_ls[0])], "methods": methods}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print("Wrote RQ4 accuracy JSON:", args.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())

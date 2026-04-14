#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2 论文级可扩展性基准：单次拍卖（batch=1 profile）计时、GPU 峰值显存、通信量估算、
堆叠时间（本地一步 | 聚合 | 拍卖求解）。

神经机制仅在对应该 N 的 checkpoint 存在时计时（自动扫描 result/*.pt 的 arch.n_agents）。

输出: run/privacy_paper/rq2/rq2_paper_data.json
作图: exp_rq/rq2_plot_paper_figures.py
"""
from __future__ import division, print_function

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from aggregation import aggr_batch
from baselines import csra_qms_batch, pac_batch, vcg_procurement_batch
from datasets_fl_benchmark import generate_privacy_paper_bids
from experiments import DEVICE, load_auc_model
from utils import allocs_instantiate_plosses


def _epoch_from_path(p):
    m = re.search(r"_(\d+)_checkpoint\.pt$", p)
    return int(m.group(1)) if m else 0


def discover_ckpts_by_n(glob_pat, model_type_key):
    """
    model_type_key: 'MFGRegretNet' or 'RegretNet' (exclude MFG in RegretNet search).
    Returns dict n_agents -> path (latest epoch per N).
    """
    by_n = {}
    for p in sorted(glob.glob(glob_pat), key=_epoch_from_path):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            mt = arch.get("model_type") or ""
            na = arch.get("n_agents")
            ni = arch.get("n_items")
            if na is None or ni != 1:
                continue
            if model_type_key == "MFGRegretNet" and mt != "MFGRegretNet":
                continue
            if model_type_key == "RegretNet" and mt == "MFGRegretNet":
                continue
            if model_type_key == "RegretNet" and mt not in ("", "RegretNet"):
                continue
            by_n[int(na)] = p
        except Exception:
            continue
    return by_n


def _run_auction_core(name, reports, budget, model, n_items, device):
    """Returns (plosses, payments, sizes) without aggr_batch."""
    budget = budget.view(-1, 1)
    n_agents = reports.shape[1]
    sizes = reports[:, :, -1].view(-1, n_agents)
    payments_out = None
    if model is not None:
        reports = reports.reshape(-1, n_agents, n_items + 2)
        allocs, payments_out = model((reports, budget))
        pbudgets = reports.view(-1, n_agents, n_items + 2)[:, :, -2]
        plosses, _ = allocs_instantiate_plosses(allocs, pbudgets)
        return plosses, payments_out, sizes
    if name == "PAC":
        plosses, payments_out = pac_batch(reports, budget)
    elif name == "VCG":
        plosses, payments_out = vcg_procurement_batch(reports, budget)
    elif name == "CSRA":
        plosses, payments_out = csra_qms_batch(reports, budget)
    elif name == "MFG-Pricing":
        from baselines.mfg_pricing import mfg_pricing_batch

        plosses, payments_out = mfg_pricing_batch(reports, budget)
    else:
        raise ValueError("unknown mech " + str(name))
    return plosses, payments_out, sizes


def time_auction_aggr_split(name, reports, budget, model, n_items, aggr_method, device, warmup, repeat):
    """
    返回 (t_auction_core, t_aggr) 的平均时间。
    t_auction_core: 拍卖求解时间
    t_aggr: 聚合时间
    """
    times_core = []
    times_aggr = []
    for i in range(warmup + repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        plosses, _, sz = _run_auction_core(name, reports, budget, model, n_items, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_core.append(t1 - t0)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        aggr_batch(plosses, sz, method=aggr_method)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        times_aggr.append(t3 - t2)
    
    # 去掉 warmup
    return float(np.mean(times_core[warmup:])), float(np.mean(times_aggr[warmup:]))


def time_local_fl_one_round(n_agents, device, hidden_dim=256):
    """N 个客户端各做 1 次小批量 forward+backward（MNIST 式 CNN 简化）。"""
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 10)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))

    model = Tiny().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(2):
        x = torch.randn(16, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(n_agents):
        x = torch.randn(16, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (16,), device=device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def time_grad_aggregate(n_agents, param_dim=50000, device="cpu"):
    """模拟服务器对 N 个梯度向量做加权求和。"""
    vecs = [torch.randn(param_dim, device=device) for _ in range(n_agents)]
    w = torch.softmax(torch.randn(n_agents, device=device), dim=0)
    
    # Warmup
    s = torch.zeros(param_dim, device=device)
    for i in range(n_agents):
        s = s + w[i] * vecs[i]
    
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    s = torch.zeros(param_dim, device=device)
    for i in range(n_agents):
        s = s + w[i] * vecs[i]
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def comm_bytes_estimate(n_agents, neural=False):
    """粗估每轮 float32：上报 bid (v,eps,size) + 下发 eps/weight/pay。"""
    # 上报：每个客户端上报 (v, epsilon, size) = 3 个 float32
    up = n_agents * 3 * 4
    # 下发：每个客户端收到 (epsilon_alloc, weight, payment, size) = 4 个 float32
    down = n_agents * 4 * 4
    # 神经网络额外开销（模型参数更新等）
    extra = n_agents * 64 * 4 if neural else 0
    return int(up + down + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default="10,50,100,200,400")
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out-dir", type=str, default="run/privacy_paper/rq2")
    ap.add_argument("--skip-large-n", action="store_true", help="skip N>200 for slow baselines")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="number of profiles per timing call (larger = more accurate, shows O(N^k) better)")
    args = ap.parse_args()

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    if args.skip_large_n:
        n_list = [n for n in n_list if n <= 200]

    os.makedirs(args.out_dir, exist_ok=True)
    device = DEVICE
    if isinstance(device, str):
        device = torch.device(device)
    n_items = 1

    # 查找所有 checkpoint
    mfg_by_n = discover_ckpts_by_n("result/mfg_regretnet_privacy_*_checkpoint.pt", "MFGRegretNet")
    # Also pick up N-specific MFG checkpoints (mfg_regretnet_nXX_*)
    for p in sorted(glob.glob("result/mfg_regretnet_n*_checkpoint.pt"), key=_epoch_from_path):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            if arch.get("model_type") == "MFGRegretNet" and arch.get("n_agents") and arch.get("n_items") == 1:
                na = int(arch["n_agents"])
                if na not in mfg_by_n:
                    mfg_by_n[na] = p
                else:
                    # prefer n-specific over privacy-generic
                    pass
        except Exception:
            continue

    reg_by_n = discover_ckpts_by_n("result/regretnet_privacy_pcost_*_checkpoint.pt", "RegretNet")
    # Also search other regretnet checkpoints
    for p in sorted(glob.glob("result/regretnet_privacy_*_checkpoint.pt"), key=_epoch_from_path):
        if "pcost" in os.path.basename(p):
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            if arch.get("model_type") != "MFGRegretNet" and arch.get("n_agents") and arch.get("n_items") == 1:
                na = int(arch["n_agents"])
                if na not in reg_by_n:
                    reg_by_n[na] = p
        except Exception:
            continue

    # Also find N-specific regretnet_pcost_nXX checkpoints
    for p in sorted(glob.glob("result/regretnet_privacy_pcost_n*_checkpoint.pt"), key=_epoch_from_path):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            if arch.get("n_agents") and arch.get("n_items") == 1:
                na = int(arch["n_agents"])
                if na not in reg_by_n or na == 10:
                    reg_by_n[na] = p
        except Exception:
            continue

    # DM-RegretNet by N
    dm_by_n = {}
    for p in sorted(glob.glob("result/dm_regretnet_privacy_pcost_n*_checkpoint.pt") +
                    glob.glob("result/dm_regretnet_privacy_pcost_*_checkpoint.pt"), key=_epoch_from_path):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            arch = d.get("arch") or {}
            if arch.get("n_agents") and arch.get("n_items") == 1:
                na = int(arch["n_agents"])
                dm_by_n[na] = p
        except Exception:
            continue

    print("=" * 60)
    print("RQ2 Scalability Benchmark")
    print("=" * 60)
    print(f"N list: {n_list}")
    print(f"Device: {device}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"\nMFG-RegretNet checkpoints found: {sorted(mfg_by_n.keys())}")
    print(f"RegretNet checkpoints found:     {sorted(reg_by_n.keys())}")
    print(f"DM-RegretNet checkpoints found:  {sorted(dm_by_n.keys())}")
    print("=" * 60)

    rows_time = []
    detail = {}

    baselines = [
        ("PAC", "PAC"),
        ("VCG", "VCG"),
        ("CSRA", "CSRA"),
        ("MFG-Pricing", "MFG-Pricing"),
    ]

    for N in n_list:
        print(f"\n>>> Running N = {N}")
        # 使用 batch_size 个 profiles 以增加工作量，获得更准确的时间测量
        rep = generate_privacy_paper_bids(N, n_items, args.batch_size, seed=42 + N).to(device)
        if rep.dim() == 2:
            rep = rep.unsqueeze(0)
        bud = args.budget * torch.ones(rep.shape[0], 1, device=device)

        key = str(N)
        detail[key] = {}

        # === 基线方法 ===
        for disp, mech_name in baselines:
            print(f"  {disp:15s} ... ", end="", flush=True)
            model = None
            tc, ta = time_auction_aggr_split(
                mech_name, rep, bud, model, n_items, "ConvlAggr", device, args.warmup, args.repeat
            )
            # 修正：Local training 在实际 FL 中是并行的，只计算单个客户端时间
            # 不应该乘以 N（那是串行的总时间）
            t_loc_single = time_local_fl_one_round(1, device)  # 单个客户端
            t_loc = t_loc_single  # 并行假设：总时间 = 单个客户端时间
            
            # Server aggregation：这是串行的，时间随 N 增长
            t_agg = time_grad_aggregate(N, param_dim=min(80000, 5000 + N * 100), device=device)
            
            mem = 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                p, _, sz = _run_auction_core(mech_name, rep, bud, None, n_items, device)
                aggr_batch(p, sz, method="ConvlAggr")
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024.0**3)
            
            cb = comm_bytes_estimate(N, neural=False)
            detail[key][disp] = {
                "t_auction_solve": tc,
                "t_aggr_fl_weights": ta,
                "t_local_train_proxy": t_loc,
                "t_server_grad_agg": t_agg,
                "peak_gpu_gb": mem,
                "comm_bytes_est": cb,
            }
            rows_time.append(
                {
                    "n_agents": N,
                    "mechanism": disp,
                    "mean_time_sec": tc + ta,
                    "t_auction_core": tc,
                    "t_mechanism_aggr": ta,
                }
            )
            print(f"t_total={tc+ta:.6f}s (auction={tc:.6f}s, aggr={ta:.6f}s)")

        # === 神经网络方法 ===
        for label, ckpt_map, arch_name in [
            ("Ours", mfg_by_n, "MFG"),
            ("RegretNet", reg_by_n, "RegretNet"),
            ("DM-RegretNet", dm_by_n, "DM-RegretNet"),
        ]:
            path = ckpt_map.get(N)
            if not path or not os.path.isfile(path):
                print(f"  {label:15s} ... SKIPPED (no checkpoint for N={N})")
                continue
            
            print(f"  {label:15s} ... ", end="", flush=True)
            model = load_auc_model(path).to(device)
            model.eval()
            tc, ta = time_auction_aggr_split(
                "PAC",  # name 只是占位符，实际会用 model
                rep,
                bud,
                model,
                n_items,
                "ConvlAggr",
                device,
                args.warmup,
                args.repeat,
            )

            t_loc_single = time_local_fl_one_round(1, device)  # 并行假设：单个客户端时间
            t_loc = t_loc_single
            t_agg = time_grad_aggregate(N, param_dim=min(80000, 5000 + N * 100), device=device)
            
            mem = 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                p, _, s = _run_auction_core("", rep, bud, model, n_items, device)
                aggr_batch(p, s, method="ConvlAggr")
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024.0**3)
            
            cb = comm_bytes_estimate(N, neural=True)
            detail[key][label] = {
                "t_auction_solve": tc,
                "t_aggr_fl_weights": ta,
                "t_local_train_proxy": t_loc,
                "t_server_grad_agg": t_agg,
                "peak_gpu_gb": mem,
                "comm_bytes_est": cb,
                "ckpt": path,
            }
            rows_time.append(
                {
                    "n_agents": N,
                    "mechanism": label,
                    "mean_time_sec": tc + ta,
                    "t_auction_core": tc,
                    "t_mechanism_aggr": ta,
                }
            )
            print(f"t_total={tc+ta:.6f}s (auction={tc:.6f}s, aggr={ta:.6f}s)")

    # === 保存结果 ===
    out = {
        "meta": {
            "note": f"Batch={args.batch_size} profiles per timing call. t_local = single-client proxy (parallel FL assumption). Server grad agg is serial O(N).",
            "n_list": n_list,
            "budget": args.budget,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "rq2_time_rows": rows_time,
        "per_n_detail": detail,
        "mfg_ckpts_used": mfg_by_n,
        "regretnet_ckpts_used": reg_by_n,
    }
    path_json = os.path.join(args.out_dir, "rq2_paper_data.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✓ Saved results to: {path_json}")
    print("=" * 60)
    print("\nSummary:")
    print(f"  Total mechanisms tested: {len(set(r['mechanism'] for r in rows_time))}")
    print(f"  Total data points: {len(rows_time)}")
    print(f"\nNext step: python exp_rq/rq2_plot_paper_figures.py --input {path_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

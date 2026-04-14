# -*- coding: utf-8 -*-
"""自动解析 RQ1 可用的 RegretNet checkpoint（排除 MFG-RegretNet）。"""
from __future__ import division, print_function

import glob
import os
import re


def _torch_load_meta(path):
    """Checkpoint load; PyTorch<2.0 has no weights_only kwarg."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_regretnet_ckpt(n_agents=10, n_items=1):
    """
    优先 result/regretnet_privacy_pcost_*_checkpoint.pt（隐私成本正确版）；
    否则 result/regretnet_privacy_*_checkpoint.pt；
    否则在 result/*_*_checkpoint.pt 中找 arch 为 RegretNet 且 n_agents/n_items 匹配的 .pt。
    """
    try:
        import torch
    except ImportError:
        return ""

    def _epoch(p):
        m = re.search(r"_(\d+)_checkpoint\.pt$", p)
        return int(m.group(1)) if m else 0

    # Prefer pcost version (correctly trained with privacy cost)
    cand_pcost = glob.glob("result/regretnet_privacy_pcost_*_checkpoint.pt")
    cand_pcost = sorted(cand_pcost, key=_epoch)
    if cand_pcost:
        p = cand_pcost[-1]
        if os.path.isfile(p):
            return p

    cand = glob.glob("result/regretnet_privacy_*_checkpoint.pt")
    cand = [c for c in cand if "pcost" not in c and "dm_" not in c]
    cand = sorted(cand, key=_epoch)
    if cand:
        p = cand[-1]
        if os.path.isfile(p):
            return p

    for p in sorted(glob.glob("result/*_*_checkpoint.pt")):
        bn = os.path.basename(p).lower()
        if "mfg_regretnet" in bn or "mfg-regret" in bn:
            continue
        try:
            d = _torch_load_meta(p)
            arch = d.get("arch") or {}
            if arch.get("model_type") == "MFGRegretNet":
                continue
            if arch.get("n_agents") == n_agents and arch.get("n_items") == n_items:
                return p
        except Exception:
            continue
    return ""


def resolve_mfg_regretnet_ckpt(n_agents=10, n_items=1):
    """result/mfg_regretnet_privacy_*_checkpoint.pt 中与 (n_agents, n_items) 匹配的最新 epoch。"""
    try:
        import torch
    except ImportError:
        return ""

    def _epoch(p):
        m = re.search(r"_(\d+)_checkpoint\.pt$", p)
        return int(m.group(1)) if m else 0

    cand = sorted(glob.glob("result/mfg_regretnet_privacy_*_checkpoint.pt"), key=_epoch)
    for p in reversed(cand):
        try:
            d = _torch_load_meta(p)
            arch = d.get("arch") or {}
            na, ni = arch.get("n_agents"), arch.get("n_items")
            if na is not None and int(na) != int(n_agents):
                continue
            if ni is not None and int(ni) != int(n_items):
                continue
            return p
        except Exception:
            continue
    return ""


def resolve_dm_regretnet_ckpt(n_agents=10, n_items=1):
    """
    DM-RegretNet：优先 result/dm_regretnet_privacy_pcost_*_checkpoint.pt（隐私成本正确版）；
    否则 result/dm_regretnet_privacy_*_checkpoint.pt。
    """
    try:
        import torch
    except ImportError:
        return ""

    def _epoch(p):
        m = re.search(r"_(\d+)_checkpoint\.pt$", p)
        return int(m.group(1)) if m else 0

    # Prefer pcost version
    cand_pcost = glob.glob("result/dm_regretnet_privacy_pcost_*_checkpoint.pt")
    cand_pcost = sorted(cand_pcost, key=_epoch)
    if cand_pcost:
        p = cand_pcost[-1]
        if os.path.isfile(p):
            return p

    cand = glob.glob("result/dm_regretnet_privacy_*_checkpoint.pt")
    cand = [c for c in cand if "pcost" not in c]
    cand = sorted(cand, key=_epoch)
    if cand:
        p = cand[-1]
        if os.path.isfile(p):
            return p

    for p in sorted(glob.glob("result/*_*_checkpoint.pt")):
        bn = os.path.basename(p).lower()
        if "dm_regretnet" not in bn and "dm-regret" not in bn:
            continue
        if "mfg" in bn:
            continue
        try:
            d = _torch_load_meta(p)
            arch = d.get("arch") or {}
            if arch.get("model_type") == "MFGRegretNet":
                continue
            if arch.get("n_agents") == n_agents and arch.get("n_items") == n_items:
                return p
        except Exception:
            continue
    return ""

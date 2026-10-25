"""Frozen MF-Net adapter. Its forward already returns epsilon, not a share."""
import torch
from torch import nn

from mf_reference import MFNet, KAPPA_RANGE
from p2_reference import kappa_feature


class MFAdapter(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.kind = "mf_p2"

    def forward(self, inputs):
        reports, budget, kappa = inputs
        dtype = next(self.network.parameters()).dtype
        return self.network(reports.to(dtype), budget.to(dtype),
                            kappa_feature(kappa.to(dtype), *KAPPA_RANGE))


def is_mf_checkpoint(saved):
    state = saved.get("state_dict", {})
    return saved.get("kind") in {"mf_p2", "MF-Net", "mf_net"} or (
        "encoder.0.weight" in state and "allocation_head.0.weight" in state)


def load_mf(saved):
    state = saved["state_dict"]
    config = saved.get("config", {})
    if config.get("population", True) is not True:
        raise ValueError("MF-Net-local is not the A1 MF-Net candidate")
    if tuple(config.get("kappa_range", KAPPA_RANGE)) != KAPPA_RANGE:
        raise ValueError("MF-Net kappa encoding differs from ln-v1")
    architecture = saved.get("architecture", config.get("architecture", {}))
    allowed = {"population", "width", "hidden", "depth", "pay_k"}
    if set(architecture) - allowed or architecture.get("population", True) is not True:
        raise ValueError("unsupported MF-Net architecture metadata")
    net = MFNet(**architecture)
    dtype = state["encoder.0.weight"].dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("unsupported MF-Net checkpoint dtype")
    net = net.to(dtype=dtype)
    net.load_state_dict(state, strict=True)
    return MFAdapter(net).eval()

"""Common contract execution, unchanged baseline economics, and learned heads."""
import numpy as np
import torch

from common import ROOT, isolated_rng
from contracts import settlement, balanced_target, MIN_EVENT_EPSILON
from models_scaled import MFGRegretNetScaled, MFGRegretNetNoMF, RegretNetScaled
from baselines import pac_batch, vcg_procurement_batch, csra_qms_batch
from baselines.mfg_pricing import mfg_pricing_batch
from baselines.lin2023.mechanism import solve_discrete_contracts

V8F = ROOT / "main_final2/revisions/method_improvement_20261015/runs/mfgV8F_seed42_ep200.pt"
OLD_MODELS = {
    "V8F": ("mfg", V8F),
    "RegretNet-old": ("regretnet_s", ROOT / "main_final2/revisions/b1_retrain_20261014/runs/regretnet_s_seed42_ep200.pt"),
    "DM-RegretNet-old": ("dm_s", ROOT / "main_final2/revisions/b1_retrain_20261014/runs/dm_s_seed42_ep150.pt"),
}


def make_model(kind):
    if kind in ("mfg", "mfg_nob"):
        cls = MFGRegretNetScaled if kind == "mfg" else MFGRegretNetNoMF
        model = cls(10, 1, hidden_layer_size=100, n_hidden_layers=3, activation="tanh", separate=True)
        model.glorot_init()
    else:
        model = RegretNetScaled(10, 1, activation="tanh", hidden_layer_size=100, n_hidden_layers=3,
                                p_activation="softmax", a_activation="deterministic" if kind == "dm_s" else "softmax",
                                separate=True, smoothing=.125, normalized_input=-1)
    return model


def load_model(path, kind=None):
    with isolated_rng(42):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        kind = kind or saved["kind"]
        model = make_model(kind)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.deter_train = False
        model.eval()
    return model


def execute(reports, exposure, payment):
    """Truthful execution; attacks must explicitly supply the true types."""
    executed, paid, _ = settlement(reports, exposure, payment)
    return executed, paid


def fill_uniform(cap, total):
    lo, hi = 0., float(cap.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.minimum(cap, mid).sum() < total:
            lo = mid
        else:
            hi = mid
    return np.minimum(cap, (lo + hi) / 2)


@torch.no_grad()
def allocate(arm, reports, budget, model_path=None, reference_path=None, return_offer=False):
    budgets = torch.full((len(reports), 1), float(budget))
    if arm == "Uniform-DP":
        exposure, _ = allocate("reference", reports, budget, reference_path or V8F)
        result = [fill_uniform(rep[:, 1].double().numpy(), float(e.sum())) for rep, e in zip(reports, exposure)]
        offered = torch.tensor(np.asarray(result))
        return torch.where(offered >= MIN_EVENT_EPSILON, offered, 0.), torch.zeros_like(exposure)
    if arm.startswith("Balanced-"):
        exposure, payment = balanced_target(reports, budgets, ceiling=float(arm.split("-")[1]))
    elif model_path is not None or arm in OLD_MODELS:
        kind = None
        if model_path is None:
            kind, model_path = OLD_MODELS[arm]
        model = load_model(model_path, kind)
        alloc, payment = model((reports, budgets))
        exposure = alloc.squeeze(-1) * reports[..., 1]
    elif arm in ("PAC", "VCG", "PAC-capped-legacy", "VCG-capped-legacy", "CSRA", "MFG-Pricing"):
        legacy_cap = arm.endswith("-capped-legacy")
        native_arm = arm.split("-")[0] if legacy_cap else arm
        fn = {"PAC": pac_batch, "VCG": vcg_procurement_batch, "CSRA": csra_qms_batch,
              "MFG-Pricing": mfg_pricing_batch}[native_arm]
        exposure, payment = fn(reports, budgets)
        if legacy_cap:
            # Retain the stronger historical performance comparator, labelled
            # explicitly; this adapted rule has no cap-truthfulness claim.
            exposure = torch.minimum(exposure, reports[..., 1])
    elif arm == "Posted-price":
        price = float(np.interp(budget, [5, 10, 15, 20, 25, 110], [.44, .63, .77, .89, 1, 1]))
        accept = (reports[..., 0] <= price).float()
        cap = reports[..., 1] * accept
        fraction = (budgets / (price * cap.sum(1, keepdim=True)).clamp_min(1e-12)).clamp(max=1)
        exposure = cap * fraction
        payment = price * exposure
    elif arm == "Lin":
        menu = solve_discrete_contracts(np.linspace(.1, 1, 10), np.full(10, .1))
        scale = min(5 / menu.epsilon.max(), budget / (10 * menu.payment.max()))
        me = torch.tensor(menu.epsilon * scale)
        mp = torch.tensor(menu.payment * scale)
        score = mp - reports[..., 0, None] * me
        valid = (me <= reports[..., 1, None]) & (me >= 1e-3)
        score = torch.where(valid, score, -torch.inf)
        best, choice = score.max(-1)
        exposure = torch.where(best > 0, me[choice], 0.)
        payment = torch.where(best > 0, mp[choice], 0.)
    else:
        raise ValueError(f"unknown mechanism {arm}")
    if not torch.isfinite(exposure).all() or not torch.isfinite(payment).all():
        raise ValueError("nonfinite proposed contract")
    if (payment < -1e-9).any() or (payment.sum(1) > float(budget) + 2e-5).any():
        raise ValueError("contract is not budget feasible")
    return (exposure, payment) if return_offer else execute(reports, exposure, payment)

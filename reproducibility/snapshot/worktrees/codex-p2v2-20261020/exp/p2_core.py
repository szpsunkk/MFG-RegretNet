"""P2 inputs, fixed-head mechanisms, reference allocations, and executed attacks.

Frozen p2_reference.py is deliberately unchanged. The only objective amendment
is the empty-release branch, documented before any experiment.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import ROOT, isolated_rng
from contracts import MIN_EVENT_EPSILON, settlement, utility
from mechanisms import V8F, allocate, make_model, load_model, fill_uniform
from regretnet import budget_projection_privacy_paper
import p2_reference as P

FAMILIES = ("mfg_p2", "mfg_nob_p2", "regretnet_s_p2")
R2 = ROOT / "run/tdsc_revision/codex_rq4_repair_20261016/round2"
REFERENCES = {
    "V8F": (V8F, "mfg"),
    "RegretNet-r2": (R2 / "mechanisms/regretnet_s_balance0.5/model.pt", "regretnet_s"),
    "No-population-r2": (R2 / "mechanisms/mfg_nob_balance0/model.pt", "mfg_nob"),
    "DM-RegretNet": (ROOT / "main_final2/revisions/b1_retrain_20261014/runs/dm_s_seed42_ep150.pt", "dm_s"),
}
STAGE0_ARMS = ("CSRA", "PAC", "Oracle-k0", "Aligned@PAC", "Aligned@CSRA",
               "Oracle-khat", "Oracle-k3", "Oracle-k30", "Oracle-k0-datasize")


class AllocationInfeasible(ValueError):
    """A declared statistical control cannot satisfy its matching constraints."""


class P2Head(nn.Module):
    """Append one scalar to both original input heads; retain all output rules."""
    def __init__(self, kind, kappa_range=(.003, 300.)):
        super().__init__()
        if kind not in FAMILIES:
            raise ValueError(kind)
        self.kind = kind
        self.kappa_range = tuple(kappa_range)
        self.network = make_model(kind[:-3])
        net = self.network
        for head in (net.allocation_head, net.payment_head):
            first = head[0]
            if not isinstance(first, nn.Linear) or first.in_features != net.input_size:
                raise AssertionError("original separate input-head architecture changed")
            head[0] = type(first)(first.in_features + 1, first.out_features, bias=first.bias is not None)
        net.input_size += 1
        if kind.startswith("mfg"):
            net.glorot_init()
        self.deter_train = False

    def forward(self, inputs):
        reports, budget, kappa = inputs
        net = self.network
        r = reports / net.in_scale
        parts = [r.flatten(1)]
        if self.kind.startswith("mfg"):
            field = r.mean(1, keepdim=True).expand_as(r)
            if self.kind == "mfg_nob_p2":
                field = torch.zeros_like(field)
            parts.append(field.flatten(1))
        parts += [budget.reshape(-1, 1) / net.b_scale,
                  P.kappa_feature(kappa.reshape(-1, 1), *self.kappa_range)]
        x = net.nn_model(torch.cat(parts, dim=1))
        allocation = net.allocation_head(x)
        if self.kind.startswith("mfg"):
            payment = budget_projection_privacy_paper(net.payment_head(x) * budget, budget)
        else:
            payment = net.payment_head(x)[:, :-1] * budget
        return allocation, payment


def load_any(path, kind=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    from a1_mf import is_mf_checkpoint, load_mf
    if kind == "mf_p2" or is_mf_checkpoint(saved):
        with isolated_rng(42):
            return load_mf(saved)
    actual_kind = kind or saved.get("kind")
    if actual_kind in FAMILIES:
        with isolated_rng(42):
            model = P2Head(actual_kind, saved["config"]["kappa_range"])
            model.load_state_dict(saved["state_dict"], strict=True)
            model.eval()
        return model
    return load_model(path, actual_kind)


def raw(model, reports, budget, kappa):
    from a1_mf import MFAdapter
    if isinstance(model, MFAdapter):
        return model((reports, budget, kappa))
    inputs = (reports, budget, kappa) if isinstance(model, P2Head) else (reports, budget)
    a, p = model(inputs)
    return a.squeeze(-1) * reports[..., 1], p


def training_term(eps, reports, kappa):
    """Exact reference surrogate when nonempty, explicit penalised empty branch."""
    original = P.p2_training_term(eps, reports, kappa)
    share = reports[..., 2] / reports[..., 2].sum(-1, keepdim=True)
    denominator = P.j_kappa_torch(reports[..., 1], share, kappa).detach()
    empty = 1 + 1 / (MIN_EVENT_EPSILON ** 2 * denominator)
    return torch.where(eps.sum(-1) > 0, original, empty)


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def exact_metrics(eps, reports, kappa):
    e = np.asarray(eps, float)
    r = np.asarray(reports, float)
    share = r[..., 2] / r[..., 2].sum(-1, keepdims=True)
    ks = np.broadcast_to(np.asarray(kappa).reshape(-1), (len(e),))
    js = np.array([P.j_kappa(x, s, k) for x, s, k in zip(e, share, ks)])
    tv = [P.misalignment(x, s)[0] for x, s in zip(e, share)]
    return {"j_kappa": float(js.mean()) if np.isfinite(js).all() else None,
            "j_infinite": bool(np.isinf(js).any()), "tv": float(np.mean(tv)),
            "epsilon_per_client_round": float(e.mean()),
            "participants_per_round": float((e > 0).sum(-1).mean()),
            "zero_release_round_fraction": float((e.sum(-1) == 0).mean())}


@torch.no_grad()
def allocation(arm, reports, budget, kappa, checkpoint=None, reference=None, target=None, model=None, raw_sizes=None):
    """Return proposed and settled contracts, retaining float64 cost equality."""
    b = torch.full((len(reports), 1), float(budget), dtype=reports.dtype)
    if arm in ("Uniform-DP", "Uniform-DP@mf_p2"):
        if reference is None:
            raise ValueError("Uniform-DP requires the selected P2 reference")
        ref = allocation("selected-reference", reports, budget, kappa, checkpoint=reference, raw_sizes=raw_sizes)
        e = torch.tensor(np.asarray([fill_uniform(row[:, 1].double().numpy(), float(x.sum()))
                                     for row, x in zip(reports, ref["executed"])]), dtype=torch.float64)
        e = torch.where(e >= MIN_EVENT_EPSILON, e, 0.)
        delta = (e.sum(-1) - ref["executed"].double().sum(-1)).abs()
        if (delta > 1e-6 + 1e-7 * e.sum(-1)).any():
            raise AllocationInfeasible("Uniform-DP exact per-round match infeasible with minimum event contract")
        return {"proposed": e, "offered_payment": torch.zeros_like(e), "executed": e,
                "payment": torch.zeros_like(e), "voluntary": True}
    if checkpoint is not None or arm in REFERENCES:
        if checkpoint is None:
            checkpoint, kind = REFERENCES[arm]
            model = model or load_any(checkpoint, kind)
        else:
            model = model or load_any(checkpoint)
        from a1_mf import MFAdapter
        model_reports = reports
        if isinstance(model, MFAdapter):
            if raw_sizes is None:
                raise ValueError("MF-Net FL allocation requires verified raw partition sizes")
            sizes = torch.as_tensor(raw_sizes, dtype=reports.dtype, device=reports.device)
            if sizes.shape != (reports.shape[1],) or not torch.isfinite(sizes).all() or (sizes < 1).any():
                raise ValueError("invalid verified raw sizes")
            model_reports = reports.clone()
            model_reports[..., 2] = sizes
        e, p = raw(model, model_reports, b, torch.full((len(reports), 1), float(kappa)))
    elif arm.startswith("Oracle-") or arm.startswith("Aligned@"):
        base_total = None
        if arm.startswith("Aligned@"):
            base, _ = allocate(arm.split("@", 1)[1], reports, budget)
            base_total = base.double().sum(-1).numpy()
        rows, pays = [], []
        for idx, row in enumerate(reports.double().numpy()):
            value, cap, size = row.T
            share = size / size.sum()
            if base_total is not None or (arm == "Oracle-P2" and target is not None):
                total = base_total[idx] if base_total is not None else 10 * target
                er, pr = P.aligned_at_total(cap, value, share, float(total), budget)
            else:
                er, pr, _ = P.p2_oracle(cap, value, share, budget, float(kappa))
            rows.append(er)
            pays.append(pr)
        e, p = torch.tensor(np.asarray(rows)), torch.tensor(np.asarray(pays))
    else:
        e, p = allocate(arm, reports, budget, return_offer=True)
    if not torch.isfinite(e).all() or not torch.isfinite(p).all():
        raise ValueError("nonfinite offer")
    if (p < -1e-9).any() or (p.sum(-1) > budget + 2e-5).any():
        raise ValueError("budget infeasible offer")
    # Reference cost payments were computed in float64, so also multiply the
    # real cost in float64 at the gate; never silently promote a float32 product.
    truth = reports.to(e.dtype)
    executed, paid, feasible = settlement(truth, e, p)
    if (arm.startswith("Oracle-") or arm.startswith("Aligned@")) and ((e > 0) & ~feasible).any():
        raise AssertionError("cost-only reference offer rejected by unchanged settlement")
    return {"proposed": e, "offered_payment": p, "executed": executed, "payment": paid,
            "voluntary": False}


def deviation_utility(model, truthful, budget, kappa, misreports):
    batch, n, features = truthful.shape
    profiles = truthful[:, None].expand(batch, n, n, features).clone()
    ix = torch.arange(n, device=truthful.device)
    profiles[:, ix, ix] = misreports
    e, p = raw(model, profiles.reshape(-1, n, features), budget.repeat_interleave(n, 0),
               kappa.repeat_interleave(n, 0))
    e, p = e.view(batch, n, n)[:, ix, ix], p.view(batch, n, n)[:, ix, ix]
    return utility(truthful, e, p)


def search(model, truthful, budget, kappa, strong=False, sizes=True):
    """Unilateral truth-inclusive grid + PGA; all true datasets stay unchanged."""
    batch, n, _ = truthful.shape
    kwargs = {"device": truthful.device, "dtype": truthful.dtype}
    vg = torch.linspace(0, 1, 5 if strong else 3, **kwargs)
    eg = torch.tensor([0., 1e-6, .01, .05, .1, .2, .4, .6, .8, 1.] if strong else [0., .0001, .1, .5, 1.], **kwargs)
    dg = torch.tensor([1., .75, .5, .25, 0.] if sizes else [1.], **kwargs)
    grid = torch.cartesian_prod(vg, eg, dg)
    with torch.no_grad():
        best = truthful.clone()
        best_u = deviation_utility(model, truthful, budget, kappa, best)
        for part in grid.split(5):
            count = len(part)
            candidate = truthful[:, None].expand(batch, count, n, 3).clone()
            candidate[..., 0] = part[None, :, None, 0]
            candidate[..., 1] = truthful[:, None, :, 1] * part[None, :, None, 1]
            candidate[..., 2] = (truthful[:, None, :, 2] * part[None, :, None, 2]).clamp_min(10)
            u = deviation_utility(model, truthful.repeat_interleave(count, 0), budget.repeat_interleave(count, 0),
                                  kappa.repeat_interleave(count, 0), candidate.reshape(-1, n, 3))
            values, indices = u.view(batch, count, n).max(1)
            chosen = candidate[torch.arange(batch, device=truthful.device)[:, None], indices,
                               torch.arange(n, device=truthful.device)[None]]
            best = torch.where((values > best_u)[..., None], chosen, best)
            best_u = torch.maximum(values, best_u)
    candidate = best.detach().clone().requires_grad_(True)
    for _ in range(25 if strong else 8):
        u = deviation_utility(model, truthful, budget, kappa, candidate)
        grad, = torch.autograd.grad(u.sum(), candidate)
        with torch.no_grad():
            best = torch.where((u > best_u)[..., None], candidate, best)
            best_u = torch.maximum(u, best_u)
            candidate[..., 0].add_(.03 * grad[..., 0]).clamp_(0, 1)
            candidate[..., 1].add_(.03 * grad[..., 1]).clamp_(min=0)
            candidate[..., 1].copy_(torch.minimum(candidate[..., 1], truthful[..., 1]))
            if sizes:
                candidate[..., 2].add_(.03 * grad[..., 2]).clamp_(min=10)
                candidate[..., 2].copy_(torch.minimum(candidate[..., 2], truthful[..., 2]))
            else:
                candidate[..., 2].copy_(truthful[..., 2])
    with torch.no_grad():
        u = deviation_utility(model, truthful, budget, kappa, candidate)
        best = torch.where((u > best_u)[..., None], candidate, best)
    return best.detach()

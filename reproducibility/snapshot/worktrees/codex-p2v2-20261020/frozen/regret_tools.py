"""Inner maximization and certified regret for the 20e327e auction networks.

Two tools, both used without modifying the original code files:

* grid_init_misreports / install_grid_pga: before the original R-step PGA of
  utils.optimize_misreports, every client's misreport is initialized at the
  best point of a grid over (v', eps', d') in its feasible report set
  [0, vbar] x [0, eps_i] x [1, d_i]. The grid always contains the truthful
  report and the cap floor eps' = 0, which the original PGA cannot reach.

* certified_regret: an upper bound on the exact ex-post regret of every client,
  by interval bound propagation (IBP) through the MFG-RegretNet heads and
  branch-and-bound over boxes of the client's own report. It returns the best
  utility found (a lower bound on the maximum) and a certified upper bound.

Utility model (paper, c(v, eps) = v * eps, n_items = 1):
    u_i = pbar_i - v_i * min(z_i * eps'_i, eps'_i)
with v_i the TRUE valuation, z_i the allocation and eps'_i the reported cap.
"""
import math

import torch

import regretnet as rn
from utils import tiled_misreport_util, optimize_misreports as _orig_optimize_misreports


# ---------------------------------------------------------------------------
# Grid-initialized PGA (training and evaluation)
# ---------------------------------------------------------------------------
def candidate_grid(gv, ge, gd, device, dtype=torch.float32):
    """Rows (v', eps_frac, size_frac); eps' = eps_frac * eps_i, d' = size_frac * d_i."""
    vg = torch.linspace(0.0, 1.0, gv, device=device, dtype=dtype)
    eg = torch.linspace(0.0, 1.0, ge, device=device, dtype=dtype)
    dg = torch.ones(1, device=device, dtype=dtype) if gd <= 1 else torch.linspace(0.0, 1.0, gd, device=device, dtype=dtype)
    return torch.cartesian_prod(vg, eg, dg)


@torch.no_grad()
def grid_init_misreports(model, reports, budget, val_type, cost_from_plosses,
                         gv=5, ge=9, gd=1, instantiation=False, chunk=64):
    """Best grid misreport per (profile, client); the truthful report competes too."""
    B_, N, F = reports.shape
    cand = candidate_grid(gv, ge, gd, reports.device, reports.dtype)
    G = cand.shape[0]
    best_u = tiled_misreport_util(reports, reports, model, budget, val_type,
                                  instantiation=instantiation, cost_from_plosses=cost_from_plosses)
    best = reports.clone()
    for s in range(0, G, chunk):
        c = cand[s:s + chunk]
        g = c.shape[0]
        mis = reports.unsqueeze(1).repeat(1, g, 1, 1)
        mis[:, :, :, :-2] = c[None, :, None, 0:1]
        mis[:, :, :, -2] = c[None, :, None, 1] * reports[:, None, :, -2]
        size = reports[:, None, :, -1]
        mis[:, :, :, -1] = torch.clamp(1.0 + c[None, :, None, 2] * (size - 1.0), min=1.0)
        flat = mis.reshape(B_ * g, N, F)
        u = tiled_misreport_util(flat, reports.repeat_interleave(g, 0), model,
                                 budget.repeat_interleave(g, 0), val_type.repeat_interleave(g, 0),
                                 instantiation=instantiation, cost_from_plosses=cost_from_plosses)
        u = u.view(B_, g, N)
        u_best, idx = u.max(dim=1)
        better = u_best > best_u
        pick = mis[torch.arange(B_, device=reports.device)[:, None], idx,
                   torch.arange(N, device=reports.device)[None, :], :]
        best = torch.where(better[:, :, None], pick, best)
        best_u = torch.maximum(best_u, u_best)
    return best


def pga_box(model, current_reports, current_misreports, budget, val_type, misreport_iter, lr,
            instantiation=False, cost_from_plosses=False, vbar=1.0):
    """utils.optimize_misreports with the report box [0, vbar] x [0, eps] x [1, d]."""
    current_misreports.requires_grad_(True)
    true_eps = current_reports[:, :, -2]
    true_size = current_reports[:, :, -1]
    for _ in range(misreport_iter):
        model.zero_grad()
        u = tiled_misreport_util(current_misreports, current_reports, model, budget, val_type,
                                 instantiation=instantiation, cost_from_plosses=cost_from_plosses)
        (g,) = torch.autograd.grad(u.sum(), current_misreports)
        g = torch.nan_to_num(g)
        with torch.no_grad():
            current_misreports += lr * g
            current_misreports[:, :, :-2].clamp_(0.0, vbar)
            current_misreports[:, :, -2] = torch.minimum(current_misreports[:, :, -2].clamp_min(0.0), true_eps)
            current_misreports[:, :, -1] = torch.minimum(current_misreports[:, :, -1].clamp_min(1.0), true_size)
    return current_misreports


def install_grid_pga(gv=5, ge=9, gd=1, vbar=1.0):
    """Patch regretnet.optimize_misreports (used by train_loop/test_loop)."""
    def patched(model, current_reports, current_misreports, budget, val_type,
                misreport_iter=10, lr=1e-1, train=True, instantiation=False, cost_from_plosses=False):
        init = grid_init_misreports(model, current_reports, budget, val_type, cost_from_plosses,
                                    gv=gv, ge=ge, gd=gd, instantiation=instantiation)
        with torch.no_grad():
            current_misreports.copy_(init)
        return pga_box(model, current_reports, current_misreports, budget, val_type,
                       misreport_iter, lr, instantiation=instantiation,
                       cost_from_plosses=cost_from_plosses, vbar=vbar)
    rn.optimize_misreports = patched
    return patched


# ---------------------------------------------------------------------------
# Exact utilities and IBP bounds for MFGRegretNet (n_items = 1)
# ---------------------------------------------------------------------------
def _mlp_layers(seq):
    """Linear/Tanh pairs and the final Linear of a separate MFGRegretNet head."""
    mods = [m for m in seq if isinstance(m, (torch.nn.Linear, torch.nn.Tanh, torch.nn.ReLU))]
    return mods


def _ibp(mods, lo, hi):
    for m in mods:
        if isinstance(m, torch.nn.Linear):
            c, r = (lo + hi) / 2, (hi - lo) / 2
            c = c @ m.weight.t() + m.bias
            r = r @ m.weight.abs().t()
            lo, hi = c - r, c + r
        elif isinstance(m, torch.nn.Tanh):
            lo, hi = torch.tanh(lo), torch.tanh(hi)
        else:
            lo, hi = torch.relu(lo), torch.relu(hi)
    return lo, hi


def _forward(mods, x):
    for m in mods:
        x = m(x)
    return x


class MFGHeads:
    """Exact and interval evaluation of a separate=True MFGRegretNet."""

    def __init__(self, model):
        assert model.separate and model.n_items == 1
        self.N = model.n_agents
        self.alloc = _mlp_layers(model.allocation_head)
        self.pay = _mlp_layers(model.payment_head)
        dev = next(model.parameters()).device
        sc = getattr(model, "in_scale", None)
        self.sc = sc.to(dev) if sc is not None else torch.ones(3, device=dev)
        self.bs = float(getattr(model, "b_scale", 1.0))

    def inputs(self, reports, budget):
        n = reports.shape[1]
        r = reports / self.sc
        mfg = r.mean(dim=1, keepdim=True).repeat(1, n, 1)
        return torch.cat([r.reshape(r.shape[0], -1), mfg.reshape(r.shape[0], -1),
                          budget.reshape(-1, 1) / self.bs], dim=1)

    def utility_exact(self, reports, budget, i, v_true):
        """u_i at the given (deviated) profiles. reports (P, N, 3), budget (P, 1), v_true (P,)."""
        x = self.inputs(reports, budget)
        a = _forward(self.alloc, x).view(-1, self.N, 2)
        z = torch.softmax(a, dim=2)[:, i, 0]
        p = torch.sigmoid(_forward(self.pay, x)) * budget.reshape(-1, 1)
        tot = p.sum(1).clamp(min=1e-12)
        pbar = p[:, i] * torch.clamp(budget.reshape(-1) / tot, max=1.0)
        eps = reports[:, i, 1]
        return pbar - v_true * torch.minimum(z * eps, eps)

    def utility_upper(self, reports, budget, i, v_true, box_lo, box_hi):
        """Upper bound of u_i over own reports in [box_lo, box_hi] (each (P, 3)), others fixed."""
        P, N, _ = reports.shape
        rs = reports / self.sc
        blo, bhi = box_lo / self.sc, box_hi / self.sc
        rlo, rhi = rs.clone(), rs.clone()
        rlo[:, i, :], rhi[:, i, :] = blo, bhi
        s_other = rs.sum(1) - rs[:, i, :]
        m_lo, m_hi = (s_other + blo) / N, (s_other + bhi) / N
        b = budget.reshape(-1, 1)
        x_lo = torch.cat([rlo.reshape(P, -1), m_lo.repeat(1, N), b / self.bs], dim=1)
        x_hi = torch.cat([rhi.reshape(P, -1), m_hi.repeat(1, N), b / self.bs], dim=1)
        a_lo, a_hi = _ibp(self.alloc, x_lo, x_hi)
        a_lo, a_hi = a_lo.view(P, N, 2), a_hi.view(P, N, 2)
        # z_i = 1 / (1 + exp(l_dummy - l_item)) is increasing in l_item, decreasing in l_dummy
        z_lo = torch.sigmoid(a_lo[:, i, 0] - a_hi[:, i, 1])
        g_lo, g_hi = _ibp(self.pay, x_lo, x_hi)
        p_lo, p_hi = torch.sigmoid(g_lo) * b, torch.sigmoid(g_hi) * b
        scale_hi = torch.clamp(b.reshape(-1) / p_lo.sum(1).clamp(min=1e-12), max=1.0)
        pbar_hi = p_hi[:, i] * scale_hi
        eps_lo = box_lo[:, 1]
        return pbar_hi - v_true * z_lo * eps_lo


@torch.no_grad()
def certified_regret(model, reports, budget, vbar=1.0, init_split=(8, 8, 2), tol=1e-3,
                     max_rounds=40, max_boxes=4_000_000):
    """Per-client lower/upper bounds on the exact ex-post regret (MFGRegretNet, n_items=1).

    Returns dict with tensors (P, N): truthful utility, best found (lower bound on
    max), certified upper bound on max, regret_lb, regret_ub, and the number of
    unresolved boxes (0 means the bound closed to within tol).
    """
    heads = MFGHeads(model)
    P, N, _ = reports.shape
    dev = reports.device
    v_true_all = reports[:, :, 0]
    u_truth = torch.stack([heads.utility_exact(reports, budget, i, v_true_all[:, i]) for i in range(N)], 1)
    lb_out = torch.zeros(P, N, device=dev)
    ub_out = torch.zeros(P, N, device=dev)
    open_out = torch.zeros(P, N, dtype=torch.long, device=dev)
    gv, ge, gd = init_split
    for i in range(N):
        eps_i, d_i = reports[:, i, 1], reports[:, i, 2]
        # initial boxes in normalized coordinates (a, b, c) in [0,1]^3:
        # v' = a * vbar, eps' = b * eps_i, d' = 1 + c * (d_i - 1)
        ea = torch.linspace(0, 1, gv + 1, device=dev)
        eb = torch.linspace(0, 1, ge + 1, device=dev)
        ec = torch.linspace(0, 1, gd + 1, device=dev)
        lo = torch.cartesian_prod(ea[:-1], eb[:-1], ec[:-1])
        hi = torch.cartesian_prod(ea[1:], eb[1:], ec[1:])
        K = lo.shape[0]
        prof = torch.arange(P, device=dev).repeat_interleave(K)
        blo, bhi = lo.repeat(P, 1), hi.repeat(P, 1)
        best = u_truth[:, i].clone()
        for rnd in range(max_rounds):
            M = prof.shape[0]
            if M == 0:
                break
            def to_abs(t):
                return torch.stack([t[:, 0] * vbar, t[:, 1] * eps_i[prof],
                                    1.0 + t[:, 2] * (d_i[prof] - 1.0)], dim=1)
            rep = reports[prof]
            bud = budget.reshape(-1, 1)[prof]
            vt = v_true_all[prof, i]
            ctr = to_abs((blo + bhi) / 2)
            rc = rep.clone()
            rc[:, i, :] = ctr
            uc = heads.utility_exact(rc, bud, i, vt)
            best.scatter_reduce_(0, prof, uc, reduce="amax")
            ub = heads.utility_upper(rep, bud, i, vt, to_abs(blo), to_abs(bhi))
            keep = ub > best[prof] + tol
            prof, blo, bhi, ub = prof[keep], blo[keep], bhi[keep], ub[keep]
            if prof.shape[0] == 0:
                break
            if rnd == max_rounds - 1 or 2 * prof.shape[0] > max_boxes:
                break
            # split each remaining box along its widest normalized side
            w = bhi - blo
            ax = w.argmax(1)
            mid = (blo.gather(1, ax[:, None]) + bhi.gather(1, ax[:, None])) / 2
            lo2, hi1 = blo.clone(), bhi.clone()
            hi1.scatter_(1, ax[:, None], mid)
            lo2.scatter_(1, ax[:, None], mid)
            blo, bhi = torch.cat([blo, lo2]), torch.cat([hi1, bhi])
            prof = torch.cat([prof, prof])
        # pruned boxes satisfy ub <= best + tol, so best + tol bounds them
        ub_i = best + tol
        if prof.shape[0] > 0:
            ub_i.scatter_reduce_(0, prof, ub, reduce="amax")
            open_out[:, i] = torch.bincount(prof, minlength=P)
        lb_out[:, i], ub_out[:, i] = best, ub_i
    return {"u_truth": u_truth, "max_lb": lb_out, "max_ub": ub_out,
            "regret_lb": (lb_out - u_truth).clamp_min(0), "regret_ub": (ub_out - u_truth).clamp_min(0),
            "open_boxes": open_out}


def hoeffding_term(urange, n, delta, n_clients=1):
    return urange * math.sqrt(math.log(n_clients / delta) / (2 * n))


def bernstein_term(sample_var, urange, n, delta, n_clients=1):
    """Maurer-Pontil (2009, Thm 4) for variables in [0, urange], union bound over clients."""
    l = math.log(2 * n_clients / delta)
    return math.sqrt(2 * sample_var * l / n) + 7 * urange * l / (3 * (n - 1))

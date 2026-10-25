"""Reference implementation for the large-population campaign (protocol ln-v1, 2026-10-19).

MF-Net is a per-client mean-field mechanism: client i's allocation and payment depend on its own report, on
population statistics that are averages over all reports, and on public context (B/N, kappa). Its parameter
count does not depend on N, so one trained network serves every population size. Codex imports or copies these
functions unchanged; the review recomputes outcomes, settlements and the mean-field-limit quantities with this
file. Pure functions apart from torch modules: no file I/O, no global random state.

Notation as in the design file REVISION_PLAN_OBJECTIVE_MFG_LARGEN_20261019.md, Sec. 4 and Appendix A.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "p2_mechanism_design_20261018" / "exp"))
import p2_reference as P2  # noqa: E402  (J_kappa, kappa feature, oracle; frozen with protocol p2-v1)

V_MAX, EPS_MAX = 1.0, 5.0
LOG_D = math.log1p(1e5)     # size feature log1p(d) / LOG_D
D_SCALE = 1e4               # population mean of raw sizes enters as mean(d) / D_SCALE
REL_CLIP = 50.0             # relative size d_i / mean(d) is clipped to [0, REL_CLIP] and divided by 10
BN_SCALE = 10.0             # per-capita budget feature (B/N) / BN_SCALE
PAY_K = 10.0                # raw payment = PAY_K * (B/N) * sigmoid(.)
MIN_EVENT_EPSILON = P2.MIN_EVENT_EPSILON


# ---------------------------------------------------------------- mechanism

def own_features(reports):
    """reports (..., 3) = (valuation, cap, size) -> (..., 3) scaled features."""
    v, cap, d = reports.unbind(-1)
    return torch.stack([v / V_MAX, cap / EPS_MAX, torch.log1p(d.clamp_min(0)) / LOG_D], -1)


def _mlp(inp, hidden, out, depth):
    layers, width = [], inp
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.Tanh()]
        width = hidden
    return nn.Sequential(*layers, nn.Linear(width, out))


class MFNet(nn.Module):
    """Per-client mean-field mechanism.

    population=True : client i reads [enc(r_i), mean_j enc(r_j), mean_j f(r_j), mean_j d_j / D_SCALE,
                      rel_i = clip(d_i / mean_j d_j) / 10, (B/N)/BN_SCALE, kappa feature].
    population=False: the population block (everything except enc(r_i) and the public context) is zero, so
                      the raw outputs of client i depend only on its own report and the public context; the
                      clients stay coupled only through the budget projection.
    """

    def __init__(self, population=True, width=64, hidden=128, depth=3, pay_k=PAY_K):
        super().__init__()
        self.population, self.width, self.pay_k = population, width, pay_k
        self.encoder = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        inp = width + (width + 5) + 2
        self.allocation_head = _mlp(inp, hidden, 2, depth)
        self.payment_head = _mlp(inp, hidden, 1, depth)

    # -- statistics -------------------------------------------------------------------------------------
    def statistics(self, reports, mask=None):
        """Population statistics s = (mean enc, mean features, mean d) over the active clients."""
        if mask is None:
            mask = torch.ones(reports.shape[:2], dtype=torch.bool, device=reports.device)
        safe = torch.where(mask[..., None], reports, torch.ones_like(reports))
        h = self.encoder(own_features(safe))
        w = mask[..., None].to(h.dtype)
        count = w.sum(1).clamp_min(1.0)
        return dict(mean_enc=(h * w).sum(1) / count, mean_feat=(own_features(safe) * w).sum(1) / count,
                    mean_d=(safe[..., 2:3] * w).sum(1) / count, count=count, enc=h, mask=mask)

    def head_inputs(self, reports, budget, kappa_feat, stats):
        n = reports.shape[1]
        enc = stats["enc"] if "enc" in stats else self.encoder(own_features(reports))
        mean_d = stats["mean_d"].clamp_min(1e-8)
        rel = (reports[..., 2] / mean_d).clamp(0.0, REL_CLIP)[..., None] / 10.0
        pop = torch.cat([stats["mean_enc"][:, None].expand(-1, n, -1),
                         stats["mean_feat"][:, None].expand(-1, n, -1),
                         (stats["mean_d"] / D_SCALE)[:, None].expand(-1, n, -1), rel], -1)
        if not self.population:
            pop = torch.zeros_like(pop)
        per_capita = budget.reshape(-1, 1) / stats["count"]
        ctx = torch.cat([per_capita / BN_SCALE, kappa_feat.reshape(-1, 1)], -1)[:, None].expand(-1, n, -1)
        return torch.cat([enc, pop, ctx], -1), per_capita

    def raw(self, reports, budget, kappa_feat, mask=None, stats=None):
        """Allocation share z in [0,1] and raw payment before the budget projection, shape (batch, N)."""
        reports = sanitize(reports, mask)
        if stats is None:
            stats = self.statistics(reports, mask)
        mask = stats.get("mask", mask)
        x, per_capita = self.head_inputs(reports, budget, kappa_feat, stats)
        z = self.allocation_head(x).softmax(-1)[..., 0]
        pay = self.pay_k * per_capita * torch.sigmoid(self.payment_head(x).squeeze(-1))
        if mask is not None:
            z, pay = z * mask, pay * mask
        return z, pay

    def forward(self, reports, budget, kappa_feat, mask=None):
        """Released epsilon z_i * cap_i and projected payments, shape (batch, N)."""
        reports = sanitize(reports, mask)
        z, pay = self.raw(reports, budget, kappa_feat, mask)
        return z * reports[..., 1], budget_projection(pay, budget)


def sanitize(reports, mask):
    """Inactive (masked) slots are replaced by a neutral finite row before any computation."""
    if mask is None:
        return reports
    return torch.where(mask[..., None], reports, torch.ones_like(reports))


def budget_projection(pay, budget):
    """p_bar = p / max(1, sum p / B); identical to regretnet.budget_projection_privacy_paper."""
    total = pay.sum(-1, keepdim=True).clamp_min(1e-12)
    return pay * (budget.reshape(-1, 1) / total).clamp(max=1.0)


def settle(true_types, eps, pay):
    """Settlement of the task-aware execution layer (contracts.settlement without calibration flags):
    a contract executes iff eps >= MIN_EVENT, eps <= true cap, payment >= 0 and true utility >= 0.
    Returns (executed eps, paid, utility, executed mask)."""
    utility = pay - true_types[..., 0] * eps
    ok = (eps >= MIN_EVENT_EPSILON) & (eps <= true_types[..., 1] + 1e-7) & (utility >= -1e-9) & (pay >= 0)
    eps_x, pay_x = torch.where(ok, eps, torch.zeros_like(eps)), torch.where(ok, pay, torch.zeros_like(pay))
    return eps_x, pay_x, pay_x - true_types[..., 0] * eps_x, ok


# ---------------------------------------------------------------- unilateral deviations

def unilateral_outcome(model, reports, budget, kappa_feat, agent, candidates, mask=None):
    """Exact outcome of client `agent[b]` when it reports each candidate, all others truthful.

    reports (B, N, 3), agent (B,) long, candidates (B, C, 3). Every candidate profile is recomputed from
    scratch (statistics, all clients' payments, projection). Returns (eps, pay) of the deviating client,
    shape (B, C). Any faster implementation must match this within 1e-5 relative.
    """
    bsz, n, _ = reports.shape
    c = candidates.shape[1]
    prof = reports[:, None].expand(bsz, c, n, 3).clone()
    idx = torch.arange(bsz, device=reports.device)
    prof[idx[:, None], torch.arange(c, device=reports.device)[None], agent[:, None]] = candidates
    rep = lambda t: t.repeat_interleave(c, 0)  # noqa: E731
    m = None if mask is None else rep(mask)
    eps, pay = model(prof.reshape(bsz * c, n, 3), rep(budget.reshape(-1, 1)), rep(kappa_feat.reshape(-1, 1)), m)
    eps, pay = eps.view(bsz, c, n), pay.view(bsz, c, n)
    return eps[idx, :, agent], pay[idx, :, agent]


def grid_candidates(true_row, n_v=21, n_cap=11, size_fracs=(1.0, .75, .5, .25, .05)):
    """One-sided report grid for one client: v' in [0, V_MAX], cap' in [0, cap], d' in size_fracs * d (>= 1)."""
    v = torch.linspace(0, V_MAX, n_v, dtype=true_row.dtype, device=true_row.device)
    cap = true_row[1] * torch.linspace(0, 1, n_cap, dtype=true_row.dtype, device=true_row.device)
    d = (true_row[2] * torch.tensor(size_fracs, dtype=true_row.dtype, device=true_row.device)).clamp_min(1.0)
    gv, gc, gd = torch.meshgrid(v, cap, d, indexing="ij")
    return torch.stack([gv.reshape(-1), gc.reshape(-1), gd.reshape(-1)], -1)


# ---------------------------------------------------------------- mean-field limit

@torch.no_grad()
def mf_limit_state(model, law, n_agents_sample, generator):
    """Population state of a law in the mean-field limit, estimated from one large i.i.d. sample.

    law: dict of one market law (see sample_laws); returns statistics (batch 1) and the limit normaliser
    max(1, mean raw payment / (B/N)) used by the projection, plus the per-capita budget and kappa feature."""
    dt = next(model.parameters()).dtype
    reports = sample_markets(law, n_agents_sample, generator, count=1).to(dt)
    beta = float(law["beta"])
    kf = P2.kappa_feature(torch.tensor([[float(law["kappa"])]], dtype=torch.float64), *KAPPA_RANGE).to(dt)
    budget = torch.tensor([[beta * n_agents_sample]], dtype=dt)
    stats = model.statistics(reports)
    _, pay = model.raw(reports, budget, kf, stats=stats)
    norm = torch.clamp(pay.mean() / beta, min=1.0)
    return dict(stats={k: stats[k] for k in ("mean_enc", "mean_feat", "mean_d")}, normaliser=norm, beta=beta,
                kappa_feat=kf)


def mf_limit_outcome(model, state, rows):
    """Outcome of clients reporting `rows` (M, 3) against a fixed mean-field state: (eps, pay), shape (M,)."""
    m = rows.shape[0]
    stats = {k: v.expand(m, -1) for k, v in state["stats"].items()}
    stats["count"] = torch.ones(m, 1, dtype=rows.dtype)
    budget = torch.full((m, 1), state["beta"], dtype=rows.dtype)   # per-capita budget with count 1
    z, pay = model.raw(rows[:, None], budget, state["kappa_feat"].expand(m, -1), stats=stats)
    return (z * rows[:, None, 1]).squeeze(1), (pay / state["normaliser"]).squeeze(1)


@torch.no_grad()
def mf_limit_regret(model, state, true_rows, **grid):
    """Found regret of each true type (rows of true_rows) against a fixed mean-field state (grid search)."""
    out = []
    for t in true_rows:
        cand = torch.cat([t[None], grid_candidates(t, **grid)])
        eps, pay = mf_limit_outcome(model, state, cand)
        _, _, u, _ = settle(t.expand_as(cand), eps, pay)
        out.append((u.max() - u[0]).clamp_min(0.0))
    return torch.stack(out)


@torch.no_grad()
def finite_regret(model, reports, budget, kappa_feat, agent, mask=None, **grid):
    """Found regret (grid search, exact recomputation) of client agent[b] in each market b, shape (B,)."""
    cands = torch.stack([torch.cat([reports[b, agent[b]][None], grid_candidates(reports[b, agent[b]], **grid)])
                         for b in range(reports.shape[0])])
    eps, pay = unilateral_outcome(model, reports, budget, kappa_feat, agent, cands, mask)
    truth = reports[torch.arange(reports.shape[0]), agent][:, None].expand_as(cands)
    _, _, u, _ = settle(truth, eps, pay)
    return (u.max(1).values - u[:, 0]).clamp_min(0.0)


# ---------------------------------------------------------------- markets

KAPPA_RANGE = (0.003, 300.0)


def _loguniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))


def sample_laws(count, rng):
    """Market laws for training and the varying-law test set; rng is a numpy Generator."""
    return [dict(a=float(_loguniform(rng, .5, 5.)), b=float(_loguniform(rng, .5, 5.)),
                 cap_lo=float(rng.uniform(.1, 1.)), cap_hi=float(rng.uniform(2., 5.)),
                 dir_alpha=float(_loguniform(rng, .05, 2.)), mean_size=float(_loguniform(rng, 20., 8000.)),
                 beta=float(_loguniform(rng, .1, 11.)), kappa=float(_loguniform(rng, *KAPPA_RANGE)))
            for _ in range(count)]


def paper_law(dir_alpha, mean_size, beta, kappa):
    """The fixed law of the paper's FL experiments: v ~ U[0,1], cap ~ U[0.1,5], Dirichlet-type sizes with the
    given mean size per client (pool / N for a pool split among N clients)."""
    return dict(a=1.0, b=1.0, cap_lo=.1, cap_hi=5., dir_alpha=float(dir_alpha), mean_size=float(mean_size),
                beta=float(beta), kappa=float(kappa))


def sample_markets(law, n, generator, count=1):
    """count i.i.d. markets of n clients from one law -> reports (count, n, 3) float64.

    generator: numpy Generator. Sizes are i.i.d. round(mean_size * Gamma(dir_alpha, 1) / dir_alpha), clipped
    below at 1: the N -> infinity law of N * mean_size * Dirichlet(dir_alpha * 1_N), so that the population law
    of a market does not depend on N (a symmetric Dirichlet split of a pool of N * mean_size records)."""
    v = generator.beta(law["a"], law["b"], (count, n)) * V_MAX
    cap = law["cap_lo"] + (law["cap_hi"] - law["cap_lo"]) * generator.uniform(0, 1, (count, n))
    g = generator.gamma(law["dir_alpha"], 1.0, (count, n)) / law["dir_alpha"]
    d = np.maximum(1.0, np.round(law["mean_size"] * g))
    return torch.tensor(np.stack([v, cap, d], -1), dtype=torch.float64)


def market_batch(laws, n, generator):
    """One market per law, all with n clients: reports (L, n, 3), budget (L, 1), kappa (L, 1)."""
    reports = torch.cat([sample_markets(law, n, generator) for law in laws])
    budget = torch.tensor([[law["beta"] * n] for law in laws], dtype=torch.float64)
    kappa = torch.tensor([[law["kappa"]] for law in laws], dtype=torch.float64)
    return reports, budget, kappa


def training_term(model, reports, budget, kappa):
    """Buyer term J_kappa(executed)/J_kappa(all caps) with shares from the true sizes (protocol ln-v1).

    Truthful profiles without padding: each training batch has one population size N."""
    eps, pay = model(reports, budget, P2.kappa_feature(kappa, *KAPPA_RANGE))
    eps_x, _, _, _ = settle(reports, eps, pay)
    return P2.p2_training_term(eps_x, reports, kappa)

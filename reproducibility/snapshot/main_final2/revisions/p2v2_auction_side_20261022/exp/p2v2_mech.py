"""MFG-RegretNet (P2-v2, QV-sel) for the auction-side experiments of the paper (RQ1-RQ3, attacks, accounting).

The mechanism is Codex's frozen A7 selector, used read-only:
  * worktrees/codex-p2v2-20261020/next/v2_model.py      (P2V2 total network, cap_projection, own_payment)
  * worktrees/codex-p2v2-20261020/a7flow/model.py        (Selector: data-share threshold tau(kappa, B))
  * run/tdsc_revision/codex_p2v2_20261020/A7_tau/table.json (frozen threshold table)
  * checkpoints run/tdsc_revision/codex_p2v2_20261020/continuation_A7/QV-sel/training/g1/<family>/model.pt
All are checked against the SHA256 values frozen by Codex (frozen_stage_inputs.json, A7_continuation_manifest.json).

SelectorModel below repeats Selector.threshold and Selector.allocate of a7flow/model.py line for line (importing
that module would pull in Codex's whole A7 state machinery); test_p2v2_mech.py checks it against Codex's class.

Added here:
  1. The data dividend of the paper's payment rule, h_i = pi_i (B - vbar Sbar_i), pi_i = d_i / sum_j d_j, vbar = 1.
     Data sizes are verified at registration (author's choice, 2026-10-23), so a client reports only its price and
     budget. Sbar_i is the purchase S = min(50 sigmoid(.), B, sum of selected caps) with client i's price and budget
     replaced by the most favorable ones, 0 and epsbar = 5, and its own (verified) data size. For fixed data sizes S is
     non-increasing in the price and non-decreasing in the budget (the selection depends only on the sizes), so
     Sbar_i >= S and sum_i (p_i + h_i) <= B; Sbar_i does not depend on client i's price or budget.
     In the robustness search of RQ1, where a client may also report a smaller data size, everything, Sbar_i
     included, uses the reported size. sbar_sup() keeps the earlier definition (largest purchase over all sizes as
     well) for reference only.
  2. A fast evaluator of one client's outcome under many candidate reports of that client, all others fixed
     (ClientContext.outcomes). It caches the other clients' embeddings and reproduces own_payment
     (64-interval trapezoid on [v', 1]); test_p2v2_mech.py checks it against own_payment.
The dividend does not change the allocation, so it leaves the FL runs (Stage 3) untouched.
Reports are rows (v, cap, d). Everything runs in float64.
"""
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FL_ROOT", HERE.parents[3]))
WORK = ROOT / "worktrees/codex-p2v2-20261020"
OUT = ROOT / "run/tdsc_revision/codex_p2v2_20261020"
CKPT_DIR = OUT / "continuation_A7/QV-sel/training/g1"

SOURCE_SHA256 = {
    "next/v2_model.py": "d2c0d6c4751905457873418ce669b69ba0f0b3da072b1bf6411d6a71be3888ee",
    "exp/p2_reference.py": "d4dfdbc7c253e3de452df4f3d900dad067f071d71e92416994be5195723fabcb",
    "a7flow/model.py": "1defd361aac95809ea713ae1be503befeebf0b106744c8fd900a0a6be94248de",
    "a7flow/tau.py": "0d0bb412774c7fbe81f5824d1e0222c7ac1bf2bb0edec202783a7923df9ac02a",
}
TAU_TABLE_SHA256 = "777b80c70115950ba1cb968bc35e9e3e66cf10c93ab6295175eb8b891d7dd3c7"
CKPT_SHA256 = {
    "mfg_p2v2": "baddeed7201f2d4dfe82bce9aeb65a0fed86d29759618cab4a7130f57a2b382b",
    "nob_p2v2": "1cd8e8d3e75ab846d780d66810a368c335be231ad4753ca9b1e58a5044e0f0c0",
    "profile_p2v2": "9eec13a7b2872edfb93a23301cc7428d06dcab4e0c59d5f445e497537c174377",
}
FAMILIES = tuple(CKPT_SHA256)

# a7flow/tau.py
KAPPAS = np.array([.003, .01, .03, .1, .3, 1, 3, 10, 30, 100, 300])
AMOUNTS = np.array([1., 2, 5, 10, 20, 30, 50, 100, 110])

VBAR, EPSBAR, DMIN, DBAR = 1.0, 5.0, 1.0, 500.0
S_MAX = 50.0
MIN_EVENT = 0.01
INTERVALS = 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _import_v2_model():
    for rel, want in SOURCE_SHA256.items():
        got = sha256(WORK / rel)
        if got != want:
            raise RuntimeError(f"{rel}: SHA256 {got} differs from the frozen {want}")
    got = sha256(OUT / "A7_tau/table.json")
    if got != TAU_TABLE_SHA256:
        raise RuntimeError(f"tau table SHA256 {got} differs from the frozen {TAU_TABLE_SHA256}")
    sys.path.insert(0, str(WORK / "next"))
    import v2_model  # noqa: E402  (inserts WORK/exp itself for p2_reference)
    return v2_model


V2 = _import_v2_model()
kappa_feature = sys.modules["p2_reference"].kappa_feature


class SelectorModel(V2.P2V2):
    """a7flow/model.py Selector; the thresholds come from the checkpoint's buffers (the frozen table)."""

    def __init__(self, family):
        super().__init__(family, 1.)
        self.register_buffer("tau_budget", torch.zeros(len(KAPPAS), len(AMOUNTS), dtype=torch.float64))
        self.register_buffer("tau_fixed_total", torch.zeros(len(KAPPAS), len(AMOUNTS), dtype=torch.float64))

    # --- verbatim from a7flow/model.py (Selector.threshold) ---
    def threshold(self, kappa, amount, mode):
        x = kappa.reshape(-1).log().clamp(np.log(KAPPAS[0]), np.log(KAPPAS[-1]))
        y = amount.reshape(-1).log().clamp(np.log(AMOUNTS[0]), np.log(AMOUNTS[-1]))
        gx = torch.as_tensor(np.log(KAPPAS), device=x.device, dtype=x.dtype)
        gy = torch.as_tensor(np.log(AMOUNTS), device=y.device, dtype=y.dtype)
        ix = (torch.searchsorted(gx, x, right=True) - 1).clamp(0, len(gx)-2)
        iy = (torch.searchsorted(gy, y, right=True) - 1).clamp(0, len(gy)-2)
        wx, wy = (x-gx[ix])/(gx[ix+1]-gx[ix]), (y-gy[iy])/(gy[iy+1]-gy[iy])
        values = getattr(self, 'tau_' + mode)
        return ((1-wx)*(1-wy)*values[ix, iy] + wx*(1-wy)*values[ix+1, iy]
                + (1-wx)*wy*values[ix, iy+1] + wx*wy*values[ix+1, iy+1])

    # --- verbatim from a7flow/model.py (Selector.allocate) ---
    def allocate(self, reports, budget, kappa, fixed_total=None):
        sizes, caps = reports[..., 2], reports[..., 1]
        shares = sizes / sizes.sum(-1, keepdim=True)
        amount = budget if fixed_total is None else torch.full_like(budget, float(fixed_total))
        threshold = self.threshold(kappa, amount, 'budget' if fixed_total is None else 'fixed_total')
        mask = shares >= threshold[:, None]
        fallback = torch.nn.functional.one_hot(shares.argmax(-1), num_classes=reports.shape[1]).bool()
        mask = torch.where(mask.any(-1, keepdim=True), mask, fallback)
        caps = torch.where(mask, caps, 0.)
        shares = torch.where(mask, shares, 0.)
        shares = shares / shares.sum(-1, keepdim=True)
        total = self.total(reports, budget, kappa) if fixed_total is None else amount
        if fixed_total is None:
            total = torch.minimum(total, budget.reshape(-1, 1))
        total = torch.minimum(total, caps.sum(-1, keepdim=True))
        e = V2.cap_projection(total, shares, caps)
        return torch.where(e >= .01, e, 0.)


def load(family):
    path = CKPT_DIR / family / "model.pt"
    got = sha256(path)
    if got != CKPT_SHA256[family]:
        raise RuntimeError(f"{path}: SHA256 {got} differs from the frozen {CKPT_SHA256[family]}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["family"], saved["gamma"], saved.get("selected_proxy"), saved.get("tau_sha256")) != \
            (family, 1.0, "QV-sel", TAU_TABLE_SHA256):
        raise RuntimeError("unexpected checkpoint metadata")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = SelectorModel(family).double()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def provenance():
    return {"sources": {rel: sha256(WORK / rel) for rel in SOURCE_SHA256},
            "tau_table": sha256(OUT / "A7_tau/table.json"),
            "checkpoints": dict(CKPT_SHA256),
            "p2v2_mech.py": sha256(Path(__file__)),
            "torch": torch.__version__, "numpy": np.__version__}


# ------------------------------------------------------------------------------------------- helpers
def features(r):
    """Codex's P2V2.total features: (1 - v, cap / 5, log(d / 10) / log 50)."""
    return torch.stack((1 - r[..., 0], r[..., 1] / 5., torch.log(r[..., 2] / 10.) / math.log(50)), -1)


def selection(shares, tau):
    """Selector mask for shares (..., N) and thresholds tau broadcastable to shares[..., 0]."""
    mask = shares >= tau[..., None]
    fallback = F.one_hot(shares.argmax(-1), num_classes=shares.shape[-1]).bool()
    return torch.where(mask.any(-1, keepdim=True), mask, fallback)


# ------------------------------------------------------------------------------- one client, many reports
class ClientContext:
    """Quantities of client position c in a batch of profiles that do not depend on c's own report."""

    def __init__(self, model, reports, client):
        self.model, self.reports, self.c = model, reports, client
        P, N, _ = reports.shape
        self.N = N
        self.others = [j for j in range(N) if j != client]
        rest = reports[:, self.others]
        fam = model.family
        if fam == "mfg_p2v2":
            self.emb_rest = model.phi(features(rest)).sum(1)                        # (P, 64)
        elif fam == "profile_p2v2":
            f = features(reports)
            self.emb_slots = {j: model.psi[j](f[:, j]) for j in self.others}        # (P, 16) each

    def public(self, budget, kappa):
        """Codex's public network on (B, kappa): coefficients softplus(out[:-1]) and bias out[-1]; and tau."""
        pub = torch.cat((budget.reshape(-1, 1) / 110., kappa_feature(kappa.reshape(-1, 1), .003, 300.)), -1)
        out = self.model.public(pub)
        tau = self.model.threshold(kappa, budget, "budget")
        return F.softplus(out[:, :-1]), out[:, -1], tau

    def embed(self, cand):
        """The client's own embedding for reports cand (P, ..., 3): phi (mfg), psi_c (profile) or None (nob)."""
        fam = self.model.family
        if fam == "mfg_p2v2":
            return self.model.phi(features(cand))
        if fam == "profile_p2v2":
            return self.model.psi[self.c](features(cand))
        return None

    def score_from(self, coef, bias, emb, shape):
        """Score of the total network given the client's embedding emb (P, ..., W); shape = emb.shape[:-1]."""
        fam = self.model.family
        expand = (slice(None),) + (None,) * (len(shape) - 1)
        if fam == "nob_p2v2":
            return bias[expand].expand(shape)
        if fam == "mfg_p2v2":
            z = (self.emb_rest[expand] + emb) / self.N
            return bias[expand] + (coef[expand] * z).sum(-1)
        c, w = self.c, 16
        fixed = sum((coef[:, w * j:w * (j + 1)] * self.emb_slots[j]).sum(-1) for j in self.others)
        own = (coef[:, w * c:w * (c + 1)][expand] * emb).sum(-1)
        return bias[expand] + fixed[expand] + own

    def score(self, coef, bias, cand):
        """Score of the total network when the client reports cand (P, ..., 3)."""
        return self.score_from(coef, bias, self.embed(cand), cand.shape[:-1])

    def place(self, own, column):
        """Full (P, ..., N) array of report column `column` with the client's values `own` (P, ...)."""
        expand = (slice(None),) + (None,) * (own.dim() - 1)
        rest = self.reports[:, self.others, column][expand]
        rest = rest.expand(*own.shape, self.N - 1)
        c = self.c
        return torch.cat((rest[..., :c], own[..., None], rest[..., c:]), -1)

    def prepare(self, cand):
        """Own-cost grid of own_payment and the client's embeddings on it; independent of (B, kappa)."""
        P, G, _ = cand.shape
        v = cand[..., 0]
        fraction = torch.linspace(0., 1., INTERVALS + 1, dtype=cand.dtype)
        grid = v[..., None] + (1 - v[..., None]) * fraction                                # (P, G, 65)
        full = cand[:, :, None, :].expand(P, G, INTERVALS + 1, 3).clone()
        full[..., 0] = grid
        return full, self.embed(full)

    def evaluate(self, budget, kappa, cand, full, emb, dividend=True):
        """Release e, Myerson payment p (own_payment) and dividend h for prepared candidates (P, G, 3)."""
        coef, bias, tau = self.public(budget, kappa)
        P, G, _ = cand.shape
        caps = self.place(cand[..., 1], 1)                                              # (P, G, N)
        sizes = self.place(cand[..., 2], 2)
        shares = sizes / sizes.sum(-1, keepdim=True)
        mask = selection(shares, tau[:, None].expand(P, G))
        caps = torch.where(mask, caps, torch.zeros_like(caps))
        shares = torch.where(mask, shares, torch.zeros_like(shares))
        shares = shares / shares.sum(-1, keepdim=True)
        S = S_MAX * torch.sigmoid(self.score_from(coef, bias, emb, full.shape[:-1]))
        S = torch.minimum(S, budget.reshape(-1, 1, 1))
        S = torch.minimum(S, caps.sum(-1, keepdim=True))
        K = INTERVALS + 1
        e_all = V2.cap_projection(S.reshape(-1, 1),
                                  shares[:, :, None].expand(P, G, K, self.N).reshape(-1, self.N),
                                  caps[:, :, None].expand(P, G, K, self.N).reshape(-1, self.N))
        x = e_all[:, self.c].reshape(P, G, K)
        x = torch.where(x >= MIN_EVENT, x, torch.zeros_like(x))
        e = x[..., 0]
        v = cand[..., 0]
        if self.model.family == "nob_p2v2":
            p = e                                                                       # Codex: payment = e
        else:
            integral = ((x[..., 1:] + x[..., :-1]) * .5).sum(-1) * (1 - v) / INTERVALS
            p = v * e + integral
        if not dividend:
            return e, p, None
        own_size = cand[..., 2].pow(self.model.gamma)
        rest_size = self.reports[:, self.others, 2].pow(self.model.gamma).sum(-1, keepdim=True)
        with torch.no_grad():
            sbar = self.purchase_at(budget, kappa, cand[..., 2].detach())
        h = own_size / (own_size + rest_size) * (budget.reshape(-1, 1) - VBAR * sbar)
        return e, p, h

    def outcomes(self, budget, kappa, cand, dividend=True):
        """Release e, Myerson payment p (own_payment) and dividend h of the client for candidate reports.

        cand (P, G, 3); budget, kappa (P, 1). Returns three (P, G) tensors (h is None if dividend=False).
        Differentiable in cand (used by the PGA steps).
        """
        full, emb = self.prepare(cand)
        return self.evaluate(budget, kappa, cand, full, emb, dividend)

    def outcomes_multi(self, settings, cand, dividend=True):
        """outcomes() for several (budget, kappa) pairs, sharing the embeddings; no gradients."""
        with torch.no_grad():
            full, emb = self.prepare(cand)
            return [self.evaluate(b, k, cand, full, emb, dividend) for b, k in settings]

    # ---------------------------------------------------------------- dividend reference purchase
    def purchase_at(self, budget, kappa, d_own, select_at=None):
        """S with the client's report (0, 5, d_own); selection evaluated at sizes `select_at` (default d_own).

        d_own, select_at: (P, M). Returns (P, M).
        """
        coef, bias, tau = self.public(budget, kappa)
        P, M = d_own.shape
        cand = torch.stack((torch.zeros_like(d_own), torch.full_like(d_own, EPSBAR), d_own), -1)
        sel_sizes = self.place(d_own if select_at is None else select_at, 2)
        shares = sel_sizes / sel_sizes.sum(-1, keepdim=True)
        caps = self.place(torch.full_like(d_own, EPSBAR), 1)
        mask = selection(shares, tau[:, None].expand(P, M))
        cap_sum = torch.where(mask, caps, torch.zeros_like(caps)).sum(-1)
        S = S_MAX * torch.sigmoid(self.score(coef, bias, cand))
        return torch.minimum(torch.minimum(S, budget.reshape(-1, 1)), cap_sum)

    def breakpoints(self, budget, kappa):
        """Sorted candidate sizes in [1, 500] where the selection can change, per profile: (P, M)."""
        _, _, tau = self.public(budget, kappa)
        d_rest = self.reports[:, self.others, 2]
        D = d_rest.sum(-1, keepdim=True)
        t = tau[:, None]
        own_cross = torch.where(t < 1, t * D / (1 - t).clamp_min(1e-300), torch.full_like(D, DBAR))
        other_cross = torch.where(t > 0, d_rest / t.clamp_min(1e-300) - D, torch.full_like(d_rest, DBAR))
        pts = torch.cat((torch.full_like(D, DMIN), torch.full_like(D, DBAR), own_cross, other_cross, d_rest), -1)
        return pts.clamp(DMIN, DBAR).sort(-1).values

    def sbar(self, budget, kappa):
        """Reference purchase of the dividend: price 0, budget epsbar, the client's own (verified) size: (P,)."""
        return self.purchase_at(budget, kappa, self.reports[:, self.c, 2:3])[:, 0]

    def sbar_sup(self, budget, kappa):
        """Earlier definition, for reference only: largest purchase over every size in [1, 500] as well: (P,)."""
        pts = self.breakpoints(budget, kappa)
        at_points = self.purchase_at(budget, kappa, pts)
        mids = 0.5 * (pts[:, 1:] + pts[:, :-1])
        right = self.purchase_at(budget, kappa, pts[:, 1:], select_at=mids)
        return torch.maximum(at_points.max(-1).values, right.max(-1).values)

    def sbar_fixed(self, budget, kappa):
        """Purchase with the client's report replaced by (0, 5, 500) (the earlier, non-maximal definition)."""
        d = torch.full((self.reports.shape[0], 1), DBAR, dtype=self.reports.dtype)
        return self.purchase_at(budget, kappa, d)[:, 0]


# ------------------------------------------------------------------------------------------- full profiles
def mechanism(model, reports, budget, kappa, chunk=256, dividend=True):
    """Codex's mechanism (release e, Myerson payment p) and the dividend h for truthful profiles."""
    es, ps, hs = [], [], []
    with torch.no_grad():
        for lo in range(0, len(reports), chunk):
            r, b, k = reports[lo:lo + chunk], budget[lo:lo + chunk], kappa[lo:lo + chunk]
            e, p = model.mechanism(r, b, k)
            es.append(e), ps.append(p)
            if dividend:
                h = torch.zeros_like(e)
                sizes = r[..., 2].pow(model.gamma)
                share = sizes / sizes.sum(-1, keepdim=True)
                for c in range(r.shape[1]):
                    sb = ClientContext(model, r, c).sbar(b, k)
                    h[:, c] = share[:, c] * (b[:, 0] - VBAR * sb)
                hs.append(h)
    return torch.cat(es), torch.cat(ps), (torch.cat(hs) if dividend else None)


def candidate_grid(reports, client, gv=21, ge=21, gd=5):
    """Grid of eval_rq1.py (regret_tools.candidate_grid): v' in linspace(0, 1, gv), eps' = frac * eps_c,
    d' = max(1, 1 + frac * (d_c - 1)); returns (P, gv * ge * gd, 3)."""
    dt = reports.dtype
    vg = torch.linspace(0.0, 1.0, gv, dtype=dt)
    eg = torch.linspace(0.0, 1.0, ge, dtype=dt)
    dg = torch.linspace(0.0, 1.0, gd, dtype=dt) if gd > 1 else torch.ones(1, dtype=dt)
    g = torch.cartesian_prod(vg, eg, dg)
    eps = reports[:, client, 1:2]
    size = reports[:, client, 2:3]
    v = g[None, :, 0].expand(len(reports), -1)
    e = g[None, :, 1] * eps
    d = torch.clamp(1.0 + g[None, :, 2] * (size - 1.0), min=1.0)
    return torch.stack((v, e, d), -1)

"""Exact cached mechanism graph; CrypTen defaults or float64 diagnostic backend.

No secret branching or secret index is opened. The only reveals are e, p, h.
The protocol uses 65 grid points, including both integration endpoints.
"""
import math
import time
from collections import Counter

import torch
import torch.nn.functional as F


def cat(xs, dim=-1):
    if isinstance(xs[0], torch.Tensor):
        return torch.cat(xs, dim=dim)
    import crypten
    return crypten.cat(list(xs), dim=dim)


def comparison(x, y, mode):
    out = getattr(x, mode)(y)
    return out.to(x.dtype) if isinstance(out, torch.Tensor) else out


def raw_minimum(a, b):
    flag = comparison(a, b, "lt")
    return b + (a - b) * flag


def raw_relu(a):
    return a * comparison(a, 0, "ge")


def bitonic_sort(x):
    """Fixed public sorting network on the last, power-of-two dimension."""
    n = x.size(-1)
    if n < 1 or n & (n - 1):
        raise ValueError("sort length must be a power of two")
    idx = torch.arange(n)
    k = 2
    while k <= n:
        j = k // 2
        while j:
            partner = idx ^ j
            i1 = idx[partner > idx]
            i2 = partner[partner > idx]
            asc = ((i1 & k) == 0).double()
            a, b = x[..., i1], x[..., i2]
            mn = raw_minimum(a, b)
            mx = a + b - mn
            order = torch.cat((i1, i2))
            inv = torch.empty_like(order)
            inv[order] = torch.arange(n)
            x = cat((mn * asc + mx * (1 - asc),
                     mx * asc + mn * (1 - asc)))[..., inv]
            j //= 2
        k *= 2
    return x


class Operations:
    """Record public operation shapes, without revealing intermediate values."""

    def __init__(self):
        self.recipe = []

    def call(self, kind, x, fn, **details):
        self.recipe.append(dict(kind=kind, shape=list(x.size()), **details))
        return fn()

    def linear(self, x, weight, bias):
        return self.call("public_linear", x,
                         lambda: x.matmul(weight.t()) + bias,
                         output_width=int(weight.size(0)))

    def tanh(self, x):
        return self.call("tanh", x, x.tanh)

    def sigmoid(self, x):
        return self.call("sigmoid", x, x.sigmoid)

    def mul(self, x, y):
        return self.call("secret_multiply", x, lambda: x * y)

    def scale(self, x, y):
        # Record the broadcast output size, rather than the smaller operand.
        shape = list(torch.broadcast_shapes(tuple(x.size()), tuple(y.size())))
        self.recipe.append(dict(kind="public_scale", shape=shape))
        return x * y

    def compare(self, x, y, mode):
        return self.call("compare_" + mode, x, lambda: comparison(x, y, mode),
                         rhs="secret" if hasattr(y, "get_plain_text") or isinstance(y, torch.Tensor) else "public")

    def minimum(self, a, b):
        return self.call("minimum", a, lambda: raw_minimum(a, b),
                         rhs="secret" if hasattr(b, "get_plain_text") or isinstance(b, torch.Tensor) else "public")

    def relu(self, x):
        return self.call("maximum_zero", x, lambda: raw_relu(x))

    def sort(self, x):
        return self.call("bitonic_sort", x, lambda: bitonic_sort(x))


def project(total, shares, caps, ops, diagnostics=None):
    """V2.cap_projection, including its full-cap/zero/fallback branches."""
    n = caps.size(-1)
    total = total.reshape(-1, 1)
    caps_sum = caps.sum(-1, keepdim=True)
    total = ops.minimum(total, caps_sum)
    if diagnostics is not None:
        # Retain the existing tensor only. No extra secure operation or reveal.
        diagnostics["S"] = total
    target = ops.scale(total, shares)
    breaks = caps - target
    n2 = 1 << (n - 1).bit_length()
    if n2 > n:
        pad = breaks[..., :1].expand(breaks.size(0), n2 - n) * 0 + 64.
        breaks = cat((breaks, pad))
    breaks = ops.sort(breaks)[..., :n]
    prefix = breaks.cumsum(-1) - breaks
    inverse_counts = 1. / torch.arange(n, 0, -1, dtype=torch.float64)
    multipliers = ops.scale(total - target.sum(-1, keepdim=True) - prefix,
                            inverse_counts)
    admissible = ops.compare(multipliers, breaks, "le")
    preceding = admissible.cumsum(-1) - admissible
    first = ops.mul(admissible, ops.compare(preceding, 1, "lt"))
    absent = ops.compare(admissible.sum(-1, keepdim=True), 1, "lt")
    last = torch.zeros(n, dtype=torch.float64)
    last[-1] = 1.
    first = first + absent * last
    lam = ops.mul(multipliers, first).sum(-1, keepdim=True)
    e = ops.minimum(ops.relu(target + lam), caps)
    full = ops.compare(total, caps_sum, "ge")
    e = e + ops.mul(caps - e, full)
    zero = ops.compare(total, 0., "le")
    e = ops.mul(e, 1 - zero)
    return ops.mul(e, ops.compare(e, .01, "ge"))


def public_context(model, reports, budget=50., kappa=3.):
    """Public quantities depend only on verified d, B, kappa, and public weights."""
    from p2v2_mech import V2
    b = torch.tensor([[budget]], dtype=torch.float64)
    k = torch.tensor([[kappa]], dtype=torch.float64)
    pub = torch.cat((b / 110., V2.kappa_feature(k, .003, 300.)), -1)
    with torch.no_grad():
        out = model.public(pub)[0]
        tau = model.threshold(k, b, "budget")[0]
    sizes = reports[:, 2]
    shares = sizes / sizes.sum()
    mask = shares >= tau
    if not bool(mask.any()):
        mask[shares.argmax()] = True
    selected = shares * mask
    selected = selected / selected.sum()
    weights = [(F.softplus(model.phi[j].raw_weight).detach(),
                model.phi[j].bias.detach()) for j in (0, 2)]
    return dict(coef=F.softplus(out[:-1]), bias=out[-1], shares=shares,
                selected_shares=selected, mask=mask.double(), weights=weights,
                data_feature=torch.log(sizes / 10.) / math.log(50),
                budget=float(budget), kappa=float(kappa), tau=float(tau))


def plaintext_purchase(model, reports, ctx):
    """Independent float64 S from the frozen model, before allocation/flooring."""
    with torch.no_grad():
        budget = reports.new_full((1, 1), ctx["budget"])
        kappa = reports.new_full((1, 1), ctx["kappa"])
        cap_sum = (reports[0, :, 1] * ctx["mask"]).sum().reshape(1, 1)
        return torch.minimum(torch.minimum(model.total(reports, budget, kappa), budget), cap_sum)


def mechanism(secret, ctx, ops=None, diagnostics=None):
    """secret: N x 2 (price, cap). All 65N counterfactuals are one batch."""
    ops = ops or Operations()
    n, grid_size = secret.size(0), 65
    values, caps = secret[:, 0], secret[:, 1]
    features = cat(((1 - values).unsqueeze(-1), (caps / 5.).unsqueeze(-1),
                    (values * 0 + ctx["data_feature"]).unsqueeze(-1)))

    def embed(f):
        for w, b in ctx["weights"]:
            f = ops.tanh(ops.linear(f, w, b))
        return f

    emb = embed(features)
    emb_sum = emb.sum(0)
    caps_selected = caps * ctx["mask"]
    caps_sum = caps_selected.sum().reshape(1, 1)

    def purchase(z, cap_total):
        score = ops.scale(z, ctx["coef"]).sum(-1, keepdim=True) + ctx["bias"]
        s = ops.sigmoid(score) * 50.
        return ops.minimum(ops.minimum(s, ctx["budget"]), cap_total)

    total = purchase((emb_sum / n).reshape(1, 64), caps_sum)
    e = project(total, ctx["selected_shares"].reshape(1, n),
                caps_selected.reshape(1, n), ops, diagnostics=diagnostics).reshape(n)

    fraction = torch.linspace(0., 1., grid_size, dtype=torch.float64)
    grid = values.unsqueeze(-1) + (1 - values).unsqueeze(-1) * fraction
    grid_features = cat(((1 - grid).unsqueeze(-1),
                         (caps / 5.).reshape(n, 1, 1).expand(n, grid_size, 1),
                         (values * 0 + ctx["data_feature"]).reshape(n, 1, 1).expand(n, grid_size, 1)))
    grid_emb = embed(grid_features.reshape(n * grid_size, 3)).reshape(n, grid_size, 64)
    grid_mean = (emb_sum.reshape(1, 1, 64) - emb.reshape(n, 1, 64) + grid_emb) / n
    grid_total = purchase(grid_mean.reshape(n * grid_size, 64), caps_sum)
    allocations = project(grid_total, ctx["selected_shares"].reshape(1, n).expand(n * grid_size, n),
                          caps_selected.reshape(1, n).expand(n * grid_size, n), ops)
    # Public indices: which client each valuation grid belongs to.
    rows = torch.arange(n * grid_size)
    clients = torch.arange(n).repeat_interleave(grid_size)
    own = allocations[rows, clients].reshape(n, grid_size)
    integral_sum = ((own[:, 1:] + own[:, :-1]) * .5).sum(-1)
    integral = ops.mul(integral_sum, 1 - values) / 64.
    p = ops.mul(values, e) + integral

    # A client's dividend uses its (public) most favorable report (0,5,d).
    # We deliberately evaluate the same phi graph (no precision shortcut).
    favorable = cat(((values * 0 + 1).unsqueeze(-1),
                     (values * 0 + 1).unsqueeze(-1),
                     (values * 0 + ctx["data_feature"]).unsqueeze(-1)))
    favorable_emb = embed(favorable)
    cap_bar = caps_sum - caps_selected.reshape(n, 1) + (5 * ctx["mask"]).reshape(n, 1)
    sbar = purchase((emb_sum.reshape(1, 64) - emb + favorable_emb) / n, cap_bar).reshape(n)
    h = ops.scale(ctx["budget"] - sbar, ctx["shares"])
    return (e, p, h), ops.recipe


def grouped_recipe(recipe):
    import json
    counts = Counter(json.dumps(op, sort_keys=True) for op in recipe)
    return [dict(operation=json.loads(key), calls=count) for key, count in counts.items()]

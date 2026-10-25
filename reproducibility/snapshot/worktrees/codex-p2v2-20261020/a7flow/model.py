"""A6 selector added to the unchanged A4 total network and Myerson payments."""
import numpy as np
import torch
from flow_state import TABLE, read, digest
from v2_model import P2V2, cap_projection
from tau import KAPPAS, AMOUNTS, interpolate, selected
import p2_reference as REF


class Selector(P2V2):
    def __init__(self, family, table=None):
        super().__init__(family, 1.)
        self.table = read(TABLE) if table is None else table
        if table is None and read(TABLE.parent / 'table_sha256.json')['sha256'] != digest(TABLE):
            raise RuntimeError('threshold table hash changed')
        for mode in ('budget', 'fixed_total'):
            self.register_buffer('tau_' + mode, torch.tensor(self.table[mode], dtype=torch.float64))

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
        e = cap_projection(total, shares, caps)
        return torch.where(e >= .01, e, 0.)


def reference_allocation(reports, budget, kappa, s_net, table, fixed_total=None):
    shares = reports[:, 2] / reports[:, 2].sum()
    amount, mode = (budget, 'budget') if fixed_total is None else (fixed_total, 'fixed_total')
    threshold = interpolate(table, kappa, amount, mode)
    mask = selected(shares[None], [threshold])[0]
    caps = np.where(mask, reports[:, 1], 0.)
    target = np.where(mask, shares, 0.); target /= target.sum()
    total = min(s_net(reports), budget, caps.sum()) if fixed_total is None else min(fixed_total, caps.sum())
    return REF.drop_below_min_event(REF._fill_for_mu(total * target, caps, total))


def reference_mechanism(reports, budget, kappa, s_net, table, fixed_total=None):
    x = reference_allocation(reports, budget, kappa, s_net, table, fixed_total)
    p = np.zeros(10)
    for i in range(10):
        grid = np.linspace(reports[i, 0], 1., 65)
        curve = []
        for cost in grid:
            changed = reports.copy(); changed[i, 0] = cost
            curve.append(reference_allocation(changed, budget, kappa, s_net, table, fixed_total)[i])
        curve = np.asarray(curve)
        p[i] = reports[i, 0] * x[i] + np.sum((curve[1:] + curve[:-1]) * .5 * np.diff(grid))
    return x, p


def load_model(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        if saved.get('selected_proxy') == 'QV-sel':
            assert saved['tau_sha256'] == digest(TABLE)
            net = Selector(saved['family']).double()
        else:
            net = P2V2(saved['family'], saved['gamma']).double()
    net.load_state_dict(saved['state_dict'], strict=True)
    return net.to(device).eval()

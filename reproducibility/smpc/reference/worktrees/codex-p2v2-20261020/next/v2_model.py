"""A4/A5 monotone totals, exact cap projection and 64-interval payments."""
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'exp'))
from p2_reference import kappa_feature, p2_training_term

FAMILIES = ('mfg_p2v2', 'nob_p2v2', 'profile_p2v2')


class PositiveLinear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        # Small positive fan-in weights avoid saturation at initialization.
        self.raw_weight = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.normal_(self.raw_weight, math.log(math.expm1(1. / inputs)), .05)

    def forward(self, x):
        return F.linear(x, F.softplus(self.raw_weight), self.bias)


def cap_projection(total, shares, caps):
    """Exact sorted-breakpoint water filling, differentiable within each face.

    The target is nonnegative and sums to total, so its fill multiplier is
    nonnegative (up to rounding). Only upper-cap breakpoints are needed.
    """
    total = torch.minimum(total.reshape(-1, 1), caps.sum(-1, keepdim=True))
    target = total * shares
    breaks = (caps - target).sort(-1).values
    prefix = torch.cat((torch.zeros_like(total), breaks.cumsum(-1)[:, :-1]), -1)
    counts = torch.arange(caps.shape[-1], 0, -1, dtype=caps.dtype, device=caps.device)
    multipliers = (total - target.sum(-1, keepdim=True) - prefix) / counts
    admissible = multipliers <= breaks
    index = admissible.to(torch.int64).argmax(-1, keepdim=True)
    index = torch.where(admissible.any(-1, keepdim=True), index, torch.full_like(index, caps.shape[-1] - 1))
    lam = multipliers.gather(-1, index)
    e = torch.minimum(caps, (target + lam).clamp_min(0))
    e = torch.where(total >= caps.sum(-1, keepdim=True), caps, e)
    return torch.where(total <= 0, torch.zeros_like(e), e)


class P2V2(nn.Module):
    def __init__(self, family, gamma):
        super().__init__()
        if family not in FAMILIES or gamma not in (0., .5, 1.):
            raise ValueError((family, gamma))
        self.family, self.gamma = family, float(gamma)
        width = 64 if family == 'mfg_p2v2' else 160 if family == 'profile_p2v2' else 0
        self.public = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, width + 1))
        if family == 'mfg_p2v2':
            self.phi = nn.Sequential(PositiveLinear(3, 64), nn.Tanh(), PositiveLinear(64, 64), nn.Tanh())
        elif family == 'profile_p2v2':
            self.psi = nn.ModuleList([nn.Sequential(PositiveLinear(3, 16), nn.Tanh(), PositiveLinear(16, 16), nn.Tanh()) for _ in range(10)])
        with torch.no_grad():
            self.public[-1].weight.mul_(.02)
            if width:
                self.public[-1].bias[:-1].fill_(math.log(math.expm1(1. / width)))
            self.public[-1].bias[-1] = -2.

    def total(self, reports, budget, kappa):
        public = torch.cat((budget.reshape(-1, 1) / 110., kappa_feature(kappa.reshape(-1, 1), .003, 300.)), -1)
        output = self.public(public)
        score = output[:, -1:]
        if self.family != 'nob_p2v2':
            f = torch.stack((1 - reports[..., 0], reports[..., 1] / 5., torch.log(reports[..., 2] / 10.) / math.log(50)), -1)
            if self.family == 'mfg_p2v2':
                z = self.phi(f).mean(1)
            else:
                z = torch.cat([net(f[:, i]) for i, net in enumerate(self.psi)], -1)
            score = score + (F.softplus(output[:, :-1]) * z).sum(-1, keepdim=True)
        return 50. * torch.sigmoid(score)

    def allocate(self, reports, budget, kappa, fixed_total=None):
        total = self.total(reports, budget, kappa) if fixed_total is None else torch.full_like(budget.reshape(-1, 1), float(fixed_total))
        if fixed_total is None:
            total = torch.minimum(total, budget.reshape(-1, 1))
        caps = reports[..., 1]
        total = torch.minimum(total, caps.sum(-1, keepdim=True))
        sizes = reports[..., 2].pow(self.gamma)
        shares = sizes / sizes.sum(-1, keepdim=True)
        e = cap_projection(total, shares, caps)
        return torch.where(e >= .01, e, 0.)

    def own_payment(self, reports, budget, kappa, clients, fixed_total=None):
        """Only requested clients' payments; used by exact unilateral searches."""
        batch, n, _ = reports.shape
        clients = torch.as_tensor(clients, dtype=torch.long, device=reports.device)
        rows = torch.arange(batch, device=reports.device)
        e = self.allocate(reports, budget, kappa, fixed_total)[rows, clients]
        if fixed_total is not None or self.family == 'nob_p2v2':
            return e, e
        values = reports[rows, clients, 0]
        fraction = torch.linspace(0., 1., 65, dtype=reports.dtype, device=reports.device)
        grid = values[:, None] + (1 - values[:, None]) * fraction
        counter = reports[:, None].expand(batch, 65, n, 3).clone()
        counter[rows[:, None], torch.arange(65, device=reports.device)[None], clients[:, None], 0] = grid
        allocations = self.allocate(counter.reshape(-1, n, 3), budget.repeat_interleave(65, 0), kappa.repeat_interleave(65, 0)).reshape(batch, 65, n)
        own = allocations[rows[:, None], torch.arange(65, device=reports.device)[None], clients[:, None]]
        integral = ((own[:, 1:] + own[:, :-1]) * .5).sum(-1) * (1 - values) / 64.
        return e, values * e + integral

    def mechanism(self, reports, budget, kappa, fixed_total=None):
        e = self.allocate(reports, budget, kappa, fixed_total)
        if fixed_total is not None or self.family == 'nob_p2v2':
            return e, e
        payments = [self.own_payment(reports, budget, kappa,
                    torch.full((len(reports),), i, device=reports.device), fixed_total)[1] for i in range(reports.shape[1])]
        return e, torch.stack(payments, -1)

    def objective(self, reports, budget, kappa):
        e = self.allocate(reports, budget, kappa)
        tempered = torch.stack((reports[..., 0], reports[..., 1], reports[..., 2].pow(self.gamma)), -1)
        return p2_training_term(e, tempered, kappa)

    def numpy_total(self, budget, kappa):
        """Independent NumPy evaluation for slow scalar reference checks."""
        state = {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}
        pub = np.array([budget / 110., np.clip((np.log(kappa) - np.log(.003)) / np.log(100000.), 0, 1)])
        for i in (0, 2, 4):
            pub = pub @ state[f'public.{i}.weight'].T + state[f'public.{i}.bias']
            if i < 4:
                pub = np.tanh(pub)
        coefficients, bias = np.logaddexp(0., pub[:-1]), pub[-1]
        def embed(f, prefix):
            for i in (0, 2):
                f = np.tanh(f @ np.logaddexp(0., state[f'{prefix}.{i}.raw_weight']).T + state[f'{prefix}.{i}.bias'])
            return f
        def total(r):
            score = bias
            if self.family != 'nob_p2v2':
                features = np.stack((1 - r[:, 0], r[:, 1] / 5., np.log(r[:, 2] / 10.) / np.log(50.)), -1)
                z = embed(features, 'phi').mean(0) if self.family == 'mfg_p2v2' else np.concatenate([embed(features[i], f'psi.{i}') for i in range(10)])
                score += coefficients @ z
            return 50. / (1 + np.exp(-score))
        return total


def load_model(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    with torch.random.fork_rng():
        torch.random.default_generator.manual_seed(42)
        model = P2V2(saved['family'], saved['gamma']).double()
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval()

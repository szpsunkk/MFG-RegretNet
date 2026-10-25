"""MFG-RegretNet with inputs scaled to [0, 1] (architecture otherwise unchanged).

Scaling: v / VBAR, eps / EPS_MAX, size / SIZE_MAX for the per-client rows and the
mean-field row, and B / B_SCALE for the budget feature. The payment scale and the
budget projection still use the raw budget, exactly as in regretnet.MFGRegretNet.
"""
import torch

from regretnet import MFGRegretNet, budget_projection_privacy_paper

VBAR, EPS_MAX, SIZE_MAX, B_SCALE = 1.0, 5.0, 500.0, 100.0


class MFGRegretNetScaled(MFGRegretNet):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.register_buffer("in_scale", torch.tensor([VBAR] * self.n_items + [EPS_MAX, SIZE_MAX]))
        self.b_scale = B_SCALE

    def forward(self, input):
        reports = input[0].reshape(-1, self.n_agents, self.n_items + 2)
        budget = input[1].reshape(-1, 1)
        r = reports / self.in_scale
        b_mfg = r.mean(dim=1, keepdim=True).repeat(1, self.n_agents, 1)
        x = torch.cat([r.reshape(-1, self.n_agents * (self.n_items + 2)),
                       b_mfg.reshape(-1, self.n_agents * (self.n_items + 2)),
                       budget / self.b_scale], dim=1)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)
        payments_raw = self.payment_head(x) * budget
        return allocs, budget_projection_privacy_paper(payments_raw, budget)


class MFGRegretNetNoMF(MFGRegretNetScaled):
    """Ablation: MFG-RegretNet with the mean-field rows of the input set to zero (no b_MFG)."""

    def forward(self, input):
        reports = input[0].reshape(-1, self.n_agents, self.n_items + 2)
        budget = input[1].reshape(-1, 1)
        r = reports / self.in_scale
        x = torch.cat([r.reshape(-1, self.n_agents * (self.n_items + 2)),
                       torch.zeros_like(r).reshape(-1, self.n_agents * (self.n_items + 2)),
                       budget / self.b_scale], dim=1)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)
        payments_raw = self.payment_head(x) * budget
        return allocs, budget_projection_privacy_paper(payments_raw, budget)


from regretnet import RegretNet  # noqa: E402


class RegretNetScaled(RegretNet):
    """RegretNet / DM-RegretNet (original trunks and softmax budget-split payment head) with the
    same fixed input scaling as MFGRegretNetScaled: v / VBAR, eps / EPS_MAX, size / SIZE_MAX per
    client and B / B_SCALE as the budget feature (input size N * (m + 2) + 1, as in RegretNet).

    Deterministic allocation (DM-RegretNet): the original code sharpens and rounds the allocation
    after the abstention column has been cut, which for a single item (m = 1) is a softmax / argmax
    over one entry and therefore always allocates 1. Here the sharpening (training) and the argmax
    (test) are taken over the m items plus the abstention option, and the abstention column is cut
    afterwards, so a client can be either bought from or not.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.register_buffer("in_scale", torch.tensor([VBAR] * self.n_items + [EPS_MAX, SIZE_MAX]))
        self.b_scale = B_SCALE

    def forward(self, input):
        reports = input[0].reshape(-1, self.n_agents, self.n_items + 2)
        budget = input[1].reshape(-1, 1)
        x = torch.cat([(reports / self.in_scale).reshape(reports.shape[0], -1), budget / self.b_scale], dim=1)
        x = self.nn_model(x)
        if self.a_activation == "deterministic":
            h = x
            mods = list(self.allocation_head.children())
            for mod in mods[:-1]:          # all layers except the final cut of the abstention column
                h = mod(h)                 # (batch, N, m + 1) probabilities
            if self.deter_train:
                full = torch.nn.functional.softmax(h / self.smoothing, dim=2)
            else:
                full = torch.nn.functional.one_hot(torch.argmax(h, dim=2), num_classes=h.shape[2]).to(h.dtype)
            allocs = full[:, :, :-1]
        else:
            allocs = self.allocation_head(x)
        payments = self.payment_head(x)[:, :-1] * budget
        return allocs, payments

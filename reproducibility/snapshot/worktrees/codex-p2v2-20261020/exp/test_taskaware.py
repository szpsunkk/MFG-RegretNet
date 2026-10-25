"""Regression tests for audited semantic failures and the new task objective."""
import math
import unittest

import torch
import torch.nn.functional as F

from common import RepairedRunner
from contracts import settlement, utility, balanced_target, MIN_EVENT_EPSILON
from mechanisms import allocate
from task_proxy import TaskProxy, EPS_GRID, STEP_GRID
from tdsc_dpsgd_runner import DPSGDConfig, EXPANDED_RDP_ORDERS
from tdsc_rdp_cache import exact_rdp_cache


class TaskAwareTests(unittest.TestCase):
    def test_pac_vcg_cap_understatement_counterexamples_cancel_payment(self):
        truth = torch.tensor([[[.1, 1., 100.], [.2, 1., 100.], [.9, 1., 100.]]])
        dev = truth.clone()
        dev[0, 0, 1] = .1
        for arm in ("PAC", "VCG"):
            e0, p0 = allocate(arm, truth, 1.)
            ed, pd = allocate(arm, dev, 1.)
            self.assertEqual(float(ed[0, 0]), 0.)
            self.assertEqual(float(pd[0, 0]), 0.)
            self.assertGreater(float(utility(truth, e0, p0)[0, 0]), 0.)

    def test_legacy_cap_adapter_remains_a_labelled_performance_control(self):
        truth = torch.tensor([[[.1, 1., 100.], [.2, 1., 100.], [.9, 1., 100.]]])
        dev = truth.clone()
        dev[0, 0, 1] = .1
        e0, p0 = allocate("PAC-capped-legacy", truth, 1.)
        ed, pd = allocate("PAC-capped-legacy", dev, 1.)
        gain = utility(truth, ed, pd)[0, 0] - utility(truth, e0, p0)[0, 0]
        self.assertAlmostEqual(float(gain), .09, places=6)

    def test_true_type_is_separate_from_report(self):
        truth = torch.tensor([[[.8, 1., 100.]]])
        # Cheap reported cost cannot override the client's private true cost.
        e, p, ok = settlement(truth, torch.tensor([[.8]]), torch.tensor([[.1]]))
        self.assertFalse(bool(ok[0, 0]))
        self.assertEqual(float(e.sum() + p.sum()), 0.)

    def test_zero_tiny_over_cap_and_failed_calibration_settle_nothing(self):
        truth = torch.tensor([[[.2, 1., 100.]] * 4])
        exposure = torch.tensor([[0., 5e-6, 1.1, .5]])
        pay = torch.ones_like(exposure)
        e, p, ok = settlement(truth, exposure, pay, torch.tensor([[True, True, True, False]]))
        self.assertFalse(ok.any())
        self.assertEqual(float(e.sum() + p.sum()), 0.)

    def test_fixed_public_teacher_budget_ir_and_capacity(self):
        g = torch.Generator().manual_seed(44)
        reports = torch.rand(100, 10, 3, generator=g)
        reports[..., 1] = .1 + 4.9 * reports[..., 1]
        reports[..., 2] = 10 + 490 * reports[..., 2]
        budget = 1 + 100 * torch.rand(100, 1, generator=g)
        for ceiling in (1., 2.):
            e, p = balanced_target(reports, budget, ceiling)
            self.assertTrue((e <= reports[..., 1] + 1e-6).all())
            self.assertTrue((p.sum(-1) <= budget[:, 0] + 1e-5).all())
            self.assertTrue((p - reports[..., 0] * e >= -1e-6).all())

    def proxy_fixture(self):
        proxy = TaskProxy.__new__(TaskProxy)
        proxy.eg, proxy.sg = EPS_GRID.double().log(), STEP_GRID.double().log()
        proxy.log_sigma = torch.log(2 / EPS_GRID.double()).expand(len(STEP_GRID), -1)
        g = torch.Generator().manual_seed(100)
        data = {"base": torch.zeros(1, 50, 10, dtype=torch.float64),
                "deltas": torch.randn(1, 10, 50, 10, dtype=torch.float64, generator=g) * .1,
                "x": torch.randn(1, 16, 50, dtype=torch.float64, generator=g),
                "y": torch.arange(16)[None] % 10,
                "steps": torch.full((1, 10), 16., dtype=torch.float64),
                "learning_rate": torch.tensor([.1], dtype=torch.float64)}
        noise = torch.randn(1, 50, 10, dtype=torch.float64, generator=g)
        return proxy, data, noise

    def test_empty_task_loss_is_unchanged_model_loss_not_zero(self):
        proxy, data, noise = self.proxy_fixture()
        e = torch.zeros(1, 10, dtype=torch.float64, requires_grad=True)
        loss = proxy.loss(e, data, noise)
        self.assertAlmostEqual(float(loss), math.log(10), places=12)
        loss.backward()
        self.assertTrue(torch.isfinite(e.grad).all())
        self.assertEqual(float(e.grad.norm()), 0.)

    def test_task_objective_has_correct_finite_difference_gradient(self):
        proxy, data, noise = self.proxy_fixture()
        e = torch.linspace(.4, .7, 10, dtype=torch.float64)[None].requires_grad_(True)
        grad, = torch.autograd.grad(proxy.loss(e, data, noise), e)
        h = 1e-5
        plus, minus = e.detach().clone(), e.detach().clone()
        plus[0, 3] += h
        minus[0, 3] -= h
        reference = (proxy.loss(plus, data, noise) - proxy.loss(minus, data, noise)) / (2 * h)
        self.assertAlmostEqual(float(grad[0, 3]), float(reference), places=7)
        self.assertGreater(float(grad.norm()), 1e-6)

    def test_minimum_contract_is_actually_calibratable_for_event_shapes(self):
        cfg = DPSGDConfig(noise_multiplier=1., max_grad_norm=1., learning_rate=.1,
                          local_epochs=1, batch_size=64, delta=1e-5,
                          rdp_orders=EXPANDED_RDP_ORDERS, dataset_name="minimum-contract-regression")
        runner = RepairedRunner(torch.nn.Linear(2, 10), 1, cfg, seed=0)
        with exact_rdp_cache(maxsize=262144):
            for size in (10, 64, 128, 1024, 8192, 32768):
                p = runner.preview_participation(client_id=0, dataset_size=size, round_index=0,
                                                  allocated_event_epsilon=MIN_EVENT_EPSILON)
                self.assertTrue(p["event_contract"]["one_sided_target_satisfied"])
        self.assertEqual(runner._external_ledgers[0].state_version, 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)

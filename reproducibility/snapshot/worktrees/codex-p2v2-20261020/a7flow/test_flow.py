"""Numerical checks against scalar references; no image data or FL outcomes."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import flow_state
from tau import KAPPAS, AMOUNTS, selected, interpolate, grid_j
from model import Selector, reference_allocation, reference_mechanism
from check_v2 import profiles
from v2_model import P2V2
import p2_reference as REF


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.table = {mode: np.full((11, 9), .12).tolist() for mode in ('budget', 'fixed_total')}

    def test_cpu_grid_matches_scalar_registered_construction(self):
        values = profiles(np.random.default_rng(123), 4)
        kappas = np.array([.003, 3., 30., 300.])
        for fixed in (False, True):
            for count in (1, 4, 10):
                for amount in (1., 8., 30.):
                    actual = grid_j(values, count, amount, kappas=kappas, grid=400, fixed=fixed)
                    expected = []
                    for row in values:
                        pi = row[:, 2] / row[:, 2].sum()
                        indices = np.argsort(-row[:, 2], kind='stable')[:count]
                        caps = np.zeros(10); caps[indices] = row[indices, 1]
                        target = np.zeros(10); target[indices] = row[indices, 2]/row[indices, 2].sum()
                        high = min(amount, caps.sum())
                        totals = [high] if fixed else np.linspace(high/400, high, 400)
                        all_j = []
                        for total in totals:
                            x = REF.drop_below_min_event(REF._fill_for_mu(total*target, caps, total))
                            all_j.append([REF.j_kappa(x, pi, k) for k in kappas])
                        expected.append(np.min(all_j, axis=0))
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_bilinear_interpolation_and_edges(self):
        xx = (np.log(KAPPAS)-np.log(KAPPAS[0]))/np.log(KAPPAS[-1]/KAPPAS[0])
        yy = (np.log(AMOUNTS)-np.log(AMOUNTS[0]))/np.log(AMOUNTS[-1]/AMOUNTS[0])
        values = .01 + .08*xx[:, None] + .09*yy[None] + .03*xx[:, None]*yy[None]
        table = {mode: values.tolist() for mode in ('budget', 'fixed_total')}
        net = Selector('nob_p2v2', table).double()
        for k, amount in ((.0001, .01), (.1, 10), (2., 17.), (1000., 500.)):
            got = float(net.threshold(torch.tensor([k], dtype=torch.float64),
                                     torch.tensor([amount], dtype=torch.float64), 'budget'))
            self.assertAlmostEqual(got, interpolate(table, k, amount, 'budget'), places=13)
        self.assertEqual(interpolate(table, 1e-9, .1, 'budget'), values[0, 0])

    def test_empty_selection_and_ties_choose_lowest_index(self):
        got = selected(np.array([[.1]*10]), [.25])
        self.assertEqual(np.flatnonzero(got[0]).tolist(), [0])
        all_clients = selected(np.array([[.1]*10]), [.1])
        self.assertTrue(all_clients.all())

    def test_threshold_objective_gather_preserves_profile_axis(self):
        bank = np.arange(4*3*2*10).reshape(4, 3, 2, 10)
        counts = np.array([1, 7, 4, 10])
        gathered = bank[np.arange(4), :, :, counts-1]
        expected = np.stack([bank[i, :, :, counts[i]-1] for i in range(4)])
        np.testing.assert_array_equal(gathered, expected)

    def test_selector_allocations_and_myerson_payments_match_reference(self):
        row = profiles(np.random.default_rng(678), 1)[0]
        for family in flow_state.FAMILIES:
            torch.manual_seed(42)
            net = Selector(family, self.table).double().eval()
            r = torch.tensor(row[None]); b = torch.tensor([[10.]]); k = torch.tensor([[3.]])
            with torch.no_grad():
                for fixed in (None, 8.):
                    e, p = net.mechanism(r, b.double(), k.double(), fixed)
                    x, pay = reference_mechanism(row, 10., 3., net.numpy_total(10., 3.), self.table, fixed)
                    np.testing.assert_allclose(e[0].numpy(), x, atol=1e-10, rtol=1e-10)
                    np.testing.assert_allclose(p[0].numpy(), pay, atol=1e-10, rtol=1e-10)
                    self.assertLessEqual(float(p.sum()), 10.+1e-9)
                    self.assertTrue(bool((p-r[..., 0]*e >= -1e-9).all()))

    def test_zero_threshold_equals_original_and_preserves_gradient(self):
        table = {mode: np.zeros((11, 9)).tolist() for mode in ('budget', 'fixed_total')}
        torch.manual_seed(42); original = P2V2('mfg_p2v2', 1.).double()
        torch.manual_seed(42); selector = Selector('mfg_p2v2', table).double()
        reports = torch.tensor(profiles(np.random.default_rng(990), 8))
        budget = torch.full((8, 1), 10., dtype=torch.float64)
        kappa = torch.full((8, 1), 3., dtype=torch.float64)
        a = original.objective(reports, budget, kappa).mean()
        b = selector.objective(reports, budget, kappa).mean()
        self.assertAlmostEqual(float(a), float(b), places=12)
        a.backward(); b.backward()
        for (_, p), (_, q) in zip(original.named_parameters(), selector.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-12)

    def test_conditional_jobs_keep_candidates_out_of_old_folders(self):
        from continue_a7 import phase_job
        for arm in flow_state.GAMMA:
            for phase in ('training', 'economics'):
                job = phase_job(phase, arm, 'mfg_p2v2')
                self.assertIn('/continuation_A7/' + arm + '/', job['out'])
                self.assertIn('/a7flow/run_phase.py', job['command'][2])
                self.assertFalse(job['gpu'])

    def test_memory_admission_reserves_4g_per_unobserved_worker(self):
        from resource_worker import reserve
        state = {'free_mib': 7167, 'ram_available_mib': 32000}
        self.assertFalse(reserve(state))
        state['free_mib'] = 7168
        self.assertTrue(reserve(state))
        self.assertFalse(reserve(state, 1))
        state['free_mib'] = 11264
        self.assertTrue(reserve(state, 1))

    def test_stage3_reuses_original_rule_with_separate_A7_allocations(self):
        import run_phase
        import stage3_v2
        import qv_jobs
        previous = {n: getattr(stage3_v2, n) for n in ('OUT', 'PAPER', 'check_identity', 'load_model', 'build_contract_job')}
        try:
            with patch.object(run_phase, 'initialize', return_value=flow_state.candidate('QV-g1')), \
                 patch.object(qv_jobs, 'build_contract_job', return_value={'sentinel': True}) as build:
                stage = run_phase.configure_stage3('QV-g1')
                stage.build_contract_job('stage3_v2', inputs=[], extra={'gamma': 1.})
                args, kwargs = build.call_args
                self.assertEqual(args[0], 'stage3_A7')
                self.assertEqual(kwargs['extra']['selected_proxy'], 'QV-g1')
                self.assertIn(flow_state.OUT / 'A7_continuation_manifest.json', kwargs['inputs'])
        finally:
            for name, value in previous.items():
                setattr(stage3_v2, name, value)


if __name__ == '__main__':
    unittest.main()

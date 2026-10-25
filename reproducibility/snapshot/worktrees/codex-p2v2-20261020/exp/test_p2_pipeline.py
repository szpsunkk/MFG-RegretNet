"""Boundary, scheduling, true-utility and selection tests on small CPU fixtures."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import sys
import contextlib
import io
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from common import OUT, FROZEN, WORK, atomic_json, digest, isolated_rng
from contracts import settlement, utility
from mechanisms import make_model
from p2_core import (P2Head, FAMILIES, raw, training_term, allocation, search, deviation_utility,
                     exact_metrics, array_hash, load_any, STAGE0_ARMS)
import p2_reference as P
from p2_state import terminal_previous, check_reference_hashes, immutable_json
from p2_jobs import calibration
from p2_results import paired_interval, stage0_decision, stage2_select
from train_mechanism import sample_profiles


class P2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.truth, self.budget = sample_profiles(2, torch.Generator().manual_seed(42), "cpu")
        self.kappa = torch.ones(2, 1)

    def test_frozen_inputs_and_execution_layer_identical(self):
        check_reference_hashes()
        source = WORK.parent / "codex-taskaware-20261017"
        for path in FROZEN.rglob("*.py"):
            self.assertEqual(digest(path), digest(source / "frozen" / path.relative_to(FROZEN)))
        self.assertEqual(digest(WORK / "exp/contracts.py"), digest(source / "exp/contracts.py"))

    def test_reference_empty_bug_is_exposed_and_wrapper_does_not_reward_it(self):
        reports = torch.tensor([[[.5, 1., 100.]] * 10], dtype=torch.float64)
        kappa = torch.tensor([.01], dtype=torch.float64)
        empty = torch.zeros(1, 10, dtype=torch.float64, requires_grad=True)
        before = P.p2_training_term(empty, reports, kappa)
        self.assertAlmostEqual(float(before), .0101, places=10)
        self.assertTrue(math.isinf(P.j_kappa(np.zeros(10), np.full(10, .1), .01)))
        after = training_term(empty, reports, kappa)
        self.assertTrue(torch.isfinite(after).all())
        self.assertGreater(float(after), 1.)
        after.sum().backward()
        self.assertTrue(torch.isfinite(empty.grad).all())
        tiny = torch.full_like(empty, .009)
        rejected = settlement(reports, tiny, reports[..., 0] * tiny)[0]
        self.assertEqual(float(rejected.sum()), 0.)
        self.assertGreater(float(training_term(rejected, reports, kappa)), 1.)

    def test_nonempty_training_unchanged(self):
        e = self.truth[..., 1] * .5
        self.assertTrue(torch.equal(training_term(e, self.truth, self.kappa), P.p2_training_term(e, self.truth, self.kappa)))
        self.assertTrue(torch.allclose(training_term(self.truth[..., 1], self.truth, self.kappa), torch.ones(2)))

    def test_k0_counterexample_is_preserved(self):
        cap, v = np.array([5.] + [.1] * 9), np.full(10, .5)
        share = np.array([.99] + [.01 / 9] * 9)
        e, _, info = P.p2_oracle(cap, v, share, 30., 0.)
        self.assertTrue(np.array_equal(e, np.array([5.] + [0.] * 9)))
        self.assertAlmostEqual(info["j"], .04)
        self.assertLess(info["j"], P.j_kappa(cap, share, 0.))

    def test_heads_preserve_original_output_equations_and_add_only_kappa(self):
        for family in FAMILIES:
            model = P2Head(family)
            base = make_model(family[:-3])
            state = base.state_dict()
            modified = model.network.state_dict()
            for name, value in state.items():
                if name in ("payment_head.0.weight", "allocation_head.0.weight"):
                    modified[name][:, :-1] = value
                    modified[name][:, -1] = 0
                    self.assertEqual(modified[name].shape[1], value.shape[1] + 1)
                else:
                    self.assertEqual(value.shape, modified[name].shape)
                    modified[name] = value
            model.network.load_state_dict(modified)
            expected = base((self.truth, self.budget))
            got = model((self.truth, self.budget, self.kappa))
            for a, b in zip(expected, got):
                self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=2e-6))
            e, p = raw(model, self.truth, self.budget, self.kappa)
            loss = training_term(settlement(self.truth, e, p)[0], self.truth, self.kappa).mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_checkpoint_requires_explicit_kappa_and_roundtrips(self):
        with tempfile.TemporaryDirectory(dir=OUT / "validation") as folder:
            model = P2Head("mfg_p2")
            path = Path(folder) / "model.pt"
            torch.save({"kind": "mfg_p2", "config": {"kappa_range": [.003, 300.]}, "state_dict": model.state_dict()}, path)
            loaded = load_any(path)
            for a, b in zip(model((self.truth, self.budget, self.kappa)), loaded((self.truth, self.budget, self.kappa))):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(ValueError):
                loaded((self.truth, self.budget))

    def test_unilateral_search_holds_true_types_and_size_limits(self):
        model = P2Head("mfg_p2")
        truth = self.truth.clone()
        truth[0, 0, 2] = 10.
        dev = search(model, truth, self.budget, self.kappa, strong=True)
        self.assertTrue(((dev[..., 0] >= 0) & (dev[..., 0] <= 1)).all())
        self.assertTrue(((dev[..., 1] >= 0) & (dev[..., 1] <= truth[..., 1])).all())
        self.assertTrue(((dev[..., 2] >= 10) & (dev[..., 2] <= truth[..., 2])).all())
        found = deviation_utility(model, truth, self.budget, self.kappa, dev)
        e, p = raw(model, truth, self.budget, self.kappa)
        self.assertTrue((found >= utility(truth, e, p) - 1e-7).all())
        # Explicit single-client recomputation must agree with the vectorized diagonal.
        profile = truth[0:1].clone()
        profile[0, 3] = dev[0, 3]
        ep, pa = raw(model, profile, self.budget[0:1], self.kappa[0:1])
        expected = utility(truth[0:1], ep, pa)[0, 3]
        self.assertAlmostEqual(float(expected), float(found[0, 3]), places=5)
        fixed = search(model, truth, self.budget, self.kappa, sizes=False)
        self.assertTrue(torch.equal(fixed[..., 2], truth[..., 2]))

    def test_reference_cost_payments_pass_gate_exactly(self):
        for arm in ("Oracle-k0", "Oracle-k3", "Aligned@PAC", "Aligned@CSRA"):
            d = allocation(arm, self.truth, 30., 0. if arm == "Oracle-k0" else 3.)
            self.assertTrue(torch.equal(d["proposed"], d["executed"]))
            self.assertTrue(torch.equal(d["payment"], self.truth.double()[..., 0] * d["executed"]))
            self.assertTrue((d["payment"].sum(-1) <= 30 + 1e-7).all())

    def test_analytic_baselines_preserve_original_settlement(self):
        from mechanisms import allocate
        for arm in ("PAC", "VCG", "PAC-capped-legacy", "VCG-capped-legacy", "CSRA", "Lin", "MFG-Pricing", "Posted-price"):
            expected_e, expected_p = allocate(arm, self.truth, 30.)
            got = allocation(arm, self.truth, 30., 3.)
            self.assertTrue(torch.equal(expected_e, got["executed"]))
            self.assertTrue(torch.equal(expected_p, got["payment"]))

    def test_privacy_calibration_reachability(self):
        r = torch.tensor([[[.1, 2., 100.]] * 10] * 4)
        feasible = calibration(r, "Posted-price", None, .8, 3.)
        self.assertTrue(feasible["feasible"])
        self.assertLessEqual(abs(feasible["relative_error_100"]), .005)
        impossible = calibration(r, "Posted-price", None, 3., 3.)
        self.assertFalse(impossible["feasible"])
        self.assertEqual(impossible["sampled_budgets"][0][0], .5)

    def test_infinite_market_not_dropped_from_j_average(self):
        e = self.truth[..., 1].clone()
        e[0] = 0
        result = exact_metrics(e, self.truth, 3.)
        self.assertTrue(result["j_infinite"])
        self.assertIsNone(result["j_kappa"])

    def test_manifest_immutable_and_allocation_hash_sensitive(self):
        with tempfile.TemporaryDirectory(dir=OUT / "validation") as folder:
            path = Path(folder) / "manifest.json"
            immutable_json(path, {"budget": 30.})
            immutable_json(path, {"budget": 30.})
            with self.assertRaises(RuntimeError):
                immutable_json(path, {"budget": 100.})
        e = np.ones((2, 10))
        original = array_hash(e)
        e[0, 0] += 1e-12
        self.assertNotEqual(original, array_hash(e))

    def test_campaign_refuses_changed_source(self):
        import p2_state
        with patch.object(p2_state, "read", return_value={"source_sha256": {"code": "original"}, "input_sha256": {}}):
            with patch.object(p2_state, "source_identity", return_value={"code": "changed"}):
                with self.assertRaises(RuntimeError):
                    p2_state.check_identity()

    def test_uniform_control_uses_selected_allocation_each_round(self):
        with tempfile.TemporaryDirectory(dir=OUT / "validation") as folder:
            model = P2Head("mfg_p2")
            path = Path(folder) / "model.pt"
            torch.save({"kind": "mfg_p2", "config": {"kappa_range": [.003, 300.]}, "state_dict": model.state_dict()}, path)
            anchor = allocation("mfg_p2", self.truth, 30., 3., checkpoint=path)
            control = allocation("Uniform-DP", self.truth, 30., 3., reference=path)
            self.assertTrue(torch.allclose(anchor["executed"].double().sum(-1), control["executed"].sum(-1), atol=1e-8, rtol=1e-7))

    def test_waits_for_full_previous_campaign_including_confirmation(self):
        old = {"complete": 252, "total": 252, "active": [], "failed": []}
        for parent in ({"status": "running", "phase": "development"},
                       {"status": "complete", "phase": "training_complete"},
                       {"status": "complete", "phase": "development_complete", "promotion_gate_pass": True},
                       {"status": "failed", "phase": "failed"}):
            self.assertFalse(terminal_previous(old, parent)[0])
        self.assertTrue(terminal_previous(old, {"status": "complete", "phase": "complete"})[0])
        self.assertTrue(terminal_previous(old, {"status": "complete", "phase": "development_complete", "promotion_gate_pass": False})[0])
        old["active"] = [1]
        self.assertFalse(terminal_previous(old, {"status": "complete", "phase": "complete"})[0])

    def test_stage0_complete_matrix_and_no_diagnostic_selection(self):
        rows = [{"dataset": ds, "alpha": alpha, "seed": seed, "arm": arm,
                 "accuracy": .9 if arm == "Oracle-k0-datasize" else .62 if arm == "Oracle-k3" else .6}
                for ds in ("MNIST", "CIFAR10") for alpha in (.5, .1) for seed in (14, 15) for arm in STAGE0_ARMS]
        hat = {"by_cell": {f"{d}_a{a}": 1. for d in ("MNIST", "CIFAR10") for a in (.5, .1)}}
        result = stage0_decision(rows, hat)
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["selected_rule"]["MNIST"], "Oracle-k3")
        with self.assertRaises(ValueError):
            stage0_decision(rows[:-1], hat)
        with self.assertRaises(ValueError):
            stage0_decision(rows[:-1] + [rows[0]], hat)

    def stage2_fixture(self):
        cells = [{"dataset": d, "alpha": a, "mode": m, "value": 30. if m == "budget" else .8}
                 for d in ("MNIST", "CIFAR10") for a in (.5, .1) for m in ("budget", "privacy")]
        candidates = [{"arm": family, "kind": family, "eligible": True} for family in FAMILIES]
        rows = [{**cell, "seed": s, "arm": arm, "accuracy": .62 if arm in FAMILIES else .6,
                 "loss": 1., "matching_pass": True}
                for cell in cells for s in (14, 15) for arm in FAMILIES + ("V8F", "CSRA", "PAC")]
        return cells, candidates, rows

    def test_stage2_does_not_promote_missing_seed_or_unmatched_cell(self):
        cells, candidates, rows = self.stage2_fixture()
        self.assertTrue(stage2_select(rows, candidates, cells)["gate_pass"])
        remove = next(r for r in rows if r["arm"] == "mfg_p2")
        missing = [r for r in rows if r is not remove]
        self.assertFalse(stage2_select(missing, candidates, cells)["gate_pass"])
        invalid = copy.deepcopy(rows)
        next(r for r in invalid if r["arm"] == "mfg_p2" and r["mode"] == "privacy")["matching_pass"] = False
        self.assertFalse(stage2_select(invalid, candidates, cells)["gate_pass"])
        failed = copy.deepcopy(rows)
        for r in failed:
            if r["dataset"] == "MNIST" and r["arm"] == "mfg_p2":
                r["accuracy"] = .59
        self.assertFalse(stage2_select(failed, candidates, cells)["gate_pass"])

    def test_seed_paired_interval(self):
        out = paired_interval([.01, .02, .03])
        self.assertEqual(out["n_seeds"], 3)
        self.assertAlmostEqual(out["mean_difference"], .02)
        self.assertLess(out["ci95_low"], 0)
        self.assertGreater(out["ci95_high"], .04)

    def test_explicit_tiny_noise_path_uses_real_runner(self):
        from estimate_kappa import CaptureRunner
        from tdsc_dpsgd_runner import DPSGDConfig, DPSGDClientRequest, EXPANDED_RDP_ORDERS
        from tdsc_rdp_cache import exact_rdp_cache
        from torch import nn
        g = torch.Generator().manual_seed(42)
        dataset = TensorDataset(torch.randn(8, 3, generator=g), torch.randint(0, 2, (8,), generator=g))
        cfg = DPSGDConfig(noise_multiplier=1e-6, max_grad_norm=1., learning_rate=.1, batch_size=4,
                          local_epochs=1, rdp_orders=EXPANDED_RDP_ORDERS, dataset_name="P2-functional-test")
        with isolated_rng(42):
            model = nn.Linear(3, 2)
        runner = CaptureRunner(model, 2, cfg, seed=42, device="cpu")
        runner.captured = []
        with exact_rdp_cache(maxsize=1000):
            preview = runner.preview_participation(client_id=0, dataset_size=8, round_index=0, allocated_event_epsilon=1.)
            self.assertGreater(preview["noise_multiplier"], 1e-6)
            result = runner.run_round(0, [DPSGDClientRequest(client_id=i, dataset=dataset, noise_multiplier=1e-6,
                                                            aggregation_weight=8.) for i in range(2)])
        self.assertEqual(result.release_count, 2)
        self.assertEqual(np.asarray(runner.captured).shape, (2, 8))
        self.assertTrue(np.isfinite(runner.captured).all())
        self.assertTrue(all(e["accountant_parity_pass"] for e in result.participation_events))

    def test_precomputed_runner_end_to_end_and_completed_resume(self):
        import run_cell
        from tdsc_rdp_cache import exact_rdp_cache
        g = torch.Generator().manual_seed(42)
        dataset = TensorDataset(torch.randn(8, 3, generator=g), torch.randint(0, 10, (8,), generator=g))
        reports = torch.tensor([[[.5, 1., 100.]] * 10] * 2)
        exposure, pay = np.full((2, 10), .5), np.full((2, 10), .25)
        spec = {"dataset": "MNIST", "alpha": .5, "seed": 14, "arm": "Oracle-k3", "kappa": 3.,
                "budget": 30., "rounds": 2, "aggregation": "eps", "stage": "smoke", "smoke_size": None,
                "reports_sha256": array_hash(reports), "split_sha256": "cpu-fixture",
                "executed": exposure.tolist(), "payment": pay.tolist(),
                "proposed": exposure.tolist(), "offered_payment": pay.tolist(),
                "allocation_sha256": array_hash(exposure), "payment_sha256": array_hash(pay),
                "input_sha256": {}, "comparison": "CPU functional fixture", "non_strategic_reference": True}
        original_read = run_cell.read
        def read_with_fixture_manifest(path):
            return {"source_sha256": {"scope": "unittest CPU fixture"}} if str(path).endswith("launch_manifest.json") else original_read(path)
        with tempfile.TemporaryDirectory(dir=OUT / "validation") as folder:
            folder = Path(folder)
            spec_path = folder / "allocation.json"
            atomic_json(spec_path, spec)
            out = folder / "cell"
            argv = ["run_cell", "--dataset", "MNIST", "--alpha", ".5", "--seed", "14", "--arm", "Oracle-k3",
                    "--budget", "30", "--rounds", "2", "--stage", "smoke", "--device", "cpu", "--out", str(out),
                    "--allocation-spec", str(spec_path), "--spec-sha256", digest(spec_path)]
            with patch.object(sys, "argv", argv), patch.object(run_cell, "check_identity"), \
                 patch.object(run_cell, "data_for_cell", return_value=([dataset] * 10, dataset, {"sha256": "cpu-fixture"}, lambda: torch.nn.Linear(3, 10))), \
                 patch.object(run_cell, "market_reports", return_value=reports), patch.object(run_cell, "read", side_effect=read_with_fixture_manifest), \
                 exact_rdp_cache(maxsize=1000), contextlib.redirect_stdout(io.StringIO()):
                run_cell.main()
                saved = original_read(out / "result.json")
                run_cell.main()
                self.assertEqual(saved, original_read(out / "result.json"))
            self.assertEqual(saved["summary"]["participants_per_round"], 10)
            self.assertAlmostEqual(saved["summary"]["epsilon_per_client_round"], .5)
            self.assertAlmostEqual(saved["summary"]["payment_per_round"], 2.5)
            self.assertTrue(saved["summary"]["accountant_parity_pass"])
            self.assertEqual(saved["events_sha256"], digest(out / "events.json"))


if __name__ == "__main__":
    (OUT / "validation").mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)

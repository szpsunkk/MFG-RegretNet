"""Adoption, MF-Net integration and immutable handoff regression checks."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from common import digest, atomic_json, ROOT, WORK, FROZEN, isolated_rng
from a1_rules import (confirmation_rule, confirmation_cells, stage2_candidate_gate,
                      adoption_priority, EXTERNAL)
from a1_mf import MFAdapter
from mf_reference import MFNet
from p2_reference import kappa_feature
from p2_core import raw, allocation, load_any, deviation_utility
from contracts import utility
from a1_inputs import verify_handoff, register_mf, LN_PROTOCOL_HASH, MF_REFERENCE_HASH


def fixture_rows(candidate="mf_p2", cells=None, seeds=(231, 232, 233)):
    cells = cells or [{"dataset": ds, "alpha": .5, "mode": "budget", "value": 30.} for ds in ("MNIST", "CIFAR10")]
    rows = [{**cell, "arm": arm, "seed": seed, "accuracy": .81 if arm == candidate else .8,
             "loss": .5, "matching_pass": True}
            for cell in cells for arm in (*EXTERNAL, candidate, "Uniform-DP", "V8F") for seed in seeds]
    return rows, cells


class A1Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_best_external_is_chosen_once_across_seeds(self):
        rows, cells = fixture_rows()
        for r in rows:
            if r["arm"] == "PAC":
                r["accuracy"] = {231: .9, 232: .7, 233: .7}[r["seed"]]
            if r["arm"] in ("Uniform-DP", "V8F"):
                r["accuracy"] = 1.
        result = confirmation_rule(rows, "mf_p2", cells)
        self.assertTrue(result["pass"])
        self.assertTrue(all(r["external_arm"] == "PAC-capped-legacy" for r in result["per_cell"]))
        np.testing.assert_allclose(result["datasets"]["CIFAR10"]["seed_differences_pp"], [1., 1., 1.])
        self.assertEqual(result["datasets"]["MNIST"]["interval_pp"]["n_seeds"], 3)

    def test_positive_average_is_not_three_positive_cifar_seeds(self):
        rows, cells = fixture_rows()
        next(r for r in rows if r["dataset"] == "CIFAR10" and r["arm"] == "mf_p2" and r["seed"] == 231)["accuracy"] = .799
        result = confirmation_rule(rows, "mf_p2", cells)
        self.assertFalse(result["pass"])
        self.assertGreater(result["datasets"]["CIFAR10"]["interval_pp"]["mean_difference"], 0)

    def test_exact_zero_cifar_gain_fails(self):
        rows, cells = fixture_rows()
        for r in rows:
            if r["dataset"] == "CIFAR10" and r["arm"] == "mf_p2":
                r["accuracy"] = .8
        self.assertFalse(confirmation_rule(rows, "mf_p2", cells)["pass"])

    def test_mnist_margin_uses_percentage_points(self):
        rows, cells = fixture_rows()
        for r in rows:
            if r["dataset"] == "MNIST" and r["arm"] == "mf_p2":
                r["accuracy"] = .797
        self.assertTrue(confirmation_rule(rows, "mf_p2", cells)["pass"])
        for r in rows:
            if r["dataset"] == "MNIST" and r["arm"] == "mf_p2":
                r["accuracy"] = .7969
        self.assertFalse(confirmation_rule(rows, "mf_p2", cells)["pass"])

    def test_missing_candidate_is_not_a_good_subset(self):
        rows, cells = fixture_rows()
        rows = [r for r in rows if not (r["dataset"] == "CIFAR10" and r["arm"] == "mf_p2" and r["seed"] == 231)]
        self.assertIsNone(confirmation_rule(rows, "mf_p2", cells)["pass"])

    def test_unreachable_external_requires_record(self):
        cells = [{"dataset": ds, "alpha": .5, "mode": "privacy", "value": .8} for ds in ("MNIST", "CIFAR10")]
        rows, _ = fixture_rows(cells=cells)
        missing = [r for r in rows if r["arm"] == "PAC" and r["seed"] == 231]
        rows = [r for r in rows if not (r["arm"] == "PAC" and r["seed"] == 231)]
        self.assertIsNone(confirmation_rule(rows, "mf_p2", cells)["pass"])
        result = confirmation_rule(rows, "mf_p2", cells, exclusions=[{**r, "reason": "unreachable"} for r in missing])
        self.assertTrue(result["pass"])
        self.assertEqual(len(result["omissions"]), 2)

    def test_actual_privacy_mismatch_is_not_unreachability(self):
        rows, cells = fixture_rows(cells=[{"dataset": ds, "alpha": .5, "mode": "privacy", "value": .8} for ds in ("MNIST", "CIFAR10")])
        rows[0]["matching_pass"] = False
        self.assertIsNone(confirmation_rule(rows, "mf_p2", cells)["pass"])

    def test_taskaware_whole_cell_unreachability(self):
        cells = [{"dataset": ds, "alpha": .5, "mode": "privacy", "value": .8} for ds in ("MNIST", "CIFAR10")]
        rows, _ = fixture_rows(cells=cells)
        rows = [r for r in rows if r["arm"] != "PAC"]
        exclusions = [{**cell, "arm": "PAC", "reason": "target not reached for every paired seed",
                       "calibrations": {str(s): {"feasible": s != 231} for s in (231, 232, 233)}} for cell in cells]
        value = confirmation_rule(rows, "mf_p2", cells, exclusions=exclusions)
        self.assertTrue(value["pass"])
        self.assertEqual(value["omissions"][0]["details"]["232"]["calibrations"], exclusions[0]["calibrations"])
        exclusions[0]["calibrations"]["231"]["feasible"] = True
        with self.assertRaises(ValueError):
            confirmation_rule(rows, "mf_p2", cells, exclusions=exclusions)

    def test_duplicate_seed_rejected(self):
        rows, cells = fixture_rows()
        with self.assertRaises(ValueError):
            confirmation_rule(rows + [rows[0]], "mf_p2", cells)

    def test_independent_mf_development_gate(self):
        rows, cells = fixture_rows(seeds=(14, 15))
        self.assertTrue(stage2_candidate_gate(rows, "mf_p2", cells)["gate_pass"])
        self.assertFalse(stage2_candidate_gate(rows, "mfg_p2", cells)["gate_pass"])

    def test_priority_is_not_highest_test_accuracy(self):
        self.assertEqual(adoption_priority(True, True, True, True)["model"], "MF-Net")
        self.assertEqual(adoption_priority(False, True, True, True)["model"], "mfg_p2")
        self.assertTrue(adoption_priority(False, True, True, True)["mf_separate_large_population_study"])
        self.assertEqual(adoption_priority(None, None, True, True)["status"], "provisional")
        self.assertEqual(adoption_priority(False, False, False, False)["model"], "V8F")

    def test_cell_mappings_are_complete(self):
        self.assertEqual(len(confirmation_cells()), 18)
        self.assertEqual(len(confirmation_cells(True)), 16)

    def test_adapter_does_not_multiply_capacity_twice(self):
        with isolated_rng(123):
            net = MFNet()
        model = MFAdapter(net)
        reports = torch.tensor([[[.2, 3., 1000.], [.8, 2., 4000.]]])
        budget, k = torch.tensor([[30.]]), torch.tensor([[3.]])
        actual = raw(model, reports, budget, k)
        expected = net(reports, budget, kappa_feature(k, .003, 300.))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)

    def test_fl_adapter_uses_raw_sizes_without_mutating_reports(self):
        with isolated_rng(124):
            model = MFAdapter(MFNet())
        reports = torch.tensor([[[.2, 3., 10.], [.8, 2., 500.]]])
        original = reports.clone()
        sizes = [20, 8000]
        detail = allocation("mf_p2", reports, 30., 3., checkpoint="fixture-unused", model=model, raw_sizes=sizes)
        expected_reports = reports.clone()
        expected_reports[..., 2] = torch.tensor(sizes)
        expected, _ = raw(model, expected_reports, torch.tensor([[30.]]), torch.tensor([[3.]]))
        torch.testing.assert_close(detail["proposed"], expected)
        torch.testing.assert_close(reports, original)
        with self.assertRaises(ValueError):
            allocation("mf_p2", reports, 30., 3., checkpoint="fixture-unused", model=model)

    def test_mf_deviation_recomputes_population_and_payments(self):
        with isolated_rng(125):
            model = MFAdapter(MFNet())
        truth = torch.tensor([[[.2, 3., 50.], [.8, 2., 400.]]])
        mis = truth.clone()
        mis[0, 0] = torch.tensor([.7, 1., 10.])
        b, k = torch.tensor([[10.]]), torch.tensor([[3.]])
        actual = deviation_utility(model, truth, b, k, mis)
        for i in range(2):
            profile = truth.clone()
            profile[0, i] = mis[0, i]
            e, p = raw(model, profile, b, k)
            expected = utility(truth, e, p)
            torch.testing.assert_close(actual[0, i], expected[0, i])

    def test_handoff_integrity_and_no_seed_switch(self):
        import a1_inputs
        with tempfile.TemporaryDirectory() as tmp:
            folder, out = Path(tmp) / "input", Path(tmp) / "out"
            folder.mkdir()
            with isolated_rng(126):
                net = MFNet()
            torch.save({"kind": "MF-Net", "state_dict": net.state_dict(), "config": {}}, folder / "selected_mf.pt")
            selection = {"selected": {"MF-Net": {"seed": 61, "checkpoint_sha256": digest(folder / "selected_mf.pt")}},
                         "selection_used_official_test": False, "dev_profile_seed": 71}
            atomic_json(folder / "stage_a_selection.json", selection)
            value = {"schema": "p2-a1-mf-handoff-v1", "protocol": "ln-v1", "family": "MF-Net",
                     "protocol_sha256": LN_PROTOCOL_HASH, "reference_sha256": MF_REFERENCE_HASH,
                     "selection_used_official_test": False, "selected_before_fl": True, "training_seed": 61,
                     "checkpoint_sha256": digest(folder / "selected_mf.pt"), "selection_sha256": digest(folder / "stage_a_selection.json")}
            atomic_json(folder / "p2_handoff.json", value)
            verify_handoff(folder)
            with patch.object(a1_inputs, "OUT", out), patch.object(a1_inputs, "INBOX", folder), patch.object(a1_inputs, "freeze_stage_inputs"):
                first = register_mf()
                self.assertEqual(first, register_mf())
                selection["selected"]["MF-Net"]["seed"] = 62
                atomic_json(folder / "stage_a_selection.json", selection)
                value.update(training_seed=62, selection_sha256=digest(folder / "stage_a_selection.json"))
                atomic_json(folder / "p2_handoff.json", value)
                with self.assertRaises(RuntimeError):
                    register_mf()
            with (folder / "selected_mf.pt").open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaises(ValueError):
                verify_handoff(folder)

    def test_execution_layer_remains_byte_identical(self):
        parent = ROOT / "worktrees/codex-p2-101-20261019"
        for path in FROZEN.rglob("*.py"):
            self.assertEqual(digest(path), digest(parent / "frozen" / path.relative_to(FROZEN)))
        for name in ("contracts.py", "train_p2.py", "p2_reference.py", "mechanisms.py"):
            self.assertEqual(digest(WORK / "exp" / name), digest(parent / "exp" / name))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Data isolation, privacy matching and complete-cell promotion regressions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import campaign
from common import OUT, data_for_cell


class CampaignTests(unittest.TestCase):
    def test_proxy_development_and_private_records_are_disjoint(self):
        saved = json.loads((OUT / "public_bank.manifest.json").read_text())["public_data"]
        for dataset in ("MNIST", "CIFAR10"):
            _, evaluation, manifest, _ = data_for_cell(dataset, .1, 12, stage="dev")
            proxy = set(manifest["public_proxy_indices"])
            validation = set(manifest["development_indices"])
            private = set(sum(manifest["full_client_indices"], []))
            self.assertEqual(sorted(proxy), saved[dataset]["proxy_indices"])
            self.assertEqual(sorted(validation), saved[dataset]["development_indices"])
            self.assertEqual(sorted(validation), list(evaluation.indices))
            self.assertFalse(proxy & validation or proxy & private or validation & private)
            self.assertEqual(len(proxy | validation | private), 60000 if dataset == "MNIST" else 50000)
            self.assertEqual(sum(map(len, manifest["full_client_indices"])), len(private))

    def test_privacy_matching_preserves_reachable_and_unreachable_targets(self):
        reports = torch.tensor([[[.5, 2., 100.]] * 10] * 4)
        found = campaign.calibrate(reports, "Balanced-1", None, .8)
        self.assertTrue(found["feasible"])
        self.assertLessEqual(abs(found["relative_error"]), .005)
        missed = campaign.calibrate(reports, "Balanced-1", None, 2.)
        self.assertFalse(missed["feasible"])
        self.assertLess(missed["observed_max"], 1.001)

    def selection_fixture(self):
        config = {"mechanism_families": ["mfg"], "development_seeds": [12, 13],
                  "development_cells": [
                      {"dataset": ds, "alpha": .1 if mode == "budget" else .5,
                       "mode": mode, "value": 30. if mode == "budget" else .8}
                      for ds in ("MNIST", "CIFAR10") for mode in ("budget", "privacy")]}
        rows = [{"arm": "V8F", "kind": "mfg", "candidate": False, "eligible": True},
                {"arm": "new", "kind": "mfg", "candidate": True, "eligible": True, "task_weight": 1.}]
        records = [{**cell, "seed": seed, "arm": arm, "accuracy": .7 if arm == "new" else .6,
                    "loss": 1., "matching_pass": True}
                   for cell in config["development_cells"] for seed in config["development_seeds"]
                   for arm in ("V8F", "new")]
        return config, rows, records

    def choose(self, config, rows, records):
        with tempfile.TemporaryDirectory(prefix="selection-test-", dir=OUT / "smoke") as folder:
            with patch.object(campaign, "OUT", Path(folder)):
                return campaign.select(config, copy.deepcopy(rows), records)

    def test_only_complete_paired_development_can_promote(self):
        config, rows, records = self.selection_fixture()
        self.assertTrue(self.choose(config, rows, records)["promotion_gate_pass"])
        missing = [r for r in records if not (r["arm"] == "new" and r["dataset"] == "CIFAR10"
                                               and r["mode"] == "privacy" and r["seed"] == 13)]
        self.assertFalse(self.choose(config, rows, missing)["promotion_gate_pass"])
        # A duplicate cannot substitute for the missing cell even at the same count.
        missing.append(copy.deepcopy(next(r for r in missing if r["arm"] == "new")))
        self.assertFalse(self.choose(config, rows, missing)["promotion_gate_pass"])

    def test_one_regime_failure_or_mismatched_privacy_cannot_promote(self):
        config, rows, records = self.selection_fixture()
        failed = copy.deepcopy(records)
        for r in failed:
            if r["arm"] == "new" and r["mode"] == "privacy":
                r["accuracy"] = .5
        self.assertFalse(self.choose(config, rows, failed)["promotion_gate_pass"])
        records[-1]["matching_pass"] = False
        self.assertFalse(self.choose(config, rows, records)["promotion_gate_pass"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    (OUT / "smoke").mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)

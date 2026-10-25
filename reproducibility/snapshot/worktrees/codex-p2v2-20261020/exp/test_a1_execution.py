"""Exercise MF-Net through actual DP execution and reject invalid artifact reuse."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from common import atomic_json, digest, isolated_rng
from p2_core import allocation, array_hash
from a1_mf import MFAdapter
from mf_reference import MFNet
from a1_inputs import reusable
from a1_evaluate import load_rows


class ExecutionChecks(unittest.TestCase):
    def test_mf_runs_actual_dp_pipeline_and_checks_raw_sizes(self):
        import run_cell
        from tdsc_rdp_cache import exact_rdp_cache
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(42)
        data = TensorDataset(torch.randn(8, 3, generator=generator), torch.randint(0, 10, (8,), generator=generator))
        reports = torch.tensor([[[.5, 1., 500.]] * 10] * 2)
        model = MFAdapter(MFNet())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        detail = allocation("mf_p2", reports, 30., 3., model=model, checkpoint="fixture", raw_sizes=[8] * 10)
        spec = {"dataset": "MNIST", "alpha": .5, "seed": 14, "arm": "mf_p2", "kappa": 3.,
                "budget": 30., "rounds": 2, "aggregation": "eps", "stage": "smoke", "smoke_size": None,
                "reports_sha256": array_hash(reports), "split_sha256": "cpu-mf-fixture",
                "verified_raw_sizes": [8] * 10, "mf_input_rule": "actual raw sizes",
                "input_sha256": {}, "comparison": "CPU functional fixture", "non_strategic_reference": False}
        for field in ("executed", "payment", "proposed", "offered_payment"):
            spec[field] = detail[field].double().tolist()
        spec.update(allocation_sha256=array_hash(np.asarray(spec["executed"])),
                    payment_sha256=array_hash(np.asarray(spec["payment"])))
        original_read = run_cell.read
        def fixture_read(path):
            return {"source_sha256": {"scope": "unittest MF CPU fixture"}} if str(path).endswith("launch_manifest.json") else original_read(path)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "selected_mf.pt"
            torch.save({"kind": "MF-Net", "state_dict": model.network.state_dict()}, checkpoint)
            path, out = folder / "allocation.json", folder / "run"
            atomic_json(path, spec)
            argv = ["run_cell", "--dataset", "MNIST", "--alpha", ".5", "--seed", "14", "--arm", "mf_p2",
                    "--checkpoint", str(checkpoint), "--budget", "30", "--rounds", "2", "--stage", "smoke", "--device", "cpu",
                    "--out", str(out), "--allocation-spec", str(path), "--spec-sha256", digest(path)]
            with patch.object(sys, "argv", argv), patch.object(run_cell, "check_identity"), \
                 patch.object(run_cell, "data_for_cell", return_value=([data] * 10, data, {"sha256": "cpu-mf-fixture"}, lambda: torch.nn.Linear(3, 10))), \
                 patch.object(run_cell, "market_reports", return_value=reports), patch.object(run_cell, "read", side_effect=fixture_read), \
                 exact_rdp_cache(maxsize=1000), contextlib.redirect_stdout(io.StringIO()):
                run_cell.main()
                saved = original_read(out / "result.json")
                run_cell.main()
                self.assertEqual(saved, original_read(out / "result.json"))
                spec["verified_raw_sizes"][0] = 9
                atomic_json(path, spec)
                argv[-1] = digest(path)
                with self.assertRaisesRegex(RuntimeError, "raw size"):
                    run_cell.main()
            self.assertTrue(saved["summary"]["accountant_parity_pass"])
            self.assertAlmostEqual(saved["summary"]["epsilon_per_client_round"], .5)
            self.assertAlmostEqual(saved["summary"]["payment_per_round"], 30.)
            self.assertEqual(saved["meta"]["verified_raw_sizes"], [8] * 10)

    def test_reuse_checks_budget_allocation_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            spec = {"dataset": "MNIST", "alpha": .5, "seed": 14, "arm": "PAC", "budget": 30., "kappa": 3.,
                    "rounds": 40, "aggregation": "eps", "stage": "dev", "smoke_size": None,
                    "split_sha256": "split", "reports_sha256": "reports", "allocation_sha256": "allocation", "payment_sha256": "payment",
                    "executed": [[1.]], "payment": [[1.]], "proposed": [[1.]], "offered_payment": [[1.]], "input_sha256": {}}
            for name in ("old", "new"):
                atomic_json(folder / (name + ".json"), spec)
            job = {"cell": {"dataset": "MNIST", "alpha": .5}, **{k: spec[k] for k in ("seed", "arm", "budget", "kappa")},
                   "out": str(folder), "done": str(folder / "result.json"), "allocation_spec": str(folder / "old.json"),
                   "allocation_spec_sha256": digest(folder / "old.json")}
            new = {**job, "allocation_spec": str(folder / "new.json")}
            for name in ("events", "privacy"):
                atomic_json(folder / (name + ".json"), {"fixture": True})
            result = {"meta": {**spec, "allocation_spec_sha256": job["allocation_spec_sha256"]},
                      "summary": {"accountant_parity_pass": True},
                      **{n + "_sha256": digest(folder / (n + ".json")) for n in ("events", "privacy")}}
            atomic_json(folder / "result.json", result)
            self.assertTrue(reusable(job, new))
            for change in ({"budget": 100.}, {"executed": [[.1]]}):
                atomic_json(folder / "new.json", {**spec, **change})
                new["allocation_spec_sha256"] = digest(folder / "new.json")
                self.assertFalse(reusable(job, new))
            atomic_json(folder / "new.json", spec)
            new["allocation_spec_sha256"] = digest(folder / "new.json")
            atomic_json(folder / "events.json", {"fixture": "changed"})
            with self.assertRaisesRegex(RuntimeError, "evidence"):
                reusable(job, new)

    def test_adoption_recomputes_endpoints_and_checks_selected_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            job = {"cell": {"dataset": "MNIST", "alpha": .5, "mode": "budget", "value": 30.},
                   "seed": 211, "arm": "mfg_c2_task1", "budget": 30., "done": str(folder / "result.json")}
            meta = {**job["cell"], "seed": 211, "arm": job["arm"], "budget": 30., "rounds": 100,
                    "stage": "test", "checkpoint_sha256": "selected", "client_sizes": [100] * 10, "split_sha256": "same"}
            summary = {"accuracy": .8, "macro_f1": .7, "loss": 1., "accountant_parity_pass": True}
            for name in ("events", "privacy"):
                atomic_json(folder / (name + ".json"), {})
            result = {"meta": meta, "summary": summary,
                      "identity": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                      "measurements": [{"round": r, **summary} for r in range(96, 101)],
                      **{n + "_sha256": digest(folder / (n + ".json")) for n in ("events", "privacy")}}
            atomic_json(folder / "result.json", result)
            self.assertEqual(len(load_rows(folder, [job], candidate=job["arm"], checkpoint_hash="selected")[0]), 1)
            with self.assertRaisesRegex(ValueError, "another checkpoint"):
                load_rows(folder, [job], candidate=job["arm"], checkpoint_hash="another")
            result["summary"]["accuracy"] = .9
            atomic_json(folder / "result.json", result)
            with self.assertRaisesRegex(ValueError, "five round"):
                load_rows(folder, [job], candidate=job["arm"], checkpoint_hash="selected")

    def test_parent_failure_is_not_scientific_rejection(self):
        import a1_campaign
        with patch.object(a1_campaign, "check_identity"), patch.object(a1_campaign, "register_mf", return_value=None), \
             patch.object(a1_campaign, "parent_active", return_value=[]), \
             patch.object(a1_campaign, "read", return_value={"status": "failed"}):
            with self.assertRaisesRegex(RuntimeError, "operational failure"):
                a1_campaign.await_parent(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

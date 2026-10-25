"""Tests of A3 isolation, safe admission, fingerprints and analysis sealing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from common import WORK, ROOT, atomic_json, digest, market_reports
from p2_state import relocate, safe_progress
from a3_fingerprint import fingerprint
from a3_resources import can_start
import a3_campaign as campaign


class A3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_portable_paths_do_not_change_values(self):
        self.assertEqual(relocate({"/data/home/skk/paper/FL-market/a": ["/data/home/skk/paper/FL-market/b", .5]}),
                         {str(ROOT / "a"): [str(ROOT / "b"), .5]})

    def test_status_removes_accuracy_and_loss(self):
        value = safe_progress({"round": 4, "accuracy": .9, "loss": .2, "macro_f1": .8, "elapsed_seconds": 10})
        self.assertEqual(value, {"round": 4, "elapsed_seconds": 10})

    def test_memory_reserves_new_processes(self):
        self.assertFalse(can_start({"free_mib": 4000, "ram_available_mib": 20000}))
        self.assertTrue(can_start({"free_mib": 5000, "ram_available_mib": 20000}))
        self.assertFalse(can_start({"free_mib": 5000, "ram_available_mib": 20000}, unobserved_reservations=1))

    def test_fingerprint_excludes_payments_but_detects_changed_treatment(self):
        split = {"client_sizes": [64] * 10}
        reports = market_reports(1, split["client_sizes"], 14)
        spec = {"dataset": "MNIST", "alpha": .5, "seed": 14, "aggregation": "eps", "stage": "smoke",
                "smoke_size": 64, "split_sha256": "fixture", "executed": [[.1] * 10], "payment": [[1.] * 10]}
        first = fingerprint(spec, split, reports)
        other = copy.deepcopy(spec)
        other["payment"] = [[2.] * 10]
        self.assertEqual(first, fingerprint(other, split, reports))
        other["executed"][0][0] = 0.
        self.assertNotEqual(first["sha256"], fingerprint(other, split, reports)["sha256"])
        self.assertEqual(len(first["payload"]["rounds"][0][0]["rng_seeds"]), 3)

    def test_analysis_never_collects_partial_results(self):
        with tempfile.TemporaryDirectory(dir=WORK) as tmp, patch.object(campaign, "OUT", Path(tmp)), patch.object(campaign, "collect") as collect:
            atomic_json(Path(tmp) / "jobs_stage2_p2.json", [{"done": str(Path(tmp) / "missing")}])
            atomic_json(Path(tmp) / "jobs_stage2_early.json", [])
            campaign.analyze_stage2()
            campaign.analyze_stage3()
            collect.assert_not_called()

    def test_direct_collection_without_full_stage_authorization_is_rejected(self):
        from p2_results import collect
        with self.assertRaisesRegex(RuntimeError, "decision-relevant"):
            collect("stage3_early", [{"done": "should-never-be-opened"}])

    def test_whole_cell_assignment_is_fixed_for_late_arms(self):
        with tempfile.TemporaryDirectory(dir=WORK) as tmp, patch.object(campaign, "OUT", Path(tmp)):
            out = Path(tmp)
            for machine, rate, k in (("101", 800, 8), ("local", 1600, 8)):
                base = out / "operations" / machine
                atomic_json(base / "PREFLIGHT_DONE.json", {"complete": True})
                atomic_json(base / "concurrency.json", {"selected_k": k, "rate_rounds_per_hour": rate})
                atomic_json(base / "environment.json", {"host": machine})
            atomic_json(out / "operations/local/regression.json", {"local_eligible": True})
            self.assertTrue(campaign.assign("stage3"))
            path = out / "assignments/stage3.json"
            before = digest(path)
            data = json.loads(path.read_text())
            self.assertEqual(len(data["blocks"]), 54)
            for block in data["blocks"]:
                self.assertEqual(block["machine"], data["cells"][campaign.cell_key(block["cell"])])
            self.assertTrue(campaign.assign("stage3"))
            self.assertEqual(before, digest(path))

    def test_frozen_dp_executor_is_unchanged(self):
        legacy = WORK.parent / "codex-p2-101-20261019"
        for path in (WORK / "frozen").rglob("*.py"):
            self.assertEqual(digest(path), digest(legacy / path.relative_to(WORK)))

    def test_candidate_test_runs_require_passed_gate(self):
        import a3_prepare as prepare
        with tempfile.TemporaryDirectory(dir=WORK) as tmp, patch.object(prepare, "OUT", Path(tmp)), \
             patch.object(prepare, "check_identity"), patch.object(prepare, "prepare") as build:
            atomic_json(Path(tmp) / "stage0_decision.json", {"gate_pass": True})
            atomic_json(Path(tmp) / "stage2_selection.json", {"gate_pass": False})
            with self.assertRaises(AssertionError):
                prepare.build("stage3_p2")
            build.assert_not_called()

    def test_worker_runs_cpu_job_once_and_exits_cleanly(self):
        import a3_worker as worker
        import sys
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            out = Path(tmp)
            done = out / "job/done.json"
            item = {"id": "cpu/fixture", "command": [sys.executable, "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')", str(done)],
                    "out": str(done.parent), "done": str(done), "gpu": False, "priority": 0}
            atomic_json(out / "queue.json", {"jobs": [item]})
            atomic_json(out / "ALL_DONE", {"fixture": True})
            resource = {"free_mib": 20000, "ram_available_mib": 20000, "pswpin": 0, "pswpout": 0}
            with patch.object(worker, "OUT", out), patch.object(worker, "check_identity"), \
                 patch.object(worker, "resources", return_value=resource), patch.object(worker, "gpu_pids", return_value=set()):
                worker.run("101")
                before = (done.stat().st_mtime_ns, (done.parent / "process.json").read_bytes())
                worker.run("101")
                self.assertEqual(before, (done.stat().st_mtime_ns, (done.parent / "process.json").read_bytes()))


if __name__ == "__main__":
    unittest.main()

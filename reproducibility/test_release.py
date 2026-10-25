"""Regression checks against the original selector and recorded web fixtures."""
import json
import unittest

import numpy as np
import torch

from runtime import HERE, WORK, SNAPSHOT, activate, definitions, verify_release

activate()
import p2v2_mech as M


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_release()
        torch.set_num_threads(1)

    def test_original_selector_and_dividend_invariants(self):
        table = json.loads((SNAPSHOT / "run/tdsc_revision/codex_p2v2_20261020/A7_tau/table.json").read_text())
        env = {"np": np, "torch": torch, "P2V2": M.V2.P2V2, "cap_projection": M.V2.cap_projection,
               "KAPPAS": M.KAPPAS, "AMOUNTS": M.AMOUNTS}
        selector = definitions(WORK / "a7flow/model.py", ["Selector"], env)["Selector"]
        gen = torch.Generator().manual_seed(625)
        r = torch.rand((4, 10, 3), generator=gen, dtype=torch.float64)
        r[..., 1] = .1 + 4.9 * r[..., 1]
        r[..., 2] = 10 + 490 * r[..., 2].square()
        for family in M.FAMILIES:
            model = M.load(family)
            original = selector(family, table).double()
            original.load_state_dict(model.state_dict(), strict=True)
            for B in (10., 30., 100.):
                for k in (3., 30.):
                    b, kap = torch.full((4, 1), B, dtype=torch.float64), torch.full((4, 1), k, dtype=torch.float64)
                    with torch.no_grad():
                        e0, p0 = original.mechanism(r, b, kap)
                        e, p, h = M.mechanism(model, r, b, kap)
                    torch.testing.assert_close(e, e0, atol=1e-11, rtol=0)
                    torch.testing.assert_close(p, p0, atol=1e-11, rtol=0)
                    self.assertTrue(bool((e <= r[..., 1] + 1e-11).all()))
                    self.assertTrue(bool(((p + h).sum(-1) <= B + 1e-10).all()))
                    self.assertTrue(bool((p + h - r[..., 0] * e >= -1e-10).all()))
                    self.assertTrue(bool((h >= -1e-10).all()))


if __name__ == "__main__":
    unittest.main()

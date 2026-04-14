#!/usr/bin/env python3
"""
Phase 2 verification: PAC and VCG baselines (no aggregation/cvxpy required).
For full pipeline (auction -> aggr_batch -> FL) ensure cvxpy is installed.
"""
import torch
from datasets_fl_benchmark import generate_privacy_paper_bids
from baselines import pac_batch, vcg_procurement_batch


def main():
    # Privacy paper setting: v~U[0,1], eps~U[0.1,5], B
    reports = generate_privacy_paper_bids(n_agents=10, n_items=1, num_profiles=4, seed=42)
    budget = torch.tensor([[10.0], [50.0], [100.0], [25.0]])

    plosses_pac, pay_pac = pac_batch(reports, budget)
    plosses_vcg, pay_vcg = vcg_procurement_batch(reports, budget)

    assert plosses_pac.shape == (4, 10) and pay_pac.shape == (4, 10)
    assert plosses_vcg.shape == (4, 10) and pay_vcg.shape == (4, 10)
    assert (pay_pac.sum(dim=1) <= budget.squeeze(1) + 1e-5).all(), "PAC should be budget-feasible"
    assert (pay_vcg.sum(dim=1) <= budget.squeeze(1) + 1e-5).all(), "VCG should be budget-feasible"

    # Edge: n_agents=1 -> no allocation (k must be in {1..n-1}, so k=0)
    r1 = generate_privacy_paper_bids(n_agents=1, n_items=1, num_profiles=2, seed=0)
    b1 = torch.tensor([[100.0], [100.0]])
    pl1_pac, pay1_pac = pac_batch(r1, b1)
    pl1_vcg, pay1_vcg = vcg_procurement_batch(r1, b1)
    assert (pl1_pac == 0).all() and (pay1_pac == 0).all(), "PAC n=1: no allocation"
    assert (pl1_vcg == 0).all() and (pay1_vcg == 0).all(), "VCG n=1: no allocation"

    # Edge: v=0 for one agent -> PAC can select (c(0,eps)=0 <= B/k); v=[0,0.5,1], B=10 -> k=2, eps=1
    r0 = torch.zeros(1, 3, 3)
    r0[0, :, 0] = torch.tensor([0.0, 0.5, 1.0])
    r0[0, :, 1], r0[0, :, 2] = 1.0, 100.0
    b0 = torch.tensor([[10.0]])
    pl0_pac, pay0_pac = pac_batch(r0, b0)
    assert pl0_pac.sum().item() == 2.0, "PAC v=[0,0.5,1] B=10: k=2, sum(plosses)=2"
    assert pay0_pac.sum().item() <= 10.0 + 1e-5, "PAC budget feasible with v=0"

    print("PAC: plosses sum per batch", plosses_pac.sum(dim=1).tolist())
    print("PAC: payment sum per batch", pay_pac.sum(dim=1).tolist())
    print("VCG: plosses sum per batch", plosses_vcg.sum(dim=1).tolist())
    print("VCG: payment sum per batch", pay_vcg.sum(dim=1).tolist())

    # Full pipeline: auction() with PAC/VCG + aggregation (requires experiments → aggregation; cvxpy only for OptAggr)
    try:
        from experiments import auction
        n_items = 1
        trade_mech_pac = ("PAC", "ConvlAggr", "", n_items)
        trade_mech_vcg = ("VCG", "ConvlAggr", "", n_items)
        plosses_auc, weights_auc = auction(reports, budget, trade_mech_pac, model=None)
        assert plosses_auc.shape == reports.shape[:2] and weights_auc.shape == reports.shape[:2]
        plosses_auc2, weights_auc2 = auction(reports, budget, trade_mech_vcg, model=None)
        assert plosses_auc2.shape == reports.shape[:2] and weights_auc2.shape == reports.shape[:2]
        print("[Phase 2] PAC and VCG baselines + auction() integration OK.")
    except ModuleNotFoundError as e:
        if "cvxpy" in str(e).lower():
            print("[Phase 2] PAC and VCG baselines OK (auction() integration skipped: cvxpy not installed).")
        else:
            raise


if __name__ == "__main__":
    main()

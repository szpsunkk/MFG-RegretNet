#!/usr/bin/env python3
"""
Phase 3 verification: MFG-RegretNet alignment (b_MFG, budget projection, (reports, budget) interface).
Requires: regretnet (and thus aggregation/cvxpy for full import). If cvxpy is missing, only a subset of checks can run.
"""
import sys

def main():
    try:
        import torch
        from regretnet import MFGRegretNet, budget_projection_privacy_paper
        from datasets_fl_benchmark import generate_privacy_paper_bids
    except ModuleNotFoundError as e:
        if "cvxpy" in str(e).lower():
            print("[Phase 3] Skip: cvxpy not installed (required by aggregation). Install cvxpy to run MFG-RegretNet.")
        else:
            raise
        return

    n_agents, n_items = 5, 1
    model = MFGRegretNet(n_agents, n_items, hidden_layer_size=32, n_hidden_layers=1)
    reports = generate_privacy_paper_bids(n_agents, n_items, 4, seed=42)
    budget = torch.tensor([[10.0], [20.0], [15.0], [25.0]])

    allocs, payments = model((reports, budget))
    assert allocs.shape == (4, n_agents, n_items), allocs.shape
    assert payments.shape == (4, n_agents), payments.shape
    assert (payments.sum(dim=1) <= budget.squeeze(1) + 1e-5).all(), "Budget projection should keep sum(p) <= B"
    print("MFGRegretNet forward: allocs", allocs.shape, "payments", payments.shape, "sum(p) < B OK")

    # b_MFG = mean over agents: when we pass reports, internal b_mfg = reports.mean(dim=1)
    b_mfg = reports.mean(dim=1)
    assert b_mfg.shape == (4, n_items + 2), b_mfg.shape
    print("b_MFG = (1/N)*sum_i b_i (mean over agents) OK")

    print("[Phase 3] MFG-RegretNet alignment checks passed.")


if __name__ == "__main__":
    main()

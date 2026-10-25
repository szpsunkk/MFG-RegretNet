"""Checks of p2v2_mech.py (exit code 0 only if every check passes).

1. SelectorModel equals Codex's a7flow Selector (allocation and payments) on random profiles.
2. ClientContext.outcomes equals own_payment (e and p) for random profiles, clients and candidate reports
   (including withholding eps' = 0 and d' = 1), for all three families, several budgets and kappas.
3. The full-profile mechanism agrees with ClientContext at the truthful report, including the dividend.
4. Dividend (data sizes verified): h >= 0; Sbar_i >= S (so sum_i (p_i + h_i) <= B); h_i does not change with
   client i's price or cap; truthful utility p + h - v e >= 0; Sbar_i is not exceeded by S at 200 random
   (price, budget) reports of client i with its own size.
5. Records (does not fail on): how often h_i decreases when d_i grows by 50%, and how often the purchase with the
   size report 500 is below the actual purchase.
"""
import sys

import numpy as np
import torch

import p2v2_mech as M

torch.set_num_threads(4)
TOL = 1e-11


def profiles(rng, count):
    """Test-style profiles: v ~ U[0,1], cap ~ U[0.1, 5], d ~ U{10..500} (as generate_privacy_paper_bids)."""
    return torch.tensor(np.stack([rng.random((count, 10)), .1 + 4.9 * rng.random((count, 10)),
                                  rng.integers(10, 501, (count, 10)).astype(float)], -1))


def skewed(rng, count):
    """Training-style sizes 10 + 490 U^2: many small clients, so the selector drops some."""
    r = profiles(rng, count)
    r[..., 2] = torch.tensor(10 + 490 * rng.random((count, 10)) ** 2)
    return r


def codex_selector_check(rng):
    sys.path.insert(0, str(M.WORK / "a7flow"))
    import model as codex  # Codex's a7flow/model.py (imports its state modules; read-only use)
    worst = 0.
    for family in M.FAMILIES:
        mine = M.load(family)
        theirs = codex.load_model(M.CKPT_DIR / family / "model.pt")
        for B in (10., 25., 50., 100.):
            for kap in (3., 30.):
                r = skewed(rng, 32)
                b = torch.full((32, 1), B, dtype=torch.float64)
                k = torch.full((32, 1), kap, dtype=torch.float64)
                with torch.no_grad():
                    e1, p1 = mine.mechanism(r, b, k)
                    e2, p2 = theirs.mechanism(r, b, k)
                worst = max(worst, float((e1 - e2).abs().max()), float((p1 - p2).abs().max()))
    return worst


def codex_own(model, reports, budget, kappa, client, cand):
    """own_payment on each candidate profile; returns (P, G) e and p."""
    P, G, _ = cand.shape
    full = reports[:, None].expand(P, G, 10, 3).clone()
    full[:, :, client] = cand
    rows = full.reshape(P * G, 10, 3)
    b, k = budget.repeat_interleave(G, 0), kappa.repeat_interleave(G, 0)
    clients = torch.full((P * G,), client, dtype=torch.long)
    e, p = model.own_payment(rows, b, k, clients)
    return e.reshape(P, G), p.reshape(P, G)


def main():
    rng = np.random.default_rng(20261022)
    ok = True
    d_sel = codex_selector_check(rng)
    print(f"SelectorModel vs Codex Selector: max |difference| {d_sel:.2e}")
    ok &= d_sel <= TOL
    worst, fixed_below, fixed_total, h_drops, h_total = {}, 0, 0, 0, 0
    with torch.no_grad():
        for family in M.FAMILIES:
            model = M.load(family)
            for B in (10., 25., 50., 100.):
                for kap in (3., 30.):
                    reports = skewed(rng, 6) if kap == 3. else profiles(rng, 6)
                    budget = torch.full((6, 1), B, dtype=torch.float64)
                    kappa = torch.full((6, 1), kap, dtype=torch.float64)
                    E, Pm, H = M.mechanism(model, reports, budget, kappa)
                    for client in (0, 3, 9):
                        grid = M.candidate_grid(reports, client, gv=5, ge=5, gd=3)
                        rand = torch.stack([torch.tensor(rng.random((6, 7))),
                                            torch.tensor(rng.random((6, 7))) * reports[:, client, 1:2],
                                            1 + torch.tensor(rng.random((6, 7))) * (reports[:, client, 2:3] - 1)], -1)
                        cand = torch.cat((reports[:, client:client + 1], grid, rand), 1)
                        ctx = M.ClientContext(model, reports, client)
                        e, p, h = ctx.outcomes(budget, kappa, cand)
                        e0, p0 = codex_own(model, reports, budget, kappa, client, cand)
                        de, dp = float((e - e0).abs().max()), float((p - p0).abs().max())
                        worst[family] = max(worst.get(family, 0.), de, dp)
                        if de > TOL or dp > TOL:
                            ok = False
                            print("MISMATCH", family, B, kap, client, de, dp)
                        dt = max(float((E[:, client] - e[:, 0]).abs().max()), float((Pm[:, client] - p[:, 0]).abs().max()),
                                 float((H[:, client] - h[:, 0]).abs().max()))
                        if dt > TOL:
                            ok = False
                            print("TRUTH MISMATCH", family, B, kap, client, dt)
                        # Sbar bounds S over the client's price and budget reports (own size fixed)
                        sb = ctx.sbar(budget, kappa)
                        cand2 = torch.stack([torch.tensor(rng.random((6, 200))), 5 * torch.tensor(rng.random((6, 200))),
                                             reports[:, client, 2:3].expand(6, 200)], -1)
                        coef, bias, _ = ctx.public(budget, kappa)
                        caps2 = ctx.place(cand2[..., 1], 1)
                        sizes2 = ctx.place(cand2[..., 2], 2)
                        mask2 = M.selection(sizes2 / sizes2.sum(-1, keepdim=True),
                                            model.threshold(kappa, budget, "budget")[:, None].expand(6, 200))
                        s2 = torch.minimum(torch.minimum(M.S_MAX * torch.sigmoid(ctx.score(coef, bias, cand2)), budget),
                                           torch.where(mask2, caps2, torch.zeros_like(caps2)).sum(-1))
                        if float((s2 - sb[:, None]).max()) > 1e-12:
                            ok = False
                            print("SBAR NOT MAXIMAL", family, B, kap, client, float((s2 - sb[:, None]).max()))
                        sf = ctx.sbar_fixed(budget, kappa)
                        # S at the actual profile, from the full mechanism's purchase
                        s_actual = torch.minimum(torch.minimum(model.total(reports, budget, kappa), budget),
                                                 actual_selected_caps(model, reports, budget, kappa))[:, 0]
                        if float((s_actual - sb).max()) > 1e-12:
                            ok = False
                            print("SBAR BELOW S", family, B, kap, client)
                        fixed_below += int((sf < s_actual - 1e-12).sum())
                        fixed_total += len(sf)
                    u = Pm + H - reports[..., 0] * E
                    checks = {"h_nonneg": float(H.min()) >= -1e-12,
                              "budget": float((Pm + H).sum(-1).max() - B) <= 1e-9,
                              "ir": float(u.min()) >= -1e-12}
                    for i in (0, 5):
                        moved = reports.clone()
                        moved[:, i, 0] = torch.tensor(rng.random(6))
                        moved[:, i, 1] = moved[:, i, 1] * torch.tensor(rng.random(6))
                        _, _, H2 = M.mechanism(model, moved, budget, kappa)
                        checks[f"h_ignores_own_price_cap_{i}"] = float((H2[:, i] - H[:, i]).abs().max()) <= 1e-12
                        bigger = reports.clone()
                        bigger[:, i, 2] = torch.minimum(bigger[:, i, 2] * 1.5, torch.tensor(500.))
                        _, _, H3 = M.mechanism(model, bigger, budget, kappa)
                        h_drops += int((H[:, i] - H3[:, i] > 1e-12).sum()); h_total += 6
                    bad = [k for k, v in checks.items() if not v]
                    if bad:
                        ok = False
                        print("PROPERTY FAIL", family, B, kap, bad)
    # outcomes_multi (shared embeddings) equals outcomes per setting
    with torch.no_grad():
        for family in M.FAMILIES:
            model = M.load(family)
            reports = skewed(rng, 5)
            sets = [(torch.full((5, 1), B, dtype=torch.float64), torch.full((5, 1), k, dtype=torch.float64))
                    for B in (25., 50.) for k in (3., 30.)]
            ctx = M.ClientContext(model, reports, 4)
            cand = M.candidate_grid(reports, 4, gv=4, ge=4, gd=2)
            multi = ctx.outcomes_multi(sets, cand)
            for (b, k), (e1, p1, h1) in zip(sets, multi):
                e2, p2, h2 = ctx.outcomes(b, k, cand)
                d = max(float((e1 - e2).abs().max()), float((p1 - p2).abs().max()), float((h1 - h2).abs().max()))
                if d > 1e-14:
                    ok = False
                    print("MULTI MISMATCH", family, d)
    print("max |difference| to own_payment per family:", {k: f"{v:.2e}" for k, v in worst.items()})
    print(f"purchase with size report 500 below the actual S: {fixed_below} of {fixed_total} client-profiles")
    print(f"h_i lower after d_i grows by 50%: {h_drops} of {h_total} (allowed: sizes are verified)")
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


def actual_selected_caps(model, reports, budget, kappa):
    sizes, caps = reports[..., 2], reports[..., 1]
    shares = sizes / sizes.sum(-1, keepdim=True)
    mask = M.selection(shares, model.threshold(kappa, budget, "budget"))
    return torch.where(mask, caps, torch.zeros_like(caps)).sum(-1, keepdim=True)


if __name__ == "__main__":
    sys.exit(main())

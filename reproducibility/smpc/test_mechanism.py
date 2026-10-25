"""Tests numerical equivalence before spending time on MPC, including branch edges."""
import argparse
import json
import torch

from secure_mechanism import Operations, bitonic_sort, grouped_recipe, mechanism, plaintext_purchase, project, public_context
from support import HERE, atomic_json, now, reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", default="10,50,100,200,500")
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, common, timing = reference()
    from p2v2_mech import V2
    results = {}
    purchase_errors = {}
    with torch.no_grad():
        for n in map(int, args.n_list.split(",")):
            reports = common.generate_privacy_paper_bids(n, 1, 1001).double()
            b = torch.full((1, 1), 50., dtype=torch.float64)
            k = torch.full((1, 1), 3., dtype=torch.float64)
            ctx = public_context(model, reports[0])
            diagnostics = {}
            actual, recipe = mechanism(reports[0, :, :2], ctx, diagnostics=diagnostics)
            expected = timing.ours_cached(model, reports, b, k)
            errors = {key: float((a - e[0]).abs().max())
                      for key, a, e in zip(("e", "p", "h"), actual, expected)}
            if max(errors.values()) > 1e-9:
                raise AssertionError((n, errors))
            s_error = float((diagnostics["S"] - plaintext_purchase(model, reports, ctx)).abs().max())
            assert s_error < 1e-9, (n, "S", s_error)
            purchase_errors[str(n)] = s_error
            previous = HERE / "operations/preflight_history/before_A2_threshold_revision/recipes" / ("N%d.json" % n)
            assert recipe == json.loads(previous.read_text())["sequential_recipe"], (n, "A2 changed secure operation graph")
            results[str(n)] = errors
            atomic_json(HERE / "recipes" / ("N%d.json" % n), dict(N=n, operations=grouped_recipe(recipe),
                         sequential_recipe=recipe, public_selected=int(ctx["mask"].sum())))
            print(dict(N=n, plaintext_parity=errors, S_error=s_error), flush=True)
        caps = torch.tensor([[0., .2, 0., .4], [1., 1., 1., 1.], [0., 0., 0., 0.],
                             [1., .5, 2., .01], [0., .1, 0., 2.]], dtype=torch.float64)
        shares = torch.tensor([[0., .6, 0., .4], [.25, .25, .25, .25], [.25, .25, .25, .25],
                               [.25, .25, .25, .25], [0., .5, 0., .5]], dtype=torch.float64)
        for amount in (0., .0001, .01, .2, 1., 4., 50.):
            totals = torch.full((5, 1), amount, dtype=torch.float64)
            expected = V2.cap_projection(totals, shares, caps)
            expected = torch.where(expected >= .01, expected, 0.)
            actual = project(totals, shares, caps, Operations())
            assert torch.allclose(actual, expected, atol=1e-12, rtol=0), (amount, actual, expected)
        g = torch.Generator().manual_seed(42)
        for n in (1, 2, 16, 64, 128, 256, 512):
            x = torch.randint(-5, 6, (5, n), generator=g).double()
            assert torch.equal(bitonic_sort(x), x.sort(-1).values)
    atomic_json(HERE / "validation" / "plaintext_tests.json",
                dict(time=now(), passed=True, full_mechanism=results, purchase_S_max_abs=purchase_errors,
                     same_secure_operation_graph_as_A1=True,
                     projection_edge_cases=35, sorting_sizes=[1, 2, 16, 64, 128, 256, 512]))


if __name__ == "__main__":
    main()

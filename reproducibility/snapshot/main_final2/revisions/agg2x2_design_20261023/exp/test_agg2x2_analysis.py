"""Synthetic checks of agg2x2_analysis.py (version 2): every case of the pre-committed reading of Blocks A, B and C,
refusal on missing or extra runs, the consistency checks on the stored records, and the privacy ratio of Block C.
Writes only to temporary directories. Version 1 of these checks is kept as test_agg2x2_analysis_v1_20261023.py.
Usage: python test_agg2x2_analysis.py
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np

import agg2x2_analysis as G

REG = {"seeds": {"0.1": [24, 25], "0.5": [24, 25]}, "block_b_arm": {"0.1": "CSRA", "0.5": "CSRA"},
       "block_c_arm": "CSRA-declared"}
EPS = {G.OURS: 0.6, G.UNIFORM: 0.6, "CSRA": 0.3, "CSRA-declared": 1.5}


def confusion(acc, n=10000):
    """10 x 10 confusion matrix with the given accuracy (in [0, 1]) to 1e-4."""
    right = int(round(acc * n))
    cm = np.zeros((10, 10), dtype=int)
    for c in range(10):
        cm[c, c] = right // 10 + (1 if c < right % 10 else 0)
    wrong = n - right
    for k in range(wrong):
        cm[k % 10, (k + 1) % 10] += 1
    return cm


def write_run(root, block, alpha, seed, arm, agg, acc_pp, eps=None, **override):
    d = Path(root) / f"{block}_{alpha}_{seed}_{arm}_{agg}"
    d.mkdir(parents=True)
    meas = []
    for r in range(1, 41):
        cm = confusion(acc_pp / 100)
        meas.append({"round": r, "accuracy": float(np.trace(cm) / cm.sum()), "confusion": cm.tolist()})
    meta = {"dataset": "CIFAR10", "alpha": alpha, "seed": seed, "arm": arm, "aggregation": agg, "stage": "dev",
            "rounds": 40, "budget": 30.0, "allocation_spec_sha256": "x"}
    meta.update(override)
    res = {"meta": meta, "measurements": meas}
    if eps is not None:
        res["summary"] = {"epsilon_per_client_round": eps}
    (d / "result.json").write_text(json.dumps(res))
    return {"block": block, "alpha": alpha, "seed": seed, "arm": arm, "aggregation": agg,
            "done": str(d / "result.json"), "allocation_spec_sha256": "x"}


def scenario(root, acc_fn, block_b=True, block_c=True, reg=REG):
    jobs = []
    for alpha in G.ALPHAS:
        for s in reg["seeds"][str(alpha)]:
            for arm in (G.OURS, G.UNIFORM):
                for agg in G.AGGS:
                    jobs.append(write_run(root, "A", alpha, s, arm, agg, acc_fn(alpha, s, arm, agg), EPS[arm]))
            if block_b:
                x = reg["block_b_arm"][str(alpha)]
                for agg in G.AGGS:
                    jobs.append(write_run(root, "B", alpha, s, x, agg, acc_fn(alpha, s, x, agg), EPS[x]))
            if block_c:
                for agg in G.AGGS:
                    jobs.append(write_run(root, "C", alpha, s, G.DECLARED, agg, acc_fn(alpha, s, G.DECLARED, agg),
                                          EPS[G.DECLARED]))
    return jobs


def run(acc_fn, block_b=True, block_c=True, reg=REG):
    with tempfile.TemporaryDirectory() as tmp:
        return G.analyse(scenario(tmp, acc_fn, block_b, block_c, reg), reg)


def close(x, y, tol=0.02):
    return abs(x - y) <= tol


def main():
    # 1. Allocation case; B persists; C ahead with the privacy ratio 1.5 / 0.3 = 5.0 in the text.
    base = {G.OURS: 52.0, G.UNIFORM: 50.0, "CSRA": 48.0, G.DECLARED: 51.0}
    acc = lambda al, s, arm, agg: base[arm] + (1.0 if agg == "datasize" else 0.0) + (0.1 if s == 25 else 0.0)
    r = run(acc)
    a = r["block_A"]
    assert a["case"] == "allocation", a["case"]
    assert close(a["mean"]["A"], 2) and close(a["mean"]["D"], 2) and close(a["mean"]["G_U"], 1) and close(a["mean"]["I"], 0)
    assert r["block_B"]["case"] == "persists" and close(r["block_B"]["by_alpha"]["0.1"]["E_ee"], 4)
    c = r["block_C"]
    assert c["case"] == "ahead" and close(c["by_alpha"]["0.5"]["F_ee"], 1) and close(c["by_alpha"]["0.1"]["R_e"], 3)
    assert close(c["privacy_ratio_to_X"], 5.0) and "5.0 times as much privacy" in r["text_C"], r["text_C"]
    print("allocation / persists / ahead ok:", r["text_C"][:90], "...")

    # 2. Aggregation case; B protocol; C behind in one alpha.
    def acc2(al, s, arm, agg):
        table = {(G.OURS, "eps"): 53.0, (G.OURS, "datasize"): 53.1, (G.UNIFORM, "eps"): 50.0, (G.UNIFORM, "datasize"): 53.0}
        if arm == "CSRA":
            return 52.0 if agg == "eps" else 54.0
        if arm == G.DECLARED:
            return 54.0 if al == 0.1 else 52.0
        return table[(arm, agg)]
    r = run(acc2)
    assert r["block_A"]["case"] == "aggregation" and close(r["block_A"]["mean"]["I"], 2.9)
    assert r["block_B"]["case"] == "protocol"
    assert r["block_C"]["case"] == "behind" and "not behind" in r["text_C"]
    print("aggregation / protocol / behind ok")

    # 3. Uniform case.
    acc3 = lambda al, s, arm, agg: {(G.OURS, "eps"): 52.0, (G.OURS, "datasize"): 52.0,
                                    (G.UNIFORM, "eps"): 50.0, (G.UNIFORM, "datasize"): 53.0}.get((arm, agg), 50.0)
    assert run(acc3)["block_A"]["case"] == "uniform"
    print("uniform case ok")

    # 4. Mean D above DELTA but positive in only 2 of 4 pairs: not the allocation case.
    acc4 = lambda al, s, arm, agg: (55.0 if (arm == G.OURS and agg == "datasize" and s == 24) else
                                    48.0 if (arm == G.OURS and agg == "datasize") else 50.0)
    a = run(acc4, block_b=False, block_c=False)["block_A"]
    assert a["D_positive_pairs"] == 2 and a["mean"]["D"] >= G.DELTA and a["case"] == "aggregation", a
    print("3-of-4 rule ok")

    # 5. Blocks B and C absent: A analysed, B and C reported with zero complete runs.
    r = run(lambda *k: 50.0, block_b=False, block_c=False)
    assert "block_A" in r and "block_B" not in r and "block_C" not in r
    assert r["counts_B"]["complete"] == 0 and r["counts_C"]["complete"] == 0
    print("blocks B and C absent ok")

    # 6. Block C complete but Block B absent: C is read without R and without the ratio.
    r = run(lambda al, s, arm, agg: 52.0 if arm == G.OURS else 50.0, block_b=False, block_c=True)
    c = r["block_C"]
    assert c["case"] == "ahead" and c["privacy_ratio_to_X"] is None and "R_e" not in c["rows"][0]
    assert "times as much privacy" not in r["text_C"]
    print("block C without B ok")

    # 7. A registration without Block C (version 1 layout): C reported as not registered.
    reg1 = {k: v for k, v in REG.items() if k != "block_c_arm"}
    r = run(lambda *k: 50.0, block_c=False, reg=reg1)
    assert r["counts_C"] == "not registered" and "block_B" in r
    print("registration without C ok")

    # 8. A missing Block A run: refuse, counts only.
    with tempfile.TemporaryDirectory() as tmp:
        jobs = scenario(tmp, lambda *k: 50.0)
        Path(jobs[0]["done"]).unlink()
        r = G.analyse(jobs, REG)
        assert "block_A" not in r and r["counts_A"] == {"registered": 16, "complete": 15}
    print("missing A run refused ok")

    # 9. A missing Block C run: A and B analysed, C refused with counts.
    with tempfile.TemporaryDirectory() as tmp:
        jobs = scenario(tmp, lambda *k: 50.0)
        Path(next(j for j in jobs if j["block"] == "C")["done"]).unlink()
        r = G.analyse(jobs, REG)
        assert "block_B" in r and "block_C" not in r and r["counts_C"] == {"registered": 8, "complete": 7}
    print("missing C run refused ok")

    # 10. Extra, unregistered runs raise (in A and in C).
    for block, arm in (("A", G.OURS), ("C", G.DECLARED)):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = scenario(tmp, lambda *k: 50.0)
            jobs.append(write_run(tmp, block, 0.1, 26, arm, "eps", 50.0))
            try:
                G.analyse(jobs, REG)
                raise AssertionError("extra run accepted")
            except RuntimeError as e:
                assert "outside the registration" in str(e)
    print("extra runs refused ok")

    # 11. Wrong stage (official test set), wrong rounds or wrong aggregation raises.
    for override in ({"stage": "test"}, {"rounds": 100}, {"aggregation": "eps"}):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = scenario(tmp, lambda *k: 50.0, block_b=False, block_c=False)
            j = next(x for x in jobs if x["aggregation"] == "datasize")
            Path(j["done"]).unlink()
            Path(j["done"]).parent.rmdir()
            jobs[jobs.index(j)] = write_run(tmp, "A", j["alpha"], j["seed"], j["arm"], "datasize", 50.0, **override)
            try:
                G.analyse(jobs, REG)
                raise AssertionError(f"accepted {override}")
            except RuntimeError as e:
                assert "differs from its registered run" in str(e)
    print("metadata checks ok")

    # 12. Accuracy field inconsistent with the confusion matrix raises.
    with tempfile.TemporaryDirectory() as tmp:
        jobs = scenario(tmp, lambda *k: 50.0, block_b=False, block_c=False)
        p = Path(jobs[0]["done"])
        d = json.loads(p.read_text())
        d["measurements"][-1]["accuracy"] += 0.01
        p.write_text(json.dumps(d))
        try:
            G.analyse(jobs, REG)
            raise AssertionError("inconsistent accuracy accepted")
        except RuntimeError as e:
            assert "confusion matrix" in str(e)
    print("confusion check ok")

    # 13. Block C registered with another arm name is rejected.
    bad = copy.deepcopy(REG)
    bad["block_c_arm"] = "CSRA"
    try:
        run(lambda *k: 50.0, reg=bad)
        raise AssertionError("wrong block C arm accepted")
    except AssertionError as e:
        assert "CSRA" in str(e) and "accepted" not in str(e)
    print("block C arm check ok")

    # 14. Markdown writer runs on a complete result.
    with tempfile.TemporaryDirectory() as tmp:
        r = run(lambda al, s, arm, agg: 50.0 + (2 if arm == G.OURS else 0))
        G.write_md(Path(tmp) / "x.md", r)
        text = (Path(tmp) / "x.md").read_text()
        assert "Block A" in text and "Block B" in text and "Block C" in text
    print("all agg2x2 analysis checks passed")


if __name__ == "__main__":
    main()

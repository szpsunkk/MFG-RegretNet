"""Pre-registered analysis of the aggregation control (Claude; version 2, 2026-10-24; task
CODEX_TASK_20261023_aggregation_2x2.md). Version 1 (2026-10-23) is kept as agg2x2_analysis_v1_20261023.py;
version 2 adds Block C and records the privacy each run releases. Blocks A and B and their reading are unchanged.

Question: does the accuracy advantage of MFG-RegretNet (arm mfg_p2v2) come from the privacy it buys or from the
aggregation weights that its allocation implies? Algorithm 2 weights each update by its epsilon; our allocation
splits epsilon in proportion to data shares, so its epsilon weights are close to data-size weights. Block C adds
CSRA with releases equal to the declared budgets: the confirmation used the one-shot CSRA whose winners are paid for
their declared budgets but release 1/(N-k); with declared releases the same payments buy far more privacy.

Design (CIFAR-10, purchased-budget setting B = 30, alpha in {0.1, 0.5}, two fresh development seeds per alpha,
40 rounds, development evaluation data, endpoint = mean accuracy of rounds 36-40 recomputed from the stored
confusion matrices):
  Block A  arms {mfg_p2v2, Uniform-DP} x aggregation {eps, datasize}            16 runs
  Block B  arm X(alpha) x aggregation {eps, datasize}, X(alpha) = the external arm that is best in the frozen
           A1.3 decision for the Stage 3 setting CIFAR10 / budget 30 / alpha (CSRA for both alphas)    8 runs
  Block C  arm CSRA-declared x aggregation {eps, datasize}                                             8 runs
Quantities per (alpha, seed), all in percentage points:
  A   = acc(ours, eps)      - acc(Uniform-DP, eps)       allocation effect under epsilon weights
  D   = acc(ours, datasize) - acc(Uniform-DP, datasize)  allocation effect under data-size weights (primary)
  G_U = acc(Uniform-DP, datasize) - acc(Uniform-DP, eps) aggregation effect for the uniform split
  G_O = acc(ours, datasize) - acc(ours, eps)             aggregation effect for our allocation
  I   = A - D                                            interaction
  E_ee = acc(ours, eps) - acc(X, eps);  E_ed = acc(ours, eps) - acc(X, datasize);  E_dd = acc(ours, datasize) - acc(X, datasize)
  F_ee = acc(ours, eps) - acc(CSRA-declared, eps)  (primary for Block C: the protocol of the paper)
  F_dd = acc(ours, datasize) - acc(CSRA-declared, datasize)
  R_e, R_d = acc(CSRA-declared, agg) - acc(X, agg) for agg = eps, datasize (when Block B is complete)
Pre-committed reading (fixed before any run; DELTA = 0.5 pp):
  Block A, case "allocation":   mean D >= DELTA and D > 0 in at least 3 of the 4 (alpha, seed) pairs
           case "aggregation":  otherwise, if mean D > -DELTA
           case "uniform":      mean D <= -DELTA
  Block B, case "persists":     mean E_dd > 0 in both alphas (means over the two seeds)
           case "protocol":     otherwise
  Block C, case "ahead":        mean F_ee > 0 in both alphas (means over the two seeds)
           case "behind":       otherwise
The analysis refuses to read a block unless every registered run of it is complete; blocks are read separately
(B and C need A; a block that is incomplete is reported as not analysed, with its counts).
Each run also records the privacy it released (summary epsilon_per_client_round of run_cell.py) when present.
Usage: python agg2x2_analysis.py --manifest jobs_agg2x2.json --registration registration.json --out-dir DIR
"""
import argparse
import hashlib
import json
import statistics as st
from pathlib import Path

import numpy as np

DATASET, MODE, VALUE, ROUNDS, ALPHAS = "CIFAR10", "budget", 30.0, 40, (0.1, 0.5)
ENDPOINT = (36, 37, 38, 39, 40)
DELTA = 0.5
OURS, UNIFORM, DECLARED = "mfg_p2v2", "Uniform-DP", "CSRA-declared"
AGGS = ("eps", "datasize")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with open(path) as fh:
        return json.load(fh)


def block_arms(reg, block, alpha):
    if block == "A":
        return (OURS, UNIFORM)
    if block == "B":
        return (reg["block_b_arm"][str(alpha)],)
    if block == "C":
        assert reg["block_c_arm"] == DECLARED, reg["block_c_arm"]
        return (DECLARED,)
    raise ValueError(block)


def expected_runs(reg, block):
    """Registered (alpha, seed, arm, aggregation) tuples of a block."""
    out = []
    for alpha in ALPHAS:
        seeds = reg["seeds"][str(alpha)]
        if len(seeds) != 2:
            raise ValueError(f"alpha {alpha}: two seeds required, got {seeds}")
        out += [(alpha, s, arm, agg) for s in seeds for arm in block_arms(reg, block, alpha) for agg in AGGS]
    return out


def endpoint(job):
    res = load(job["done"])
    meta = res["meta"]
    want = {"dataset": DATASET, "alpha": job["alpha"], "seed": job["seed"], "arm": job["arm"],
            "aggregation": job["aggregation"], "stage": "dev", "rounds": ROUNDS, "budget": VALUE}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise RuntimeError(f"{job['done']}: result differs from its registered run: {bad}")
    if job.get("allocation_spec_sha256") and meta.get("allocation_spec_sha256") != job["allocation_spec_sha256"]:
        raise RuntimeError(f"{job['done']}: allocation specification hash differs")
    end = [m for m in res["measurements"] if m["round"] in ENDPOINT]
    if sorted(m["round"] for m in end) != list(ENDPOINT):
        raise RuntimeError(f"{job['done']}: endpoint rounds incomplete")
    accs = []
    for m in end:
        cm = np.asarray(m["confusion"], dtype=float)
        acc = float(np.trace(cm) / cm.sum())
        if abs(acc - m["accuracy"]) > 1e-12:
            raise RuntimeError(f"{job['done']}: accuracy differs from its confusion matrix")
        accs.append(acc)
    eps = (res.get("summary") or {}).get("epsilon_per_client_round")
    return 100 * float(np.mean(accs)), eps, sha(job["done"])


def collect(manifest, reg, block):
    """Endpoints of one block, or None (with counts) when any registered run is missing."""
    want = expected_runs(reg, block)
    jobs = {(j["alpha"], j["seed"], j["arm"], j["aggregation"]): j for j in manifest if j.get("block") == block}
    extra = sorted(set(jobs) - set(want), key=str)
    if extra:
        raise RuntimeError(f"block {block}: runs outside the registration: {extra}")
    done = [k for k in want if k in jobs and jobs[k].get("done") and Path(jobs[k]["done"]).exists()]
    if len(done) != len(want):
        return None, None, {"registered": len(want), "complete": len(done)}
    acc, eps, hashes = {}, {}, {}
    for k in want:
        acc[k], eps[k], hashes["|".join(map(str, k))] = endpoint(jobs[k])
    return acc, eps, {"registered": len(want), "complete": len(done), "result_sha256": hashes}


def block_a(acc, reg):
    pairs = []
    for alpha in ALPHAS:
        for s in reg["seeds"][str(alpha)]:
            a = {(arm, agg): acc[(alpha, s, arm, agg)] for arm in (OURS, UNIFORM) for agg in AGGS}
            A = a[(OURS, "eps")] - a[(UNIFORM, "eps")]
            D = a[(OURS, "datasize")] - a[(UNIFORM, "datasize")]
            pairs.append({"alpha": alpha, "seed": s, "accuracy": {f"{arm}|{agg}": v for (arm, agg), v in a.items()},
                          "A": A, "D": D, "G_U": a[(UNIFORM, "datasize")] - a[(UNIFORM, "eps")],
                          "G_O": a[(OURS, "datasize")] - a[(OURS, "eps")], "I": A - D})
    mean = {q: st.mean(p[q] for p in pairs) for q in ("A", "D", "G_U", "G_O", "I")}
    by_alpha = {str(alpha): {q: st.mean(p[q] for p in pairs if p["alpha"] == alpha) for q in ("A", "D", "G_U", "G_O", "I")}
                for alpha in ALPHAS}
    positive = sum(p["D"] > 0 for p in pairs)
    if mean["D"] >= DELTA and positive >= 3:
        case = "allocation"
    elif mean["D"] > -DELTA:
        case = "aggregation"
    else:
        case = "uniform"
    return {"pairs": pairs, "mean": mean, "by_alpha": by_alpha, "D_positive_pairs": positive, "case": case}


def block_b(acc_a, acc_b, reg):
    rows = []
    for alpha in ALPHAS:
        x = reg["block_b_arm"][str(alpha)]
        for s in reg["seeds"][str(alpha)]:
            o_e, o_d = acc_a[(alpha, s, OURS, "eps")], acc_a[(alpha, s, OURS, "datasize")]
            x_e, x_d = acc_b[(alpha, s, x, "eps")], acc_b[(alpha, s, x, "datasize")]
            rows.append({"alpha": alpha, "seed": s, "arm": x, "accuracy": {f"{x}|eps": x_e, f"{x}|datasize": x_d},
                         "E_ee": o_e - x_e, "E_ed": o_e - x_d, "E_dd": o_d - x_d})
    by_alpha = {str(alpha): {q: st.mean(r[q] for r in rows if r["alpha"] == alpha) for q in ("E_ee", "E_ed", "E_dd")}
                for alpha in ALPHAS}
    case = "persists" if all(by_alpha[str(a)]["E_dd"] > 0 for a in ALPHAS) else "protocol"
    return {"rows": rows, "by_alpha": by_alpha, "case": case}


def block_c(acc_a, acc_b, acc_c, eps_b, eps_c, reg):
    rows = []
    for alpha in ALPHAS:
        x = reg["block_b_arm"][str(alpha)] if reg.get("block_b_arm") else None
        for s in reg["seeds"][str(alpha)]:
            o_e, o_d = acc_a[(alpha, s, OURS, "eps")], acc_a[(alpha, s, OURS, "datasize")]
            c_e, c_d = acc_c[(alpha, s, DECLARED, "eps")], acc_c[(alpha, s, DECLARED, "datasize")]
            row = {"alpha": alpha, "seed": s, "accuracy": {f"{DECLARED}|eps": c_e, f"{DECLARED}|datasize": c_d},
                   "F_ee": o_e - c_e, "F_dd": o_d - c_d, "eps_per_client_round": eps_c[(alpha, s, DECLARED, "eps")]}
            if acc_b is not None:
                row.update({"R_e": c_e - acc_b[(alpha, s, x, "eps")], "R_d": c_d - acc_b[(alpha, s, x, "datasize")],
                            "eps_per_client_round_X": eps_b[(alpha, s, x, "eps")]})
            rows.append(row)
    qs = ("F_ee", "F_dd") + (("R_e", "R_d") if acc_b is not None else ())
    by_alpha = {str(alpha): {q: st.mean(r[q] for r in rows if r["alpha"] == alpha) for q in qs} for alpha in ALPHAS}
    case = "ahead" if all(by_alpha[str(a)]["F_ee"] > 0 for a in ALPHAS) else "behind"
    ratio = None
    if acc_b is not None and all(r["eps_per_client_round"] and r["eps_per_client_round_X"] for r in rows):
        ratio = st.mean(r["eps_per_client_round"] for r in rows) / st.mean(r["eps_per_client_round_X"] for r in rows)
    return {"rows": rows, "by_alpha": by_alpha, "case": case, "privacy_ratio_to_X": ratio}


TEXT = {
    "allocation": "With data-size-weighted aggregation for both, our allocation still exceeds a uniform split of the "
                  "same per-round privacy by {D:.2f} pp (development data, CIFAR-10, B=30); under the epsilon "
                  "weights of Algorithm 2 the difference is {A:.2f} pp, so the remainder comes from the weights.",
    "aggregation": "Most of the gain over a uniform split of the same per-round privacy comes from the aggregation "
                   "weights that our allocation implies: {A:.2f} pp under the epsilon weights of Algorithm 2 and "
                   "{D:.2f} pp when both use data-size weights. Accuracy claims are stated for Algorithm 2.",
    "uniform": "With data-size-weighted aggregation for both, a uniform split of the same per-round privacy is "
               "better by {negD:.2f} pp; the gain under Algorithm 2 ({A:.2f} pp) comes from the aggregation "
               "weights. Stated as a limitation; accuracy claims are stated for Algorithm 2.",
    "persists": "When the strongest baseline of each setting also uses data-size weights, our method (with the "
                "epsilon weights of Algorithm 2) differs from it by {E_ed} pp and, with data-size weights for "
                "both, by {E_dd} pp.",
    "protocol": "When the strongest baseline also uses data-size weights, it is not below our method in every "
                "setting ({E_dd} pp with data-size weights for both); the accuracy advantage is stated for the "
                "epsilon-weighted aggregation of Algorithm 2.",
    "ahead": "Against CSRA with releases equal to the declared budgets{ratio}, MFG-RegretNet is ahead under the "
             "epsilon weights of Algorithm 2 by {F_ee} pp (development data, CIFAR-10, B=30; {F_dd} pp with "
             "data-size weights for both).",
    "behind": "CSRA with releases equal to the declared budgets{ratio} is not behind MFG-RegretNet in every setting "
              "under the epsilon weights of Algorithm 2 ({F_ee} pp; development data, CIFAR-10, B=30; {F_dd} pp "
              "with data-size weights for both); the CSRA of the confirmation is the 1/(N-k) adaptation.",
}


def write_md(path, res):
    lines = ["# Aggregation control (pre-registered)", ""]
    a = res.get("block_A")
    if a is None:
        lines.append(f"Block A incomplete: {res['counts_A']}")
    else:
        lines += ["## Block A", "", "| alpha | seed | ours/eps | ours/datasize | Uniform/eps | Uniform/datasize | A | D | G_U | G_O | I |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in a["pairs"]:
            c = p["accuracy"]
            lines.append(f"| {p['alpha']} | {p['seed']} | {c[OURS + '|eps']:.2f} | {c[OURS + '|datasize']:.2f} | "
                         f"{c[UNIFORM + '|eps']:.2f} | {c[UNIFORM + '|datasize']:.2f} | {p['A']:+.2f} | {p['D']:+.2f} | "
                         f"{p['G_U']:+.2f} | {p['G_O']:+.2f} | {p['I']:+.2f} |")
        m = a["mean"]
        lines += ["", f"Means: A {m['A']:+.2f}, D {m['D']:+.2f}, G_U {m['G_U']:+.2f}, G_O {m['G_O']:+.2f}, I {m['I']:+.2f}; "
                  f"D > 0 in {a['D_positive_pairs']} of 4 pairs. Case: **{a['case']}**.", "", "Text: " + res["text_A"], ""]
    b = res.get("block_B")
    if b is None:
        lines.append(f"Block B not analysed: {res.get('counts_B')}")
    else:
        lines += ["## Block B", "", "| alpha | seed | arm X | X/eps | X/datasize | E_ee | E_ed | E_dd |", "|---|---|---|---|---|---|---|---|"]
        for r in b["rows"]:
            c = r["accuracy"]
            lines.append(f"| {r['alpha']} | {r['seed']} | {r['arm']} | {c[r['arm'] + '|eps']:.2f} | "
                         f"{c[r['arm'] + '|datasize']:.2f} | {r['E_ee']:+.2f} | {r['E_ed']:+.2f} | {r['E_dd']:+.2f} |")
        lines += ["", f"Case: **{b['case']}**.", "", "Text: " + res["text_B"], ""]
    c = res.get("block_C")
    if c is None:
        lines.append(f"Block C not analysed: {res.get('counts_C')}")
    else:
        lines += ["## Block C", "", "| alpha | seed | declared/eps | declared/datasize | F_ee | F_dd | R_e | R_d | eps/client/round |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in c["rows"]:
            acc = r["accuracy"]
            fmt = lambda k: f"{r[k]:+.2f}" if k in r else "--"
            lines.append(f"| {r['alpha']} | {r['seed']} | {acc[DECLARED + '|eps']:.2f} | {acc[DECLARED + '|datasize']:.2f} | "
                         f"{fmt('F_ee')} | {fmt('F_dd')} | {fmt('R_e')} | {fmt('R_d')} | {r['eps_per_client_round']} |")
        lines += ["", f"Case: **{c['case']}**.", "", "Text: " + res["text_C"]]
    Path(path).write_text("\n".join(lines) + "\n")


def by_alpha_text(block, q):
    return " / ".join(f"{block['by_alpha'][str(al)][q]:+.2f} (alpha {al})" for al in ALPHAS)


def analyse(manifest, reg):
    res = {"registration": reg, "delta_pp": DELTA}
    acc_a, _, counts_a = collect(manifest, reg, "A")
    res["counts_A"] = counts_a
    if acc_a is None:
        return res
    a = block_a(acc_a, reg)
    res["block_A"] = a
    res["text_A"] = TEXT[a["case"]].format(A=a["mean"]["A"], D=a["mean"]["D"], negD=-a["mean"]["D"])
    acc_b = eps_b = None
    if not reg.get("block_b_arm"):
        res["counts_B"] = "not registered"
    else:
        acc_b, eps_b, counts_b = collect(manifest, reg, "B")
        res["counts_B"] = counts_b
        if acc_b is not None:
            b = block_b(acc_a, acc_b, reg)
            res["block_B"] = b
            res["text_B"] = TEXT[b["case"]].format(E_ed=by_alpha_text(b, "E_ed"), E_dd=by_alpha_text(b, "E_dd"))
    if not reg.get("block_c_arm"):
        res["counts_C"] = "not registered"
        return res
    acc_c, eps_c, counts_c = collect(manifest, reg, "C")
    res["counts_C"] = counts_c
    if acc_c is None:
        return res
    c = block_c(acc_a, acc_b, acc_c, eps_b, eps_c, reg)
    res["block_C"] = c
    ratio = (f", which release {c['privacy_ratio_to_X']:.1f} times as much privacy per round as the 1/(N-k) adaptation"
             if c["privacy_ratio_to_X"] else "")
    res["text_C"] = TEXT[c["case"]].format(ratio=ratio, F_ee=by_alpha_text(c, "F_ee"), F_dd=by_alpha_text(c, "F_dd"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    reg = load(a.registration)
    manifest = load(a.manifest)
    manifest = manifest["jobs"] if isinstance(manifest, dict) else manifest
    res = analyse(manifest, reg)
    res["inputs_sha256"] = {"manifest": sha(a.manifest), "registration": sha(a.registration), "analysis": sha(__file__)}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agg2x2_result.json").write_text(json.dumps(res, indent=1))
    write_md(out / "agg2x2_result.md", res)
    if "block_A" not in res:
        print("Block A incomplete:", res["counts_A"])
    else:
        print("Block A case:", res["block_A"]["case"], "| Block B:", res.get("block_B", {}).get("case", res.get("counts_B")),
              "| Block C:", res.get("block_C", {}).get("case", res.get("counts_C")))


if __name__ == "__main__":
    main()

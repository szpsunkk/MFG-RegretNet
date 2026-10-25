"""Numbers for the experiments section of the P2-v2 manuscript (PAPER_EXPERIMENTS_PLAN_20261023.md).

Baselines come from the finished evaluation files that the current tables already use; MFG-RegretNet comes only from
the P2-v2 auction-side runs (main_final2/revisions/p2v2_auction_side_20261022, results copied back from 101). Rows of
the earlier regret-trained variant in the baseline files ("MFG-RegretNet", "mfg_seed*", "mfgV8F") are excluded:
the paper reports one mechanism under that name. Every number keeps its source file.

Usage: python collect_paper_data.py --results DIR --out data/paper_data.json [--smoke] [--parts rq1,rq3,...]
--smoke accepts incomplete runs (layout tests only); the output is then marked "smoke": true and must not be used.
"""
import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REV = ROOT / "main_final2/revisions"
AUX = REV / "p2v2_auction_side_20261022/exp"
sys.path.insert(0, str(AUX))
import summarize_rq1 as S1  # noqa: E402
import common_p2v2 as C  # noqa: E402

EARLIER = ("MFG-RegretNet", "mfg_seed", "mfgV8F", "mfg")   # the earlier variant in baseline files
BASE_MECHS = ["PAC", "VCG", "CSRA", "MFG-Pricing", "RegretNet"]
MODES = ["overbid", "underbid", "epsilon_inflate", "epsilon_deflate", "underbid_epsilon_deflate",
         "underbid_epsilon_inflate"]
INCAP = {"overbid", "underbid", "epsilon_deflate", "underbid_epsilon_deflate"}
INFL = {"epsilon_inflate", "underbid_epsilon_inflate"}
SOURCES = defaultdict(list)


def read(path):
    SOURCES["files"].append(str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path))
    return json.load(open(path))


# ------------------------------------------------------------------------------------------- collusion
def collusion_summary(rows):
    """Per mechanism: gain % per mode at extreme magnitude (pooled over seeds and ratios), largest in-cap gain %."""
    out = {}
    for mech in sorted({r["mechanism_key"] for r in rows}):
        rs = [r for r in rows if r["mechanism_key"] == mech]
        per_mode = {}
        for mode in MODES:
            g = [r for r in rs if r["attack_mode"] == mode and r["attack_magnitude"] == "extreme"]
            if g:
                per_mode[mode] = 100 * st.mean(r["coalition_gain_mean"] for r in g) / st.mean(
                    r["coalition_truthful_utility_mean"] for r in g)
        rel = [100 * r["coalition_gain_mean"] / r["coalition_truthful_utility_mean"] for r in rs
               if r["attack_mode"] in INCAP and r["coalition_truthful_utility_mean"] > 0]
        out[mech] = {"extreme_gain_pct": per_mode, "incap_max_pct": max(rel) if rel else None,
                     "incap_positive_share": (sum(x > 1e-4 for x in rel) / len(rel)) if rel else None,
                     "inflate_extreme_max_pct": max((per_mode[m] for m in INFL if m in per_mode), default=None),
                     "configs": len(rs)}
    return out


def baseline_collusion():
    d = read(ROOT / "run/tdsc_revision/collusion_v8f_20261016/collusion_robustness.json")
    rows = [dict(r, mechanism_key=r["mechanism"]) for r in d["results"] if r["mechanism"] in BASE_MECHS]
    return collusion_summary(rows)


def ours_collusion(results):
    rows = []
    for kap in ("3", "30"):
        f = Path(results) / f"collusion_mfg_p2v2_kappa{kap}.json"
        if f.exists():
            for r in read(f)["results"]:
                if r["variant"] == "with_dividend":
                    rows.append(dict(r, mechanism_key=f"Ours_kappa{kap}"))
    return collusion_summary(rows) if rows else {}


# ------------------------------------------------------------------------------------------- accountant
def accountant_summary(rows):
    out = {}
    for mech in sorted({r["mechanism_key"] for r in rows}):
        rs = [r for r in rows if r["mechanism_key"] == mech]
        first = {}
        for cap in ("10.0", "30.0", "60.0"):
            vals = [min([x for x in r["exhaustion_round_by_cap"][cap] if x is not None], default=None) for r in rs]
            first[cap] = vals
        rounds = sorted({c["round"] for r in rs for c in r["curve"]})
        curve = {k: [st.mean(next(c[k] for c in r["curve"] if c["round"] == t) for r in rs) for t in rounds]
                 for k in ("mean_eps_total", "max_eps_total")}
        out[mech] = {"seeds": [r["seed"] for r in rs], "first_client_reaches": first,
                     "excluded_by_check": [r["excluded_by_pre_release_check"] for r in rs],
                     "rounds": rounds, **curve}
    return out


def baseline_accountant(noise):
    d = read(ROOT / f"run/tdsc_revision/accountant_v8f_20261016/privacy_accountant_{noise}.json")
    return accountant_summary([dict(r, mechanism_key=r["mechanism"]) for r in d["results"]
                               if r["mechanism"] in BASE_MECHS])


def ours_accountant(results, noise):
    f = Path(results) / f"accountant_mfg_p2v2_{noise}.json"
    if not f.exists():
        return {}
    return accountant_summary([dict(r, mechanism_key=f"Ours_kappa{r['kappa']:g}") for r in read(f)["results"]])


# ------------------------------------------------------------------------------------------- adaptive
def adaptive_pct(m):
    t = m["truthful_utility"]
    return {"commit_pct": 100 * m["gain_commit"] / t, "learner_pct": 100 * m["gain_learner"] / t,
            "best_policy_pct": 100 * m["gain_best_policy"] / t, "best_response_pct": 100 * m["gain_best_response"] / t,
            "commit_abs": m["gain_commit"], "truthful_utility": t}


def baseline_adaptive(B=25):
    out = {}
    s10 = read(REV / "step10_attacks_20261014/results/adaptive_attack.json")["results"][f"{float(B)}"]
    for k in ("PAC", "CSRA", "MFG-Pricing"):
        out[k] = adaptive_pct(s10[k]["mean"] if "mean" in s10[k] else s10[k])
    for name, f, key in (("RegretNet", "adaptive_regretnet_s.json", "regretnet_s_seed42"),
                         ("DM-RegretNet", "adaptive_dm_s.json", "dm_s_seed42")):
        d = read(REV / "b1_retrain_20261014/eval" / f)["results"][f"{float(B)}"][key]
        out[name] = adaptive_pct(d["mean"] if "mean" in d else d)
    return out


def ours_adaptive(results, smoke):
    files = sorted(glob.glob(str(Path(results) / "adaptive" / "adaptive_mfg_p2v2_traj*.json")))
    if not files:
        return {}
    if len(files) != 10 and not smoke:
        raise SystemExit(f"adaptive: {len(files)} of 10 trajectories")
    per = defaultdict(list)
    for f in files:
        for key, m in read(f)["results"].items():
            per[key].append(m)
    out = {}
    for key, ms in per.items():
        mean = {k: st.mean(m[k] for m in ms) for k in ("gain_commit", "gain_learner", "gain_best_policy",
                                                        "gain_best_response", "truthful_utility")}
        out[key] = dict(adaptive_pct(mean), trajectories=len(ms))
    return out


# ------------------------------------------------------------------------------------------- RQ1, RQ3, hetero
def ours_regret(results, pattern, smoke):
    out = {}
    for d in sorted(glob.glob(str(Path(results) / pattern))):
        meta, merged, complete, ranges, seconds = S1.load_dir(d)
        if not complete and not smoke:
            raise SystemExit(f"{d}: incomplete shards {ranges}")
        reports = C.test_profiles() if meta["scenario"] == "test" else C.hetero_profiles(meta["scenario"])
        lo = ranges[0][0]
        n = merged[0]["e"].shape[0]
        eps = reports[lo:lo + n, :, 1]
        for s, stt in enumerate(meta["settings"]):
            key = f"{meta['family']}|{meta['scenario']}|B{stt['budget_table']:g}|kappa{stt['kappa']:g}"
            out[key] = dict(S1.summarize(merged[s], eps, stt["budget_effective"]), profiles=n, complete=complete)
    return out


def ours_rq3(results):
    out = {}
    for f in sorted(glob.glob(str(Path(results) / "rq3_*.json"))):
        d = read(f)
        out[d["family"]] = {"profiles": d["profiles"], "results": d["results"]}
    return out


def baseline_rq1(B=25):
    E = REV / "method_improvement_20261015/eval"
    out = {}
    base = read(E / f"rq1test_base_B{B}.json")["results"]
    base2 = read(E / f"rq1test_base2_B{B}.json")["results"]
    for name, r in (("PAC", base["PAC"]), ("VCG", base2["VCG"]), ("CSRA", base2["CSRA"]),
                    ("MFG-Pricing", base["MFG-Pricing"])):
        out[name] = {"regret_mean_x1e3": 1e3 * r["regret_mean"], "ir_violation_rate": r["ir_violation_rate"],
                     "budget_used": r["budget_used"], "seeds": 1}
    for name, f, kind in (("RegretNet", f"rq1test_regretnet_s_B{B}.json", "regretnet_s"),
                          ("DM-RegretNet", f"rq1test_dm_s_B{B}.json", "dm_s")):
        rs = list(read(E / f)["results"][kind].values())
        reg = [1e3 * r["regret_mean"] for r in rs]
        out[name] = {"regret_mean_x1e3": st.mean(reg), "regret_sd_x1e3": st.pstdev(reg),
                     "ir_violation_rate": st.mean(r["ir_violation_rate"] for r in rs),
                     "budget_used": st.mean(r["budget_used"] for r in rs), "seeds": len(rs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parts", default="rq1,hetero,rq3,collusion,accountant,adaptive",
                    help="our parts to collect; the others are left empty and recorded as not collected")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    assert parts <= {"rq1", "hetero", "rq3", "collusion", "accountant", "adaptive"}, parts
    data = {
        "smoke": a.smoke,
        "baselines": {"rq1_B25": baseline_rq1(25), "collusion": baseline_collusion(),
                      "accountant": {n: baseline_accountant(n) for n in ("gaussian", "laplace")},
                      "adaptive_B25": baseline_adaptive(25)},
        "ours": {"rq1": ours_regret(a.results, "rq1_test_*", a.smoke) if "rq1" in parts else {},
                 "hetero": ours_regret(a.results, "hetero_*", a.smoke) if "hetero" in parts else {},
                 "rq3": ours_rq3(a.results) if "rq3" in parts else {},
                 "collusion": ours_collusion(a.results) if "collusion" in parts else {},
                 "accountant": {n: ours_accountant(a.results, n) if "accountant" in parts else {}
                                for n in ("gaussian", "laplace")},
                 "adaptive": ours_adaptive(a.results, a.smoke) if "adaptive" in parts else {}},
        "parts_collected": sorted(parts),
        "excluded_rows": "baseline files' rows of the earlier regret-trained variant (MFG-RegretNet / mfg_seed* / mfgV8F)",
        "sources": sorted(set(SOURCES["files"])),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1, default=float))
    print({"out": a.out, "smoke": a.smoke, "ours_rq1": len(data["ours"]["rq1"]), "ours_hetero": len(data["ours"]["hetero"]),
           "ours_rq3": list(data["ours"]["rq3"]), "ours_collusion": list(data["ours"]["collusion"]),
           "ours_adaptive": len(data["ours"]["adaptive"])})


if __name__ == "__main__":
    main()

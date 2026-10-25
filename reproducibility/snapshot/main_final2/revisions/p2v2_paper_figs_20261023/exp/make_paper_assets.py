"""Figures and LaTeX snippets of the P2-v2 experiments section from data/paper_data.json (collect_paper_data.py).

Format follows the first submission (Fig. 6 heatmap of RQ1; Fig. 8 accounting curves; Tables of RQ1 and RQ3), in the
shared style of revisions/three_figs_v8f_20261016/exp/refstyle.py (STIX, 8 pt, MFG-RegretNet in #205D91, no red).
Choices fixed in PAPER_EXPERIMENTS_PLAN_20261023.md; nothing here depends on the values.

Outputs (in --out): fig_collusion_heatmap_p2v2.pdf, fig_accountant_{gaussian,laplace}_p2v2.pdf,
rq1_rows.tex, rq3_rows.tex, auction_macros.tex, supp_tables.tex.
With a smoke input (paper_data["smoke"]), every figure carries a "SMOKE TEST" watermark and every snippet a comment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "three_figs_v8f_20261016/exp"))
import refstyle as RS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

MODES = ["overbid", "underbid", "epsilon_inflate", "epsilon_deflate", "underbid_epsilon_deflate",
         "underbid_epsilon_inflate"]
MODE_LABEL = {"overbid": "OB", "underbid": "UB", "epsilon_inflate": r"$\epsilon^{+}$", "epsilon_deflate": r"$\epsilon^{-}$",
              "underbid_epsilon_deflate": r"UB+$\epsilon^{-}$", "underbid_epsilon_inflate": r"UB+$\epsilon^{+}$"}
ROWS = [("PAC/VCG", "PAC"), ("CSRA", "CSRA"), ("MFG-Pricing", "MFG-Pricing"), ("RegretNet", "RegretNet"),
        (r"Ours, $\kappa{=}3$", "Ours_kappa3"), (r"Ours, $\kappa{=}30$", "Ours_kappa30")]
DIVERGING = LinearSegmentedColormap.from_list("blue_white_ochre", ["#205D91", "#FFFFFF", "#997522"])


def watermark(fig, smoke):
    if smoke:
        fig.text(0.5, 0.5, "SMOKE TEST - NOT RESULTS", ha="center", va="center", fontsize=14, color="#888888",
                 alpha=0.35, rotation=20, zorder=20)


def fmt_pct(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.1f}" if signed else f"{x:.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def sci(x, nd=1):
    """LaTeX scientific notation for small nonnegative values: 0, below 1e-12, or a.b x 10^e."""
    if x == 0:
        return "0"
    if abs(x) < 1e-12:
        return r"$<10^{-12}$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, nd) >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.{nd}f}{{\times}}10^{{{e}}}$"


def tex_num(x, nd=2):
    s = f"{x:.{nd}f}"
    return s.replace("-", "$-$")


# ------------------------------------------------------------------------------------------- figures
def collusion_heatmap(data, out, smoke):
    base, ours = data["baselines"]["collusion"], data["ours"]["collusion"]
    src = dict(base, **ours)
    rows = [(label, key) for label, key in ROWS if key in src]
    M = np.array([[src[key]["extreme_gain_pct"].get(m, np.nan) for m in MODES] for _, key in rows])
    fig, ax = plt.subplots(figsize=(3.45, 0.32 * len(rows) + 0.75))
    lo = min(-1.0, np.nanmin(M)); hi = max(1.0, np.nanmax(M))
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            strong = abs(v) > 0.55 * (hi if v > 0 else -lo)
            ax.text(j, i, fmt_pct(v), ha="center", va="center", fontsize=6.3,
                    color="white" if strong else "#222222")
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([MODE_LABEL[m] for m in MODES])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    for k, (_, key) in enumerate(rows):
        if key.startswith("Ours"):
            ax.get_yticklabels()[k].set_fontweight("bold")
            ax.get_yticklabels()[k].set_color(RS.COLOR["Ours"])
    ax.tick_params(length=0, labelsize=7)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.07, pad=0.16, aspect=40)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("coalition gain (% of truthful utility)", fontsize=7)
    cb.outline.set_linewidth(0.4)
    watermark(fig, smoke)
    fig.savefig(out / "fig_collusion_heatmap_p2v2.pdf")
    fig.savefig(out / "fig_collusion_heatmap_p2v2.png", dpi=300)
    plt.close(fig)


def accountant_figure(data, noise, out, smoke, legend_loc="upper left"):
    base, ours = data["baselines"]["accountant"][noise], data["ours"]["accountant"][noise]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    order = [("PAC/VCG", "PAC", "PAC"), ("CSRA", "CSRA", "CSRA"), ("MFG-Pricing", "MFG-Pricing", "MFG-Pricing"),
             ("RegretNet", "RegretNet", "RegretNet")]
    for label, key, style in order:
        if key in base:
            ax.plot(base[key]["rounds"], base[key]["mean_eps_total"], label=label,
                    **dict(RS.line_kw(style), markevery=max(1, len(base[key]["rounds"]) // 8)))
    for label, key, ls in ((r"Ours, $\kappa{=}3$", "Ours_kappa3", "-"), (r"Ours, $\kappa{=}30$", "Ours_kappa30", "--")):
        if key in ours:
            kw = RS.line_kw("Ours"); kw["ls"] = ls
            ax.plot(ours[key]["rounds"], ours[key]["mean_eps_total"], label=label,
                    **dict(kw, markevery=max(1, len(ours[key]["rounds"]) // 8)))
    for cap in (10, 30, 60):
        ax.axhline(cap, color="#999999", lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel("FL round")
    ax.set_ylabel(r"mean cumulative $\epsilon$")
    RS.axes_style(ax)
    RS.legend(ax, loc=legend_loc, ncol=2)
    watermark(fig, smoke)
    fig.savefig(out / f"fig_accountant_{noise}_p2v2.pdf")
    fig.savefig(out / f"fig_accountant_{noise}_p2v2.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- tables and macros
def rq1_rows(data, B=25):
    lines = []
    for kap in (3, 30):
        key = f"mfg_p2v2|test|B{B}|kappa{kap}"
        r = data["ours"]["rq1"].get(key)
        if r is None:
            lines.append(f"% {key}: missing")
            continue
        m = r["with_dividend"]
        reg = m["regret_mean_x1e3"]
        reg_txt = sci(reg) if reg < 1e-3 else tex_num(reg, 3)
        ir = m["ir_violation_rate"]
        ir_txt = "0" if ir == 0 else f"{100 * ir:.1f}\\%"
        lines.append(rf"  \rowcolor{{gray!15}}" "\n"
                     rf"  \textbf{{MFG-RegretNet}}, $\kappa{{=}}{kap}$ & {reg_txt} & {ir_txt} & {m['budget_used']:.3f} \\")
    return "\n".join(lines) + "\n"


def rq3_rows(data):
    r = data["ours"]["rq3"].get("mfg_p2v2")
    if r is None:
        return "% rq3 missing\n"
    lines = []
    for kap in (3, 30):
        rows = {B: r["results"][f"kappa{kap:g}_B{B:g}"]["with_dividend"] for B in (25., 50., 75., 100.)}
        use = [v["budget_used"] for v in rows.values()]
        ir = max(v["ir_violation_rate"] for v in rows.values())
        ir_txt = "0" if ir == 0 else f"{100 * ir:.1f}\\%"
        sw = " & ".join(f"{rows[B]['welfare']:.2f}" for B in (25., 50., 75., 100.))
        lines.append(rf"  \rowcolor{{gray!15}}" "\n"
                     rf"  \textbf{{MFG-RegretNet}}, $\kappa{{=}}{kap}$ & {min(use):.2f}--{max(use):.2f} & {ir_txt} & {sw} \\")
    return "\n".join(lines) + "\n"


def macros(data):
    out = []
    ours, base = data["ours"], data["baselines"]
    for kap, tag in ((3, "Three"), (30, "Thirty")):
        for B in (25, 50):
            r = ours["rq1"].get(f"mfg_p2v2|test|B{B}|kappa{kap}")
            if r:
                m = r["with_dividend"]
                out.append(rf"\newcommand{{\OursRegretMaxB{B}K{tag}}}{{{m['regret_max']:.1e}}}")
                out.append(rf"\newcommand{{\OursRegretSizeMaxB{B}K{tag}}}{{{m['regret_size_only_max']:.1e}}}")
                out.append(rf"\newcommand{{\OursBudgetUseB{B}K{tag}}}{{{m['budget_used']:.3f}}}")
        a = ours["adaptive"].get(f"B25_kappa{kap}_with_dividend")
        if a:
            out.append(rf"\newcommand{{\OursAdaptiveCommitK{tag}}}{{{a['commit_pct']:+.2f}}}")
            out.append(rf"\newcommand{{\OursAdaptiveBestPolicyK{tag}}}{{{a['best_policy_pct']:.2f}}}")
        c = ours["collusion"].get(f"Ours_kappa{kap}")
        if c:
            out.append(rf"\newcommand{{\OursCoalitionIncapMaxK{tag}}}{{{c['incap_max_pct']:.1f}}}")
            if c["inflate_extreme_max_pct"] is not None:
                out.append(rf"\newcommand{{\OursCoalitionInflateK{tag}}}{{{c['inflate_extreme_max_pct']:.1f}}}")
        g = ours["accountant"]["gaussian"].get(f"Ours_kappa{kap}")
        if g:
            first = [x for x in g["first_client_reaches"]["10.0"] if x is not None]
            out.append(rf"\newcommand{{\OursFirstTenK{tag}}}{{{min(first) if first else 'none within 200'}}}")
    full = [v for k, v in base["accountant"]["gaussian"].items() if k in ("MFG-Pricing", "RegretNet")]
    if full:
        rounds = [x for v in full for x in v["first_client_reaches"]["10.0"] if x is not None]
        out.append(rf"\newcommand{{\FullSpendFirstTen}}{{{min(rounds)}--{max(rounds)}}}")
    return "\n".join(out) + "\n"


def rq1_detail_table(data):
    """Supplementary RQ1 detail of MFG-RegretNet (with the dividend), B in {25, 50}, kappa in {3, 30}: mean and largest
    regret of the full search, largest regret with verified data sizes (the report set of the guarantee), largest
    gain from data-size misreports alone, IR violations, budget use, markets without a purchase, payment per unit of
    released privacy and the share of contracts paid at the price cap."""
    lines = [r"\begin{tabular}{ccccccccccc}", r"\toprule",
             r"& & \multicolumn{4}{c}{Regret} & & & No & \multicolumn{2}{c}{Payment per unit} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){10-11}",
             r"$B$ & $\kappa$ & mean & max & verified, max & size only, max & IR viol. & $\mathcal{R}/B$ & purchase & mean & at cap \\",
             r"\midrule"]
    for B in (25, 50):
        for kap in (3, 30):
            r = data["ours"]["rq1"].get(f"mfg_p2v2|test|B{B}|kappa{kap}")
            if r is None:
                lines.append(f"% B{B} kappa{kap}: missing")
                continue
            m = r["with_dividend"]
            ir = "0" if m["ir_violation_rate"] == 0 else f"{100 * m['ir_violation_rate']:.2f}\\%"
            nop = "0" if m["zero_purchase_share"] == 0 else f"{100 * m['zero_purchase_share']:.2f}\\%"
            lines.append(f"{B} & {kap} & {sci(m['regret_mean'])} & {sci(m['regret_max'])} & "
                         f"{sci(m['regret_verified_sizes_grid_max'])} & {sci(m['regret_size_only_max'])} & {ir} & "
                         f"{m['budget_used']:.3f} & {nop} & {m['pay_per_eps_mean']:.3f} & "
                         f"{100 * m['pay_per_eps_at_cap_share']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def rq3_decomposition_table(data, extra):
    """Welfare decomposition per round (truthful reports, 10,000 profiles): total payment, payment for the release,
    dividend, privacy cost sum_i v_i e_i under the true valuations, surplus without and with the dividend, released
    privacy and the share of markets without a purchase; MFG-RegretNet from rq3_p2v2.py, the analytical baselines
    from rq3_decomposition_extra.py (same profiles)."""
    ours = data["ours"]["rq3"].get("mfg_p2v2")
    lines = [r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"$B$ & Mechanism & $\mathcal{R}$ & Release & Dividend & Privacy cost & $\overline{\mathrm{SW}}$ w/o div. & "
             r"$\overline{\mathrm{SW}}$ & $\sum_i\epsilon_i^{\mathrm{out}}$ & No purchase \\", r"\midrule"]
    pct = lambda x: "0" if x == 0 else f"{100 * x:.1f}\\%"
    for B in (25, 50, 75, 100):
        first = True
        if ours is not None:
            for kap in (3, 30):
                row = ours["results"][f"kappa{kap}_B{B}"]
                w, wo = row["with_dividend"], row["without_dividend"]
                zero = extra["ours_no_purchase_share"][f"kappa{kap}_B{B}"]
                lines.append(f"{B if first else ''} & MFG-RegretNet, $\\kappa{{=}}{kap}$ & {w['revenue']:.2f} & "
                             f"{wo['revenue']:.2f} & {w['revenue'] - wo['revenue']:.2f} & "
                             f"{wo['revenue'] - wo['welfare']:.2f} & {wo['welfare']:.2f} & {w['welfare']:.2f} & "
                             f"{w['released_eps_sum']:.2f} & {pct(zero)} \\\\")
                first = False
        base = extra["baselines"][f"B{B}"]
        for key, label in (("PAC", "PAC / VCG"), ("CSRA", "CSRA, release $1/(N{-}k)$"),
                           ("CSRA-declared", "CSRA, declared release"), ("MFG-Pricing", "MFG-Pricing")):
            r = base[key]
            if key == "PAC":
                assert abs(r["welfare"] - base["VCG"]["welfare"]) < 1e-9
            lines.append(f"{B if first else ''} & {label} & {r['revenue']:.2f} & {r['revenue']:.2f} & -- & "
                         f"{r['privacy_cost']:.2f} & {r['welfare']:.2f} & {r['welfare']:.2f} & "
                         f"{r['released_eps_sum']:.2f} & {pct(r['no_purchase_share'])} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def collusion_table(data):
    """Coalition gain at the extreme magnitude per attack mode (pooled over seeds and coalition ratios)."""
    src = dict(data["baselines"]["collusion"], **data["ours"]["collusion"])
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             "Mechanism & " + " & ".join(MODE_LABEL[m] for m in MODES) + r" \\", r"\midrule"]
    for label, key in ROWS:
        if key in src:
            lab = label.replace("Ours", r"\textbf{MFG-RegretNet}") if key.startswith("Ours") else label
            lines.append(f"{lab} & " + " & ".join(fmt_pct(src[key]["extreme_gain_pct"].get(m)) for m in MODES)
                         .replace("-", "$-$") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def adaptive_table(data):
    """Adaptive attacker at B = 25: gain per round over truthful reporting, in % of the truthful utility."""
    ad = dict(data["baselines"]["adaptive_B25"])
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Mechanism & After learning & Whole horizon & Best fixed policy & Best response \\", r"\midrule"]
    rows = list(ad.items())
    for kap in (3, 30):
        a = data["ours"]["adaptive"].get(f"B25_kappa{kap}_with_dividend")
        if a:
            rows.append((rf"\textbf{{MFG-RegretNet}}, $\kappa{{=}}{kap}$", a))
    for name, a in rows:
        lines.append(f"{name} & " + " & ".join(tex_num(a[k]) for k in ("commit_pct", "learner_pct", "best_policy_pct",
                                                                      "best_response_pct")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def hetero_table(data, n_agents=10):
    """Heterogeneous valuation scenarios (5,000 profiles each, utilities with the dividend): regret of the full
    search, with verified data sizes, and from data-size misreports alone; the number of client-profile pairs whose
    size misreport gains anything; IR violations, budget use, markets without a purchase, payment per unit."""
    lines = [r"\begin{tabular}{lcccccccccc}", r"\toprule",
             r"& & \multicolumn{3}{c}{Regret, max} & Size gain & & & No & \multicolumn{2}{c}{Payment per unit} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){10-11}",
             r"Scenario, $B$ & $\kappa$ & full & verified & size only & (clients) & IR viol. & $\mathcal{R}/B$ & purchase & mean & at cap \\",
             r"\midrule"]
    for scen, B in (("bimodal", 50), ("bimodal", 100), ("realistic", 100)):
        for kap in (3, 30):
            r = data["ours"]["hetero"].get(f"mfg_p2v2|{scen}|B{B}|kappa{kap}")
            if r is None:
                lines.append(f"% {scen} B{B} kappa{kap}: missing")
                continue
            m = r["with_dividend"]
            clients = round(m["regret_size_only_positive_share"] * r["profiles"] * n_agents)
            ir = "0" if m["ir_violation_rate"] == 0 else f"{100 * m['ir_violation_rate']:.2f}\\%"
            nop = "0" if m["zero_purchase_share"] == 0 else f"{100 * m['zero_purchase_share']:.2f}\\%"
            lines.append(f"{scen.capitalize()}, {B} & {kap} & {sci(m['regret_max'])} & "
                         f"{sci(m['regret_verified_sizes_grid_max'])} & {sci(m['regret_size_only_max'])} & {clients} & "
                         f"{ir} & {m['budget_used']:.3f} & {nop} & {m['pay_per_eps_mean']:.3f} & "
                         f"{100 * m['pay_per_eps_at_cap_share']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def accounting_table(data, noise):
    """Round at which the first client reaches eps_tot = 10, 30, 60 (range over the three seeds; -- if not within
    200 rounds in any seed) and client releases refused by the pre-release check (mean over seeds)."""
    base, ours = data["baselines"]["accountant"][noise], data["ours"]["accountant"][noise]
    rows = []
    if "PAC" in base and "VCG" in base:
        same = base["PAC"]["first_client_reaches"] == base["VCG"]["first_client_reaches"]
        rows.append(("PAC / VCG" if same else "PAC", base["PAC"]))
        if not same:
            rows.append(("VCG", base["VCG"]))
    for key, label in (("CSRA", "CSRA"), ("MFG-Pricing", "MFG-Pricing"), ("RegretNet", "RegretNet")):
        if key in base:
            rows.append((label, base[key]))
    for kap in (3, 30):
        if f"Ours_kappa{kap}" in ours:
            rows.append((rf"\textbf{{MFG-RegretNet}}, $\kappa{{=}}{kap}$", ours[f"Ours_kappa{kap}"]))

    def rng(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return "--"
        txt = f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"
        return txt if len(v) == len(vals) else txt + r"$^{*}$"
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Mechanism & $\epsilon^{\mathrm{tot}}{=}10$ & $30$ & $60$ & Refused \\", r"\midrule"]
    for label, v in rows:
        f = v["first_client_reaches"]
        lines.append(f"{label} & {rng(f['10.0'])} & {rng(f['30.0'])} & {rng(f['60.0'])} & "
                     f"{np.mean(v['excluded_by_check']):.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def supp_tables(data):
    """Collusion per mode, adaptive gains, hetero and accounting rows for the supplement placeholders."""
    lines = ["% collusion: extreme magnitude, gain % per mode (pooled over seeds and coalition ratios)"]
    src = dict(data["baselines"]["collusion"], **data["ours"]["collusion"])
    for label, key in ROWS:
        if key in src:
            lines.append(f"{label} & " + " & ".join(fmt_pct(src[key]['extreme_gain_pct'].get(m)) for m in MODES)
                         .replace("-", "$-$") + r" \\")
    lines.append("% adaptive attacker, B = 25: commit, learner, best fixed policy, best response (% of truthful utility)")
    ad = dict(data["baselines"]["adaptive_B25"])
    for kap in (3, 30):
        a = data["ours"]["adaptive"].get(f"B25_kappa{kap}_with_dividend")
        if a:
            ad[f"Ours, $\\kappa{{=}}{kap}$"] = a
    for name, a in ad.items():
        lines.append(f"{name} & " + " & ".join(tex_num(a[k]) for k in ("commit_pct", "learner_pct", "best_policy_pct",
                                                                      "best_response_pct")) + r" \\")
    lines.append("% heterogeneous valuations: regret max, IR violation rate, budget used (with dividend)")
    for key, r in sorted(data["ours"]["hetero"].items()):
        m = r["with_dividend"]
        lines.append(f"% {key}: regret_max {m['regret_max']:.2e}, ir {m['ir_violation_rate']:.4f}, "
                     f"budget_used {m['budget_used']:.3f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="rq3_decomposition_extra.json (baselines and no-purchase shares)")
    a = ap.parse_args()
    data = json.load(open(a.data))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(data.get("smoke"))
    head = "% SMOKE TEST: layout only, not results\n" if smoke else "% generated by make_paper_assets.py\n"
    collusion_heatmap(data, out, smoke)
    for noise in ("gaussian", "laplace"):
        # Laplace: every curve rises steeply on the left, so the legend goes to the empty lower right (author, 10-25)
        accountant_figure(data, noise, out, smoke, "lower right" if noise == "laplace" else "upper left")
    (out / "rq1_rows.tex").write_text(head + rq1_rows(data))
    (out / "rq3_rows.tex").write_text(head + rq3_rows(data))
    (out / "auction_macros.tex").write_text(head + macros(data))
    (out / "supp_tables.tex").write_text(head + supp_tables(data))
    if data["ours"]["rq1"]:
        (out / "rq1_detail_table.tex").write_text(head + rq1_detail_table(data))
    if data["ours"]["hetero"]:
        (out / "hetero_table.tex").write_text(head + hetero_table(data))
    if data["ours"]["collusion"]:
        (out / "collusion_table.tex").write_text(head + collusion_table(data))
    if data["ours"]["adaptive"]:
        (out / "adaptive_table.tex").write_text(head + adaptive_table(data))
    if data["ours"]["rq3"] and a.extra:
        (out / "rq3_decomposition_table.tex").write_text(head + rq3_decomposition_table(data, json.load(open(a.extra))))
    for noise in ("gaussian", "laplace"):
        if data["ours"]["accountant"][noise]:
            (out / f"accounting_table_{noise}.tex").write_text(head + accounting_table(data, noise))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

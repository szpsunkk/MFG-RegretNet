"""RQ2 figure of the P2-v2 manuscript: (a) time to clear a batch of 100 profiles, (b) incremental GPU memory, against N.

Input: p2v2_auction_side_20261022/results/rq2_timing.json (rq2_timing_p2v2.py on host 101, one RTX A5000, all
mechanisms interleaved in the same process; median of seven calls). Format of the first submission's Fig. 7, shared
style of three_figs_v8f_20261016/exp/refstyle.py (MFG-RegretNet in #205D91, no red). PAC and VCG are drawn as one
line (their times differ by less than 4%; both are listed in the JSON). Mechanisms that run on the CPU allocate
almost no device memory; panel (b) shows them at their measured values.
Usage: python make_rq2_figure.py --input rq2_timing.json --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "three_figs_v8f_20261016/exp"))
import refstyle as RS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

LINES = [("Ours", "MFG-RegretNet"), ("PAC", "PAC"), ("CSRA", "CSRA"), ("MFG-Pricing", "MFG-Pricing"),
         ("RegretNet", "RegretNet"), ("DM-RegretNet", "DM-RegretNet")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.input).read_text())
    Ns = sorted(int(n) for n in d["median_s"])
    for n in Ns:
        pac, vcg = d["median_s"][str(n)]["PAC"], d["median_s"][str(n)]["VCG"]
        assert abs(pac - vcg) / pac < 0.05, (n, pac, vcg)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    fig.subplots_adjust(wspace=0.62)
    for style, key in LINES:
        kw = RS.line_kw(style)
        label = RS.NAME[style]
        axes[0].plot(Ns, [1e3 * d["median_s"][str(n)][key] for n in Ns], label=label, **kw)
        axes[1].plot(Ns, [max(d["memory_mib"][str(n)][key], 1e-2) for n in Ns], label=label, **kw)
    for ax, ylabel, tag in ((axes[0], "time (ms)", "(a)"), (axes[1], "GPU memory (MiB)", "(b)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$N$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.minorticks_off()
        RS.axes_style(ax)
        RS.subcaption(ax, tag)
    RS.legend(fig, handles=axes[0].get_legend_handles_labels()[0], ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RS.finalize(fig)
    fig.savefig(out / "fig_rq2_scalability_p2v2.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_rq2_scalability_p2v2.png", dpi=300, bbox_inches="tight")
    print("wrote", out / "fig_rq2_scalability_p2v2.pdf")


if __name__ == "__main__":
    main()

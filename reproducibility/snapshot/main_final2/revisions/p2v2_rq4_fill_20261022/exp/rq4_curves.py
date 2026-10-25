"""RQ4 convergence figures of the P2-v2 manuscript (format of the first submission's Figs. 9 and 11).

Fixed beforehand (PAPER_EXPERIMENTS_PLAN_20261023.md), independent of the results:
  * main text: one figure per dataset at B = 30, panels alpha = 0.5 and alpha = 0.1;
  * supplement: every setting of the dataset (5 MNIST and 4 CIFAR-10 settings per alpha);
  * lines: MFG-RegretNet, PAC, CSRA, MFG-Pricing, posted price, Lin et al., DM-RegretNet, RegretNet;
  * test accuracy per round, mean over seeds 231-233, no smoothing.
Reads only after every Stage 3 run is complete and Codex's decision file exists (same rule as rq4_fill.py).
Usage: python rq4_curves.py --base ... --old ... --decision ... --out DIR
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "three_figs_v8f_20261016/exp"))
import refstyle as RS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

LINES = [("mfg_p2v2", "Ours"), ("PAC", "PAC"), ("CSRA", "CSRA"), ("MFG-Pricing", "MFG-Pricing"),
         ("Posted-price", "Posted-price"), ("Lin", "Lin"), ("DM-RegretNet", "DM-RegretNet"),
         ("RegretNet-r2", "RegretNet")]
SEEDS = (231, 232, 233)


def read(path):
    with open(path) as fh:
        return json.load(fh)


def curves(jobs, dataset, mode, value, alpha):
    acc = defaultdict(dict)
    for j in jobs:
        c = j["cell"]
        if (c["dataset"], c["mode"], float(c["value"]), float(c["alpha"])) != (dataset, mode, float(value), float(alpha)):
            continue
        if j["arm"] not in dict(LINES):
            continue
        m = sorted(read(j["done"])["measurements"], key=lambda r: r["round"])
        acc[j["arm"]][j["seed"]] = (np.array([r["round"] for r in m]), np.array([100 * r["accuracy"] for r in m]))
    out = {}
    for arm, by_seed in acc.items():
        if set(by_seed) != set(SEEDS):
            continue
        rounds = by_seed[SEEDS[0]][0]
        if any(not np.array_equal(by_seed[s][0], rounds) for s in SEEDS):
            continue
        out[arm] = (rounds, np.mean([by_seed[s][1] for s in SEEDS], axis=0))
    return out


def panel(ax, data, title):
    for arm, style in LINES:
        if arm in data:
            r, y = data[arm]
            ax.plot(r, y, label=RS.NAME[style], **dict(RS.line_kw(style), markevery=max(1, len(r) // 8)))
    ax.set_xlabel("FL round")
    ax.set_ylabel("test accuracy (%)")
    RS.axes_style(ax)
    RS.subcaption(ax, title)


def figure(jobs, dataset, settings, path, smoke):
    n = len(settings)
    cols = 2
    rows_ = (n + 1) // 2
    fig, axes = plt.subplots(rows_, cols, figsize=(3.45, 1.75 * rows_ + 0.35), squeeze=False)
    for k, (mode, value, alpha) in enumerate(settings):
        ax = axes[k // cols][k % cols]
        lab = (f"$B{{=}}{value:g}$" if mode == "budget" else rf"$\bar\epsilon{{=}}{value:g}$")
        panel(ax, curves(jobs, dataset, mode, value, alpha), f"({chr(97 + k)}) {lab}, $\\alpha{{=}}{alpha:g}$")
    for k in range(n, rows_ * cols):
        axes[k // cols][k % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(h_pad=2.2, w_pad=0.8)
    top = 1 - 0.42 / fig.get_size_inches()[1]
    fig.subplots_adjust(top=top)
    RS.legend(fig, handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    if smoke:
        fig.text(0.5, 0.5, "SYNTHETIC - NOT RESULTS", ha="center", va="center", fontsize=14, color="#888888",
                 alpha=0.35, rotation=20)
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="test data only: watermark the figures")
    ap.add_argument("--remap", default="", help="OLD=NEW prefix of the result paths (runs recorded on another host)")
    a = ap.parse_args()
    jobs = read(Path(a.base) / "jobs_stage3_v2.json") + read(Path(a.old) / "jobs_stage3_early.json")
    if a.remap:
        src, dst = a.remap.split("=", 1)
        for j in jobs:
            if j["done"].startswith(src):
                j["done"] = dst + j["done"][len(src):]
    missing = [j["id"] for j in jobs if not Path(j["done"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(jobs)} runs incomplete; nothing read (A3.5)")
    if not Path(a.decision).exists():
        raise SystemExit("Codex's decision file does not exist yet; nothing read")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for dataset, tag, budgets in (("MNIST", "mnist", (10., 30., 100.)), ("CIFAR10", "cifar10", (30., 100.))):
        main_settings = [("budget", 30., .5), ("budget", 30., .1)]
        figure(jobs, dataset, main_settings, out / f"fig_rq4_curves_{tag}_B30.pdf", a.synthetic)
        full = [("budget", b, al) for al in (.5, .1) for b in budgets] + \
               [("privacy", v, al) for al in (.5, .1) for v in (.5, .8)]
        figure(jobs, dataset, full, out / f"fig_rq4_curves_{tag}_all.pdf", a.synthetic)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

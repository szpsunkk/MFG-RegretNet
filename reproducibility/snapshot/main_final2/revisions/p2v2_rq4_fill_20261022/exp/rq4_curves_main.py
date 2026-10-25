"""RQ4 curves of the main text, version 2 (Claude, 2026-10-25): four settings per dataset in one row, smoothed.

Presentation change requested by the author on 2026-10-25, after the results (it departs from
PAPER_EXPERIMENTS_PLAN_20261023.md, which fixed the B = 30 panels and no smoothing for the main text):
  * Fig. rq4-curves-mnist: panels (d), (f), (i), (j) of Supplementary Fig. S1, i.e. alpha = 0.1 with B = 10 and 100
    and with purchased privacy 0.5 and 0.8;
  * Fig. rq4-curves-cifar: panels (c), (d), (g), (h) of Supplementary Fig. S2, i.e. alpha = 0.1 with B = 30 and 100
    and with purchased privacy 0.5 and 0.8;
  * both as double-column figures with the four panels side by side;
  * smoothing for display only: the accuracy is measured every five rounds and in rounds 96-100; the curve takes
    rounds 0, 5, ..., 95 and, as its last point, the mean of rounds 96-100 (the final accuracy A_final of the
    tables), and averages each point with its neighbors within +-H rounds, the window narrowed symmetrically at
    the ends (round 0 and the last point stay as measured). No number in the text or the tables changes; the last
    point of every curve is checked against the independent recomputation of the decision (rq4_recomputed.json).
  * --what all (author's request of the same day, so that the supplement matches the main figures): Supplementary
    Figs. S1 and S2 (every setting, the layout of rq4_curves.py) smoothed in the same way, and the unsmoothed
    per-round accuracy of every drawn mechanism, setting, and seed in rq4_curves_raw.csv.
The selection, the lines, and the seed mean are those of rq4_curves.py; its unsmoothed figures are kept on disk.
Reads only after every Stage 3 run is complete and Codex's decision file exists (same rule as rq4_curves.py).
Usage: python rq4_curves_main.py --base ... --old ... --decision ... --remap OLD=NEW --out DIR [--half-width H]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import rq4_curves as C  # noqa: E402
RS = C.RS
plt = C.plt

GRID = list(range(0, 100, 5))
TAIL = [96, 97, 98, 99, 100]
LETTERS = {"MNIST": "dfij", "CIFAR10": "cdgh"}
BUDGETS = {"MNIST": (10., 30., 100.), "CIFAR10": (30., 100.)}


def full_settings(dataset):
    """Panel order of Supplementary Figs. S1 and S2 (rq4_curves.main)."""
    return ([("budget", b, al) for al in (.5, .1) for b in BUDGETS[dataset]] +
            [("privacy", v, al) for al in (.5, .1) for v in (.5, .8)])


def smooth(rounds, y, half_width):
    """Grid points 0, 5, ..., 95 and the mean of rounds 96-100, averaged within +-half_width rounds."""
    rounds = list(rounds)
    assert rounds == GRID + TAIL, rounds
    pos = {r: i for i, r in enumerate(rounds)}
    g = np.array([y[pos[r]] for r in GRID] + [np.mean([y[pos[r]] for r in TAIL])])
    x = np.array(GRID + [100])
    k, n = half_width // 5, len(g)
    assert half_width % 5 == 0
    out = np.array([g[i - min(k, i, n - 1 - i): i + min(k, i, n - 1 - i) + 1].mean() for i in range(n)])
    return x, out, g


def check_final(dataset, setting, data, recomputed):
    """The last point of MFG-RegretNet's curve is its A_final in the recomputation of the decision."""
    mode, value, alpha = setting
    cell = next(c for c in recomputed if c["cell"] == [dataset, alpha, mode, value])
    _, _, g = smooth(*data["mfg_p2v2"], 5)
    assert abs(g[-1] - 100 * cell["candidate_mean"]) < 1e-6, (setting, g[-1], cell["candidate_mean"])
    return cell


def figure(jobs, dataset, recomputed, path, half_width, smoke):
    full = full_settings(dataset)
    settings = [full[ord(c) - 97] for c in LETTERS[dataset]]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.72))
    info = []
    for k, (ax, setting) in enumerate(zip(axes, settings)):
        mode, value, alpha = setting
        data = C.curves(jobs, dataset, mode, value, alpha)
        # a comparator that no budget brings to the purchased privacy in every seed has no curve ("--" in
        # Supplementary Table S1), as in Supplementary Figs. S1 and S2
        assert "mfg_p2v2" in data and (mode == "privacy" or len(data) == len(C.LINES)), sorted(data)
        cell = check_final(dataset, setting, data, recomputed)
        info.append(dict(panel=chr(97 + k), supp_panel=LETTERS[dataset][k], setting=list(setting),
                         no_curve=sorted(arm for arm, _ in C.LINES if arm not in data),
                         best_external=cell["external_arm"], best_external_drawn=cell["external_arm"] in data,
                         margin_pp=100 * (cell["candidate_mean"] - cell["external_mean"])))
        for arm, style in C.LINES:
            if arm not in data:
                continue
            r, y = data[arm]
            x, ys, _ = smooth(r, y, half_width) if half_width else (np.array(r), np.array(y), None)
            ax.plot(x, ys, label=RS.NAME[style], **dict(RS.line_kw(style), markevery=max(1, len(x) // 7)))
        ax.set_xlabel("FL round")
        if k == 0:
            ax.set_ylabel("test accuracy (%)")
        ax.set_xlim(-3, 103)
        RS.axes_style(ax)
        lab = (f"$B{{=}}{value:g}$" if mode == "budget" else rf"$\bar\epsilon{{=}}{value:g}$")
        RS.subcaption(ax, f"({chr(97 + k)}) {lab}, $\\alpha{{=}}{alpha:g}$", offset=-25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    fig.subplots_adjust(top=1 - 0.30 / fig.get_size_inches()[1], bottom=0.33)
    RS.legend(fig, handles=handles, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    if smoke:
        fig.text(0.5, 0.5, "SYNTHETIC - NOT RESULTS", ha="center", va="center", fontsize=14, color="#888888",
                 alpha=0.35, rotation=20)
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    return info


def figure_all(jobs, dataset, recomputed, path, half_width, smoke):
    """Supplementary Figs. S1 and S2: every setting in the layout of rq4_curves.figure, smoothed as the main figures."""
    settings = full_settings(dataset)
    n, cols = len(settings), 2
    rows_ = (n + 1) // 2
    fig, axes = plt.subplots(rows_, cols, figsize=(3.45, 1.75 * rows_ + 0.35), squeeze=False)
    info = []
    for k, setting in enumerate(settings):
        ax = axes[k // cols][k % cols]
        mode, value, alpha = setting
        data = C.curves(jobs, dataset, mode, value, alpha)
        assert "mfg_p2v2" in data and (mode == "privacy" or len(data) == len(C.LINES)), sorted(data)
        check_final(dataset, setting, data, recomputed)
        info.append(dict(panel=chr(97 + k), setting=list(setting),
                         no_curve=sorted(arm for arm, _ in C.LINES if arm not in data)))
        for arm, style in C.LINES:
            if arm not in data:
                continue
            x, ys, _ = smooth(*data[arm], half_width)
            ax.plot(x, ys, label=RS.NAME[style], **dict(RS.line_kw(style), markevery=max(1, len(x) // 7)))
        ax.set_xlabel("FL round")
        ax.set_ylabel("test accuracy (%)")
        ax.set_xlim(-3, 103)
        RS.axes_style(ax)
        lab = (f"$B{{=}}{value:g}$" if mode == "budget" else rf"$\bar\epsilon{{=}}{value:g}$")
        RS.subcaption(ax, f"({chr(97 + k)}) {lab}, $\\alpha{{=}}{alpha:g}$")
    for k in range(n, rows_ * cols):
        axes[k // cols][k % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(h_pad=2.2, w_pad=0.8)
    fig.subplots_adjust(top=1 - 0.42 / fig.get_size_inches()[1])
    RS.legend(fig, handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    if smoke:
        fig.text(0.5, 0.5, "SYNTHETIC - NOT RESULTS", ha="center", va="center", fontsize=14, color="#888888",
                 alpha=0.35, rotation=20)
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    return info


def write_raw(jobs, path):
    """Unsmoothed test accuracy (%) per round of every drawn mechanism in every setting, per seed and seed mean."""
    names = dict(C.LINES)
    rows = ["dataset,mode,value,alpha,mechanism,round," + ",".join(f"seed{s}" for s in C.SEEDS) + ",mean"]
    for dataset in ("MNIST", "CIFAR10"):
        for mode, value, alpha in full_settings(dataset):
            per = {}
            for j in jobs:
                c = j["cell"]
                if (c["dataset"], c["mode"], float(c["value"]), float(c["alpha"])) != (dataset, mode, value, alpha):
                    continue
                if j["arm"] in names:
                    m = sorted(C.read(j["done"])["measurements"], key=lambda r: r["round"])
                    per.setdefault(j["arm"], {})[j["seed"]] = [(r["round"], 100 * r["accuracy"]) for r in m]
            drawn = C.curves(jobs, dataset, mode, value, alpha)
            for arm, _ in C.LINES:
                if arm not in drawn:
                    continue
                seeds = per[arm]
                assert set(seeds) == set(C.SEEDS)
                for i, (rd, _) in enumerate(seeds[C.SEEDS[0]]):
                    vals = [seeds[s][i][1] for s in C.SEEDS]
                    assert all(seeds[s][i][0] == rd for s in C.SEEDS)
                    assert abs(np.mean(vals) - drawn[arm][1][i]) < 1e-9
                    rows.append(f"{dataset},{mode},{value:g},{alpha:g},{RS.NAME[names[arm]]},{rd}," +
                                ",".join(f"{v:.4f}" for v in vals) + f",{np.mean(vals):.4f}")
    Path(path).write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--recomputed", default=str(HERE.parent / "results/rq4_recomputed.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--half-width", type=int, default=10, help="rounds on each side; 0 = no smoothing")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--what", choices=("main", "all"), default="main",
                    help="main: Figs. 9 and 10 of the main text; all: Supplementary Figs. S1 and S2 and the raw CSV")
    ap.add_argument("--synthetic", action="store_true", help="test data only: watermark the figures")
    ap.add_argument("--remap", default="", help="OLD=NEW prefix of the result paths (runs recorded on another host)")
    a = ap.parse_args()
    jobs = C.read(Path(a.base) / "jobs_stage3_v2.json") + C.read(Path(a.old) / "jobs_stage3_early.json")
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
    rec = C.read(a.recomputed)
    assert rec["differences_to_codex"] == [] and rec["runs"] == 880
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(half_width_rounds=a.half_width, endpoint_checked_against="rq4_recomputed.json", synthetic=a.synthetic,
                panels={})
    if a.what == "all":
        for dataset, tag in (("MNIST", "mnist"), ("CIFAR10", "cifar10")):
            meta["panels"][dataset] = figure_all(jobs, dataset, rec["decision"]["per_cell"],
                                                 out / f"fig_rq4_curves_{tag}_{a.tag}.pdf", a.half_width, a.synthetic)
        meta["raw_csv_rows"] = write_raw(jobs, out.parent / "rq4_curves_raw.csv")
        (out / f"fig_rq4_curves_{a.tag}.json").write_text(json.dumps(meta, indent=1))
        print("panels", {d: len(p) for d, p in meta["panels"].items()}, "raw rows", meta["raw_csv_rows"])
        print("wrote", sorted(p.name for p in out.iterdir() if a.tag in p.name))
        return
    for dataset, tag in (("MNIST", "mnist"), ("CIFAR10", "cifar10")):
        info = figure(jobs, dataset, rec["decision"]["per_cell"], out / f"fig_rq4_curves_{tag}_{a.tag}.pdf",
                      a.half_width, a.synthetic)
        meta["panels"][dataset] = info
        for x in info:
            print(dataset, x["panel"], x["setting"], f"margin {x['margin_pp']:+.2f}", "no curve:", x["no_curve"],
                  "best external:", x["best_external"], "drawn" if x["best_external_drawn"] else "NOT DRAWN")
    (out / f"fig_rq4_curves_{a.tag}.json").write_text(json.dumps(meta, indent=1))
    print("wrote", sorted(p.name for p in out.iterdir() if a.tag in p.name))


if __name__ == "__main__":
    main()

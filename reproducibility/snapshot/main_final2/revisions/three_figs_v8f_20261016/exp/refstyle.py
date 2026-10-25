"""Plot style borrowed from Fig. 8 of TMC-2026-05-1886 (plot_figure8.py), shared by make_figs.py and make_curves.py.

Kept from the reference: STIX serif text and math, 8 pt labels, 7.5 pt ticks, 7 pt legends, all four spines at
0.6 pt, no grid, outward ticks on the left and bottom axes, 0.9 pt lines with hollow markers (size 3, edge 0.6 pt),
bars with a light tint of the series colour (0.78 + 0.22 rgb), a 0.65 pt edge and a hatch in the series colour,
framed legends inside the axes (grey #777777 edge, 0.45 pt), sub-captions "(a) ..." placed below the x label.
Changed: the reference's red (#B14442) is not used (author: no red); three of its colours are kept
(blue #205D91 for MFG-RegretNet, teal #258579, ochre #997522) and five more were chosen greedily with the dataviz
validator for the largest colour distance to the others (non-reddish hues only). Pairs that stay close (blue vs teal,
as in the reference) are told apart by marker, line style and hatch.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

os.environ["PATH"] = "/usr/local/texlive/2022/bin/x86_64-linux:" + os.environ.get("PATH", "")

RC = {
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "hatch.linewidth": 0.45, "axes.grid": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}
plt.rcParams.update(RC)

COLOR = {"Ours": "#205D91", "MFG-Pricing": "#258579", "RegretNet": "#997522", "PAC": "#B0B0B0",
         "DM-RegretNet": "#222222", "CSRA": "#8F77CF", "Lin": "#5D1C62", "Posted-price": "#6A4F39"}
MARKER = {"Ours": "o", "MFG-Pricing": "D", "RegretNet": "v", "PAC": "p", "DM-RegretNet": "^", "CSRA": "s",
          "Lin": "X", "Posted-price": "P"}
LINESTYLE = {"Ours": "-", "MFG-Pricing": "-.", "RegretNet": "--", "PAC": "-", "DM-RegretNet": ":",
             "CSRA": (0, (1, 1.2)), "Lin": (0, (3, 1, 1, 1)), "Posted-price": (0, (5, 1.5))}
HATCH = {"Ours": "///", "MFG-Pricing": "\\\\\\", "DM-RegretNet": "xx", "Lin": "---", "RegretNet": "|||",
         "CSRA": "++", "PAC": None, "Posted-price": "//"}
NAME = {"Ours": "MFG-RegretNet", "MFG-Pricing": "MFG-Pricing", "RegretNet": "RegretNet", "PAC": "PAC/VCG",
        "DM-RegretNet": "DM-RegretNet", "CSRA": "CSRA", "Lin": "Lin et al.", "Posted-price": "Posted price"}


def tint(name):
    return tuple(0.78 + 0.22 * c for c in to_rgb(COLOR[name]))


def line_kw(name, emphasis=True):
    ours = emphasis and name == "Ours"
    return dict(color=COLOR[name], marker=MARKER[name], markerfacecolor="white", markeredgewidth=0.6,
                markeredgecolor=COLOR[name], lw=1.2 if ours else 0.9, ls=LINESTYLE[name], ms=3.4 if ours else 3.0,
                zorder=6 if ours else 3)


def bar_kw(name):
    return dict(color=tint(name), edgecolor=COLOR[name], linewidth=0.65, hatch=HATCH[name], zorder=2)


def axes_style(ax):
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", pad=2, labelsize=7.5, top=False, right=False)
    ax.tick_params(axis="both", which="minor", top=False, right=False)
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 3


def subcaption(ax, text, offset=-29):
    ax.annotate(text, (0.5, 0), xycoords="axes fraction", xytext=(0, offset), textcoords="offset points",
                fontsize=7.5, ha="center", va="top", annotation_clip=False)


def legend(ax_or_fig, handles=None, **kw):
    opts = dict(loc="best", fontsize=7, frameon=True, facecolor="white", edgecolor="#777777", framealpha=1,
                fancybox=False, borderpad=0.25, handlelength=1.05, columnspacing=0.55, handletextpad=0.35,
                labelspacing=0.2, markerscale=0.8, borderaxespad=0.4)
    opts.update(kw)
    leg = ax_or_fig.legend(handles=handles, **opts) if handles is not None else ax_or_fig.legend(**opts)
    leg.get_frame().set_linewidth(0.45)
    return leg


class BoundedHatch(Artist):
    """Vector hatching of plot_figure8.py (reference): line spacing 1/(6 n) inch for an n-character hatch,
    0.45 pt strokes in the edge colour, clipped to the rectangle. Extended with '|' and '+'; works for bars of
    negative height."""

    def __init__(self, rectangle):
        super().__init__()
        self.rectangle = rectangle
        self.hatch = rectangle.get_hatch()
        self.set_zorder(rectangle.get_zorder() + 0.01)
        self.set_in_layout(False)
        rectangle.set_hatch(None)

    def draw(self, renderer):
        rectangle, hatch, fig = self.rectangle, self.hatch, self.figure
        e = rectangle.get_window_extent(renderer).extents / fig.dpi
        x0, x1 = sorted((e[0], e[2])); y0, y1 = sorted((e[1], e[3]))
        if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
            return
        segments = []
        spacing = 1.0 / (6 * len(hatch))
        kind = hatch[0]
        directions = {"x": "/\\", "+": "-|"}.get(kind, kind)
        for direction in directions:
            if direction == "-":
                for y in np.arange(np.ceil(y0 / spacing) * spacing, y1, spacing):
                    segments.append([(x0, y), (x1, y)])
                continue
            if direction == "|":
                for x in np.arange(np.ceil(x0 / spacing) * spacing, x1, spacing):
                    segments.append([(x, y0), (x, y1)])
                continue
            lower, upper = ((y0 - x1, y1 - x0) if direction == "/" else (y0 + x0, y1 + x1))
            for intercept in np.arange(np.ceil(lower / spacing) * spacing, upper, spacing):
                if direction == "/":
                    left, right = max(x0, y0 - intercept), min(x1, y1 - intercept)
                    endpoints = [(left, left + intercept), (right, right + intercept)]
                else:
                    left, right = max(x0, intercept - y1), min(x1, intercept - y0)
                    endpoints = [(left, intercept - left), (right, intercept - right)]
                if right > left:
                    segments.append(endpoints)
        strokes = LineCollection(segments, colors=[rectangle.get_edgecolor()], linewidths=0.45,
                                 transform=fig.dpi_scale_trans)
        strokes.set_figure(fig)
        strokes.draw(renderer)


def finalize(fig):
    """Replace every hatch (bars and legend keys) by the bounded vector hatch of the reference."""
    fig.canvas.draw()
    axes = fig.get_axes()
    for rectangle in fig.findobj(Rectangle):
        if rectangle.get_hatch():
            texture = BoundedHatch(rectangle)
            owner = next((ax for ax in axes if rectangle in ax.patches), None)
            if owner is None:
                texture.set_zorder(6)
                fig.add_artist(texture)
            else:
                owner.add_artist(texture)

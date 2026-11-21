"""Feature-space maps for the report (2 x 2, one column wide), from lsbfd.featurespace outputs.
Usage: fs_maps.py <featurespace out dir> <fig dir>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

src, out = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9})
INK, MUTED, GRID = "#1F2933", "#52606D", "#D9DEE4"
CLASSES = ["healthy", "inner_ring", "multiple", "outer_ring"]
NAMES = {"healthy": "Healthy", "inner_ring": "Inner ring", "multiple": "Multiple",
         "outer_ring": "Outer ring"}
COL = {"healthy": "#009E73", "inner_ring": "#0072B2", "multiple": "#CC79A7",
       "outer_ring": "#E69F00"}
MK = {"healthy": "o", "inner_ring": "s", "multiple": "D", "outer_ring": "^"}
TEST = {"K001": "healthy", "KI21": "inner_ring", "KB23": "multiple", "KA30": "outer_ring"}

M = json.loads((src / "metrics.json").read_text())
P = np.load(src / "projections.npz", allow_pickle=False)
rows = {r["space"]: r for r in M["rows"]}
bearing, y = P["bearing"], P["y"]
title = {"generic": "(a) Generic statistics", "phys_clip4": "(b) Envelope-order, despiked",
         "erm": "(c) CNN embedding (ERM)", "dann": "(d) DANN embedding"}
order = [k for k in title if k in P.files]

fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.7), dpi=300)
axes = axes.ravel()
tr = ~np.isin(bearing, list(TEST))
for ax, key in zip(axes, order):
    E = P[key]
    for ci, c in enumerate(CLASSES):
        m = tr & (y == ci)
        ax.scatter(E[m, 0], E[m, 1], s=0.6, c=COL[c], marker=MK[c], alpha=0.30, lw=0,
                   rasterized=True)
    for b, c in TEST.items():
        m = bearing == b
        ax.scatter(E[m, 0], E[m, 1], s=4, c=COL[c], marker=MK[c], edgecolors=INK, lw=0.25,
                   alpha=0.95, zorder=3, rasterized=True)
    r = rows[key]
    ax.set_title(title[key], fontsize=8.5, color=INK, loc="left", pad=2)
    ax.text(0.0, -0.03, f"class {r['silhouette_class']:+.3f}, bearing "
            f"{r['silhouette_bearing_in_class']:+.3f}", transform=ax.transAxes, va="top",
            fontsize=7, color=MUTED)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
for ax in axes[len(order):]:
    ax.axis("off")
h = [Line2D([], [], ls="", marker=MK[c], color=COL[c], ms=4, label=NAMES[c]) for c in CLASSES]
h += [Line2D([], [], ls="", marker="o", color="#9AA5B1", mec=INK, mew=0.5, ms=4,
             label="test bearing")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7,
           handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(h_pad=1.6, w_pad=0.6, rect=(0, 0.04, 1, 1))
fig.savefig(out / "fs_maps.png", bbox_inches="tight")
print("ok", order, M["projection"])

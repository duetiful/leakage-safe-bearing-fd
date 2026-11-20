"""Feature-space figures for the deck, from lsbfd.featurespace outputs (metrics.json,
projections.npz). Usage: fs_plots.py <featurespace out dir> <fig dir>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

src, out = Path(sys.argv[1]), Path(sys.argv[2])
for f in fm.findSystemFonts():
    if "Carlito" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Carlito", "font.size": 13})
INK, MUTED, GRID = "#1F2933", "#52606D", "#E4E7EB"
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
order = [k for k in ("generic", "phys", "phys_clip4", "erm", "dann") if k in P.files]
short = {"generic": "Generic statistics (11-d)", "phys": "Envelope-order features (85-d)",
         "phys_clip4": "Envelope-order, despiked (85-d)", "erm": "CNN embedding (128-d)",
         "dann": "DANN embedding (128-d)"}

fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), dpi=170)
axes = axes.ravel()
for ax, key in zip(axes, order):
    E = P[key]
    tr = ~np.isin(bearing, list(TEST))
    for ci, c in enumerate(CLASSES):
        m = tr & (y == ci)
        ax.scatter(E[m, 0], E[m, 1], s=4, c=COL[c], marker=MK[c], alpha=0.35, lw=0,
                   rasterized=True)
    for b, c in TEST.items():
        m = bearing == b
        ax.scatter(E[m, 0], E[m, 1], s=26, c=COL[c], marker=MK[c], edgecolors=INK, lw=0.7,
                   alpha=0.95, zorder=3)
    r = rows[key]
    ax.set_title(short[key], fontsize=15, color=INK, fontweight="bold", loc="left")
    ax.text(0.01, -0.02, f"silhouette class {r['silhouette_class']:+.2f}  ·  bearing "
            f"{r['silhouette_bearing_in_class']:+.2f}  ·  kNN F1 {r['knn_f1_fold0']:.2f}",
            transform=ax.transAxes, va="top", fontsize=12, color=MUTED)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
for ax in axes[len(order):]:
    ax.axis("off")
leg = axes[-1] if len(order) < 6 else None
if leg is not None:
    h = [Line2D([], [], ls="", marker=MK[c], color=COL[c], ms=8, label=NAMES[c]) for c in CLASSES]
    h += [Line2D([], [], ls="", marker="o", color="#9AA5B1", ms=4, alpha=0.6,
                 label="training bearings (small)"),
          Line2D([], [], ls="", marker="o", color="#9AA5B1", mec=INK, ms=8,
                 label="unseen test bearings\nK001, KI21, KB23, KA30")]
    leg.legend(handles=h, loc="center left", frameon=False, fontsize=14)
    leg.text(0.02, 0.06, f"{M['projection']} of {M['n_points']:,} windows (4 per recording),\n"
             "fitted without labels. Numbers under each map\nare computed in the full space.",
             transform=leg.transAxes, fontsize=12, color=MUTED)
fig.tight_layout(h_pad=2.2)
fig.savefig(out / "fs_maps.png", bbox_inches="tight")

# ------------------------------------------------------------ metric bars
keys = [k for k in ("generic", "generic_clip4", "phys", "phys_clip4", "phys_clip4_real", "erm",
                    "dann") if k in rows]
lab = {"generic": "Generic (11)", "generic_clip4": "Generic, despiked", "phys": "Envelope (85)",
       "phys_clip4": "Envelope, despiked", "phys_clip4_real": "Envelope, despiked,\nreal-damage only",
       "erm": "CNN embedding", "dann": "DANN embedding"}
col = {"generic": "#9AA5B1", "generic_clip4": "#9AA5B1", "phys": "#2F7D4F", "phys_clip4": "#2F7D4F",
       "phys_clip4_real": "#2F7D4F", "erm": "#1F5FAD", "dann": "#C0392B"}
panels = [("silhouette_class", "Class silhouette  (higher = compact,\nseparated classes)"),
          ("silhouette_bearing_in_class", "Bearing silhouette within class\n(lower = less bearing identity)"),
          ("knn_f1_fold0", "kNN transfer macro-F1, fold 0\n(unseen bearings, 15 neighbours)")]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), dpi=170, sharey=True)
yy = np.arange(len(keys))[::-1]
for ax, (m, title) in zip(axes, panels):
    v = np.array([rows[k][m] for k in keys])
    ax.barh(yy, v, height=0.62, color=[col[k] for k in keys], zorder=3)
    for yi, vi in zip(yy, v):
        ax.text(vi + (0.01 if vi >= 0 else -0.01), yi, f"{vi:.2f}", va="center",
                ha="left" if vi >= 0 else "right", fontsize=12, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    lo, hi = min(0, v.min()), max(v.max(), 0)
    pad = (hi - lo) * 0.28 + 0.02
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(yy, [lab[k] for k in keys], fontsize=12.5)
fig.tight_layout()
fig.savefig(out / "fs_bars.png", bbox_inches="tight")
print("ok", order, keys)

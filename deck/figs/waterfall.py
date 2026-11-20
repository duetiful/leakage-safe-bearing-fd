"""Chronological waterfall charts for the v4 deck. Numbers are copied from the repo's
results files (reproduction.md, adaptation.md, novel.md, explore.md) and the viva guide."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
import sys

out = sys.argv[1]
for f in fm.findSystemFonts():
    if "Carlito" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Carlito", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9AA5B1",
                     "xtick.color": "#52606D", "ytick.color": "#52606D"})
INK, MUTED, GRID = "#1F2933", "#52606D", "#E4E7EB"
UP, DOWN, TOTAL, START = "#2F7D4F", "#C0392B", "#1F5FAD", "#9AA5B1"


def bar(ax, x, lo, hi, color, hatch=None, fill=True, w=0.62):
    b = min(lo, hi)
    ax.bar(x, abs(hi - lo), bottom=b, width=w, color=color if fill else "white",
           edgecolor=color, linewidth=1.6, hatch=hatch, zorder=3)


def label(ax, x, y, text, color=INK, dy=0.012, va="bottom", size=12.5, bold=False):
    ax.text(x, y + (dy if va == "bottom" else -dy), text, ha="center", va=va, color=color,
            fontsize=size, fontweight="bold" if bold else "normal", zorder=5)


# ---------------------------------------------------------------- fold 0 (benchmark split)
steps = [  # (xlabel, kind, lo, hi, value text, delta text)
    ("Published\nLR (paper)", "start", 0, 0.355, "0.355", "CNN 0.326"),
    ("My reproduction\n(3 channels)", "up", 0.355, 0.388, "0.388", "+0.033"),
    ("Vibration\nonly", "up", 0.388, 0.508, "0.508", "+0.120"),
    ("Leaky split\n(seen bearings)", "leak", 0, 0.70, "CNN 0.999", "invalid test"),
    ("CNN + DANN\n(within class)", "dead", 0.508, 0.348, "0.348", "−0.160"),
    ("CNN + SimCLR", "dead", 0.508, 0.408, "0.408", "−0.100"),
    ("Envelope CNN\n+ AdaBN", "dead", 0.508, 0.393, "0.393", "−0.115"),
    ("Envelope features\n+ trees", "down", 0.508, 0.490, "0.490", "−0.018"),
    ("Train on real\ndamage only", "up", 0.490, 0.560, "0.560", "+0.070"),
    ("+ Despiking", "up", 0.560, 0.590, "0.590", "+0.030"),
]
fig, ax = plt.subplots(figsize=(12.4, 5.1), dpi=200)
ax.set_ylim(0, 0.72)
ax.yaxis.grid(True, color=GRID, zorder=0)
ax.set_axisbelow(True)
level = None
for i, (name, kind, lo, hi, v, d) in enumerate(steps):
    if kind == "start":
        bar(ax, i, 0, hi, START)
        label(ax, i, hi, v, bold=True)
        ax.plot([i - 0.31, i + 0.31], [0.326] * 2, color=DOWN, lw=2, zorder=4)
        ax.text(i, 0.326 - 0.012, d, ha="center", va="top", color="white", fontsize=11, zorder=5)
    elif kind == "leak":
        bar(ax, i, 0, hi, MUTED, hatch="///", fill=False)
        ax.text(i, 0.705, "0.999 ↑", ha="center", va="bottom", color=MUTED, fontsize=12.5,
                fontweight="bold")
        ax.text(i, 0.36, "not a valid\ncomparison", ha="center", va="center", color=MUTED,
                fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    elif kind == "dead":
        bar(ax, i, lo, hi, DOWN, hatch="//", fill=False)
        label(ax, i, hi, v, va="top", color=INK)
        label(ax, i, hi - 0.045, d + " vs LR", va="top", color=DOWN, size=11.5)
    else:
        bar(ax, i, lo, hi, UP if kind == "up" else DOWN)
        top = max(lo, hi)
        label(ax, i, top, v, bold=True)
        label(ax, i, top + 0.032, d, color=UP if kind == "up" else DOWN, size=12)
    if level is not None and kind not in ("leak",):
        ax.plot([i - 1 + 0.31, i - 0.31], [level, level], color=MUTED, lw=1, ls=":", zorder=2)
    if kind in ("start", "up", "down"):
        level = hi
    elif kind == "leak":
        pass
ax.plot([9 - 0.31, 9 + 0.31], [0.590] * 2, color=INK, lw=0)
ax.axhline(0.508, color=MUTED, lw=1, ls="--", zorder=1)
ax.text(9.45, 0.508, "LR bar\n0.508", va="center", ha="left", color=MUTED, fontsize=12)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=11.5, color=INK)
ax.set_xlim(-0.6, 10.1)
ax.set_ylabel("Macro-F1 on K001, KA30, KI21, KB23", color=MUTED)
for x0, x1, t in [(-0.4, 3.4, "Phase 1: reproduce"), (3.6, 6.4, "Phase 2–3: deep models"),
                  (6.6, 7.4, "Phase 3"), (7.6, 9.4, "Phase 4 (exploratory)")]:
    ax.annotate("", xy=(x0, -0.215), xytext=(x1, -0.215), xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-", color=GRID, lw=3))
    ax.text((x0 + x1) / 2, -0.25, t, transform=ax.get_xaxis_transform(), ha="center",
            va="top", color=MUTED, fontsize=12.5)
ax.legend(handles=[Patch(fc=UP, label="kept, improves"), Patch(fc=DOWN, label="kept, lower on fold 0"),
                   Patch(fc="white", ec=DOWN, hatch="//", label="tried, dropped (vs LR 0.508)"),
                   Patch(fc=START, label="published benchmark")],
          loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=12.5, ncol=4)
fig.tight_layout()
fig.savefig(f"{out}/waterfall_fold0.png", bbox_inches="tight")

# ---------------------------------------------------------------- 10 folds
steps = [
    ("LR, generic\nfeatures", "start", 0, 0.355, "0.355", "± 0.101"),
    ("Swap in envelope-\norder features", "up", 0.355, 0.448, "0.448", "+0.093  (8/10, p=0.027)"),
    ("LR → boosted\ntrees", "up", 0.448, 0.528, "0.528", "+0.080"),
    ("Train on real\ndamage only", "up", 0.528, 0.563, "0.563", "+0.035"),
    ("+ Despiking", "up", 0.563, 0.570, "0.570", "+0.007"),
]
fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
ax.set_ylim(0, 0.74)
ax.yaxis.grid(True, color=GRID, zorder=0)
level = None
for i, (name, kind, lo, hi, v, d) in enumerate(steps):
    bar(ax, i, lo, hi, START if kind == "start" else UP, w=0.6)
    label(ax, i, hi, v, bold=True)
    if kind != "start":
        label(ax, i, hi + 0.034, d.split("  ")[0], color=UP, size=12)
    if level is not None:
        ax.plot([i - 1 + 0.3, i - 0.3], [level, level], color=MUTED, lw=1, ls=":")
    level = hi
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=11.5, color=INK)
ax.set_ylabel("Mean macro-F1, 10 held-out bearing sets", color=MUTED)
ax.text(-0.3, 0.70, "Envelope + trees vs LR generic: +0.174, better on 9 of 10 folds, p = 0.004",
        ha="left", color=INK, fontsize=12.5)
ax.text(3.5, 0.655, "exploratory (Phase 4)", ha="center", color=MUTED, fontsize=12)
ax.plot([2.65, 4.35], [0.648, 0.648], color=MUTED, lw=1)
fig.tight_layout()
fig.savefig(f"{out}/waterfall_10fold.png", bbox_inches="tight")
print("ok")

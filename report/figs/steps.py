"""Chronological step chart for the report (one column): one row per step, a dot at its
macro-F1 on the benchmark split. Numbers from results/{reproduction,adaptation,novel,explore}.md.
Usage: steps.py <fig dir>"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8})
INK, MUTED, GRID, KEPT, DROP, PUB = "#1F2933", "#616E7C", "#E4E7EB", "#1F5FAD", "#9AA5B1", "#1F2933"
BAR = 0.508

# (label, score, kind, delta text); kind: pub | kept | drop ; None = phase header
rows = [
    ("Phase 1: reproduce the benchmark", None, None, ""),
    ("Published LR (paper)", 0.355, "pub", ""),
    ("Published 1D CNN (paper)", 0.326, "pub", ""),
    ("Reproduced LR (2 currents + vibration)", 0.388, "kept", "-0.120"),
    ("Vibration-only LR (sets the bar)", 0.508, "kept", "0"),
    ("Phase 2: domain adaptation (CNN, 3 seeds)", None, None, ""),
    ("CNN, plain training (control)", 0.404, "drop", "-0.104"),
    ("CNN + SimCLR pre-training", 0.408, "drop", "-0.100"),
    ("CNN + DANN (within-class)", 0.348, "drop", "-0.160"),
    ("Phase 3: physics-informed input", None, None, ""),
    ("Envelope CNN + AdaBN", 0.393, "drop", "-0.115"),
    ("Envelope-order features + trees", 0.490, "kept", "-0.018"),
    ("Phase 4: data curation (post hoc)", None, None, ""),
    ("Train on healthy + real damage only", 0.560, "kept", "+0.052"),
    ("+ despiked input", 0.590, "kept", "+0.082"),
]

fig, ax = plt.subplots(figsize=(3.5, 3.3), dpi=300)
n = len(rows)
ys = list(range(n))[::-1]
ax.axvline(BAR, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(BAR, n - 0.4, "LR bar", ha="center", va="bottom", fontsize=7, color=MUTED)
ax.text(1.0, n - 0.4, "vs bar", ha="right", va="bottom", fontsize=7, color=MUTED,
        transform=ax.get_yaxis_transform())
path, labels = [], []
for y, (lab, v, kind, d) in zip(ys, rows):
    labels.append(lab)
    if v is None:
        continue
    ax.hlines(y, 0.30, 0.62, color=GRID, lw=0.5, zorder=0)
    if kind == "pub":
        ax.scatter(v, y, s=14, color=PUB, zorder=3, marker="s")
    elif kind == "kept":
        ax.scatter(v, y, s=20, color=KEPT, zorder=3)
        path.append((v, y))
    else:
        ax.scatter(v, y, s=20, facecolor="white", edgecolor=DROP, lw=1.1, zorder=3)
    left = v > 0.55
    ax.text(v + (-0.013 if left else 0.013), y, f"{v:.3f}", ha="right" if left else "left",
            va="center", fontsize=6.6, color=INK, zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.4))
    if d:
        ax.text(1.0, y, d.replace("-", "\u2212"), ha="right", va="center", fontsize=6.6, color=MUTED,
                transform=ax.get_yaxis_transform())
xs, yv = zip(*path)
ax.plot(xs, yv, color=KEPT, lw=0.8, alpha=0.5, zorder=2)
ax.set_yticks(ys, labels)
for t, (lab, v, kind, d) in zip(ax.get_yticklabels(), rows):
    t.set_fontsize(7.2 if v is not None else 7.4)
    t.set_color(INK)
    if v is None:
        t.set_fontweight("bold")
ax.tick_params(axis="y", length=0, pad=3)
ax.set_xlim(0.30, 0.67)
ax.set_ylim(-0.6, n + 0.2)
ax.set_xticks([0.3, 0.4, 0.5, 0.6])
ax.tick_params(axis="x", labelsize=7, colors=MUTED, length=2)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_bounds(0.3, 0.6)
ax.spines["bottom"].set_color(MUTED)
ax.set_xlabel("macro-F1 on the benchmark split", fontsize=7.2, color=MUTED)
fig.tight_layout(pad=0.2)
fig.savefig(f"{out}/steps.pdf", bbox_inches="tight")
fig.savefig(f"{out}/steps.png", bbox_inches="tight")
print("ok")

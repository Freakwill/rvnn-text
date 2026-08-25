"""Render the demo GENIA tree as an image for the README (data-format section).

Usage: PYTHONPATH=. python3 scripts/plot_tree.py
Output: assets/genia_tree.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"  # bundled, has real bold variants

from rvnn_text.genia import _prune_punct, load_trees, parse_ptb

DATA = Path("data/genia/genia-dist/division/train.trees")
OUT = Path("assets/genia_tree.png")
TARGET = "High-risk patients can be recognized morphologically".split()

trees = load_trees(DATA)


def words(n):
    return [n.word] if n.word is not None else [w for c in n.children for w in words(c)]


# Find the first occurrence of the demo sentence.
tree = next(t for t in trees if words(t) == TARGET)
print("tree found, leaves:", TARGET)

# Find its line number in the raw file for the caption.
raw = DATA.read_text().splitlines()
line_no = None
for i, line in enumerate(raw):
    n = _prune_punct(parse_ptb(line.strip()))
    if n is not None and words(n) == TARGET:
        line_no = i + 1
        break
print("raw line:", line_no)

# ---- layout ------------------------------------------------------------
leaf_idx = 0


def assign_x(n):
    global leaf_idx
    if n.word is not None:
        n.x = float(leaf_idx)
        leaf_idx += 1
    else:
        for c in n.children:
            assign_x(c)
        n.x = sum(c.x for c in n.children) / len(n.children)


def assign_depth(n, d=0):
    n.depth = d
    for c in n.children:
        assign_depth(c, d + 1)


assign_x(tree)
assign_depth(tree)

# ---- draw ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
ax.axis("off")


def draw(n):
    x, y = n.x, -n.depth
    for c in n.children:
        ax.plot([x, c.x], [y, -c.depth], color="#4a5568", lw=1.1, zorder=1)
        draw(c)
    if n.word is not None:
        # POS tag (bold, tinted) with the word beneath in italic.
        ax.text(
            x, y, n.symbol, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="#1d4ed8",
            bbox=dict(boxstyle="round,pad=0.22", fc="#dbeafe", ec="#93c5fd", lw=0.8),
            zorder=2,
        )
        ax.text(
            x, y - 0.42, n.word, ha="center", va="center", fontsize=9.5,
            fontstyle="italic", color="#1a202c", zorder=2,
        )
    else:
        ax.text(
            x, y, n.symbol, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="#111827",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#94a3b8", lw=0.9),
            zorder=2,
        )


draw(tree)
ax.set_title(
    f"GENIA treebank — train.trees line {line_no}\n"
    f"\"High-risk patients can be recognized morphologically\" (S1-rooted constituent tree)",
    fontsize=10.5, pad=10,
)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved:", OUT.resolve())

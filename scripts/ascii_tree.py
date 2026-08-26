"""Render a Node tree as ASCII box-drawing art (chat/README display).

`ascii_tree(tree)` returns the art; `words(n)` lists leaves in order.
Running this file directly re-renders the GENIA demo sentence.
"""
from pathlib import Path

from rvnn_text.genia import load_trees


def words(n):
    return [n.word] if n.word is not None else [w for c in n.children for w in words(c)]


def _collect_leaves(n, out):
    if n.word is not None:
        out.append(n)
    else:
        for c in n.children:
            _collect_leaves(c, out)


def ascii_tree(tree, W=20):
    """Draw the tree with box-drawing characters. W = char width per leaf slot."""
    leaves = []
    _collect_leaves(tree, leaves)
    x = {id(n): i * W + W // 2 for i, n in enumerate(leaves)}
    depth = {}

    def assign(n, d=0):
        depth[id(n)] = d
        if n.word is None:
            for c in n.children:
                assign(c, d + 1)
            x[id(n)] = (x[id(n.children[0])] + x[id(n.children[-1])]) // 2

    assign(tree)

    rows = (max(depth.values()) + 1) * 2 - 1
    ncols = len(leaves) * W
    grid = [[" "] * ncols for _ in range(rows)]

    def render(n):
        d = depth[id(n)]
        cx = x[id(n)]
        lab = n.word if n.word is not None else n.symbol
        start = cx - len(lab) // 2
        for j, ch in enumerate(lab):
            grid[d * 2][start + j] = ch
        if n.word is not None:
            return
        er = d * 2 + 1  # edge row
        cxs = [x[id(c)] for c in n.children]
        if len(cxs) == 1:
            grid[er][cxs[0]] = "│"
        else:
            lo, hi = min(cxs), max(cxs)
            for j in range(lo, hi + 1):
                grid[er][j] = "─"
            for cxx in cxs:
                if cxx == lo:
                    grid[er][cxx] = "┌"
                elif cxx == hi:
                    grid[er][cxx] = "┐"
                else:
                    grid[er][cxx] = "│"
            grid[er][cx] = "┴"
        for c in n.children:
            render(c)

    render(tree)
    return "\n".join("".join(r).rstrip() for r in grid)


if __name__ == "__main__":
    trees = load_trees(Path("data/genia/genia-dist/division/train.trees"))
    TARGET = ["High-risk", "patients", "can", "be", "recognized", "morphologically"]
    tree = next(t for t in trees if words(t) == TARGET)
    print(ascii_tree(tree))

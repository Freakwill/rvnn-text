"""Empirical check: S1/S/S structure + punctuation attachment in raw GENIA."""
from collections import Counter
from pathlib import Path

from rvnn_text.genia import parse_ptb

lines = Path("data/genia/genia-dist/division/train.trees").read_text().splitlines()
parsed = [parse_ptb(l.strip()) for l in lines if l.strip()]

# 1. root symbol distribution
root = Counter(t.symbol for t in parsed)
print("root symbols:", root.most_common(6))

# 2. what are the children of the root?
kids = Counter(tuple(c.symbol for c in t.children) for t in parsed if t.children)
print("root -> children:", kids.most_common(6))

# 3. among S1 roots: what's under the depth-1 node?
def under(t, n=1):
    node = t
    for _ in range(n):
        node = node.children[0] if node.children else None
    return node

s1 = [t for t in parsed if t.symbol == "S1"]
d1 = Counter((under(t, 1).symbol if under(t, 1) else None) for t in s1)
print("S1 -> :", d1.most_common(5))

# 4. among S1 -> S: what does that S contain?
s1s = [t for t in s1 if t.children and t.children[0].symbol == "S"]
d2 = Counter(tuple(c.symbol for c in t.children[0].children) for t in s1s if t.children[0].children)
print("S1->S->children:", d2.most_common(6))

# 5. where do punct leaves attach? (sibling of clause at depth 2 vs inside phrase)
def punct_positions(t, node, path):
    """Return (depth, is_last_sibling, parent_symbol) for each punct leaf."""
    out = []
    for i, c in enumerate(node.children):
        if c.word is not None and c.symbol in {".", ",", ":", "'", "`", "-LRB-", "-RRB-"}:
            out.append((len(path), i == len(node.children) - 1, node.symbol))
        else:
            out += punct_positions(t, c, path + [node.symbol])
    return out

allpos = []
for t in s1:
    allpos += punct_positions(t, t, [])
cnt = Counter((d, last, par) for d, last, par in allpos)
print("punct attach (depth, is_last_sibling, parent):", cnt.most_common(8))
print("total punct leaves:", len(allpos))

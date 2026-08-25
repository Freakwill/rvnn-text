"""Detail: children of the depth-2 S in the typical S1->S->S pattern."""
from collections import Counter
from pathlib import Path

from rvnn_text.genia import parse_ptb

lines = Path("data/genia/genia-dist/division/train.trees").read_text().splitlines()
parsed = [parse_ptb(l.strip()) for l in lines if l.strip()]


def node_at(t, path):
    n = t
    for i in path:
        n = n.children[i]
    return n


# S1 -> S -> S : children of the depth-2 S
pat = Counter()
for t in parsed:
    if len(t.children) == 1 and t.children[0].symbol == "S":
        s1 = t.children[0]
        if len(s1.children) == 1 and s1.children[0].symbol == "S":
            inner = s1.children[0]
            pat[tuple(c.symbol for c in inner.children)] += 1
print("S1->S->S children:", pat.most_common(8))

# how often does the inner S end with a punct leaf?
ends = Counter()
for t in parsed:
    if len(t.children) == 1 and t.children[0].symbol == "S":
        s1 = t.children[0]
        if len(s1.children) == 1 and s1.children[0].symbol == "S":
            inner = s1.children[0]
            if inner.children:
                last = inner.children[-1]
                ends[(last.symbol, last.word is not None)] += 1
print("inner S last child:", ends.most_common(5))

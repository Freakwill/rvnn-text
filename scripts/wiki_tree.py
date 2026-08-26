"""Pick a short, simple 6-word sentence from simplewiki.ptb and render its tree.

Usage: PYTHONPATH=. python3 scripts/wiki_tree.py
"""
import sys
from pathlib import Path

from rvnn_text.genia import _prune_punct, binarize, parse_ptb

sys.path.insert(0, str(Path(__file__).parent))
from ascii_tree import ascii_tree, words  # noqa: E402

DATA = Path("data/simplewiki.ptb")

cands = []
with DATA.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = parse_ptb(line)
        except Exception:
            continue
        if t.symbol != "ROOT":
            continue
        p = _prune_punct(t)
        if p is None:
            continue
        p = binarize(p)
        ws = words(p)
        if len(ws) == 6 and p.children and p.children[0].symbol == "S":
            cands.append((line, p, ws))

print(f"candidates (ROOT->S, 6 words): {len(cands)}")

# Score by word commonness: prefer frequent, lowercase, generic words.
import collections

freq = collections.Counter()
for _, _, ws in cands:
    for w in ws:
        freq[w] += 1


def score(ws):
    return sum(freq.get(w, 0) for w in ws) / 6 + 1.0 * sum(
        w.islower() and w.isalpha() for w in ws
    ) / 6


best = max(cands, key=lambda c: (score(c[2]), len(set(w.lower() for w in c[2]))))
raw, tree, ws = best
print("chosen:", " ".join(ws))
print("raw   :", raw)
print()
print(ascii_tree(tree))

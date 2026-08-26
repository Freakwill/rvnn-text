"""List all plain-pattern candidates from simplewiki.ptb (sorted)."""
import sys
from pathlib import Path

from rvnn_text.genia import _prune_punct, binarize, parse_ptb

sys.path.insert(0, str(Path(__file__).parent))
import wiki_tree  # noqa: E402
from ascii_tree import words  # noqa: E402

cands = []
with wiki_tree.DATA.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = parse_ptb(line)
        except Exception:
            continue
        p = _prune_punct(t)
        if p is None:
            continue
        p = binarize(p)
        if wiki_tree.match(p):
            cands.append((line, p, words(p)))

for line, tree, ws in sorted(cands, key=lambda c: str(c[2])):
    v = tree.children[0].children[1].children[0].word
    print(f"{v:8s} | {' '.join(ws)}")

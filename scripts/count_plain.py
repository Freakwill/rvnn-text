"""Count plain-style sentences in simplewiki.ptb under the exp-4 filter."""
import sys
from pathlib import Path

from rvnn_text.genia import _prune_punct, binarize, parse_ptb

sys.path.insert(0, str(Path(__file__).parent))
from ascii_tree import words  # noqa: E402

DATA = Path("data/simplewiki.ptb")

CLAUSE = {"S", "SBAR", "SBARQ", "SINV", "SQ", "PRN", "UCP", "FRAG", "RRC", "INTJ"}


def plain(t):
    if t.symbol != "ROOT" or len(t.children) != 1:
        return False
    s = t.children[0]
    if s.symbol != "S" or len(s.children) < 2:
        return False
    if s.children[0].symbol != "NP" or s.children[1].symbol != "VP":
        return False
    ws = words(s)
    if not (4 <= len(ws) <= 9):
        return False

    def check(n):
        if n.word is not None:
            return n.symbol != "NNP" and n.symbol != "NNPS" and not n.word.isdigit()
        if n is not s and n.symbol in CLAUSE:
            return False
        return all(check(c) for c in n.children)

    return check(s)


n_ok = n_all = 0
lengths = []
with DATA.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_all += 1
        try:
            t = parse_ptb(line)
        except Exception:
            continue
        p = _prune_punct(t)
        if p is None:
            continue
        p = binarize(p)
        if plain(p):
            n_ok += 1
            lengths.append(len(words(p)))

import statistics

print(f"total lines: {n_all}")
print(f"plain sentences: {n_ok} ({n_ok / n_all:.1%})")
print(f"length: median={statistics.median(lengths)} mean={statistics.mean(lengths):.1f}")

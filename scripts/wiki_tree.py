"""Pick a plain, template-style 6-word sentence from simplewiki.ptb.

Wanted shape (like "every clever girl likes a cat"):
  (ROOT (S (NP (DT *) (JJ *) (NN *)) (VP (VBP|VBZ *) (NP (DT *) (NN *)))))
Usage: PYTHONPATH=. python3 scripts/wiki_tree.py
"""
import sys
from pathlib import Path

from rvnn_text.genia import _prune_punct, binarize, parse_ptb

sys.path.insert(0, str(Path(__file__).parent))
from ascii_tree import ascii_tree, words  # noqa: E402

DATA = Path("data/simplewiki.ptb")


def simple_nn(n):
    """NN leaf, or binarized tail NP'(JJ, NN)."""
    if n.word is not None:
        return n.symbol in {"NN", "NNS"}
    return (
        n.symbol == "NP"
        and len(n.children) == 2
        and n.children[0].word is not None
        and n.children[0].symbol == "JJ"
        and n.children[1].word is not None
        and n.children[1].symbol in {"NN", "NNS"}
    )


def simple_np(n):
    """NP with shape DT NN or DT JJ NN (binarized right-branch)."""
    if n.symbol != "NP" or len(n.children) != 2:
        return False
    d, x = n.children
    return d.word is not None and d.symbol == "DT" and simple_nn(x)


def match(t):
    """ROOT -> S -> (plain-NP, VP(verb, plain-NP)): like 'every clever girl likes a cat'."""
    if t.symbol != "ROOT" or len(t.children) != 1:
        return False
    s = t.children[0]
    if s.symbol != "S" or len(s.children) != 2:
        return False
    subj, vp = s.children
    if not simple_np(subj) or vp.symbol != "VP" or len(vp.children) != 2:
        return False
    verb, obj = vp.children
    if verb.word is None or verb.symbol not in {"VBP", "VBZ"}:
        return False
    return simple_np(obj)


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
        p = _prune_punct(t)
        if p is None:
            continue
        p = binarize(p)
        if match(p):
            cands.append((line, p, words(p)))

print(f"plain-pattern candidates: {len(cands)}")

import collections

freq = collections.Counter()
for _, _, ws in cands:
    for w in ws:
        freq[w] += 1

ACTION = {
    "like", "likes", "want", "wants", "need", "needs", "use", "uses", "used",
    "see", "sees", "watch", "watches", "make", "makes", "made", "find", "finds",
    "love", "loves", "hate", "hates", "eat", "eats", "drink", "drinks", "buy",
    "buys", "sell", "sells", "take", "takes", "read", "reads",
    "write", "writes", "build", "builds", "play", "plays", "own", "owns",
    "span", "spans", "link", "links", "describe", "describes", "mean", "means",
}


def score(ws, verb):
    s = sum(freq.get(w, 0) for w in ws)
    s += 10 * sum(w.islower() and w.isalpha() for w in ws)
    s -= 3 * (len(set(ws)) != len(ws))
    if verb in ACTION:
        s += 60
    if verb in {"is", "are", "was", "were"}:
        s -= 120
    return s


def verb_of(tree):
    s = tree.children[0]
    return s.children[1].children[0].word


best = max(cands, key=lambda c: score(c[2], verb_of(c[1])))
for line, tree, ws in sorted(cands, key=lambda c: -score(c[2], verb_of(c[1])))[:5]:
    print(" top:", " ".join(ws), f"(verb={verb_of(tree)})")
print()
raw, tree, ws = best
print("chosen:", " ".join(ws))
print("raw   :", raw)
print()
print(ascii_tree(tree))

"""Pick a short typical GENIA sentence and print its tree (README data-format demo)."""
from pathlib import Path
from rvnn_text.genia import load_trees

trees = load_trees(Path("data/genia/genia-dist/division/train.trees"))
print("total:", len(trees))


def words(n):
    return [n.word] if n.word is not None else [w for c in n.children for w in words(c)]


def to_ptb(n):
    """Re-serialize a Node back to PTB format."""
    if n.word is not None:
        return f"({n.symbol} {n.word})"
    return "(" + n.symbol + " " + " ".join(to_ptb(c) for c in n.children) + ")"


def is_sentence(t):
    """Real sentence: S1-rooted, more than one word (skips section headers)."""
    return t.symbol == "S1" and len(words(t)) >= 2


cands = sorted((len(words(t)), i, t) for i, t in enumerate(trees) if is_sentence(t))
for L, _, t in cands[:12]:
    print(L, " ".join(words(t)))
print()
best = next(t for L, _, t in cands if L == 6)
print("chosen:", to_ptb(best))

"""Quick check: plain-sentence count + exemplars after the punct-word fix."""
from pathlib import Path

from rvnn_text.grammar import flatten
from rvnn_text.simplewiki import load_plain_trees

plain = load_plain_trees(Path("data/simplewiki/simplewiki.ptb"))
print("plain sentences after fix:", len(plain))
for t in plain[:4]:
    print("  exemplar:", " ".join(flatten(t)))

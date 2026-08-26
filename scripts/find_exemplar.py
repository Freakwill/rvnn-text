"""Find the index of the style exemplar among plain sentences."""
from pathlib import Path

from rvnn_text.grammar import flatten
from rvnn_text.simplewiki import load_plain_trees

plain = load_plain_trees(Path("data/simplewiki/simplewiki.ptb"))
target = ("a", "bridge", "spans", "this", "river")
for i, t in enumerate(plain):
    if tuple(w.lower() for w in flatten(t)) == target:
        print(f"'a bridge spans this river' at plain index {i}")
        break
else:
    print("NOT FOUND")
print("total plain:", len(plain))

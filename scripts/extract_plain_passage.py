"""Compose a plain-style paragraph from simplewiki.ptb: sample template sentences
spread across the whole file (each from a different region => different articles).

Style target: "A bridge spans this river" — plain S-V-O, no idioms, no rhetoric.
"""
import sys
from pathlib import Path

from rvnn_text.genia import _prune_punct, binarize, parse_ptb

sys.path.insert(0, str(Path(__file__).parent))
import wiki_tree  # noqa: E402
from ascii_tree import words  # noqa: E402

DATA = Path("data/simplewiki.ptb")
N = 8  # sentences in the paragraph

# First pass: collect (line_no, sentence, verb) for every plain-pattern match.
matches = []
with DATA.open() as f:
    for i, line in enumerate(f):
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
            v = p.children[0].children[1].children[0].word
            matches.append((i, " ".join(words(p)), v))

print(f"plain-pattern matches: {len(matches)}")

# Everyday-word filter: drop sentences with rare/obscure content words.
freq = {}
with DATA.open() as f:
    for i, line in enumerate(f):
        if i > matches[-1][0]:
            break
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
        for w in words(p):
            freq[w.lower()] = freq.get(w.lower(), 0) + 1

COMMON = {w for w, c in freq.items() if c >= 10}
if len(COMMON) < 2000:
    # partial file: fall back to the most frequent 60% of the vocab
    ranked = sorted(freq, key=freq.get, reverse=True)
    COMMON = set(ranked[: max(1500, int(0.6 * len(ranked)))])
print(f"common-words set: {len(COMMON)}")


def everyday(m):
    _, s, v = m
    ws = [w.lower() for w in s.split()]
    return v in wiki_tree.ACTION and all(w in COMMON for w in ws)


matches = [m for m in matches if everyday(m)]
print(f"after everyday filter: {len(matches)}")

# Pick N spread across the file, preferring action verbs.
total_lines = matches[-1][0] + 1 if matches else 1
picked = []
for k in range(N):
    lo = int(k / N * len(matches))
    hi = int((k + 1) / N * len(matches))
    seg = matches[lo:hi]
    if not seg:
        continue
    seg.sort(key=lambda m: (m[2] in wiki_tree.ACTION, m[0]))
    picked.append(seg[-1])

for i, s, v in picked:
    print(f"  line {i:7d} | {s}")
print()
print("PARAGRAPH:")
print(" ".join(s + "." for _, s, _ in picked))

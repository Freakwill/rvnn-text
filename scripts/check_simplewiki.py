"""Analyze simplewiki.ptb: tree count, roots, sentence lengths, vocab, samples."""
from collections import Counter
from pathlib import Path

from rvnn_text.genia import parse_ptb, _prune_punct, binarize

PATH = Path("data/simplewiki.ptb")
MAX_CHECK = 200_000  # cap for speed

n = 0
roots = Counter()
lengths = []
vocab: Counter = Counter()
sample: list[str] = []


def words(n):
    return [n.word] if n.word is not None else [w for c in n.children for w in words(c)]


with PATH.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            t = parse_ptb(line)
        except Exception:
            continue  # tolerate truncated last line in a partial download
        roots[t.symbol] += 1
        t = _prune_punct(t)
        if t is None:
            continue
        t = binarize(t)
        ws = words(t)
        lengths.append(len(ws))
        for w in ws:
            vocab[w] += 1
        if len(ws) <= 8 and len(sample) < 6:
            sample.append(" ".join(ws))
        n += 1
        if n >= MAX_CHECK:
            break

print(f"trees scanned: {n}")
print("roots:", roots.most_common(5))
import statistics

print(
    f"len: min={min(lengths)} median={statistics.median(lengths)} "
    f"mean={statistics.mean(lengths):.1f} max={max(lengths)}"
)
print("vocab size:", len(vocab), "| top:", vocab.most_common(10))
print("short samples:")
for s in sample:
    print("  ", s)

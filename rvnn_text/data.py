"""Synthetic-corpus construction for the RvNN text model.

The model consumes parse *trees*, not flat sequences. Because batching
recursive structures of differing shapes is non-trivial (each tree has its own
topology), training iterates over trees one at a time. The corpus here is
generated from a small context-free grammar; real corpora (e.g. the Stanford
Sentiment Treebank) can be swapped in by converting their parse trees to the
:class:`rvnn_text.grammar.Node` representation.
"""

from __future__ import annotations

import random

from .grammar import Grammar, Node


def make_corpus(grammar: Grammar, n: int, seed: int = 0) -> list[Node]:
    """Sample ``n`` parse trees from the grammar."""
    rng = random.Random(seed)
    return [grammar.sample_tree(rng) for _ in range(n)]


def train_test_split(
    trees: list[Node], val_frac: float = 0.1, seed: int = 0
) -> tuple[list[Node], list[Node]]:
    """Shuffle and split trees into train/validation sets."""
    rng = random.Random(seed)
    idx = list(range(len(trees)))
    rng.shuffle(idx)
    n_val = int(len(trees) * val_frac)
    val = [trees[i] for i in idx[:n_val]]
    train = [trees[i] for i in idx[n_val:]]
    return train, val

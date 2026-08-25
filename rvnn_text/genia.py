"""GENIA Treebank experiment — structure-constrained generation on real data.

Experiment 2 (SST-5) showed that generation collapses into word soup when the
trees carry *no syntactic categories*: without a grammar that constrains the
recursive expansion, the decoder has no structure prior.  The GENIA Treebank
fixes exactly that — real PTB-style constituency trees **with full category
labels** (``S``, ``NP``, ``VP``, ``PP``, ...) over biomedical English.

This module induces the grammar from the treebank and trains the same
recursive autoencoder as experiment 1 (rule/word cross-entropy + reconstruction
+ masked leaves), then demonstrates structure-constrained generation, cloze and
auto-completion on *real* sentences.

Data: GENIA Treebank 1.0 (biomedical Medline abstracts), PTB conversion by
Illes Solt, parsed divisions by James Clarke — distributed from
<http://bllip.cs.brown.edu/download/genia1.0-division-rel1.tar.gz>
(provenance: GENIA Treebank project <http://www.geniaproject.org/>).
14,326 train / 1,361 dev / 1,360 test / 1,494 future-use sentences.

Run::

    python -m rvnn_text.genia --max_train 2500 --epochs 5
"""

from __future__ import annotations

import random
import tarfile
import urllib.request
import zipfile  # noqa: F401  (kept for parity with sst.ensure_data)
from pathlib import Path

import fire

from .grammar import Grammar, Node, flatten, to_sentence
from .sst import binarize, build_vocab, map_oov
from .train import train_model
from .utils import get_device, set_seed

URL = "http://bllip.cs.brown.edu/download/genia1.0-division-rel1.tar.gz"
UNK = "<unk>"
MAX_WORDS = 60  # skip very long sentences for training speed

# Punctuation POS tags are dropped in preprocessing (standard treebank
# practice): they carry no lexical content, and symbols like ``.`` or ``,``
# collide with nn.ModuleDict key rules anyway.
PUNCT_POS = {"''", ",", "-LRB-", "-RRB-", ".", ":", "``"}


# -- data loading ----------------------------------------------------------

def ensure_data(data_dir: str | Path) -> Path:
    """Download (if needed) and extract the GENIA tree divisions."""
    data_dir = Path(data_dir)
    division = data_dir / "genia-dist" / "division"
    if (division / "train.trees").exists():
        return division
    data_dir.mkdir(parents=True, exist_ok=True)
    tarball = data_dir / "genia1.0-division-rel1.tar.gz"
    if not tarball.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, tarball)
    with tarfile.open(tarball) as tf:
        tf.extractall(data_dir)
    return division


def parse_ptb(text: str) -> Node:
    """Parse one parenthesized PTB tree ``(S (NP (DT the) (NN cat)) ...)``.

    Every node carries a syntactic category: internal nodes have children,
    leaves are ``(POS word)``.
    """

    def rec(i: int) -> tuple[Node, int]:
        assert text[i] == "(", f"expected '(' at {i}: {text[i:i+20]!r}"
        i += 1
        j = i
        while not text[j].isspace() and text[j] not in "()":
            j += 1
        symbol = text[i:j]
        i = j
        while text[i].isspace():
            i += 1
        if text[i] == "(":                       # internal node
            children: list[Node] = []
            while text[i] == "(":
                child, i = rec(i)
                children.append(child)
                while text[i].isspace():
                    i += 1
            assert text[i] == ")", f"expected ')' at {i}"
            return Node(symbol, children=children), i + 1
        j = i                                     # leaf: word token
        while text[j] not in " \t\n)":
            j += 1
        word = text[i:j]
        while text[j].isspace():
            j += 1
        assert text[j] == ")"
        return Node(symbol, word=word), j + 1

    root, _ = rec(0)
    return root


def _prune_punct(node: Node) -> Node | None:
    """Remove punctuation leaves (and nodes that become empty) in place."""
    if node.is_leaf:
        return None if node.symbol in PUNCT_POS else node
    children = [c for c in (_prune_punct(c) for c in node.children) if c is not None]
    if not children:
        return None
    node.children = children
    return node


def load_trees(path: Path, max_words: int = MAX_WORDS) -> list[Node]:
    """Load, prune punctuation and binarize GENIA trees; drop very long ones."""
    trees: list[Node] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        tree = _prune_punct(parse_ptb(line.strip()))
        if tree is None:
            continue
        tree = binarize(tree)
        if len(flatten(tree)) <= max_words:
            trees.append(tree)
    return trees


def build_grammar(train_trees: list[Node]) -> Grammar:
    """Induce a grammar from the training trees (category labels included).

    Internal categories keep their real labels (``S1``, ``NP``, ``VP``, ...);
    every POS tag that appears as a leaf symbol becomes its own preterminal
    (``NN -> cat``, ``DT -> the``, ...), and ``<unk>`` is available under every
    preterminal so OOV words stay encodable.
    """
    productions: dict[str, set[tuple[str, ...]]] = {}
    preterminal_words: dict[str, set[str]] = {}

    def walk(node: Node) -> None:
        if node.is_leaf:
            preterminal_words.setdefault(node.symbol, set()).add(node.word)  # type: ignore[arg-type]
            return
        productions.setdefault(node.symbol, set()).add(
            tuple(c.symbol for c in node.children))
        for c in node.children:
            walk(c)

    for tree in train_trees:
        walk(tree)
    raw = {k: sorted(v) for k, v in productions.items()}
    leaf_symbols = set(preterminal_words)
    for sym, ws in preterminal_words.items():
        raw[sym] = sorted((w,) for w in (ws | {UNK}))
    # sanity: no symbol is both an internal category and a leaf symbol
    mixed = set(productions) & leaf_symbols
    if mixed:
        raise ValueError(f"symbols are both internal and leaf: {sorted(mixed)}")
    return Grammar(raw, start=train_trees[0].symbol, preterminals=sorted(leaf_symbols))


# -- generation helpers (trees are already parsed; no left-recursive parsing) -

def cloze_tree(model, tree: Node, mask_indices: list[int],
               greedy: bool = True) -> tuple[str, str, str]:
    """Cloze on an already-parsed tree (bypasses Grammar.parse)."""
    original, filled, markup = model._fill_masked(tree, mask_indices, greedy, 1.0)
    return " ".join(markup), " ".join(filled), " ".join(original)


def continue_tree(model, tree: Node, keep: int,
                  greedy: bool = True) -> tuple[str, str, str]:
    """Auto-completion on an already-parsed tree."""
    n = len(model._leaf_pairs(tree))
    original, filled, markup = model._fill_masked(
        tree, list(range(keep, n)), greedy, 1.0)
    return " ".join(markup), " ".join(filled), " ".join(original)


# -- CLI -------------------------------------------------------------------

def main(
    data_dir: str = "data/genia",
    max_train: int = 2500,
    max_dev: int = 400,
    dim: int = 64,
    epochs: int = 5,
    lr: float = 1e-3,
    mask_frac: float = 0.15,
    min_word_count: int = 2,
    n_samples: int = 5,
    n_tasks: int = 3,
    seed: int = 42,
) -> None:
    """Train a structure-constrained RvNN on the GENIA treebank and generate."""
    set_seed(seed)
    device = get_device()
    division = ensure_data(data_dir)

    train_all = load_trees(division / "train.trees")
    dev_all = load_trees(division / "dev.trees")
    print(f"GENIA loaded: {len(train_all)} train / {len(dev_all)} dev trees, device={device}")
    print(f"using subset: {max_train} train / {max_dev} dev, dim={dim}")

    train_trees = train_all[:max_train]
    dev_trees = dev_all[:max_dev]
    vocab = build_vocab(train_trees, min_word_count)
    for tree in train_trees + dev_trees:
        map_oov(tree, vocab)
    grammar = build_grammar(train_trees)
    n_cats = len(grammar.productions)
    n_rules = sum(len(r) for r in grammar.productions.values())
    print(f"vocab={len(vocab)} (min_count={min_word_count} + <unk>), "
          f"categories={n_cats}, rules={n_rules}, start={grammar.start}")

    model, losses = train_model(
        grammar, train_trees, val_trees=dev_trees, dim=dim, epochs=epochs,
        lr=lr, mask_frac=mask_frac, device=device, sample_interval=epochs,
    )
    model.eval()
    acc = model.evaluate(dev_trees)
    print(f"\nheld-out rule-prediction accuracy: {acc['rule_acc']:.1%}")
    print(f"held-out word-prediction accuracy: {acc['word_acc']:.1%}")

    print()
    print("=" * 70)
    print("Structure-constrained generation (real grammar, novel sentences)")
    print("=" * 70)
    for _ in range(n_samples):
        tree = model.generate(seed_noise=1.0, max_depth=14, temperature=0.8)
        print(f"  {to_sentence(tree)}")

    print()
    print("=" * 70)
    print("Cloze on real GENIA sentences (mask -> encode-decode fill)")
    print("=" * 70)
    for tree in dev_trees[:n_tasks]:
        idx = max(1, len(flatten(tree)) // 2)
        masked, filled, _ = cloze_tree(model, tree, [idx])
        print(f"  masked:  {masked}")
        print(f"  filled:  {filled}")
        print()

    print("=" * 70)
    print("Auto-completion on real GENIA sentences (mask the tail)")
    print("=" * 70)
    for tree in dev_trees[:n_tasks]:
        n = len(flatten(tree))
        keep = n // 2
        masked, filled, _ = continue_tree(model, tree, keep)
        print(f"  masked:  {masked}")
        print(f"  filled:  {filled}")
        print()


if __name__ == "__main__":
    fire.Fire(main)


def cli() -> None:
    """Console-script entry point."""
    fire.Fire(main)

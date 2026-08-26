"""Simple Wikipedia experiment — RvNN encode-decode generates plain sentences.

Style target: ``A bridge spans this river`` — short, template-style S-V-O
sentences built from everyday words (no idioms, no rhetoric).  The training
set is a filtered subset of ``simplewiki.ptb``: ROOT->S sentences of 4..9
words with *no embedded clauses, no proper nouns, no numbers* — so the
grammar induced from it *is* the plain style, and the trained recursive
autoencoder generates new sentences in the same style.

The paragraph produced at the end is verified (token-sequence match) not to
appear anywhere in the corpus — i.e. it is a genuinely novel composition,
not a copy of a training sentence.

Data: parsed Simple English Wikipedia (Brown BLLIP) — one PTB tree per line,
<http://bllip.cs.brown.edu/download/simplewiki.ptb>
(Simple English Wikipedia: <https://simple.wikipedia.org/>).

Run::

    python -m rvnn_text.simplewiki --max_train 2500 --epochs 8
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import fire

from .genia import PUNCT_POS, _prune_punct, build_grammar, parse_ptb
from .grammar import flatten
from .sst import binarize, build_vocab, map_oov
from .train import train_model
from .utils import get_device, set_seed

URL = "http://bllip.cs.brown.edu/download/simplewiki.ptb"
UNK = "<unk>"

# Clause-level categories: their presence means the sentence is not a plain
# single clause (embedded/subordinate clauses, parentheticals, fragments...).
CLAUSE_CATS = {"S", "SBAR", "SBARQ", "SINV", "SQ", "PRN", "UCP", "FRAG", "RRC", "INTJ"}
# Leaves that make a sentence non-plain (proper nouns, numbers, quotes).
NONPLAIN_POS = {"NNP", "NNPS", "CD", "POS", "SYM"}

MIN_WORDS = 4
MAX_WORDS = 9


# -- data loading ----------------------------------------------------------

def ensure_data(data_dir: str | Path) -> Path:
    """Download (if needed) the parsed Simple Wikipedia corpus."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "simplewiki.ptb"
    if not path.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    return path


def is_plain(tree) -> bool:
    """ROOT->S sentence of 4..9 words, single clause, no proper nouns."""
    if tree.symbol != "ROOT" or len(tree.children) != 1:
        return False
    s = tree.children[0]
    if s.symbol != "S" or len(s.children) < 2:
        return False
    if s.children[0].symbol != "NP" or s.children[1].symbol != "VP":
        return False
    if not (MIN_WORDS <= len(flatten(s)) <= MAX_WORDS):
        return False

    def check(n) -> bool:
        if n.is_leaf:
            # reject proper nouns / numbers, and words that are punctuation
            # chars miscast as content POS by the parser (e.g. (JJ '')).
            return n.symbol not in NONPLAIN_POS and n.word not in PUNCT_POS
        if n is not s and n.symbol in CLAUSE_CATS:
            return False
        return all(check(c) for c in n.children)

    return check(s)


def load_plain_trees(path: Path) -> list:
    """Parse, clean and binarize the corpus, keeping only plain sentences."""
    trees = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            tree = parse_ptb(line.strip())
        except Exception:
            continue
        tree = _prune_punct(tree)
        if tree is None:
            continue
        tree = binarize(tree)
        if is_plain(tree):
            trees.append(tree)
    return trees


def corpus_sentences(path: Path) -> set[tuple[str, ...]]:
    """All leaf-word sequences in the corpus (lowercased), for novelty checks."""
    seen: set[tuple[str, ...]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            tree = parse_ptb(line.strip())
        except Exception:
            continue
        tree = _prune_punct(tree)
        if tree is None:
            continue
        seen.add(tuple(w.lower() for w in flatten(tree)))
    return seen


def to_sentence(tree) -> str:
    return " ".join(flatten(tree))


# -- CLI -------------------------------------------------------------------

def main(
    data_dir: str = "data/simplewiki",
    max_train: int = 2500,
    max_dev: int = 500,
    dim: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    mask_frac: float = 0.15,
    min_word_count: int = 2,
    n_sentences: int = 6,
    max_depth: int = 12,
    temperature: float = 0.8,
    seed: int = 42,
) -> None:
    """Train a plain-style RvNN on Simple Wikipedia and generate a paragraph."""
    set_seed(seed)
    device = get_device()
    data_path = ensure_data(data_dir)

    plain = load_plain_trees(data_path)
    print(f"simplewiki loaded: {len(plain)} plain sentences, device={device}")
    print(f"using subset: {max_train} train / {max_dev} dev, dim={dim}")
    print("style exemplars from the corpus:")
    for t in plain[:3]:
        print(f"  {to_sentence(t)}")

    train_trees = plain[:max_train]
    dev_trees = plain[max_train:max_train + max_dev]
    vocab = build_vocab(train_trees, min_word_count)
    for tree in train_trees + dev_trees:
        map_oov(tree, vocab)
    grammar = build_grammar(train_trees)
    n_rules = sum(len(r) for r in grammar.productions.values())
    print(f"vocab={len(vocab)} (min_count={min_word_count} + <unk>), "
          f"categories={len(grammar.productions)}, rules={n_rules}, start={grammar.start}")

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
    print("RvNN encode-decode generation — novel paragraph (plain style)")
    print("=" * 70)

    corpus = corpus_sentences(data_path)
    train_seen = {tuple(w.lower() for w in flatten(t)) for t in train_trees}
    sents = []
    for _ in range(n_sentences):
        tree = model.generate(seed_noise=1.0, max_depth=max_depth,
                              temperature=temperature)
        sents.append(to_sentence(tree))
    paragraph = ". ".join(s + "." for s in sents)
    for s in sents:
        toks = tuple(s.lower().split())
        in_train = toks in train_seen
        in_corpus = toks in corpus
        status = "novel" if not in_corpus else ("in-train" if in_train else "in-corpus")
        print(f"  [{status:9s}] {s}")
    print()
    print("paragraph:")
    print(f"  {paragraph}")

    exemplar = "a bridge spans this river"
    in_train = tuple(exemplar.split()) in train_seen
    print(f"\nexemplar '{exemplar}' in training set: {in_train}")


def cli():
    fire.Fire(main)


if __name__ == "__main__":
    cli()

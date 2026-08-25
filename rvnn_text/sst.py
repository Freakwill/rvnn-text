"""SST-5 sentiment experiment — the classic *discriminative* RvNN task.

The Stanford Sentiment Treebank (Socher et al., 2013) annotates every node of
a constituency tree with a 0-4 sentiment label.  The canonical RvNN recipe is
exactly what this module does: parse the (label-only) trees, binarize them,
induce a generic grammar (every internal node is the same symbol ``N``), and
train the shared recursive composition function with per-node sentiment
cross-entropy.  The root embedding then carries sentence sentiment, and root
accuracy is the reported metric.

Data: ``trainDevTestTrees_PTB.zip`` from
<https://nlp.stanford.edu/sentiment/> — parenthesized trees with per-node
sentiment labels (0-4), 8544 train / 1101 dev / 2210 test sentences.
Auto-downloaded on first run into ``--data_dir``.

Run::

    python -m rvnn_text.sst --max_train 1500 --max_dev 400 --epochs 5
"""

from __future__ import annotations

import random
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grammar import Grammar, Node, flatten
from .model import RvNNText
from .utils import get_device, set_seed

URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
INTERNAL = "N"   # generic symbol for every internal node (label-only trees)
PRETERM = "W"    # generic preterminal symbol for every word leaf
UNK = "<unk>"


# -- data loading ----------------------------------------------------------

def ensure_data(data_dir: str | Path) -> Path:
    """Download (if needed) and extract the SST trees; return the trees dir."""
    data_dir = Path(data_dir)
    trees_dir = data_dir / "trees"
    if (trees_dir / "train.txt").exists():
        return trees_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "trees.zip"
    if not zip_path.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return trees_dir


def parse_tree(text: str) -> Node:
    """Parse one parenthesized SST tree ``(label (child) ...)`` into a Node.

    Leaves are ``(label word)``; internal nodes carry no syntactic category,
    so every internal node gets the generic symbol ``N`` and every leaf the
    preterminal ``W``.
    """

    def rec(i: int) -> tuple[Node, int]:
        assert text[i] == "(", f"expected '(' at {i}: {text[i:i+20]!r}"
        i += 1
        j = i
        while text[j].isdigit():
            j += 1
        label = int(text[i:j])
        i = j
        while text[i].isspace():
            i += 1
        if text[i] == "(":                      # internal node
            children: list[Node] = []
            while text[i] == "(":
                child, i = rec(i)
                children.append(child)
                while text[i].isspace():
                    i += 1
            assert text[i] == ")", f"expected ')' at {i}"
            return Node(INTERNAL, children=children, label=label), i + 1
        j = i                                    # leaf: word token
        while text[j] not in " \t\n)":
            j += 1
        word = text[i:j]
        while text[j].isspace():
            j += 1
        assert text[j] == ")"
        return Node(PRETERM, word=word, label=label), j + 1

    root, _ = rec(0)
    if root.is_leaf:                            # single-word sentence
        root = Node(INTERNAL, children=[root], label=root.label)
    return root


def load_trees(path: Path) -> list[Node]:
    """Load all trees from an SST split file."""
    return [parse_tree(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def binarize(node: Node) -> Node:
    """Right-branching binarization; new inner nodes inherit the parent label.

    The model composes binary nodes only, so n-ary nodes from the real
    treebank are folded into a right spine, e.g. ``N(c1 c2 c3)`` becomes
    ``N(c1, N(c2, c3))``.
    """
    if node.is_leaf:
        return node
    children = [binarize(c) for c in node.children]
    if len(children) <= 2:
        return Node(node.symbol, children=children, label=node.label)
    tail = Node(node.symbol, children=children[-2:], label=node.label)
    for c in reversed(children[:-2]):
        tail = Node(node.symbol, children=[c, tail], label=node.label)
    return tail


def build_vocab(train_trees: list[Node], min_count: int = 2) -> dict[str, int]:
    """Word -> id for frequent words plus a shared ``<unk>`` token."""
    counts: Counter[str] = Counter()
    for tree in train_trees:
        counts.update(w for w in flatten(tree))
    words = [UNK] + sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i for i, w in enumerate(words)}


def map_oov(tree: Node, vocab: dict[str, int]) -> None:
    """Rewrite OOV leaf words to ``<unk>`` in place."""
    if tree.is_leaf:
        if tree.word not in vocab:
            tree.word = UNK
        return
    for c in tree.children:
        map_oov(c, vocab)


def build_grammar(train_trees: list[Node]) -> Grammar:
    """Induce a generic binary grammar from the binarized training trees.

    ``N -> X Y`` for every occurring child pair (plus unary ``N -> W`` for
    single-word sentences) and ``W -> <word>`` for every word.
    """
    n_rules: set[tuple[str, ...]] = {("W",)}
    words: set[str] = set()

    def walk(node: Node) -> None:
        if node.is_leaf:
            words.add(node.word)  # type: ignore[arg-type]
            return
        n_rules.add(tuple(c.symbol for c in node.children))
        for c in node.children:
            walk(c)

    for tree in train_trees:
        walk(tree)
    productions = {INTERNAL: sorted(n_rules), PRETERM: sorted((w,) for w in words)}
    return Grammar(productions, start=INTERNAL)


# -- model -----------------------------------------------------------------

class SentimentRvNN(nn.Module):
    """RvNN encoder + root/global sentiment head (Socher-style discriminative RvNN)."""

    def __init__(self, grammar: Grammar, dim: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = RvNNText(grammar, dim=dim)
        self.head = nn.Linear(dim, num_classes)

    def _nodes(self, root: Node):
        yield root
        for c in root.children:
            yield from self._nodes(c)

    def loss(self, root: Node) -> torch.Tensor:
        """Mean sentiment cross-entropy over all labelled nodes."""
        hs = self.encoder._encode_tree(root)
        total = torch.zeros((), device=self.encoder.word_emb.weight.device)
        n = 0
        for node in self._nodes(root):
            if node.label is None:
                continue
            logits = self.head(hs[id(node)]).unsqueeze(0)
            target = torch.tensor([node.label], device=logits.device)
            total = total + F.cross_entropy(logits, target)
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def predict_root(self, root: Node) -> int:
        h = self.encoder.encode(root)
        return int(self.head(h).argmax().item())


# -- training --------------------------------------------------------------

def train_model(
    model: SentimentRvNN,
    train_trees: list[Node],
    dev_trees: list[Node],
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Train the sentiment RvNN; report dev root accuracy each epoch."""
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    accs: list[float] = []
    for epoch in range(1, epochs + 1):
        trees = train_trees[:]
        random.Random(epoch).shuffle(trees)
        epoch_loss = 0.0
        for tree in trees:
            opt.zero_grad(set_to_none=True)
            loss = model.loss(tree)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += loss.item()
        acc = accuracy(model, dev_trees, device)
        accs.append(acc)
        print(f"epoch {epoch:2d}/{epochs}  loss={epoch_loss / len(trees):.4f}  "
              f"dev_root_acc={acc:.2%}")
    return accs


@torch.no_grad()
def accuracy(model: SentimentRvNN, trees: list[Node], device: torch.device) -> float:
    model.eval()
    correct = sum(model.predict_root(t) == t.label for t in trees)
    model.train()
    return correct / max(len(trees), 1)


# -- CLI -------------------------------------------------------------------

def main(
    data_dir: str = "data/sst",
    max_train: int = 1500,
    max_dev: int = 400,
    dim: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    min_word_count: int = 2,
    seed: int = 42,
) -> None:
    """Train an RvNN sentiment classifier on a subset of SST-5."""
    set_seed(seed)
    device = get_device()
    trees_dir = ensure_data(data_dir)

    train_all = [binarize(t) for t in load_trees(trees_dir / "train.txt")]
    dev_all = [binarize(t) for t in load_trees(trees_dir / "dev.txt")]
    print(f"SST-5 loaded: {len(train_all)} train / {len(dev_all)} dev trees, device={device}")
    print(f"using subset: {max_train} train / {max_dev} dev, dim={dim}")

    train_trees = train_all[:max_train]
    dev_trees = dev_all[:max_dev]
    vocab = build_vocab(train_trees, min_word_count)
    for tree in train_trees + dev_trees:
        map_oov(tree, vocab)
    grammar = build_grammar(train_trees)
    print(f"vocab={len(vocab)} (words >= {min_word_count} occurrences + <unk>), "
          f"grammar rules: N={len(grammar.productions['N'])}")

    model = SentimentRvNN(grammar, dim=dim)
    train_model(model, train_trees, dev_trees, epochs=epochs, lr=lr, device=device)

    # majority-class baseline
    majority = Counter(t.label for t in train_trees).most_common(1)[0][0]
    base = sum(t.label == majority for t in dev_trees) / len(dev_trees)
    final = accuracy(model, dev_trees, device)
    print(f"\nresults (root accuracy on {len(dev_trees)} dev sentences):")
    print(f"  majority-class baseline: {base:.2%} (always predict class {majority})")
    print(f"  RvNN sentiment accuracy: {final:.2%}")

    print("\nsample predictions (dev):")
    for tree in dev_trees[:10]:
        pred = model.predict_root(tree)
        mark = "ok " if pred == tree.label else "mis"
        sent = " ".join(flatten(tree))
        print(f"  [{mark}] {sent[:58]:58s} true={tree.label} pred={pred}")


if __name__ == "__main__":
    fire.Fire(main)


def cli() -> None:
    """Console-script entry point."""
    fire.Fire(main)

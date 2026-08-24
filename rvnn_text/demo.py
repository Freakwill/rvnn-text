"""End-to-end demo: grammar -> train -> novel generation -> cloze -> 续写.

Runs in *story mode*: the corpus consists of multi-sentence paragraphs
(``Story -> S | Story S``), so the RvNN is recursive at the paragraph level.
After training it demonstrates (a) grammar-constrained generation of *novel*
stories, (b) cloze — masked words reconstructed via encode -> decode, and
(c) auto-completion — the tail of a story masked and filled in.
"""

from __future__ import annotations

import fire

from .data import make_corpus, train_test_split
from .grammar import STORY_PRODUCTIONS, Grammar, Node, render, to_sentence
from .train import train_model
from .utils import get_device, set_seed


def _paragraph(tree: Node) -> str:
    """Render a Story tree as a readable paragraph (sentences joined with '.')."""
    sents: list[str] = []

    def rec(n: Node) -> None:
        if n.symbol == "S":
            sents.append(to_sentence(n))
            return
        for c in n.children:
            rec(c)

    rec(tree)
    return ". ".join(sents) + "."


def main(
    num_sentences: int = 1500,
    epochs: int = 20,
    dim: int = 64,
    seed: int = 42,
    n_samples: int = 6,
    mask_frac: float = 0.2,
) -> None:
    """Run a complete walkthrough of the RvNN text model (story mode)."""
    set_seed(seed)
    device = get_device()

    print("=" * 70)
    print("1. Context-free grammar (story mode)")
    print("=" * 70)
    grammar = Grammar(STORY_PRODUCTIONS, start="Story")
    for lhs in grammar.nonterminals:
        rules = " | ".join(" ".join(rhs) for rhs in grammar.productions[lhs])
        print(f"  {lhs:8s} -> {rules}")

    print()
    print("=" * 70)
    print("2. Sampled parse trees (training data = synthetic stories)")
    print("=" * 70)
    for tree in make_corpus(grammar, 3, seed=1):
        print(render(tree))
        print(f"   story: {_paragraph(tree)}")
        print()

    print("=" * 70)
    print(f"3. Training the RvNN (device={device}, mask_frac={mask_frac})")
    print("=" * 70)
    corpus = make_corpus(grammar, num_sentences, seed=seed)
    train_trees, val_trees = train_test_split(corpus, val_frac=0.1, seed=seed)
    model, _ = train_model(
        grammar,
        train_trees,
        val_trees=val_trees,
        dim=dim,
        epochs=epochs,
        device=device,
        mask_frac=mask_frac,
        sample_interval=epochs,
    )
    model.eval()

    print()
    print("=" * 70)
    print("3b. How well did the RvNN learn the grammar?")
    print("=" * 70)
    acc = model.evaluate(val_trees)
    print(f"  held-out rule-prediction accuracy: {acc['rule_acc']:.1%}")
    print(f"  held-out word-prediction accuracy: {acc['word_acc']:.1%}")

    print()
    print("=" * 70)
    print("4. Generated stories (novel sentences, grammar-constrained)")
    print("=" * 70)
    for i in range(n_samples):
        tree = model.generate(seed_noise=1.0)
        print(f"  {_paragraph(tree)}")
        if i == 0:
            print(render(tree))
            print()

    print()
    print("=" * 70)
    print("5. Cloze - mask words, reconstruct via encode -> decode")
    print("=" * 70)
    examples = [
        ("the happy cat sees a dog", [1]),
        ("every clever girl likes the robot", [3]),
        ("a red tree chases quickly", [1]),
        ("Mary likes a small clever cat", [2]),
    ]
    for sent, idxs in examples:
        r = model.cloze(sent, mask_indices=idxs, greedy=True)
        if r is None:
            print(f"  cannot parse: {sent}")
            continue
        note = "" if r["output"] == r["original"] else "   <- different from original!"
        print(f"  masked:  {r['input']}")
        print(f"  filled:  {r['output']}{note}")
        print()

    r = model.cloze("the happy cat sees a dog", mask_indices=[1], greedy=False, temperature=0.5)
    if r:
        print(f"  sampled fill (temperature=0.5): {r['output']}")
        print()

    print("=" * 70)
    print("6. Auto-completion - mask the tail, reconstruct (continuation)")
    print("=" * 70)
    cont_examples = [
        ("the happy small clever cat sees a red tree", 5),
        ("Alice likes a cat Bob chases the dog", 4),
    ]
    for sent, keep in cont_examples:
        r = model.continue_sentence(sent, keep=keep, greedy=True)
        if r is None:
            print(f"  cannot parse: {sent}")
            continue
        note = "" if r["output"] == r["original"] else "   <- different from original!"
        print(f"  story:     {r['original']}")
        print(f"  masked:    {r['input']}")
        print(f"  completed: {r['output']}{note}")
        print()


if __name__ == "__main__":
    fire.Fire(main)


def cli() -> None:
    """Console-script entry point."""
    fire.Fire(main)

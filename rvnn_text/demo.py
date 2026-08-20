"""End-to-end demo: grammar -> train -> generate -> parse/encode -> regenerate."""

from __future__ import annotations

import fire

from .data import make_corpus, train_test_split
from .grammar import DEFAULT_PRODUCTIONS, Grammar, render, to_sentence
from .train import train_model
from .utils import get_device, set_seed


def main(
    num_sentences: int = 1500,
    epochs: int = 20,
    dim: int = 64,
    seed: int = 42,
    n_samples: int = 6,
) -> None:
    """Run a complete walkthrough of the RvNN text model."""
    set_seed(seed)
    device = get_device()

    print("=" * 70)
    print("1. Context-free grammar")
    print("=" * 70)
    grammar = Grammar(DEFAULT_PRODUCTIONS)
    for lhs in grammar.nonterminals:
        rules = " | ".join(" ".join(rhs) for rhs in grammar.productions[lhs])
        print(f"  {lhs:8s} -> {rules}")

    print()
    print("=" * 70)
    print("2. Sampled parse trees (training data)")
    print("=" * 70)
    for tree in make_corpus(grammar, 3, seed=1):
        print(render(tree))
        print(f"   sentence: {to_sentence(tree)}")
        print()

    print("=" * 70)
    print(f"3. Training the RvNN (device={device})")
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
    print("4. Generated sentences (grammar-constrained)")
    print("=" * 70)
    for _ in range(n_samples):
        tree = model.generate(seed_noise=1.0)
        print(f"  {to_sentence(tree)}")
        print(render(tree))
        print()

    print("=" * 70)
    print("5. Parse -> encode -> regenerate (round-trip)")
    print("=" * 70)
    for sent in ["the happy cat sees a dog", "every clever girl likes the robot"]:
        root = model.encode_sentence(sent)
        if root is None:
            print(f"  cannot parse: {sent}")
            continue
        regen = model.generate(root_embedding=root, greedy=True)
        print(f"  input:  {sent}")
        print(f"  output: {to_sentence(regen)}")
        print()


if __name__ == "__main__":
    fire.Fire(main)


def cli() -> None:
    """Console-script entry point."""
    fire.Fire(main)

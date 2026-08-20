"""Generate sentences from a trained RvNN text model (CLI via fire)."""

from __future__ import annotations

import fire

from .checkpoint import load_model
from .grammar import render, to_sentence
from .utils import get_device, set_seed


def main(
    checkpoint: str = "checkpoints/rvnn_text.pt",
    n: int = 8,
    temperature: float = 1.0,
    greedy: bool = False,
    show_tree: bool = True,
    max_depth: int = 12,
    seed: int = 42,
    from_sentence: str | None = None,
    seed_noise: float = 1.0,
) -> None:
    """Sample sentences from a trained model.

    Args:
        checkpoint: path to a ``.pt`` file saved by ``train.py``.
        n: number of sentences to generate.
        temperature: sampling temperature (lower = more conservative).
        greedy: take the most probable rule/word at every step.
        show_tree: also print the parse tree of each sentence.
        max_depth: maximum tree depth (caps the right-recursive NOM rule).
        seed: random seed for reproducible sampling.
        from_sentence: instead of sampling from the prior, encode this sentence
            and regenerate from its vector (paraphrase / round-trip demo).
        seed_noise: std of noise added to the start vector when sampling (gives
            diversity; ignored when ``from_sentence`` is set).
    """
    set_seed(seed)
    device = get_device()
    model = load_model(checkpoint, device)
    model.eval()

    for _ in range(n):
        if from_sentence is not None:
            root = model.encode_sentence(from_sentence)
            if root is None:
                raise ValueError(f"cannot parse sentence: {from_sentence!r}")
            tree = model.generate(
                temperature=temperature,
                greedy=greedy,
                max_depth=max_depth,
                root_embedding=root,
                seed_noise=seed_noise,
            )
        else:
            tree = model.generate(
                temperature=temperature,
                greedy=greedy,
                max_depth=max_depth,
                seed_noise=seed_noise,
            )
        print(to_sentence(tree))
        if show_tree:
            print(render(tree))
            print()


if __name__ == "__main__":
    fire.Fire(main)


def cli() -> None:
    """Console-script entry point."""
    fire.Fire(main)
